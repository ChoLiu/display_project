"""Real-time mic visualizer on the Display HAT Mini.

Audio comes from arecord (piped raw PCM), captured in a background thread.

  A = cycle mode (WAVEFORM / GROWTH / CONSTELLATION / SPECTROGRAM)
  X = jump to CONSTELLATION (re-press to respawn the network)
  Y = jump to SPECTROGRAM (re-press to clear the waterfall)
  B = pause display
  Ctrl-C to quit.
"""
import math
import subprocess
import sys
import threading
import time
import numpy as np
from PIL import Image, ImageDraw
from displayhatmini import DisplayHATMini

# Swallow rpi-lgpio's PWM.__del__ TypeError that fires during interpreter shutdown.
_orig_unraisable = sys.unraisablehook
def _quiet_unraisable(u):
    if isinstance(u.exc_value, TypeError) and "NoneType" in str(u.exc_value):
        return
    _orig_unraisable(u)
sys.unraisablehook = _quiet_unraisable

SAMPLE_RATE = 16000
CARD = "plughw:3,0"      # Anker PowerConf C300 capture
CHUNK = 512              # samples per arecord read (~32 ms at 16 kHz)
WINDOW = 2048            # samples retained in ring buffer for one frame

W, H = DisplayHATMini.WIDTH, DisplayHATMini.HEIGHT  # 320 x 240


class Capture:
    def __init__(self):
        self.lock = threading.Lock()
        self.ring = np.zeros(WINDOW, dtype=np.int16)
        self.proc = subprocess.Popen(
            ["arecord", "-D", CARD, "-f", "S16_LE", "-c", "1",
             "-r", str(SAMPLE_RATE), "-t", "raw", "-q"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
        )
        self.alive = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.alive:
            raw = self.proc.stdout.read(CHUNK * 2)
            if not raw:
                break
            samples = np.frombuffer(raw, dtype=np.int16)
            with self.lock:
                self.ring = np.roll(self.ring, -len(samples))
                self.ring[-len(samples):] = samples

    def snapshot(self):
        with self.lock:
            return self.ring.copy()

    def close(self):
        self.alive = False
        try:
            self.proc.terminate()
        except Exception:
            pass


def draw_waveform(draw, samples):
    n = len(samples)
    idx = np.linspace(0, n - 1, W).astype(np.int32)
    sub = samples[idx].astype(np.int32)
    ys = ((32768 - sub) * (H - 1) // 65535).tolist()
    draw.line([(0, H // 2), (W - 1, H // 2)], fill=(40, 40, 40))
    draw.line(list(zip(range(W), ys)), fill=(0, 220, 120), width=1)


# ---------- Spectrum-driven cellular growth ----------

GROWTH_CELL_SIZE = 16
GROWTH_COLS = W // GROWTH_CELL_SIZE                # 20 columns @ 16 px
GROWTH_ROWS = H // GROWTH_CELL_SIZE                # 15 rows
GROWTH_MAX_AGE = 80
GROWTH_INITIAL_DENSITY = 0.025                     # 2.5% seed at start
GROWTH_FADE = 135                                  # /256 ≈ 0.92 per frame (trail)
GROWTH_FREQ_LO = 60                                # lowest band → leftmost column
GROWTH_FREQ_HI = 7000                              # highest band → rightmost column
GROWTH_BIRTH_BASE = 0.025                          # spontaneous birth scaled by column energy
GROWTH_BIRTH_NEIGHBOR = 0.045                      # extra per alive neighbour, × energy
GROWTH_COL_PEAK_DECAY = 0.997                      # per-column peak auto-normalisation
GROWTH_COL_PEAK_FLOOR = 0.0010                     # peak never drops below this — gates quiet-section false fires
GROWTH_DRAW_ALPHA = 0.75                           # how strongly fresh shapes overwrite the trail
GROWTH_KICK_AVG_ALPHA = 0.06                       # slow moving avg per column for onset detection
GROWTH_KICK_RATIO = 1.45                           # energy must exceed ratio × avg to count as kick
GROWTH_KICK_FLOOR = 0.18                           # normalised energy floor (0..1) for kicks
GROWTH_KICK_BURST = 5                              # cells force-spawned in a column on each kick
GROWTH_KICK_COOL = 5                               # frames of per-column lockout after a kick
GROWTH_KICK_DECAY = 0.85                           # per-column kick pulse multiplier each frame
GROWTH_KICK_BRIGHTNESS = 0.55                      # how much fresh kicks lift colour toward white

# Cyber Sunset palette: orange → magenta-purple → cyan, on deep plum background
GROWTH_PALETTE = (
    (255, 140, 60),
    (180, 60,  180),
    (60,  180, 220),
)
GROWTH_BG = (24, 12, 36)


class GrowthGrid:
    """Conway-style cellular automaton on a coarse grid. Each column maps to
    a log-distributed audio band: loud frequencies raise that column's birth
    rate, so the cells form an organic 'spectrum painting' that flows over
    time. Trail is a persistent numpy buffer faded toward the background each
    frame.
    """

    def __init__(self):
        self.cols = GROWTH_COLS
        self.rows = GROWTH_ROWS
        self.alive = np.random.random((self.cols, self.rows)) < GROWTH_INITIAL_DENSITY
        self.age = np.zeros((self.cols, self.rows), dtype=np.int32)
        self.shape = np.random.randint(0, 3, (self.cols, self.rows), dtype=np.int32)
        self.col_peak = np.full(self.cols, 0.0005, dtype=np.float32)
        self.col_energy = np.zeros(self.cols, dtype=np.float32)
        self.col_avg = np.zeros(self.cols, dtype=np.float32)
        self.col_kick = np.zeros(self.cols, dtype=np.float32)   # decaying pulse for render
        self.col_cool = np.zeros(self.cols, dtype=np.int32)
        self.acc = np.full((H, W, 3), GROWTH_BG, dtype=np.uint8)
        self.frame = 0
        # Pre-compute the frequency-edge → column mapping once per buffer length
        self._freq_edges = np.logspace(
            np.log10(GROWTH_FREQ_LO), np.log10(GROWTH_FREQ_HI), self.cols + 1)
        self._bg = np.array(GROWTH_BG, dtype=np.int16)

    def reset(self):
        self.alive = np.random.random((self.cols, self.rows)) < GROWTH_INITIAL_DENSITY
        self.age[:] = 0
        self.shape = np.random.randint(0, 3, (self.cols, self.rows), dtype=np.int32)
        self.col_peak[:] = 0.0005
        self.col_energy[:] = 0
        self.col_avg[:] = 0
        self.col_kick[:] = 0
        self.col_cool[:] = 0
        self.acc[:] = self._bg
        self.frame = 0

    def update(self, samples):
        self.frame += 1

        # --- per-column audio energy from a single FFT ---
        m = min(len(samples), 1024)
        if m >= 64:
            recent = samples[-m:].astype(np.float32) / 32768.0
            mag = np.abs(np.fft.rfft(recent * np.hanning(m)))
            bin_hz = SAMPLE_RATE / m
            for c in range(self.cols):
                lo = max(1, int(self._freq_edges[c] / bin_hz))
                hi = max(lo + 1, int(self._freq_edges[c + 1] / bin_hz))
                self.col_energy[c] = float(mag[lo:hi].mean()) / m
            # auto-normalise each column against its own decaying peak,
            # but never let the peak drift below an absolute floor — otherwise
            # in quiet sections any tiny noise spike looks "loud" relative to
            # the decayed peak and causes phantom kicks.
            self.col_peak = np.maximum(self.col_peak * GROWTH_COL_PEAK_DECAY,
                                       self.col_energy)
            np.maximum(self.col_peak, GROWTH_COL_PEAK_FLOOR, out=self.col_peak)
            norm_energy = np.clip(self.col_energy / self.col_peak, 0.0, 1.0)
        else:
            norm_energy = np.zeros(self.cols, dtype=np.float32)

        # --- per-column onset detection: kicks burst-spawn cells in that column ---
        a_avg = GROWTH_KICK_AVG_ALPHA
        self.col_avg = (1.0 - a_avg) * self.col_avg + a_avg * norm_energy
        self.col_kick *= GROWTH_KICK_DECAY
        self.col_cool = np.maximum(0, self.col_cool - 1)
        for c in range(self.cols):
            if (self.col_cool[c] == 0
                    and norm_energy[c] > GROWTH_KICK_FLOOR
                    and norm_energy[c] > GROWTH_KICK_RATIO * self.col_avg[c]):
                self.col_kick[c] = 1.0
                self.col_cool[c] = GROWTH_KICK_COOL
                # Force-spawn fresh cells in this column's dead rows
                dead_rows = np.where(~self.alive[c])[0]
                if dead_rows.size > 0:
                    n_spawn = int(min(GROWTH_KICK_BURST, dead_rows.size))
                    chosen = np.random.choice(dead_rows, size=n_spawn, replace=False)
                    self.alive[c, chosen] = True
                    self.age[c, chosen] = 0
                    self.shape[c, chosen] = np.random.randint(0, 3, n_spawn, dtype=np.int32)

        # --- evolve cells ---
        # Age living cells, kill those that hit max age
        np.add(self.age, self.alive.astype(np.int32), out=self.age)
        dead = self.age >= GROWTH_MAX_AGE
        self.alive[dead] = False
        self.age[dead] = 0

        # Count toroidal neighbours
        a = self.alive.astype(np.int32)
        neighbours = (
            np.roll(a, 1, 0) + np.roll(a, -1, 0)
            + np.roll(a, 1, 1) + np.roll(a, -1, 1)
            + np.roll(np.roll(a, 1, 0), 1, 1)
            + np.roll(np.roll(a, 1, 0), -1, 1)
            + np.roll(np.roll(a, -1, 0), 1, 1)
            + np.roll(np.roll(a, -1, 0), -1, 1)
        )

        # Birth probability driven by this column's energy
        energy_col = norm_energy[:, None]  # (cols, 1) broadcasts over rows
        birth_p = energy_col * (GROWTH_BIRTH_BASE + neighbours * GROWTH_BIRTH_NEIGHBOR)
        candidate = ~self.alive
        born = candidate & (np.random.random((self.cols, self.rows)) < birth_p)
        if born.any():
            self.alive[born] = True
            self.age[born] = 0
            self.shape[born] = np.random.randint(0, 3, int(born.sum()), dtype=np.int32)

    def _flowing_color(self, col, row, age):
        """Time + position + age driven palette blend."""
        px = col / self.cols
        py = row / self.rows
        t = self.frame * 0.01
        n = math.sin(px * 2 + py * 2 + t) * 0.5 + 0.5
        age_factor = min(age / GROWTH_MAX_AGE, 1.0)
        blend = (age_factor + n) * 0.5
        pal_len = len(GROWTH_PALETTE)
        scaled = blend * (pal_len - 1)
        idx_a = int(scaled) % pal_len
        idx_b = (idx_a + 1) % pal_len
        mix = scaled - int(scaled)
        ca = GROWTH_PALETTE[idx_a]
        cb = GROWTH_PALETTE[idx_b]
        return (
            int(ca[0] + (cb[0] - ca[0]) * mix),
            int(ca[1] + (cb[1] - ca[1]) * mix),
            int(ca[2] + (cb[2] - ca[2]) * mix),
        )

    def _draw_shape(self, ddraw, shape_type, cx, cy, size, color):
        s = max(1.0, size)
        if shape_type == 0:  # circle
            r = s * 0.5
            ddraw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
        elif shape_type == 1:  # triangle
            h = s * 0.8660254  # sqrt(3)/2
            ddraw.polygon([
                (cx, cy - h * 0.5),
                (cx - s * 0.5, cy + h * 0.5),
                (cx + s * 0.5, cy + h * 0.5),
            ], fill=color)
        else:  # rect
            half = s * 0.5
            ddraw.rectangle((cx - half, cy - half, cx + half, cy + half), fill=color)

    def render_to(self, buffer):
        # 1. Fade trail buffer toward background
        diff = self.acc.astype(np.int16) - self._bg
        self.acc = (self._bg + (diff * GROWTH_FADE) // 256).astype(np.uint8)

        # 2. Render alive cells into a fresh delta image
        delta = Image.new("RGB", (W, H), (0, 0, 0))
        ddraw = ImageDraw.Draw(delta)
        live = np.argwhere(self.alive)
        cs = GROWTH_CELL_SIZE
        for col, row in live:
            age = int(self.age[col, row])
            shape = int(self.shape[col, row])
            cx = col * cs + cs * 0.5
            cy = row * cs + cs * 0.5
            size = 1.0 + (cs - 1) * (age / GROWTH_MAX_AGE)
            color = self._flowing_color(int(col), int(row), age)
            # Cells in a column riding a fresh kick get pulled toward white
            # AND drawn slightly bigger for an extra "punch" on each beat.
            kick = float(self.col_kick[int(col)])
            if kick > 0.1:
                lift = kick * GROWTH_KICK_BRIGHTNESS
                color = (
                    int(color[0] + (255 - color[0]) * lift),
                    int(color[1] + (255 - color[1]) * lift),
                    int(color[2] + (255 - color[2]) * lift),
                )
                size += cs * 0.25 * kick
            self._draw_shape(ddraw, shape, cx, cy, size, color)

        # 3. Alpha-blend delta into acc (only where shapes were drawn)
        delta_arr = np.asarray(delta)
        mask = (delta_arr.sum(axis=2) > 0)
        alpha = GROWTH_DRAW_ALPHA
        self.acc[mask] = (
            self.acc[mask].astype(np.float32) * (1.0 - alpha)
            + delta_arr[mask].astype(np.float32) * alpha
        ).astype(np.uint8)

        # 4. Push to display buffer
        buffer.paste(Image.fromarray(self.acc))


def rms_level(samples):
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) / 32768


def led_for_level(dhm, level):
    if level < 0.02:
        dhm.set_led(0, 0.05, 0)
    elif level < 0.2:
        t = (level - 0.02) / 0.18
        dhm.set_led(t, 1.0, 0)
    else:
        t = min(1.0, (level - 0.2) / 0.5)
        dhm.set_led(1.0, max(0.0, 1.0 - t), 0)


# ---------- Audio-reactive constellation ----------

CONSTELLATION_N = 65
CONSTELLATION_CONNECT_DIST = 35         # pixels — pairs closer than this get linked
CONSTELLATION_MARGIN = 6                 # px keep-out from the screen edge
CONSTELLATION_ROT_BASE = 0.005           # rad/frame slow rotation around screen centre
CONSTELLATION_ROT_GAIN = 0.018           # extra rad/frame at full drive
CONSTELLATION_SPRING_K = 0.010           # softness of return-to-pref spring (2D)
CONSTELLATION_JITTER_BASE = 0.15         # brownian wiggle in silence
CONSTELLATION_JITTER_GAIN = 0.45         # extra wiggle at full drive
CONSTELLATION_KICK_DECAY = 0.9          # per-band kick_pulse multiplier each frame
CONSTELLATION_BEAT_MOVE_FRAC = 0.25      # of in-range particles, fraction that re-target on each band kick
CONSTELLATION_BEAT_KICK_BOOST = 8.0     # extra speed toward new pref location (the springing portion)
CONSTELLATION_BEAT_TELEPORT_FRAC = 0.8   # of picks, fraction that snap to new pos instantly (rest spring)
CONSTELLATION_RETARGET_SPREAD = 25       # px of slop added on either side of the zone's [r_lo, r_hi] on
                                          # re-target — fuzzes the zone boundaries so rings overlap and
                                          # cross-zone connection lines survive
CONSTELLATION_WILD_FRAC = 0.45           # fraction of particles immune to beat re-targeting — they just
                                          # drift via rotation + jitter, providing a permanent "free scatter"
                                          # outside the zone clusters

# Per-band radial zones from the screen centre. Low = inner disc, mid = middle
# ring, high = outer ring + corners. Format: (lo_hz, hi_hz, RGB, r_lo, r_hi, kick_floor)
# High band's r_hi reaches 200 so the rectangle's corners (sqrt(160²+120²)≈200
# from centre) belong to a zone — otherwise corner particles get sucked inward
# over time and the outer scatter slowly empties.
CONSTELLATION_BANDS = [
    ((60,   300),  (255, 60,  60),    0,  45, 0.00035),   # low / kick — inner disc, red
    ((300,  2000), (255, 160, 60),   65, 120, 0.00120),   # mid — middle ring, orange
    ((2000, 8000), (240, 220, 80),   95, 200, 0.00060),   # high / hat — outer ring + corners, yellow
]
CONSTELLATION_BAND_DRIVE_ALPHA = 0.40    # smoothing for displayed drive
CONSTELLATION_BAND_AVG_ALPHA = 0.04      # slow moving avg for kick detection
CONSTELLATION_BAND_MAX_DECAY = 0.993     # peak decays this per frame (auto-normalisation)
CONSTELLATION_BAND_MAX_FLOOR_MULT = 3.0  # band_max never drops below mult × that band's kick_floor
CONSTELLATION_BAND_KICK_RATIO = 1.2      # band energy / moving avg → kick
CONSTELLATION_BAND_KICK_FLOOR_RATIO = 0.25  # kick floor auto-scales to 25% of band_max
                                            # (kicks must be at least this loud relative to recent peak)


_NUM_BANDS = len(CONSTELLATION_BANDS)


class Constellation:
    """Network of points organised in CONCENTRIC radial zones around the screen
    centre. Low band ⇒ inner disc, mid ⇒ middle ring, high ⇒ outer ring +
    corners. Each band detects its own onsets and re-targets a random subset
    of particles currently in its zone — when bass fires, the centre pulses;
    when hats fire, the outer ring lights up. Ripples from the inside out.
    """

    def __init__(self):
        self.center = np.array([W / 2.0, H / 2.0], dtype=np.float32)
        # Radial zone bounds per band (from screen centre)
        self.band_r_lo = np.array(
            [cfg[2] for cfg in CONSTELLATION_BANDS], dtype=np.float32)
        self.band_r_hi = np.array(
            [cfg[3] for cfg in CONSTELLATION_BANDS], dtype=np.float32)
        # Per-band absolute floor for band_max — auto-derived from each band's
        # kick_floor so a quiet section can't decay the peak below ambient noise.
        self.band_max_floor = np.array(
            [cfg[4] * CONSTELLATION_BAND_MAX_FLOOR_MULT for cfg in CONSTELLATION_BANDS],
            dtype=np.float32,
        )
        self.band_drive = np.zeros(_NUM_BANDS, dtype=np.float32)
        self.band_avg = np.zeros(_NUM_BANDS, dtype=np.float32)
        self.band_max = self.band_max_floor.copy()
        self.band_kick = np.zeros(_NUM_BANDS, dtype=np.float32)
        self.band_cool = np.zeros(_NUM_BANDS, dtype=np.int32)
        self.drive = 0.0     # global drive = max band drive
        self.beat_kick = np.zeros(CONSTELLATION_N, dtype=np.float32)
        self.beat_band = np.zeros(CONSTELLATION_N, dtype=np.int32)
        self._spawn()

    def _spawn(self):
        n = CONSTELLATION_N
        m = CONSTELLATION_MARGIN
        # Spread uniformly across the full rectangle — no circular boundary
        self.pos = np.column_stack([
            np.random.uniform(m, W - m, n),
            np.random.uniform(m, H - m, n),
        ]).astype(np.float32)
        # Each particle's preferred resting position (XY, not just radius)
        self.pos_pref = self.pos.copy()
        self.vel = np.zeros((n, 2), dtype=np.float32)
        # Mark a random subset as "wild" — never gets pulled into a band zone
        # by beat re-targeting, just drifts around its spawn point.
        self.wild = np.random.random(n) < CONSTELLATION_WILD_FRAC

    def reset(self):
        self.band_drive[:] = 0
        self.band_avg[:] = 0
        self.band_max[:] = self.band_max_floor
        self.band_kick[:] = 0
        self.band_cool[:] = 0
        self.drive = 0.0
        self.beat_kick[:] = 0
        self.beat_band[:] = 0
        self._spawn()

    def update(self, samples, rms):
        n = CONSTELLATION_N

        # ---------- 1. Single FFT, extract per-band energies ----------
        m = min(len(samples), 1024)
        if m >= 64:
            recent = samples[-m:].astype(np.float32) / 32768.0
            mag = np.abs(np.fft.rfft(recent * np.hanning(m)))
            bin_hz = SAMPLE_RATE / m
            band_e = np.empty(_NUM_BANDS, dtype=np.float32)
            for i, ((lo_hz, hi_hz), _c, _x, _y, _fl) in enumerate(CONSTELLATION_BANDS):
                lo = max(1, int(lo_hz / bin_hz))
                hi = max(lo + 1, int(hi_hz / bin_hz))
                band_e[i] = float(mag[lo:hi].mean()) / m
        else:
            band_e = np.zeros(_NUM_BANDS, dtype=np.float32)

        # ---------- 2. Per-band smoothing, peak tracking, onset detection ----------
        a_d = CONSTELLATION_BAND_DRIVE_ALPHA
        a_avg = CONSTELLATION_BAND_AVG_ALPHA
        for i in range(_NUM_BANDS):
            e = float(band_e[i])
            self.band_drive[i] = (1.0 - a_d) * self.band_drive[i] + a_d * e
            self.band_avg[i] = (1.0 - a_avg) * self.band_avg[i] + a_avg * e
            self.band_max[i] = max(
                self.band_max[i] * CONSTELLATION_BAND_MAX_DECAY,
                e,
                float(self.band_max_floor[i]),
            )

            # Adaptive kick floor: static safety floor OR a fraction of the
            # recent peak, whichever is HIGHER. So in a loud climax the bar
            # rises (only real heavy hits register), and in a quiet intro it
            # falls back to the static floor (small kicks can still trigger).
            floor = max(
                CONSTELLATION_BANDS[i][4],
                CONSTELLATION_BAND_KICK_FLOOR_RATIO * self.band_max[i],
            )
            is_kick = (
                self.band_cool[i] == 0
                and e > floor
                and e > CONSTELLATION_BAND_KICK_RATIO * self.band_avg[i]
            )
            if is_kick:
                self.band_kick[i] = 1.0
                self.band_cool[i] = 4
                # Pick particles from ANYWHERE on the canvas and pull them
                # INTO this band's radial zone. Outside particles see a
                # visible migration toward the centre on bass kicks, toward
                # the outer ring on treble. Net effect: each beat reshapes
                # the whole constellation by band-specific radius.
                # Wild particles are immune — they keep their free scatter.
                pick = (np.random.random(n) < CONSTELLATION_BEAT_MOVE_FRAC) & ~self.wild
                if pick.any():
                    cnt = int(pick.sum())
                    mm = CONSTELLATION_MARGIN
                    angles = np.random.uniform(0.0, 2.0 * np.pi, cnt).astype(np.float32)
                    # Fuzz the zone's [r_lo, r_hi] with extra slop on both
                    # sides so particles can land just outside the strict
                    # band — keeps inter-zone connection lines alive.
                    sp = CONSTELLATION_RETARGET_SPREAD
                    radii = np.random.uniform(
                        max(0.0, float(self.band_r_lo[i]) - sp),
                        float(self.band_r_hi[i]) + sp,
                        cnt).astype(np.float32)
                    new_x = np.clip(
                        self.center[0] + radii * np.cos(angles), mm, W - mm)
                    new_y = np.clip(
                        self.center[1] + radii * np.sin(angles), mm, H - mm)
                    self.pos_pref[pick, 0] = new_x
                    self.pos_pref[pick, 1] = new_y
                    # Subset that snaps instantly — gives each kick a sharp visual event
                    teleport = np.random.random(cnt) < CONSTELLATION_BEAT_TELEPORT_FRAC
                    pick_idx = np.where(pick)[0]
                    snap_idx = pick_idx[teleport]
                    if snap_idx.size > 0:
                        self.pos[snap_idx, 0] = new_x[teleport]
                        self.pos[snap_idx, 1] = new_y[teleport]
                    self.beat_kick[pick] = 1.0
                    self.beat_band[pick] = i
            if self.band_cool[i] > 0:
                self.band_cool[i] -= 1
            self.band_kick[i] *= CONSTELLATION_KICK_DECAY
        self.beat_kick *= CONSTELLATION_KICK_DECAY

        # Per-band normalised drive [0..1] relative to recent peak
        norm_drive = self.band_drive / np.maximum(self.band_max, 1e-6)
        norm_drive = np.clip(norm_drive, 0.0, 1.0)
        self.drive = float(norm_drive.max())  # global = loudest band

        # ---------- 3. Motion composition ----------
        # (a) Slow rotation around screen centre — gentle global drift
        offsets_c = self.pos - self.center
        d_c = np.linalg.norm(offsets_c, axis=1)
        d_c_safe = np.maximum(d_c, 0.5)
        ux_c = offsets_c[:, 0] / d_c_safe
        uy_c = offsets_c[:, 1] / d_c_safe
        tx, ty = -uy_c, ux_c
        ang_vel = CONSTELLATION_ROT_BASE + CONSTELLATION_ROT_GAIN * self.drive
        rot_speed = ang_vel * d_c_safe
        v_rot_x = tx * rot_speed
        v_rot_y = ty * rot_speed

        # (b) Soft 2D spring toward each particle's preferred resting position
        v_sp_x = (self.pos_pref[:, 0] - self.pos[:, 0]) * CONSTELLATION_SPRING_K
        v_sp_y = (self.pos_pref[:, 1] - self.pos[:, 1]) * CONSTELLATION_SPRING_K

        # (c) Per-particle beat boost — pushes toward newly assigned pos_pref
        to_pref_x = self.pos_pref[:, 0] - self.pos[:, 0]
        to_pref_y = self.pos_pref[:, 1] - self.pos[:, 1]
        to_pref_d = np.sqrt(to_pref_x * to_pref_x + to_pref_y * to_pref_y)
        to_pref_d_safe = np.maximum(to_pref_d, 1e-3)
        beat_force = self.beat_kick * CONSTELLATION_BEAT_KICK_BOOST
        v_pk_x = to_pref_x / to_pref_d_safe * beat_force
        v_pk_y = to_pref_y / to_pref_d_safe * beat_force

        # (d) Brownian jitter, audio-modulated so loud music wiggles more
        jitter_amp = CONSTELLATION_JITTER_BASE + CONSTELLATION_JITTER_GAIN * self.drive
        ja = np.random.uniform(0.0, 2.0 * np.pi, n).astype(np.float32)
        v_j_x = np.cos(ja) * jitter_amp
        v_j_y = np.sin(ja) * jitter_amp

        self.vel[:, 0] = v_rot_x + v_sp_x + v_pk_x + v_j_x
        self.vel[:, 1] = v_rot_y + v_sp_y + v_pk_y + v_j_y
        self.pos += self.vel

        # Rectangular boundary clamp — never leave the screen
        m = CONSTELLATION_MARGIN
        np.clip(self.pos[:, 0], m, W - m, out=self.pos[:, 0])
        np.clip(self.pos[:, 1], m, H - m, out=self.pos[:, 1])

    def render(self, draw):
        # Pair distances
        diffs = self.pos[:, None, :] - self.pos[None, :, :]
        d2 = (diffs ** 2).sum(-1)
        thr2 = CONSTELLATION_CONNECT_DIST ** 2
        mask = np.triu(d2 < thr2, k=1)
        i_idx, j_idx = np.where(mask)

        # Default red lines, brightness fades with pair distance
        for i, j in zip(i_idx.tolist(), j_idx.tolist()):
            x0, y0 = float(self.pos[i, 0]), float(self.pos[i, 1])
            x1, y1 = float(self.pos[j, 0]), float(self.pos[j, 1])
            t = d2[i, j] / thr2
            v = max(40, int(255 * (1.0 - 0.75 * t)))
            draw.line([(x0, y0), (x1, y1)], fill=(v, 0, 0), width=1)

        # Stars: recently beat-kicked → coloured by their band, else dim red
        for k in range(CONSTELLATION_N):
            x = float(self.pos[k, 0])
            y = float(self.pos[k, 1])
            bk = float(self.beat_kick[k])
            if bk > 0.15:
                rgb = CONSTELLATION_BANDS[int(self.beat_band[k])][1]
                # Brighten toward white as bk → 1
                fade = bk
                r = int(rgb[0] + (255 - rgb[0]) * fade * 0.3)
                g = int(rgb[1] + (255 - rgb[1]) * fade * 0.3)
                b = int(rgb[2] + (255 - rgb[2]) * fade * 0.3)
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(r, g, b))
            else:
                draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(180, 50, 50))

        # Always-visible red rings at each band's outer radius — static
        # reference for the zone layout. Their brightness pulses with that
        # band's current activity so they double as a beat indicator.
        cx, cy = int(self.center[0]), int(self.center[1])
        for i in range(_NUM_BANDS):
            inten = float(self.band_drive[i] / max(self.band_max[i], 1e-6))
            inten = max(0.0, min(1.0, inten))
            r_val = int(80 + 170 * inten)   # dim red baseline → bright red when band fires
            rr = int(self.band_r_hi[i])
            draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr),
                         outline=(r_val, 25, 25), width=1)


# ---------- Spectrogram waterfall ----------

SPECTROGRAM_FFT_SIZE = 1024
SPECTROGRAM_FREQ_LO = 60        # Hz — leftmost column
SPECTROGRAM_FREQ_HI = 7500      # Hz — rightmost column (below Nyquist 8 kHz)
SPECTROGRAM_DB_FLOOR = -50      # below this magnitude → background
SPECTROGRAM_DB_CEIL = 5         # above this → saturates palette top

# Magma-ish palette: black → deep purple → magenta → orange → yellow → near-white
SPECTROGRAM_PALETTE = np.array([
    (0,   0,   0),
    (15,  5,   40),
    (60,  10,  90),
    (140, 25,  120),
    (220, 60,  90),
    (250, 150, 50),
    (255, 220, 100),
    (255, 255, 220),
], dtype=np.float32)


class SpectrogramWaterfall:
    """Log-frequency scrolling spectrogram. Each frame computes one FFT of
    the most recent samples, maps log-spaced frequency bands across the
    full screen width, and prepends the resulting row at the top. The
    persistent image is rolled down so older rows slide toward the bottom.
    """

    def __init__(self):
        self.img = np.zeros((H, W, 3), dtype=np.uint8)
        # Pre-compute log-frequency column → FFT-bin ranges
        n_bins = SPECTROGRAM_FFT_SIZE // 2 + 1
        bin_hz = SAMPLE_RATE / SPECTROGRAM_FFT_SIZE
        edges = np.logspace(
            np.log10(SPECTROGRAM_FREQ_LO),
            np.log10(SPECTROGRAM_FREQ_HI),
            W + 1,
        )
        self._col_lo = np.clip(
            (edges[:-1] / bin_hz).astype(np.int32), 1, n_bins - 1)
        self._col_hi = np.clip(
            np.maximum(self._col_lo + 1, (edges[1:] / bin_hz).astype(np.int32)),
            2, n_bins,
        )
        self._widths = (self._col_hi - self._col_lo).astype(np.float32)
        self._window = np.hanning(SPECTROGRAM_FFT_SIZE).astype(np.float32)
        self._lut = self._build_lut()

    @staticmethod
    def _build_lut():
        pal = SPECTROGRAM_PALETTE
        steps = len(pal) - 1
        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            t = i / 255.0 * steps
            a = int(t)
            b = min(a + 1, steps)
            f = t - a
            lut[i] = (pal[a] * (1.0 - f) + pal[b] * f).astype(np.uint8)
        return lut

    def reset(self):
        self.img[:] = 0

    def update(self, samples):
        n = SPECTROGRAM_FFT_SIZE
        if len(samples) < n:
            return
        recent = samples[-n:].astype(np.float32) / 32768.0
        mag = np.abs(np.fft.rfft(recent * self._window))
        # Per-column mean magnitude via cumsum — no Python loop over columns.
        cum = np.concatenate(([0.0], np.cumsum(mag)))
        col_vals = (cum[self._col_hi] - cum[self._col_lo]) / self._widths
        db = 20.0 * np.log10(col_vals + 1e-8)
        norm = np.clip(
            (db - SPECTROGRAM_DB_FLOOR)
            / (SPECTROGRAM_DB_CEIL - SPECTROGRAM_DB_FLOOR),
            0.0, 1.0,
        )
        idx = (norm * 255.0).astype(np.uint8)
        new_row = self._lut[idx]
        # Scroll everything down one pixel, write the fresh row on top.
        self.img = np.roll(self.img, 1, axis=0)
        self.img[0] = new_row

    def render_to(self, buffer):
        buffer.paste(Image.fromarray(self.img))


def main():
    cap = Capture()
    buffer = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(buffer)
    dhm = DisplayHATMini(buffer)
    dhm.set_led(0, 0, 0)   # keep the HAT's RGB LED off — soft-PWM flicker is distracting

    modes = ["WAVEFORM", "GROWTH", "CONSTELLATION", "SPECTROGRAM"]
    mode_idx = 0
    paused = False
    constellation = None  # lazy-init on first CONSTELLATION entry
    growth = None         # lazy-init on first GROWTH entry
    spectrogram = None    # lazy-init on first SPECTROGRAM entry
    btn_state = {dhm.BUTTON_A: False, dhm.BUTTON_B: False,
                 dhm.BUTTON_X: False, dhm.BUTTON_Y: False}

    print("Running. A=mode  X=constellation  Y=spectrogram  B=pause  Ctrl-C=quit")
    try:
        while True:
            for pin in btn_state:
                p = dhm.read_button(pin)
                if p and not btn_state[pin]:
                    if pin == dhm.BUTTON_A:
                        mode_idx = (mode_idx + 1) % len(modes)
                    elif pin == dhm.BUTTON_X:
                        if modes[mode_idx] == "CONSTELLATION" and constellation is not None:
                            constellation.reset()
                        else:
                            mode_idx = modes.index("CONSTELLATION")
                    elif pin == dhm.BUTTON_Y:
                        if modes[mode_idx] == "SPECTROGRAM" and spectrogram is not None:
                            spectrogram.reset()
                        else:
                            mode_idx = modes.index("SPECTROGRAM")
                    elif pin == dhm.BUTTON_B:
                        paused = not paused
                btn_state[pin] = p

            samples = cap.snapshot()
            rms = rms_level(samples)
            mode = modes[mode_idx]

            if not paused:
                if mode == "GROWTH":
                    # GrowthGrid keeps its own persistent buffer — don't clear
                    if growth is None:
                        growth = GrowthGrid()
                    growth.update(samples)
                    growth.render_to(buffer)
                elif mode == "SPECTROGRAM":
                    # SpectrogramWaterfall also owns its persistent buffer
                    if spectrogram is None:
                        spectrogram = SpectrogramWaterfall()
                    spectrogram.update(samples)
                    spectrogram.render_to(buffer)
                else:
                    draw.rectangle((0, 0, W, H), fill=(0, 0, 0))
                    if mode == "CONSTELLATION":
                        if constellation is None:
                            constellation = Constellation()
                        constellation.update(samples, rms)
                        constellation.render(draw)
                    else:  # WAVEFORM
                        draw_waveform(draw, samples)

            dhm.display()
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        cap.close()
        dhm.set_led(0, 0, 0)
        draw.rectangle((0, 0, W, H), fill=(0, 0, 0))
        dhm.display()
        print("\nBye.")


if __name__ == "__main__":
    main()

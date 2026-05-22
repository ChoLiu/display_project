"""Print live low-band energy, moving avg, ratio, and RMS for 10 seconds.

Play techno into the mic while this runs and watch the numbers — if low_energy
never rises substantially above low_avg during kicks, the C300's AGC/noise
suppression is flattening the dynamics and we need a different detection
strategy.
"""
import subprocess
import time
import numpy as np

SAMPLE_RATE = 16000
CARD = "plughw:3,0"
WINDOW_SAMPLES = 2048           # ~128 ms snapshot
CHUNK_SAMPLES = 512
KICK_LO_HZ = 50
KICK_HI_HZ = 180
SHORT_WINDOW = 800              # last ~50 ms used for FFT


def low_band_energy(samples):
    m = min(len(samples), SHORT_WINDOW)
    recent = samples[-m:].astype(np.float32) / 32768.0
    mag = np.abs(np.fft.rfft(recent * np.hanning(m)))
    bin_hz = SAMPLE_RATE / m
    lo = max(1, int(KICK_LO_HZ / bin_hz))
    hi = max(lo + 1, int(KICK_HI_HZ / bin_hz))
    return float(mag[lo:hi].mean()) / m


def broadband_energy(samples):
    m = min(len(samples), SHORT_WINDOW)
    recent = samples[-m:].astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(recent ** 2)))


def rms(samples):
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) / 32768.0


proc = subprocess.Popen(
    ["arecord", "-D", CARD, "-f", "S16_LE", "-c", "1",
     "-r", str(SAMPLE_RATE), "-t", "raw", "-q"],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
)

ring = np.zeros(WINDOW_SAMPLES, dtype=np.int16)
low_avg = 0.0
broad_avg = 0.0
ALPHA = 0.03
peak_low = 0.0
peak_broad = 0.0

print(f"{'t':>5} {'rms':>7} {'low':>9} {'lo_avg':>9} {'ratio':>6} {'broad':>9} {'br_avg':>9} {'br_ratio':>8}")
print("-" * 80)

start = time.time()
last_print = start
last_kick_lo = 0
last_kick_br = 0
total_low = 0
total_br = 0

try:
    while time.time() - start < 10:
        raw = proc.stdout.read(CHUNK_SAMPLES * 2)
        if not raw:
            break
        samples = np.frombuffer(raw, dtype=np.int16)
        ring = np.roll(ring, -len(samples))
        ring[-len(samples):] = samples

        cur_rms = rms(ring)
        cur_low = low_band_energy(ring)
        cur_broad = broadband_energy(ring)
        low_avg = (1 - ALPHA) * low_avg + ALPHA * cur_low
        broad_avg = (1 - ALPHA) * broad_avg + ALPHA * cur_broad

        peak_low = max(peak_low, cur_low)
        peak_broad = max(peak_broad, cur_broad)

        ratio_lo = cur_low / low_avg if low_avg > 1e-9 else 0
        ratio_br = cur_broad / broad_avg if broad_avg > 1e-9 else 0

        if ratio_lo > 1.5 and cur_low > 0.001:
            total_low += 1
        if ratio_br > 1.4 and cur_broad > 0.01:
            total_br += 1

        now = time.time()
        if now - last_print >= 0.2:
            t = now - start
            print(f"{t:5.1f} {cur_rms:7.4f} {cur_low:9.5f} {low_avg:9.5f} {ratio_lo:6.2f} "
                  f"{cur_broad:9.5f} {broad_avg:9.5f} {ratio_br:8.2f}")
            last_print = now
finally:
    proc.terminate()

print()
print(f"peak low-band energy:  {peak_low:.5f}")
print(f"peak broadband energy: {peak_broad:.5f}")
print(f"low-band spikes (ratio>1.5, abs>0.001): {total_low}")
print(f"broadband spikes (ratio>1.4, abs>0.01): {total_br}")

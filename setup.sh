#!/usr/bin/env bash
# One-shot setup for the Display HAT Mini audio visualizer.
# Target: Raspberry Pi (Bookworm / Trixie) with SPI enabled.
# Usage:   bash setup.sh
# Re-runs are safe — existing venv and packages are kept up to date.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "==> Project dir: $PROJECT_DIR"

# 1. System packages (apt)
echo "==> Installing system packages (sudo apt)..."
sudo apt update
sudo apt install -y \
    python3-venv \
    python3-pip \
    python3-dev \
    alsa-utils \
    libopenjp2-7 \
    libtiff6

# 2. Make sure SPI is enabled (Display HAT Mini needs it)
if ! lsmod | grep -q '^spi_bcm2835'; then
    echo "==> WARNING: SPI kernel module not loaded."
    echo "    Enable it with: sudo raspi-config nonint do_spi 0"
    echo "    Then reboot and re-run this script."
fi

# 3. Create venv (with system-site-packages so we can fall back to apt's
#    python3-* packages if pip wheels are unavailable on this arch)
if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating venv at $VENV_DIR (with --system-site-packages)..."
    python3 -m venv --system-site-packages "$VENV_DIR"
else
    echo "==> venv already exists, skipping creation"
fi

# 4. Install Python deps
echo "==> Upgrading pip and installing requirements..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

# 5. Sanity check
echo "==> Verifying imports..."
"$VENV_DIR/bin/python" - <<'PY'
import importlib, sys
for mod in ("displayhatmini", "PIL", "numpy", "RPi.GPIO"):
    try:
        importlib.import_module(mod)
        print(f"  OK   {mod}")
    except Exception as e:
        print(f"  FAIL {mod}: {e}")
        sys.exit(1)
PY

cat <<EOF

==> Done.

Run the visualizer:
    source .venv/bin/activate
    python audio_viz.py

Or without activating:
    .venv/bin/python audio_viz.py

EOF

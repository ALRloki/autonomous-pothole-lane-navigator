#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo "AI-Enhanced Autonomous Navigator: Environment Setup"
echo "============================================================"

# Ensure local bin directory exists
mkdir -p "$PROJECT_DIR/.bin"
export PATH="$PROJECT_DIR/.bin:$HOME/.local/bin:$PATH"

# 1. Install / Check uv (fast standalone Python environment manager)
if ! command -v uv &> /dev/null; then
    echo "[+] Installing standalone 'uv' package & Python manager..."
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PROJECT_DIR/.bin" sh
    export PATH="$PROJECT_DIR/.bin:$PATH"
fi

echo "[+] Using uv: $(command -v uv)"

# 2. Setup Python 3.11 virtual environment
echo "[+] Setting up Python 3.11 environment..."
uv python install 3.11
if [ ! -d "$PROJECT_DIR/.venv" ]; then
    uv venv "$PROJECT_DIR/.venv" --python 3.11
fi

echo "[+] Activating virtual environment..."
source "$PROJECT_DIR/.venv/bin/activate"

# 3. Install project dependencies
echo "[+] Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "============================================================"
echo "Environment setup complete!"
echo "To activate manually: source $PROJECT_DIR/.venv/bin/activate"
echo "============================================================"

#!/usr/bin/env bash
# install/build_llama_cpp.sh
#
# Builds llama.cpp with CUDA and OpenSSL support on the Jetson Orin Nano.
# OpenSSL is required for the --hf-repo / --hf-file flags to download
# models directly from HuggingFace.
#
# Run this once. The resulting llama-server binary lives at:
#   llama.cpp/build/bin/llama-server
#
# After building, optionally run install_llama_cpp.sh to add
# llama-server to your PATH.
#
# Usage:
#   chmod +x install/build_llama_cpp.sh
#   ./install/build_llama_cpp.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Prerequisites check
# ---------------------------------------------------------------------------
echo "Checking prerequisites..."

for cmd in git cmake nvcc; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "Error: '$cmd' not found." >&2
        case "$cmd" in
            git)   echo "  Install with: sudo apt install git" >&2 ;;
            cmake) echo "  Install with: sudo apt install cmake" >&2 ;;
            nvcc)  echo "  nvcc is part of the CUDA toolkit. Verify JetPack is installed." >&2 ;;
        esac
        exit 1
    fi
done

# OpenSSL dev files are required for --hf-repo / --hf-file HTTPS downloads.
if ! pkg-config --exists openssl 2>/dev/null; then
    echo "OpenSSL development files not found. Installing libssl-dev..."
    sudo apt install -y libssl-dev
fi

echo "All prerequisites found."
echo ""

# ---------------------------------------------------------------------------
# Clone
# ---------------------------------------------------------------------------
INSTALL_DIR="${HOME}/llama.cpp"

if [[ -d "${INSTALL_DIR}" ]]; then
    echo "Directory ${INSTALL_DIR} already exists. Pulling latest changes..."
    git -C "${INSTALL_DIR}" pull
else
    echo "Cloning llama.cpp into ${INSTALL_DIR}..."
    git clone https://github.com/ggml-org/llama.cpp "${INSTALL_DIR}"
fi

cd "${INSTALL_DIR}"
echo ""

# ---------------------------------------------------------------------------
# Configure
# ---------------------------------------------------------------------------
echo "Configuring build with CUDA and OpenSSL support..."

cmake -B build \
    -DGGML_CUDA=ON \
    -DLLAMA_OPENSSL=ON \
    -DCMAKE_BUILD_TYPE=Release

echo ""

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo "Building with $(nproc) parallel jobs..."
echo "This will take several minutes on the Orin Nano."
echo ""

cmake --build build \
    --config Release \
    --parallel "$(nproc)"

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
BINARY="${INSTALL_DIR}/build/bin/llama-server"

if [[ -f "${BINARY}" ]]; then
    echo ""
    echo "Build successful."
    echo "Binary: ${BINARY}"
    echo ""
    echo "To make llama-server available system-wide, run:"
    echo "  sudo ln -sf ${BINARY} /usr/local/bin/llama-server"
    echo ""
    echo "Then start the server with:"
    echo "  ./scripts/llama_server.sh"
else
    echo ""
    echo "Error: build completed but llama-server binary not found at ${BINARY}." >&2
    echo "Check the build output above for errors." >&2
    exit 1
fi

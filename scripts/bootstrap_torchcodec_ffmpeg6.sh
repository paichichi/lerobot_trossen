#!/usr/bin/env bash
# Install the missing Ubuntu FFmpeg 6 runtime libraries without sudo.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
install_root="${ACT_FFMPEG_ROOT:-$repo_root/.local/ffmpeg6}"
lib_dir="$install_root/usr/lib/x86_64-linux-gnu"

if LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$repo_root/.venv/bin/python" -c "import torchcodec" >/dev/null 2>&1; then
  printf 'TorchCodec FFmpeg runtime ready: %s\n' "$lib_dir"
  exit 0
fi

download_dir="$(mktemp -d)"
trap 'rm -rf "$download_dir"' EXIT
mkdir -p "$install_root"
cd "$download_dir"

apt download \
  libavdevice60 \
  libjack-jackd2-0 \
  libopenal1 \
  libdc1394-25 \
  libsdl2-2.0-0 \
  libsndio7.0

for archive in ./*.deb; do
  dpkg-deb -x "$archive" "$install_root"
done

LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$repo_root/.venv/bin/python" -c "import torchcodec"
printf 'TorchCodec FFmpeg runtime installed: %s\n' "$lib_dir"

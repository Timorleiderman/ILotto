#!/usr/bin/env bash
set -euo pipefail
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$source_dir/../.." && pwd)"
cuda_dir="${ILOTTO_CUDA_ROOT:-$project_dir/.cuda-12.8/toolkit}"
cuda_arch="${ILOTTO_CUDA_ARCH:-sm_86}"
"$cuda_dir/bin/nvcc" -O3 -std=c++17 -arch="$cuda_arch" -L"$cuda_dir/lib" \
  -Xptxas=-v "$source_dir/main.cu" -o "$source_dir/ilotto-cuda-search"

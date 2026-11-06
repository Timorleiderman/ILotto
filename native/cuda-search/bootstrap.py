"""Install pinned NVIDIA compiler components under a project-local directory."""

import hashlib
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.request

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
PACKAGES = (
    ("cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz",
     "9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3"),
    ("cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz",
     "8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68"),
    ("cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz",
     "0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b"),
)


def main():
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for relative, digest in PACKAGES:
        archive = root / Path(relative).name
        if not archive.exists():
            print(f"Downloading {archive.name}", flush=True)
            with urllib.request.urlopen(BASE + relative, timeout=120) as source:
                with archive.open("wb") as target:
                    shutil.copyfileobj(source, target)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Checksum mismatch: {archive}")
        with tarfile.open(archive) as source:
            source.extractall(root / "components", filter="data")
        component = root / "components" / archive.name.removesuffix(".tar.xz")
        shutil.copytree(component, root / "toolkit", dirs_exist_ok=True, symlinks=True)
    print(root / "toolkit/bin/nvcc")


if __name__ == "__main__":
    main()

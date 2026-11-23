import hashlib
import os
import tarfile
import urllib.request
import shutil
from pathlib import Path

# Optional integrity pin. The Imagenette-320 tarball is ~326 MB; set the
# expected SHA-256 here (or pass expected_sha256=...) to verify it after
# download. Leave as None to skip verification (HTTPS + size check only).
EXPECTED_SHA256 = None
EXPECTED_MIN_BYTES = 1_000_000


def _verify_download(tar_dest, expected_sha256=None):
    """Verify the downloaded tarball: exists, non-trivial size, optional hash."""
    p = Path(tar_dest)
    if not p.exists():
        raise FileNotFoundError(f"download missing: {p}")
    size = p.stat().st_size
    if size < EXPECTED_MIN_BYTES:
        raise ValueError(f"downloaded archive too small ({size} bytes): {p}")
    if expected_sha256:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
        if digest != expected_sha256:
            raise ValueError(
                f"archive checksum mismatch: got {digest}, expected {expected_sha256}"
            )


def _safe_extract_tar(tar_path, extract_dir):
    """Extract a tarball, rejecting path traversal, links, and non-file members.

    Defense in depth: manual member validation plus the ``filter="data"``
    argument on Python 3.12+ (which additionally rejects device files and
    absolute/out-of-tree links).
    """
    extract_dir = Path(extract_dir).resolve()
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar.getmembers():
            target = (extract_dir / m.name).resolve()
            if target != extract_dir and extract_dir not in target.parents:
                raise ValueError(f"unsafe path in archive: {m.name!r}")
            if m.issym() or m.islnk():
                raise ValueError(f"links not allowed in archive: {m.name!r}")
            if not (m.isfile() or m.isdir()):
                raise ValueError(f"unsupported member type in archive: {m.name!r}")
        try:
            tar.extractall(path=extract_dir, filter="data")
        except TypeError:
            # Python < 3.12 has no filter= argument; manual validation above
            # is the guard.
            tar.extractall(path=extract_dir)

def download_and_extract_dataset(target_dir="./data/test_images", num_images=1500):
    os.makedirs("./data", exist_ok=True)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    existing_files = list(target_path.glob("*.jpg")) + list(target_path.glob("*.png"))
    if len(existing_files) >= num_images:
        print(f"[+] Found {len(existing_files)} existing images in {target_dir}. Ready!")
        return

    tar_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    tar_dest = "./data/imagenette2-320.tgz"
    extract_dir = "./data/imagenette_extracted"

    if not os.path.exists(tar_dest):
        print(f"[*] Downloading Imagenette-320 dataset (~326 MB)...")
        urllib.request.urlretrieve(tar_url, tar_dest)
        print("[+] Download complete!")
    _verify_download(tar_dest, EXPECTED_SHA256)

    if not os.path.exists(extract_dir):
        print("[*] Extracting archive...")
        _safe_extract_tar(tar_dest, extract_dir)
        print("[+] Extraction complete!")

    print(f"[*] Collecting {num_images} images into {target_dir}...")
    extracted_path = Path(extract_dir)
    image_paths = list(extracted_path.rglob("*.JPEG")) + list(extracted_path.rglob("*.jpg"))
    
    count = 0
    for img_p in image_paths[:num_images]:
        dest_file = target_path / f"img_{count:05d}.jpg"
        shutil.copy2(img_p, dest_file)
        count += 1

    print(f"[+] Successfully prepared {count} images in '{target_dir}'!")

if __name__ == "__main__":
    download_and_extract_dataset()

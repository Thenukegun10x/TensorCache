import os
import tarfile
import urllib.request
import shutil
from pathlib import Path

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

    if not os.path.exists(extract_dir):
        print("[*] Extracting archive...")
        with tarfile.open(tar_dest, "r:gz") as tar:
            tar.extractall(path=extract_dir)
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

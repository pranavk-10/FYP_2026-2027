"""
HAM10000 Dataset & Metadata Downloader

Downloads official HAM10000 metadata and high-resolution dermoscopy images
from Harvard Dataverse and Hugging Face mirrors with multi-threading, integrity checking,
and resume capability.
"""

import os
import sys
import argparse
import urllib.request
import concurrent.futures
import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download, HfApi

HF_DATASET_REPO = "aditya022/HAM10000_Dataset"
HARVARD_DATAVERSE_META_URL = "https://dataverse.harvard.edu/api/access/datafile/4338392"


def get_default_data_dir():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "data")


def download_metadata(dest_dir=None):
    """
    Downloads HAM10000_metadata.csv to dest_dir.
    """
    if dest_dir is None:
        dest_dir = get_default_data_dir()
    os.makedirs(dest_dir, exist_ok=True)
    target_path = os.path.join(dest_dir, "HAM10000_metadata.csv")

    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        print(f"[OK] Metadata already exists at: {target_path}")
        return target_path

    print("[*] Downloading HAM10000 metadata...")
    try:
        cached_file = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename="archive/HAM10000_metadata.csv",
            repo_type="dataset"
        )
        import shutil
        shutil.copy(cached_file, target_path)
        print(f"[OK] Metadata downloaded successfully to: {target_path}")
        return target_path
    except Exception as e:
        print(f"[!] HF Hub download failed ({e}), attempting direct download...")
        req = urllib.request.Request(
            HARVARD_DATAVERSE_META_URL,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req) as resp, open(target_path, "wb") as out:
            out.write(resp.read())
        print(f"[OK] Metadata downloaded from Harvard Dataverse: {target_path}")
        return target_path


def download_single_image(file_info):
    """
    Downloads a single image file if not already present and non-empty.
    """
    remote_rel_path, local_path = file_info
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return True, local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    url = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/{remote_rel_path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp, open(local_path, "wb") as out_f:
                out_f.write(resp.read())
            return True, local_path
        except Exception:
            if attempt == 2:
                return False, local_path
    return False, local_path


def download_images(dest_dir=None, max_images=None, stratified=False, max_workers=16):
    """
    Downloads images from HAM10000 repository.

    Args:
        dest_dir: Destination folder for images.
        max_images: Total number of images to download (None = all 10,015 images).
        stratified: If True and max_images is set, downloads balanced samples across all 7 classes.
        max_workers: Number of concurrent download threads.
    """
    if dest_dir is None:
        dest_dir = get_default_data_dir()
    
    meta_path = download_metadata(dest_dir=dest_dir)
    df = pd.read_csv(meta_path)

    images_dir = os.path.join(dest_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print("[*] Querying dataset repository file index...")
    api = HfApi()
    repo_files = api.list_repo_files(HF_DATASET_REPO, repo_type="dataset")
    jpg_files = {os.path.basename(f): f for f in repo_files if f.endswith(".jpg")}

    # Build selection list
    if max_images is not None and max_images < len(df):
        if stratified:
            samples_per_class = max(1, max_images // len(df['dx'].unique()))
            selected_dfs = []
            for _, group in df.groupby('dx'):
                selected_dfs.append(group.sample(min(len(group), samples_per_class), random_state=42))
            selected_df = pd.concat(selected_dfs, ignore_index=True)
            if len(selected_df) < max_images:
                remaining = df[~df['image_id'].isin(selected_df['image_id'])].sample(
                    max_images - len(selected_df), random_state=42
                )
                selected_df = pd.concat([selected_df, remaining], ignore_index=True)
        else:
            selected_df = df.head(max_images)
    else:
        selected_df = df

    tasks = []
    for _, row in selected_df.iterrows():
        img_filename = f"{row['image_id']}.jpg"
        if img_filename in jpg_files:
            remote_path = jpg_files[img_filename]
            local_path = os.path.join(images_dir, img_filename)
            tasks.append((remote_path, local_path))

    print(f"[*] Starting download of {len(tasks)} images using {max_workers} threads...")
    success_count = 0
    fail_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_single_image, task) for task in tasks]
        with tqdm(total=len(futures), desc="Downloading HAM10000", unit="img") as pbar:
            for fut in concurrent.futures.as_completed(futures):
                success, _ = fut.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                pbar.update(1)

    print(f"[OK] Completed: {success_count} images ready in {images_dir} ({fail_count} failed)")
    return images_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HAM10000 Dataset & Metadata")
    parser.add_argument("--dest", type=str, default=None, help="Destination directory for data")
    parser.add_argument("--max-images", type=int, default=None, help="Number of images to download (default: all)")
    parser.add_argument("--stratified", action="store_true", help="Download stratified samples per diagnosis class")
    parser.add_argument("--workers", type=int, default=16, help="Number of concurrent download threads")
    parser.add_argument("--metadata-only", action="store_true", help="Only download metadata CSV")

    args = parser.parse_args()

    if args.metadata_only:
        download_metadata(dest_dir=args.dest)
    else:
        download_images(
            dest_dir=args.dest,
            max_images=args.max_images,
            stratified=args.stratified,
            max_workers=args.workers
        )

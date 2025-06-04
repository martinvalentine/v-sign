import os
import re
import csv
import cv2
import time
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def extract_raw_frames(video_path, output_folder, sample_rate=1, flip_vertical=False, crop_dims=None):
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    os.makedirs(output_folder, exist_ok=True)

    count = 0
    saved = 0
    while True:
        success, frame = vidcap.read()
        if not success:
            break
        if count % sample_rate == 0:
            if flip_vertical:
                frame = cv2.flip(frame, 0)

            if crop_dims:
                target_w, target_h = crop_dims
                original_h, original_w = frame.shape[:2] # Get original height and width

                # Check if the frame is large enough for the requested crop
                if original_w >= target_w and original_h >= target_h:
                    start_x = (original_w - target_w) // 2
                    end_x = start_x + target_w
                    start_y = (original_h - target_h) // 2
                    end_y = start_y + target_h
                    frame = frame[start_y:end_y, start_x:end_x]
                    # Verify cropped dimensions (optional, for debugging)
                    # print(f"Cropped frame shape: {frame.shape}")
                else:
                    # Log a warning if the frame is too small to crop as requested
                    logging.warning(f"Video {video_path}: Frame {count} ({original_w}x{original_h}) is smaller than target crop dimensions ({target_w}x{target_h}). Skipping crop for this frame.")

            frame_name = f"{saved:04d}.png"
            frame_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_path, frame)
            saved += 1
        count += 1
    vidcap.release()

def process_single_video_extraction_raw(task):
    video_path, gloss_label, output_root, split_output, sample_rate, flip_vertical, crop_dims = task

    try:
        video_filename = Path(video_path).stem
        output_folder = Path(split_output) / video_filename
        output_folder.mkdir(parents=True, exist_ok=True)

        extract_raw_frames(video_path, output_folder, sample_rate, flip_vertical, crop_dims)

        # Compute relative path from output_root
        relative_path = output_folder.relative_to(Path(output_root))

        return (video_filename, str(relative_path), gloss_label)
    except Exception as e:
        logging.error(f"Failed processing video {video_path}: {e}")
        return None


def collect_video_tasks_flat(root_dir, output_dir, split, sample_rate=1, flip_vertical=False, crop_dims=None):
    tasks = []
    root_path = Path(root_dir) / split
    split_output = Path(output_dir) / split
    exts = {".mov", ".mp4", ".avi", ".mkv"}

    if not root_path.is_dir():
        logging.warning(f"split '{split}' not found at {root_path}")
        return tasks

    sentence_folders = sorted(p for p in root_path.iterdir() if p.is_dir())
    for sent_f in sentence_folders:
        gloss_label = sent_f.name
        vids = [p for p in sent_f.iterdir() if p.is_file() and p.suffix.lower() in exts]
        if not vids:
            continue

        for video_file in sorted(vids):
            tasks.append((
                str(video_file),
                gloss_label,
                str(output_dir),
                str(split_output),
                sample_rate,
                flip_vertical,
                crop_dims
            ))

    return tasks


def process_split_optimized_raw_flat(root_dir, output_dir, csv_dir, split, num_procs, sample_rate, chunk_size, flip_vertical, crop_dims):
    start = time.time()
    split_output = Path(output_dir) / split
    split_output.mkdir(parents=True, exist_ok=True)
    Path(csv_dir).mkdir(parents=True, exist_ok=True)

    tasks = collect_video_tasks_flat(root_dir, output_dir, split, sample_rate, flip_vertical, crop_dims)
    if not tasks:
        logging.warning(f"No tasks for split '{split}' — skipping")
        return

    total = len(tasks)
    logging.info(f"→ {total} videos queued in '{split}'")
    if not chunk_size or chunk_size < 1:
        chunk_size = max(1, total // (num_procs * 4))
        logging.info(f"  auto‐calculated chunksize: {chunk_size}")

    results = []
    with Pool(num_procs) as pool:
        for res in tqdm(pool.imap_unordered(process_single_video_extraction_raw, tasks, chunksize=chunk_size),
                        total=total, desc=f"Extracting {split}"):
            if res:
                results.append(res)

    results.sort(key=lambda r: r[0])
    csv_path = Path(csv_dir) / f"{split}_annotations.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as fp:
        wr = csv.writer(fp)
        wr.writerow(["Video_ID", "Frames_Path", "Gloss_Label"])
        wr.writerows(results)

    logging.info(f"★ '{split}' done in {time.time() - start:.1f}s, CSV → {csv_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_root', default="/home/martinvalentine/Desktop/v-sign/data/raw/VSL_V0", type=str)
    parser.add_argument('--output_root', default="/home/martinvalentine/Desktop/v-sign/data/interim/frames/VSL_V0", type=str)
    parser.add_argument('--csv_root', default="/home/martinvalentine/Desktop/v-sign/data/splits/VSL_V0/csv", type=str)
    parser.add_argument('--splits', nargs='+', default=["train", "test", "dev"])
    parser.add_argument('--sample_rate', type=int, default=2)
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 1))
    parser.add_argument('--chunksize', type=int, default=0)
    parser.add_argument('--flip_vertical', action='store_true', help="Flip video frames vertically before saving")
    parser.add_argument('--crop_width', type=int, default=0, help="Target width for cropping")
    parser.add_argument('--crop_height', type=int, default=0, help="Target height for cropping")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    crop_dims = (args.crop_width, args.crop_height) if args.crop_width > 0 and args.crop_height > 0 else None

    for split in args.splits:
        logging.info(f"--- processing split '{split}' ---")
        process_split_optimized_raw_flat(
            args.video_root,
            args.output_root,
            args.csv_root,
            split,
            num_procs=args.workers,
            sample_rate=args.sample_rate,
            chunk_size=args.chunksize,
            flip_vertical=args.flip_vertical,
            crop_dims=crop_dims
        )

    logging.info("All splits processed and frame annotations generated.")


if __name__ == "__main__":
    main()
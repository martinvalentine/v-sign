import os
import cv2
import re
import glob
import csv
import time
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np
from functools import partial
from pathlib import Path
from collections import defaultdict


def resize_img(img_path, dsize='256x256px'):
    """Resize image from file path."""
    dsize = tuple(int(res) for res in re.findall("\d+", dsize))
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)
    return img


def extract_and_resize_frames(video_path, output_folder, target_size=(256, 256), sample_rate=1, flip_vertical=False, crop_dims=None):
    """Extract frames from video and resize them directly to target size.
    
    Args:
        video_path: Source video file
        output_folder: Where to save extracted frames
        target_size: Target dimensions for resizing (width, height)
        sample_rate: Extract every Nth frame
        flip_vertical: Whether to flip frames vertically
        crop_dims: (width, height) for center cropping before resizing
    """
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    os.makedirs(output_folder, exist_ok=True)
    
    # Get video metadata
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    count = 0  # Total frame counter
    saved = 0  # Saved frame counter
    
    while True:
        success, frame = vidcap.read()
        if not success:
            break
            
        if count % sample_rate == 0:  # Only process frames matching sample rate
            # Apply vertical flip if requested
            if flip_vertical:
                frame = cv2.flip(frame, 0)

            # Apply center cropping if requested
            if crop_dims:
                target_w, target_h = crop_dims
                original_h, original_w = frame.shape[:2]

                if original_w >= target_w and original_h >= target_h:
                    start_x = (original_w - target_w) // 2
                    end_x = start_x + target_w
                    start_y = (original_h - target_h) // 2
                    end_y = start_y + target_h
                    frame = frame[start_y:end_y, start_x:end_x]
                else:
                    logging.warning(f"Video {video_path}: Frame {count} ({original_w}x{original_h}) too small for crop to ({target_w}x{target_h})")

            # Resize frame directly to target size
            try:
                # Use INTER_LANCZOS4 for better quality resizing like in vsl_preprocess.py
                resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
            except Exception as e:
                logging.error(f"Error resizing frame {count} from {video_path}: {e}")
                count += 1
                continue

            # Save resized frame
            frame_name = f"{saved:04d}.png"
            frame_path = os.path.join(output_folder, frame_name)
            compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
            cv2.imwrite(frame_path, resized_frame, compression_params)
            saved += 1
            
        count += 1
            
    vidcap.release()
    return saved


def process_single_video_extraction_resize(task):
    """Process a single video in a worker process with direct resizing.
    
    Args:
        task: Tuple containing (video_path, gloss_label, output_root, split_output, 
              sample_rate, flip_vertical, crop_dims, target_size)
    
    Returns:
        Tuple with (video_id, relative_path, gloss_label) or None if failed
    """
    video_path, gloss_label, output_root, split_output, sample_rate, flip_vertical, crop_dims, target_size = task

    try:
        video_filename = Path(video_path).stem
        output_folder = Path(split_output) / video_filename
        output_folder.mkdir(parents=True, exist_ok=True)

        saved_frames = extract_and_resize_frames(
            video_path, output_folder, target_size, sample_rate, flip_vertical, crop_dims
        )

        # Compute relative path from output_root
        relative_path = output_folder.relative_to(Path(output_root))

        return (video_filename, str(relative_path), gloss_label)
    except Exception as e:
        logging.error(f"Failed processing video {video_path}: {e}")
        return None


def collect_video_tasks_flat(root_dir, output_dir, split, sample_rate=1, flip_vertical=False, crop_dims=None, target_size=(256, 256)):
    """Collect all video processing tasks for a given dataset split."""
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
                crop_dims,
                target_size
            ))

    return tasks


def process_split_with_resize(root_dir, output_dir, csv_dir, split, num_procs, sample_rate, chunk_size, flip_vertical, crop_dims, target_size):
    """Process videos in a dataset split with direct resizing during extraction."""
    start = time.time()
    split_output = Path(output_dir) / split
    split_output.mkdir(parents=True, exist_ok=True)
    Path(csv_dir).mkdir(parents=True, exist_ok=True)

    tasks = collect_video_tasks_flat(root_dir, output_dir, split, sample_rate, flip_vertical, crop_dims, target_size)
    if not tasks:
        logging.warning(f"No tasks for split '{split}' — skipping")
        return

    total = len(tasks)
    logging.info(f"→ {total} videos queued in '{split}' with direct resize to {target_size}")
    
    if not chunk_size or chunk_size < 1:
        if total < 50:
            chunk_size = max(5, total // (num_procs * 2))
        elif total < 200:
            chunk_size = max(3, total // (num_procs * 3))
        else:
            chunk_size = max(1, total // (num_procs * 4))
        logging.info(f"  auto‐calculated chunksize: {chunk_size}")

    results = []
    with Pool(num_procs) as pool:
        for res in tqdm(pool.imap_unordered(process_single_video_extraction_resize, tasks, chunksize=chunk_size),
                        total=total, desc=f"Extracting+Resizing {split}"):
            if res:
                results.append(res)

    results.sort(key=lambda r: r[0])
    csv_path = Path(csv_dir) / f"{split}_annotations.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as fp:
        wr = csv.writer(fp)
        wr.writerow(["Video_ID", "Frames_Path", "Gloss_Label"])
        wr.writerows(results)

    logging.info(f"★ '{split}' done in {time.time() - start:.1f}s, CSV → {csv_path}")


class Preprocessing:
    @staticmethod
    def annotation2dict(dataset_root, anno_path, split):
        # Load annotation data
        df = pd.read_csv(anno_path)

        print(f"Generate information dict from {anno_path}")

        info_dict = {}  # For storing data in dict format

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            video_id = str(row["Video_ID"])
            relative_path = row["Frames_Path"]
            frame_folder = os.path.join(dataset_root, relative_path)  # absolute path frame

            gloss = str(row['Gloss_Label'])
            label = gloss.replace("-", " ").lower()  # Extract gloss label

            if not os.path.exists(frame_folder):
                print(f'Warning: Frames folder not found -> {frame_folder}')

            # Count number of frame
            num_frames = len([f for f in os.listdir(frame_folder) if f.lower().endswith((".jpg", ".png"))])

            # Extract Signer ID
            signer_id_raw = video_id.split('_')[0]  # Extract person ID from video ID
            signer_id = signer_id_raw.lower()

            # Store structured data in dictionary format
            info_dict[idx] = {
                "fileid": video_id,  # Unique file ID
                "folder": os.path.join(relative_path, "*.*"),  # Match all image types
                "signer": signer_id,  # Signer ID
                "label": label,  # Label (gloss)
                "num_frames": num_frames,  # Number of frames
                "original_info": f"{video_id}|{num_frames}|{gloss}"  # Original info string
            }

        return info_dict

    @staticmethod
    def generate_stm(info_dict, save_path):
        try:
            with open(save_path, 'w') as f:
                for k, v in info_dict.items():
                    if not isinstance(k, int):  # Ensure key is an integer index
                        continue
                    f.writelines(f"{v['fileid']} 1 {v['signer']} 0.0 1.79769e+308 {v['label']}\n")
                    # 0.0 1.79769e+308 mean from start to the end of the video

            print(f"Ground truth STM saved to {save_path}")
            print("STM generation completed successfully!")
        except Exception as e:
            print(f"An error occurred while generating the STM file: {e}")

    @staticmethod
    def gloss_dict_update(total_dict, info_dict):
        next_id = 1  # Start from 1, as 0 is reserved for the blank token
        for k, v in info_dict.items():
            if not isinstance(k, int):
                continue

            # Split the label by whitespace; if the label contains multiple tokens, count each separately
            tokens = v['label'].split()
            for token in tokens:
                token = token.strip()
                if not token:
                    continue
                if token not in total_dict:
                    total_dict[token] = [next_id, 1]  # [Gloss ID, Occurrence Count]
                    next_id += 1
                else:
                    total_dict[token][1] += 1  # Increment occurrence count

        return total_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Combined frame extraction and preprocessing with direct resizing for VSL dataset.')

    # Original vsl_preprocess arguments
    parser.add_argument('--dataset-prefix', type=str, default='vsl_v3',
                        help='Save prefix for ground truth file')
    parser.add_argument('--processed-feature-root', type=str,
                        default='/home/martinvalentine/Desktop/v-sign/data/processed/VSL_V3',
                        help='Path to save the processed feature')
    parser.add_argument('--dataset-root', type=str,
                        default='/home/martinvalentine/Desktop/v-sign/data/interim/256x256px/VSL_V3',
                        help='Path to the dataset root (where frame folders are located)')
    parser.add_argument('--annotation-prefix', type=str,
                        default='/home/martinvalentine/Desktop/v-sign/data/splits/VSL_V2/csv/{}_annotations.csv',
                        help='Path template for CSV annotations with mode placeholder (train/test/dev)')
    parser.add_argument('--output-res', type=str, default='256x256px',
                        help='Resize resolution for image sequences, e.g., 256x256px')

    # Frame extraction arguments
    parser.add_argument('--video-root', type=str, 
                        default='/home/martinvalentine/Desktop/v-sign/data/raw/VSL_V3',
                        help='Root directory containing source videos')
    parser.add_argument('--extract-frames', action='store_true',
                        help='Extract frames from videos with direct resizing')
    parser.add_argument('--splits', nargs='+', default=["train", "test", "dev"],
                        help='Dataset splits to process')
    parser.add_argument('--sample-rate', type=int, default=2,
                        help='Frame sampling rate (extract every Nth frame)')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 2),
                        help='Number of worker processes')
    parser.add_argument('--reserve-cores', type=int, default=2,
                        help='CPU cores to reserve for system')
    parser.add_argument('--chunksize', type=int, default=0,
                        help='Task chunk size for multiprocessing')
    parser.add_argument('--flip-vertical', action='store_true',
                        help='Flip video frames vertically before saving')
    parser.add_argument('--crop-width', type=int, default=0,
                        help='Target width for cropping before resizing')
    parser.add_argument('--crop-height', type=int, default=0,
                        help='Target height for cropping before resizing')
    
    # Legacy arguments for compatibility
    parser.add_argument('--multiprocessing', '-m', action='store_true',
                        help='Use multiprocessing (always enabled for frame extraction)')

    return parser.parse_args()


def main():
    """Main entry point for combined frame extraction and preprocessing."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    # Parse target size from output_res
    target_size = tuple(int(res) for res in re.findall("\d+", args.output_res))
    
    # If frame extraction is requested
    if args.extract_frames:
        # Calculate optimal worker count
        total_cores = cpu_count()
        available_cores = max(1, total_cores - args.reserve_cores)
        workers = min(args.workers, available_cores)
        
        logging.info(f"CPU optimization: Using {workers} worker processes (reserving {args.reserve_cores} of {total_cores} cores)")
        logging.info(f"Extracting frames with direct resize to {target_size}")

        # Set up cropping if dimensions provided
        crop_dims = (args.crop_width, args.crop_height) if args.crop_width > 0 and args.crop_height > 0 else None

        # Update paths for frame extraction
        csv_root = args.annotation_prefix.replace('/{}_annotations.csv', '')
        
        for split in args.splits:
            logging.info(f"--- Extracting frames for split '{split}' ---")
            process_split_with_resize(
                args.video_root,
                args.dataset_root,
                csv_root,
                split,
                num_procs=workers,
                sample_rate=args.sample_rate,
                chunk_size=args.chunksize,
                flip_vertical=args.flip_vertical,
                crop_dims=crop_dims,
                target_size=target_size
            )

    # Original vsl_preprocess logic for annotation processing
    modes = ["dev", "test", "train"]
    sign_dict = {}

    os.makedirs(args.processed_feature_root, exist_ok=True)

    for mode in modes:
        anno_path = args.annotation_prefix.format(mode)
        
        # Check if annotation file exists
        if not os.path.exists(anno_path):
            logging.warning(f"Annotation file not found: {anno_path}, skipping {mode}")
            continue

        # Load annotation info from CSV and convert to dict
        info_dict = Preprocessing.annotation2dict(args.dataset_root, anno_path, split=mode)
        np.save(os.path.join(args.processed_feature_root, f"{mode}_info.npy"), info_dict)

        # Update global gloss dictionary
        Preprocessing.gloss_dict_update(sign_dict, info_dict)

        # Save ground truth STM file
        stm_path = os.path.join(args.processed_feature_root, f"{args.dataset_prefix}-ground-truth-{mode}.stm".lower())
        Preprocessing.generate_stm(info_dict, stm_path)

        # Save ground truth to evaluation folder
        eval_path = "/home/martinvalentine/Desktop/v-sign/src/vsign/evaluation/slr_eval/"
        os.makedirs(eval_path, exist_ok=True)
        stm_eval_path = os.path.join(eval_path, f"{args.dataset_prefix}-ground-truth-{mode}.stm".lower())
        Preprocessing.generate_stm(info_dict, stm_eval_path)

    # Save sorted gloss dictionary (sorted by gloss name)
    if sign_dict:
        sorted_gloss = sorted(sign_dict.items(), key=lambda x: x[0])
        save_dict = {k: [i + 1, v[1]] for i, (k, v) in enumerate(sorted_gloss)}
        np.save(os.path.join(args.processed_feature_root, "gloss_dict.npy"), save_dict)
        logging.info("Gloss dictionary saved successfully!")

    logging.info("All processing completed!")


if __name__ == "__main__":
    main() 
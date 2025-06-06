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
    """Extract frames from video with memory-efficient batch processing.
    
    Args:
        video_path: Source video file
        output_folder: Where to save extracted frames
        sample_rate: Extract every Nth frame
        flip_vertical: Whether to flip frames vertically
        crop_dims: (width, height) for center cropping
    """
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    os.makedirs(output_folder, exist_ok=True)
    
    # Get video metadata for memory optimization
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate batch size based on frame dimensions to optimize memory use
    # Larger frames = smaller batches to manage memory better
    frame_size_mb = (frame_width * frame_height * 3) / (1024 * 1024)  # Approx MB per frame (RGB)
    batch_size = max(5, min(50, int(256 / max(1, frame_size_mb))))    # Target ~256MB per batch
    
    count = 0  # Total frame counter
    saved = 0  # Saved frame counter
    
    # Process frames in batches to optimize memory usage
    while True:
        batch_frames = []
        batch_indices = []
        
        # Read a batch of frames
        for _ in range(batch_size):
            success, frame = vidcap.read()
            if not success:
                break
                
            if count % sample_rate == 0:  # Only keep frames matching sample rate
                batch_frames.append(frame)
                batch_indices.append(count)
            count += 1
            
        if not batch_frames:
            break
            
        # Process each frame in the batch
        for frame, frame_idx in zip(batch_frames, batch_indices):
            if flip_vertical:
                frame = cv2.flip(frame, 0)

            if crop_dims:
                target_w, target_h = crop_dims
                original_h, original_w = frame.shape[:2]  # Get original dimensions

                # Center-crop frame if it's large enough
                if original_w >= target_w and original_h >= target_h:
                    start_x = (original_w - target_w) // 2
                    end_x = start_x + target_w
                    start_y = (original_h - target_h) // 2
                    end_y = start_y + target_h
                    frame = frame[start_y:end_y, start_x:end_x]
                else:
                    # Log a warning if frame is too small
                    logging.warning(f"Video {video_path}: Frame {frame_idx} ({original_w}x{original_h}) too small for crop to ({target_w}x{target_h})")

            # Save frame with optimized compression (level 3 balances speed and size)
            frame_name = f"{saved:04d}.png"
            frame_path = os.path.join(output_folder, frame_name)
            compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
            cv2.imwrite(frame_path, frame, compression_params)
            saved += 1
            
            # Free memory explicitly
            del frame
            
    vidcap.release()

def process_single_video_extraction_raw(task):
    """Process a single video in a worker process.
    
    Args:
        task: Tuple containing (video_path, gloss_label, output_root, split_output, 
              sample_rate, flip_vertical, crop_dims)
    
    Returns:
        Tuple with (video_id, relative_path, gloss_label) or None if failed
    """
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
    """Collect all video processing tasks for a given dataset split.
    
    Args:
        root_dir: Root directory of video dataset
        output_dir: Directory to save extracted frames
        split: Dataset split ('train', 'test', 'dev')  
        sample_rate: Frame sampling rate
        flip_vertical: Whether to flip frames vertically
        crop_dims: Optional dimensions for cropping frames
        
    Returns:
        List of tasks for parallel processing
    """
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
    """Process videos in a dataset split with parallel CPU+RAM-optimized extraction.
    
    Uses multiprocessing with adaptive chunk sizes to balance CPU cores and memory usage.
    Designed to work efficiently on systems with 32GB RAM while preventing freezes.
    
    Args:
        root_dir: Root directory containing videos
        output_dir: Output directory for extracted frames
        csv_dir: Directory for annotation CSV files
        split: Dataset split ('train', 'test', 'dev')
        num_procs: Number of parallel processes to use
        sample_rate: Extract every Nth frame
        chunk_size: Task chunk size (0=auto)
        flip_vertical: Whether to flip frames
        crop_dims: (width, height) tuple for cropping
    """
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
        # Optimize chunk size for 32GB RAM system:
        # - Smaller chunks (1-3) for better load balancing with many videos
        # - Larger chunks (5-10) for fewer videos to reduce overhead
        if total < 50:  # Small dataset
            chunk_size = max(5, total // (num_procs * 2))
        elif total < 200:  # Medium dataset
            chunk_size = max(3, total // (num_procs * 3))
        else:  # Large dataset
            chunk_size = max(1, total // (num_procs * 4))
        logging.info(f"  auto‐calculated chunksize: {chunk_size} for optimal performance")

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
    parser.add_argument('--video_root', default="/home/martinvalentine/Desktop/v-sign/data/raw/VSL_V2", type=str)
    parser.add_argument('--output_root', default="/home/martinvalentine/Desktop/v-sign/data/interim/frames/VSL_V2", type=str)
    parser.add_argument('--csv_root', default="/home/martinvalentine/Desktop/v-sign/data/splits/VSL_V2/csv", type=str)
    parser.add_argument('--splits', nargs='+', default=["train", "test", "dev"])
    parser.add_argument('--sample_rate', type=int, default=2)
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 2))
    parser.add_argument('--reserve_cores', type=int, default=2, help="CPU cores to reserve for system (prevents freezing)")
    parser.add_argument('--chunksize', type=int, default=0)
    parser.add_argument('--flip_vertical', action='store_true', help="Flip video frames vertically before saving")
    parser.add_argument('--crop_width', type=int, default=0, help="Target width for cropping")
    parser.add_argument('--crop_height', type=int, default=0, help="Target height for cropping")
    return parser.parse_args()


def main():
    """Main entry point for frame extraction with CPU optimization."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    # Calculate optimal worker count to prevent system freeze
    total_cores = cpu_count()
    available_cores = max(1, total_cores - args.reserve_cores)
    # Use the smaller of available cores or user-specified workers
    workers = min(args.workers, available_cores)
    
    logging.info(f"CPU optimization: Using {workers} worker processes (reserving {args.reserve_cores} of {total_cores} cores)")
    logging.info(f"Memory optimization: Using batch processing to optimize 32GB RAM usage")

    # Set up cropping if dimensions provided
    crop_dims = (args.crop_width, args.crop_height) if args.crop_width > 0 and args.crop_height > 0 else None

    for split in args.splits:
        logging.info(f"--- processing split '{split}' ---")
        process_split_optimized_raw_flat(
            args.video_root,
            args.output_root,
            args.csv_root,
            split,
            num_procs=workers,
            sample_rate=args.sample_rate,
            chunk_size=args.chunksize,
            flip_vertical=args.flip_vertical,
            crop_dims=crop_dims
        )

    logging.info("All splits processed and frame annotations generated.")


if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def print_gloss_dict(gloss_dict_path):
    """Print the full gloss dictionary with IDs and occurrence counts."""
    if not os.path.exists(gloss_dict_path):
        print(f"❌ Gloss dictionary not found at: {gloss_dict_path}")
        return
    
    print("=" * 80)
    print("📚 GLOSS DICTIONARY")
    print("=" * 80)
    
    gloss_dict = np.load(gloss_dict_path, allow_pickle=True).item()
    
    print(f"Total unique glosses: {len(gloss_dict)}")
    print(f"Format: [Gloss ID, Occurrence Count]")
    print("-" * 80)
    
    # Sort by gloss name for better readability
    sorted_glosses = sorted(gloss_dict.items(), key=lambda x: x[0])
    
    for i, (gloss, info) in enumerate(sorted_glosses, 1):
        gloss_id, count = info
        print(f"{i:3d}. {gloss:<30} | ID: {gloss_id:3d} | Count: {count:4d}")
    
    print("-" * 80)
    
    # Statistics
    counts = [info[1] for info in gloss_dict.values()]
    total_occurrences = sum(counts)
    avg_count = total_occurrences / len(counts)
    min_count = min(counts)
    max_count = max(counts)
    
    print(f"📊 Statistics:")
    print(f"   Total occurrences: {total_occurrences}")
    print(f"   Average per gloss: {avg_count:.2f}")
    print(f"   Min occurrences:   {min_count}")
    print(f"   Max occurrences:   {max_count}")
    
    # Most and least frequent glosses
    most_frequent = max(gloss_dict.items(), key=lambda x: x[1][1])
    least_frequent = min(gloss_dict.items(), key=lambda x: x[1][1])
    
    print(f"   Most frequent:     '{most_frequent[0]}' ({most_frequent[1][1]} times)")
    print(f"   Least frequent:    '{least_frequent[0]}' ({least_frequent[1][1]} times)")
    print()


def print_split_info(info_path, split_name):
    """Print information about a specific split (train/test/dev)."""
    if not os.path.exists(info_path):
        print(f"❌ Info file not found for {split_name}: {info_path}")
        return
    
    print("=" * 80)
    print(f"📁 {split_name.upper()} SPLIT INFORMATION")
    print("=" * 80)
    
    info_dict = np.load(info_path, allow_pickle=True).item()
    
    print(f"Total videos: {len(info_dict)}")
    print("-" * 80)
    
    # Collect statistics
    signers = set()
    labels = []
    frame_counts = []
    
    for idx, data in info_dict.items():
        if not isinstance(idx, int):
            continue
            
        signers.add(data['signer'])
        labels.append(data['label'])
        frame_counts.append(data['num_frames'])
    
    print(f"📊 Statistics:")
    print(f"   Unique signers:    {len(signers)}")
    print(f"   Unique labels:     {len(set(labels))}")
    print(f"   Total frames:      {sum(frame_counts)}")
    print(f"   Avg frames/video:  {np.mean(frame_counts):.2f}")
    print(f"   Min frames:        {min(frame_counts)}")
    print(f"   Max frames:        {max(frame_counts)}")
    print()
    
    print(f"👥 Signers: {sorted(signers)}")
    print()
    
    # Show first few samples
    print("📋 Sample entries:")
    print("-" * 80)
    sample_count = min(5, len(info_dict))
    for i, (idx, data) in enumerate(list(info_dict.items())[:sample_count]):
        if not isinstance(idx, int):
            continue
        print(f"{i+1}. Video ID: {data['fileid']}")
        print(f"   Signer: {data['signer']}")
        print(f"   Label: {data['label']}")
        print(f"   Frames: {data['num_frames']}")
        print(f"   Folder: {data['folder']}")
        print()


def print_stm_info(stm_path, split_name):
    """Print information about STM (ground truth) files."""
    if not os.path.exists(stm_path):
        print(f"❌ STM file not found for {split_name}: {stm_path}")
        return
    
    print("=" * 80)
    print(f"📄 {split_name.upper()} STM FILE")
    print("=" * 80)
    
    with open(stm_path, 'r') as f:
        lines = f.readlines()
    
    print(f"Total entries: {len(lines)}")
    print(f"File path: {stm_path}")
    print()
    
    print("📋 Sample STM entries:")
    print("-" * 80)
    sample_count = min(3, len(lines))
    for i, line in enumerate(lines[:sample_count]):
        parts = line.strip().split()
        if len(parts) >= 6:
            video_id = parts[0]
            channel = parts[1]
            speaker = parts[2]
            start_time = parts[3]
            end_time = parts[4]
            label = ' '.join(parts[5:])
            
            print(f"{i+1}. {video_id} | Speaker: {speaker} | Label: '{label}'")
    print()


def print_csv_info(csv_path, split_name):
    """Print information about CSV annotation files."""
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found for {split_name}: {csv_path}")
        return
    
    print("=" * 80)
    print(f"📊 {split_name.upper()} CSV ANNOTATIONS")
    print("=" * 80)
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Total entries: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print()
        
        # Show unique values counts
        if 'Gloss_Label' in df.columns:
            unique_labels = df['Gloss_Label'].nunique()
            print(f"Unique gloss labels: {unique_labels}")
        
        if 'Video_ID' in df.columns:
            unique_videos = df['Video_ID'].nunique()
            print(f"Unique video IDs: {unique_videos}")
        
        print()
        print("📋 Sample CSV entries:")
        print("-" * 80)
        print(df.head(3).to_string(index=False))
        print()
        
    except Exception as e:
        print(f"Error reading CSV: {e}")


def print_directory_structure(processed_root):
    """Print the directory structure of processed files."""
    print("=" * 80)
    print("📂 DIRECTORY STRUCTURE")
    print("=" * 80)
    
    if not os.path.exists(processed_root):
        print(f"❌ Processed root directory not found: {processed_root}")
        return
    
    print(f"Root: {processed_root}")
    print()
    
    for root, dirs, files in os.walk(processed_root):
        level = root.replace(processed_root, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        subindent = ' ' * 2 * (level + 1)
        for file in sorted(files):
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            
            # Format file size
            if file_size < 1024:
                size_str = f"{file_size}B"
            elif file_size < 1024**2:
                size_str = f"{file_size/1024:.1f}KB"
            elif file_size < 1024**3:
                size_str = f"{file_size/(1024**2):.1f}MB"
            else:
                size_str = f"{file_size/(1024**3):.1f}GB"
            
            print(f"{subindent}{file} ({size_str})")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Print information from processed VSL dataset files.')
    
    parser.add_argument('--processed-root', type=str,
                        default='/home/kafka/Desktop/v-sign/data/processed/VSL_V2',
                        help='Path to processed feature root')
    parser.add_argument('--csv-root', type=str,
                        default='/home/kafka/Desktop/v-sign/data/splits/VSL_V2/csv',
                        help='Path to CSV annotations root')
    parser.add_argument('--dataset-prefix', type=str, default='vsl_v2',
                        help='Dataset prefix for STM files')
    parser.add_argument('--splits', nargs='+', default=["train", "test", "dev"],
                        help='Dataset splits to analyze')
    parser.add_argument('--show-gloss-dict', action='store_true', default=True,
                        help='Show full gloss dictionary')
    parser.add_argument('--show-splits', action='store_true', default=True,
                        help='Show split information')
    parser.add_argument('--show-stm', action='store_true', default=True,
                        help='Show STM file information')
    parser.add_argument('--show-csv', action='store_true', default=True,
                        help='Show CSV file information')
    parser.add_argument('--show-structure', action='store_true', default=True,
                        help='Show directory structure')
    
    return parser.parse_args()


def main():
    """Main function to print all processed file information."""
    args = parse_args()
    
    print("🔍 VSL DATASET PROCESSED FILES INFORMATION")
    print("=" * 80)
    print(f"Processed root: {args.processed_root}")
    print(f"CSV root: {args.csv_root}")
    print(f"Dataset prefix: {args.dataset_prefix}")
    print()
    
    # Show directory structure
    if args.show_structure:
        print_directory_structure(args.processed_root)
    
    # Show gloss dictionary
    if args.show_gloss_dict:
        gloss_dict_path = os.path.join(args.processed_root, "gloss_dict.npy")
        print_gloss_dict(gloss_dict_path)
    
    # Show information for each split
    for split in args.splits:
        if args.show_splits:
            info_path = os.path.join(args.processed_root, f"{split}_info.npy")
            print_split_info(info_path, split)
        
        if args.show_stm:
            stm_path = os.path.join(args.processed_root, f"{args.dataset_prefix}-ground-truth-{split}.stm")
            print_stm_info(stm_path, split)
        
        if args.show_csv:
            csv_path = os.path.join(args.csv_root, f"{split}_annotations.csv")
            print_csv_info(csv_path, split)
    
    print("✅ Analysis complete!")


if __name__ == "__main__":
    main() 
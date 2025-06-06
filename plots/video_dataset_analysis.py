#!/usr/bin/env python3
"""
Video Dataset Analysis Script for VSL_V0/VSL_V1
Analyzes video duration, vocabulary size, and dataset statistics
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import time

def get_video_duration(video_path):
    """Get video duration in seconds using OpenCV"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        
        if fps > 0:
            duration = frame_count / fps
            return duration
        return None
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

def analyze_video_durations(video_dir):
    """Analyze video durations in the dataset"""
    durations = []
    
    # Walk through all subdirectories to find video files
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                try:
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    
                    if fps > 0:
                        duration = frame_count / fps
                        durations.append(duration)
                    
                    cap.release()
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
    
    return durations

def analyze_dataset_stats(video_dir):
    """Analyze dataset statistics: number of signers and average words per sample"""
    signers = set()
    word_counts = []
    
    # Walk through all subdirectories to find video files
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith('.mp4'):
                # Extract information from filename
                filename_base = os.path.splitext(file)[0]
                parts = filename_base.split('_')
                
                # Extract signer information from filename
                if len(parts) >= 1 and parts[0].startswith('Signer'):
                    signer_num = parts[0].replace('Signer', '')
                    signers.add(signer_num)
                
                # Extract phrase and count words (skip signer and sequence number)
                if len(parts) >= 3:
                    # Join all parts except signer (first) and sequence number (last part)
                    phrase_parts = parts[1:-1]
                    phrase = '_'.join(phrase_parts)
                    
                    # Count words by splitting ONLY on hyphens (-), not underscores (_)
                    words = phrase.split('-')  # Only split on hyphen
                    word_counts.append(len(words))
    
    num_signers = len(signers)
    avg_words = np.mean(word_counts) if word_counts else 0
    
    return num_signers, avg_words

def load_vocabulary_size(gloss_dict_path):
    """Load vocabulary size from gloss dictionary"""
    try:
        gloss_dict = np.load(gloss_dict_path, allow_pickle=True).item()
        return len(gloss_dict)
    except Exception as e:
        print(f"Error loading gloss dictionary: {e}")
        return 0

def create_dataset_summary_table(dataset_name, stats, output_path):
    """Create a summary table visualization"""
    
    # Create figure and axis with minimal size
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    # Create table data (no title row needed)
    table_data = [
        ["Metric", "Value"],
        ["Max Duration (seconds)", f"{stats['max_duration']:.1f}"],
        ["Min Duration (seconds)", f"{stats['min_duration']:.1f}"],
        ["Vocabulary Size", f"{stats['vocab_size']}"],
        ["Number of Signers", f"{stats['num_signers']}"],
        ["Avg Words/Sample", f"{stats['avg_words']:.1f}"],
        ["Total Videos", f"{stats['total_videos']:,}"],
        ["Total Hours", f"{stats['total_hours']:.2f}"]
    ]
    
    # Table styling with minimal spacing
    row_height = 0.5
    start_y = 4.5
    
    # Draw table
    for i, row in enumerate(table_data):
        y_pos = start_y - i * row_height
        
        # Header row styling
        if i == 0:
            # Header background
            header_rect = patches.Rectangle((0.5, y_pos - 0.2), 7, 0.4, 
                                          linewidth=1, edgecolor='black', 
                                          facecolor='lightgray', alpha=0.7)
            ax.add_patch(header_rect)
            
            # Header text
            ax.text(1, y_pos, row[0], fontsize=11, fontweight='bold', va='center')
            ax.text(5.5, y_pos, row[1], fontsize=11, fontweight='bold', va='center', ha='center')
        else:
            # Data row background (alternating colors)
            if i % 2 == 0:
                row_rect = patches.Rectangle((0.5, y_pos - 0.2), 7, 0.4, 
                                           linewidth=1, edgecolor='gray', 
                                           facecolor='white', alpha=0.8)
            else:
                row_rect = patches.Rectangle((0.5, y_pos - 0.2), 7, 0.4, 
                                           linewidth=1, edgecolor='gray', 
                                           facecolor='#f8f8f8', alpha=0.8)
            ax.add_patch(row_rect)
            
            # Data text
            ax.text(1, y_pos, row[0], fontsize=10, va='center')
            ax.text(5.5, y_pos, row[1], fontsize=10, va='center', ha='center', fontweight='bold')
    
    # Add border around entire table
    table_border = patches.Rectangle((0.5, start_y - len(table_data) * row_height + 0.3), 
                                   7, len(table_data) * row_height - 0.1, 
                                   linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(table_border)
    
    # Save the figure with minimal margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight', 
                pad_inches=0, facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"Dataset summary saved to: {output_path}")

def analyze_dataset(dataset_name, video_dir, gloss_dict_path, output_path):
    """Analyze a complete dataset and create summary"""
    
    print(f"Analyzing {dataset_name} dataset...")
    print(f"Video directory: {video_dir}")
    print(f"Gloss dictionary: {gloss_dict_path}")
    print("-" * 50)
    
    # Analyze video durations
    print("Analyzing video durations...")
    durations = analyze_video_durations(video_dir)
    
    if not durations:
        print("No video files found!")
        return
    
    # Calculate duration statistics
    max_duration = max(durations)
    min_duration = min(durations)
    total_hours = sum(durations) / 3600
    total_videos = len(durations)
    
    # Load vocabulary size
    print("Loading vocabulary size...")
    vocab_size = load_vocabulary_size(gloss_dict_path)
    
    # Analyze dataset statistics
    print("Analyzing dataset statistics...")
    num_signers, avg_words = analyze_dataset_stats(video_dir)
    
    # Compile statistics
    stats = {
        'max_duration': max_duration,
        'min_duration': min_duration,
        'vocab_size': vocab_size,
        'num_signers': num_signers,
        'avg_words': avg_words,
        'total_videos': total_videos,
        'total_hours': total_hours
    }
    
    # Print results
    print(f"\n{dataset_name} Dataset Analysis Results:")
    print(f"Max duration: {max_duration:.1f} seconds")
    print(f"Min duration: {min_duration:.1f} seconds")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of signers: {num_signers}")
    print(f"Average words per sample: {avg_words:.1f}")
    print(f"Total videos: {total_videos:,}")
    print(f"Total hours: {total_hours:.2f}")
    
    # Create visualization
    print(f"\nCreating summary table...")
    create_dataset_summary_table(dataset_name, stats, output_path)
    
    return stats

def main():
    """Main function to analyze the dataset"""
    
    # Define datasets to analyze - change this to switch between datasets
    # For VSL_V0: uncomment the first set, comment the second
    # For VSL_V1: uncomment the second set, comment the first
    
    # VSL_V0 Configuration
    dataset_name = "VSL_V0"
    video_dir = "/home/martinvalentine/Desktop/v-sign/data/raw/VSL_V0"
    gloss_dict_path = "/home/martinvalentine/Desktop/v-sign/data/processed/VSL_V0/gloss_dict.npy"
    output_path = "/home/martinvalentine/Desktop/v-sign/plots/vsl_v0_dataset_summary.svg"
    
    # VSL_V1 Configuration (current)
    # dataset_name = "VSL_V1"
    # video_dir = "/home/martinvalentine/Desktop/v-sign/data/raw/VSL_V1"
    # gloss_dict_path = "/home/martinvalentine/Desktop/v-sign/data/processed/VSL_V1/gloss_dict.npy"
    # output_path = "/home/martinvalentine/Desktop/v-sign/plots/vsl_v1_dataset_summary.svg"
    
    # Check if paths exist
    if not os.path.exists(video_dir):
        print(f"Error: Video directory does not exist: {video_dir}")
        return
    
    if not os.path.exists(gloss_dict_path):
        print(f"Error: Gloss dictionary does not exist: {gloss_dict_path}")
        return
    
    # Create plots directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Video Dataset Analysis")
    print("=" * 50)
    
    # Analyze the dataset
    stats = analyze_dataset(dataset_name, video_dir, gloss_dict_path, output_path)
    
    print("\n" + "=" * 50)
    print("Analysis completed!")

if __name__ == "__main__":
    main() 
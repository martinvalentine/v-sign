#!/usr/bin/env python3
"""
Dataset Sample Visualization Script for VSL_V0
Creates a clean grid showing random sample images from the dataset
"""

import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

def find_all_image_directories(base_path):
    """Find all directories containing images from all signers"""
    all_dirs = []
    
    # Search in all splits (train, dev, test)
    for split in ['train', 'dev', 'test']:
        split_path = os.path.join(base_path, split)
        if os.path.exists(split_path):
            for item in os.listdir(split_path):
                item_path = os.path.join(split_path, item)
                if os.path.isdir(item_path) and item.startswith('Signer'):
                    all_dirs.append(item_path)
    
    return all_dirs

def get_sample_image(directory_path):
    """Get a sample image from a directory (preferably middle frame)"""
    try:
        image_files = [f for f in os.listdir(directory_path) if f.endswith('.png')]
        if not image_files:
            return None
        
        # Sort files and pick middle frame for better representation
        image_files.sort()
        middle_idx = len(image_files) // 2
        sample_image = os.path.join(directory_path, image_files[middle_idx])
        
        return sample_image
    except Exception as e:
        print(f"Error getting sample from {directory_path}: {e}")
        return None

def create_sample_grid(base_path, output_path):
    """Create a clean grid of random sample images"""
    
    # Find all available directories
    all_dirs = find_all_image_directories(base_path)
    
    if len(all_dirs) < 12:
        print(f"Warning: Only found {len(all_dirs)} directories, but need 12 for the grid")
    
    # Randomly sample 12 directories
    num_samples = min(12, len(all_dirs))
    selected_dirs = random.sample(all_dirs, num_samples)
    
    # Create figure with better spacing and cleaner look
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Remove the main title for cleaner look
    # fig.suptitle('VSL_V0 Dataset Sample Images', fontsize=20, fontweight='bold', y=0.95)
    
    sample_count = 0
    
    # Fill the grid with random samples
    for i in range(12):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        if i < len(selected_dirs):
            # Get sample image from this directory
            sample_image_path = get_sample_image(selected_dirs[i])
            
            if sample_image_path and os.path.exists(sample_image_path):
                try:
                    # Load and display image
                    img = mpimg.imread(sample_image_path)
                    ax.imshow(img, aspect='equal')
                    sample_count += 1
                    
                except Exception as e:
                    print(f"Error loading image {sample_image_path}: {e}")
                    # Create a placeholder for failed images
                    ax.set_facecolor('#f0f0f0')
                    ax.text(0.5, 0.5, '✗', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=40, color='#999')
            else:
                # Create a placeholder for missing images
                ax.set_facecolor('#f0f0f0')
                ax.text(0.5, 0.5, '○', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=40, color='#999')
        else:
            # Create empty placeholder
            ax.set_facecolor('#f8f8f8')
        
        # Remove all axis elements for clean look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # Adjust layout for better spacing
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, 
                       wspace=0.05, hspace=0.05)
    
    # Save the figure
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight', 
                pad_inches=0.1, facecolor='white')
    plt.show()
    
    print(f"Sample grid saved to: {output_path}")
    print(f"Total samples displayed: {sample_count}")
    
    return sample_count

def main():
    """Main function to create the sample visualization"""
    
    # Define paths
    base_path = "/home/kafka/Desktop/v-sign/data/interim/256x256px/VSL_V1"
    output_path = "/home/kafka/Desktop/v-sign/plots/vsl_v1_dataset_samples.svg"
    
    # Check if base path exists
    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}")
        return
    
    # Create plots directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Creating VSL_V0 dataset sample visualization...")
    print(f"Source: {base_path}")
    print(f"Output: {output_path}")
    print("="*60)
    
    # Create the sample grid
    sample_count = create_sample_grid(base_path, output_path)
    
    print("="*60)
    print("Dataset sample visualization completed!")
    
    if sample_count > 0:
        print(f"Successfully displayed {sample_count} random sample images")
        print("Clean grid layout: 4 columns × 3 rows")
        print("Each sample shows a random frame from the sign language dataset")
    else:
        print("Warning: No samples could be displayed")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Gloss Visualization Script for VSL_V0 Dataset
Visualizes gloss dictionary and sentence information from the dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import os

def load_gloss_dictionary(gloss_dict_path):
    """Load and analyze the gloss dictionary"""
    try:
        gloss_dict = np.load(gloss_dict_path, allow_pickle=True).item()
        print(f"Loaded gloss dictionary with {len(gloss_dict)} entries")
        return gloss_dict
    except Exception as e:
        print(f"Error loading gloss dictionary: {e}")
        return {}

def load_dev_info(dev_info_path):
    """Load and analyze the development set information"""
    try:
        dev_info = np.load(dev_info_path, allow_pickle=True).item()
        print(f"Loaded dev info with {len(dev_info)} entries")
        return dev_info
    except Exception as e:
        print(f"Error loading dev info: {e}")
        return {}

def analyze_gloss_statistics(gloss_dict):
    """Analyze statistics from the gloss dictionary"""
    if not gloss_dict:
        return {}
    
    # Extract word frequencies and indices
    words = list(gloss_dict.keys())
    indices = [gloss_dict[word][0] for word in words]
    frequencies = [gloss_dict[word][1] for word in words]
    
    stats = {
        'total_words': len(words),
        'words': words,
        'indices': indices,
        'frequencies': frequencies,
        'total_occurrences': sum(frequencies),
        'avg_frequency': np.mean(frequencies),
        'max_frequency': max(frequencies),
        'min_frequency': min(frequencies)
    }
    
    return stats

def analyze_sentence_statistics(dev_info):
    """Analyze statistics from the sentence information"""
    if not dev_info:
        return {}
    
    # Extract information from dev_info
    signers = []
    labels = []
    num_frames = []
    
    for entry in dev_info.values():
        signers.append(entry['signer'])
        labels.append(entry['label'])
        num_frames.append(entry['num_frames'])
    
    # Count unique signers and sentences
    unique_signers = list(set(signers))
    unique_sentences = list(set(labels))
    
    # Count words in sentences
    word_counts = []
    for label in labels:
        words = label.split()
        word_counts.append(len(words))
    
    stats = {
        'total_samples': len(dev_info),
        'unique_signers': unique_signers,
        'num_signers': len(unique_signers),
        'unique_sentences': unique_sentences,
        'num_unique_sentences': len(unique_sentences),
        'avg_frames': np.mean(num_frames),
        'max_frames': max(num_frames),
        'min_frames': min(num_frames),
        'avg_words_per_sentence': np.mean(word_counts),
        'word_counts': word_counts,
        'sentence_distribution': Counter(labels),
        'signer_distribution': Counter(signers)
    }
    
    return stats

def create_gloss_frequency_chart(gloss_stats, output_path):
    """Create a bar chart showing gloss word frequencies"""
    if not gloss_stats:
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    words = gloss_stats['words']
    frequencies = gloss_stats['frequencies']
    
    # Sort by frequency for better visualization
    sorted_data = sorted(zip(words, frequencies), key=lambda x: x[1], reverse=True)
    sorted_words, sorted_frequencies = zip(*sorted_data)
    
    # Create bar chart
    bars = ax.bar(range(len(sorted_words)), sorted_frequencies, 
                  color='steelblue', alpha=0.7, edgecolor='black')
    
    # Customize the chart
    ax.set_xlabel('Gloss Words', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('VSL_V0 Gloss Dictionary - Word Frequencies', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(sorted_words)))
    ax.set_xticklabels(sorted_words, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, freq) in enumerate(zip(bars, sorted_frequencies)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(freq), ha='center', va='bottom', fontweight='bold')
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Gloss frequency chart saved to: {output_path}")

def create_sentence_distribution_chart(sentence_stats, output_path):
    """Create a chart showing sentence distribution across signers"""
    if not sentence_stats:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Chart 1: Sentence frequency distribution
    sentence_dist = sentence_stats['sentence_distribution']
    sentences = list(sentence_dist.keys())
    counts = list(sentence_dist.values())
    
    # Sort by frequency
    sorted_data = sorted(zip(sentences, counts), key=lambda x: x[1], reverse=True)
    sorted_sentences, sorted_counts = zip(*sorted_data)
    
    bars1 = ax1.bar(range(len(sorted_sentences)), sorted_counts, 
                    color='lightcoral', alpha=0.7, edgecolor='black')
    
    ax1.set_xlabel('Sentences', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax1.set_title('Sentence Distribution in VSL_V0 Dev Set', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(sorted_sentences)))
    ax1.set_xticklabels(sorted_sentences, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars1, sorted_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Chart 2: Signer distribution
    signer_dist = sentence_stats['signer_distribution']
    signers = list(signer_dist.keys())
    signer_counts = list(signer_dist.values())
    
    bars2 = ax2.bar(signers, signer_counts, 
                    color='lightgreen', alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel('Signers', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax2.set_title('Signer Distribution in VSL_V0 Dev Set', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars2, signer_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Sentence distribution chart saved to: {output_path}")

def create_summary_table(gloss_stats, sentence_stats, output_path):
    """Create a comprehensive summary table"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'VSL_V0 Dataset - Gloss and Sentence Analysis Summary', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Create two-column layout
    # Left column: Gloss Dictionary Statistics
    ax.text(2.5, 10.5, 'Gloss Dictionary Statistics', 
            fontsize=14, fontweight='bold', ha='center')
    
    gloss_data = [
        ["Total Vocabulary Size", f"{gloss_stats.get('total_words', 0)}"],
        ["Total Word Occurrences", f"{gloss_stats.get('total_occurrences', 0)}"],
        ["Average Frequency", f"{gloss_stats.get('avg_frequency', 0):.1f}"],
        ["Max Frequency", f"{gloss_stats.get('max_frequency', 0)}"],
        ["Min Frequency", f"{gloss_stats.get('min_frequency', 0)}"]
    ]
    
    # Right column: Sentence Statistics
    ax.text(7.5, 10.5, 'Sentence Statistics', 
            fontsize=14, fontweight='bold', ha='center')
    
    sentence_data = [
        ["Total Samples", f"{sentence_stats.get('total_samples', 0)}"],
        ["Unique Sentences", f"{sentence_stats.get('num_unique_sentences', 0)}"],
        ["Number of Signers", f"{sentence_stats.get('num_signers', 0)}"],
        ["Avg Words/Sentence", f"{sentence_stats.get('avg_words_per_sentence', 0):.1f}"],
        ["Avg Frames/Sample", f"{sentence_stats.get('avg_frames', 0):.1f}"]
    ]
    
    # Draw tables
    row_height = 0.6
    start_y = 9.5
    
    # Left table (Gloss Dictionary)
    for i, row in enumerate(gloss_data):
        y_pos = start_y - i * row_height
        
        # Background
        if i % 2 == 0:
            rect = patches.Rectangle((0.5, y_pos - 0.25), 4, 0.5, 
                                   linewidth=1, edgecolor='gray', 
                                   facecolor='#f8f8f8', alpha=0.8)
        else:
            rect = patches.Rectangle((0.5, y_pos - 0.25), 4, 0.5, 
                                   linewidth=1, edgecolor='gray', 
                                   facecolor='white', alpha=0.8)
        ax.add_patch(rect)
        
        # Text
        ax.text(0.7, y_pos, row[0], fontsize=10, va='center')
        ax.text(4.3, y_pos, row[1], fontsize=10, va='center', ha='right', fontweight='bold')
    
    # Right table (Sentence Statistics)
    for i, row in enumerate(sentence_data):
        y_pos = start_y - i * row_height
        
        # Background
        if i % 2 == 0:
            rect = patches.Rectangle((5.5, y_pos - 0.25), 4, 0.5, 
                                   linewidth=1, edgecolor='gray', 
                                   facecolor='#f8f8f8', alpha=0.8)
        else:
            rect = patches.Rectangle((5.5, y_pos - 0.25), 4, 0.5, 
                                   linewidth=1, edgecolor='gray', 
                                   facecolor='white', alpha=0.8)
        ax.add_patch(rect)
        
        # Text
        ax.text(5.7, y_pos, row[0], fontsize=10, va='center')
        ax.text(9.3, y_pos, row[1], fontsize=10, va='center', ha='right', fontweight='bold')
    
    # Add borders
    left_border = patches.Rectangle((0.5, start_y - len(gloss_data) * row_height + 0.25), 
                                  4, len(gloss_data) * row_height, 
                                  linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(left_border)
    
    right_border = patches.Rectangle((5.5, start_y - len(sentence_data) * row_height + 0.25), 
                                   4, len(sentence_data) * row_height, 
                                   linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(right_border)
    
    # Add vocabulary list at the bottom
    if gloss_stats.get('words'):
        ax.text(5, 5.5, 'Vocabulary Words:', fontsize=12, fontweight='bold', ha='center')
        vocab_text = ', '.join(gloss_stats['words'])
        ax.text(5, 5, vocab_text, fontsize=11, ha='center', style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Summary table saved to: {output_path}")

def print_detailed_analysis(gloss_stats, sentence_stats):
    """Print detailed analysis to console"""
    print("\n" + "="*60)
    print("VSL_V0 DATASET - DETAILED GLOSS AND SENTENCE ANALYSIS")
    print("="*60)
    
    # Gloss Dictionary Analysis
    print("\n📚 GLOSS DICTIONARY ANALYSIS:")
    print("-" * 40)
    if gloss_stats:
        print(f"Total vocabulary size: {gloss_stats['total_words']}")
        print(f"Total word occurrences: {gloss_stats['total_occurrences']}")
        print(f"Average frequency per word: {gloss_stats['avg_frequency']:.1f}")
        print(f"Most frequent word frequency: {gloss_stats['max_frequency']}")
        print(f"Least frequent word frequency: {gloss_stats['min_frequency']}")
        
        print("\nWord-to-frequency mapping:")
        for word, freq in zip(gloss_stats['words'], gloss_stats['frequencies']):
            print(f"  '{word}': {freq} occurrences")
    
    # Sentence Analysis
    print("\n📝 SENTENCE ANALYSIS:")
    print("-" * 40)
    if sentence_stats:
        print(f"Total samples in dev set: {sentence_stats['total_samples']}")
        print(f"Number of unique sentences: {sentence_stats['num_unique_sentences']}")
        print(f"Number of signers: {sentence_stats['num_signers']}")
        print(f"Average words per sentence: {sentence_stats['avg_words_per_sentence']:.1f}")
        print(f"Average frames per sample: {sentence_stats['avg_frames']:.1f}")
        print(f"Frame range: {sentence_stats['min_frames']} - {sentence_stats['max_frames']}")
        
        print(f"\nSigners: {', '.join(sentence_stats['unique_signers'])}")
        
        print("\nSentence distribution:")
        for sentence, count in sentence_stats['sentence_distribution'].most_common():
            print(f"  '{sentence}': {count} samples")
        
        print("\nSigner distribution:")
        for signer, count in sentence_stats['signer_distribution'].most_common():
            print(f"  {signer}: {count} samples")

def main():
    """Main function to analyze and visualize gloss data"""
    
    # File paths
    gloss_dict_path = "/home/kafka/Desktop/v-sign/data/processed/VSL_V1/gloss_dict.npy"
    dev_info_path = "/home/kafka/Desktop/v-sign/data/processed/VSL_V1/train_info.npy"
    
    # Output paths
    plots_dir = "/home/kafka/Desktop/v-sign/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    frequency_chart_path = os.path.join(plots_dir, "vsl_v0_gloss_frequencies.svg")
    distribution_chart_path = os.path.join(plots_dir, "vsl_v0_sentence_distribution.svg")
    summary_table_path = os.path.join(plots_dir, "vsl_v0_gloss_summary.svg")
    
    print("VSL_V0 Gloss and Sentence Analysis")
    print("=" * 50)
    
    # Load data
    print("\n📂 Loading data files...")
    gloss_dict = load_gloss_dictionary(gloss_dict_path)
    dev_info = load_dev_info(dev_info_path)
    
    # Analyze data
    print("\n🔍 Analyzing data...")
    gloss_stats = analyze_gloss_statistics(gloss_dict)
    sentence_stats = analyze_sentence_statistics(dev_info)
    
    # Print detailed analysis
    print_detailed_analysis(gloss_stats, sentence_stats)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    if gloss_stats:
        create_gloss_frequency_chart(gloss_stats, frequency_chart_path)
    
    if sentence_stats:
        create_sentence_distribution_chart(sentence_stats, distribution_chart_path)
    
    if gloss_stats and sentence_stats:
        create_summary_table(gloss_stats, sentence_stats, summary_table_path)
    
    print("\n✅ Analysis completed!")
    print(f"📁 All visualizations saved to: {plots_dir}")

if __name__ == "__main__":
    main()

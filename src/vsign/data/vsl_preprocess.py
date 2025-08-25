import os
import cv2
import re
import glob  # File and Directory pattern
import pandas as pd
from tqdm import tqdm  # Progress bar
from multiprocessing import Pool  # Parallel processing
import numpy as np
from functools import partial
import argparse


# TODO: REMEMBER TO RENAME THE FOLDER PATH CORRESPONDING TO YOUR PATH IN LINE 98 & 102

def resize_img(img_path, dsize='256x256px'):
    dsize = tuple(int(res) for res in re.findall("\d+", dsize))
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)
    return img


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
                print('Warning: Frames folder not found -> {frame_folders}')

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

    def resize_dataset(video_idx, dsize, info_dict):
        info = info_dict[video_idx]
        prefix = '/home/kafka/Desktop/v-sign/data/interim/frames/VSL_V0'
        img_list = glob.glob(f"{prefix}/{info['folder']}")
        for img_path in img_list:
            rs_img = resize_img(img_path, dsize=dsize)
            rs_img_path = img_path.replace("frames/VSL_V0", dsize + "/VSL_V0")  # Path to save the resized images
            rs_img_dir = os.path.dirname(rs_img_path)
            if not os.path.exists(rs_img_dir):
                os.makedirs(rs_img_dir)
                cv2.imwrite(rs_img_path, rs_img)
            else:
                cv2.imwrite(rs_img_path, rs_img)

    # Executes a function in parallel across multiple processes using multiprocessing.Pool and displays a progress bar
    def run_mp_cmd(processes, process_func, process_args):
        with Pool(processes) as p:
            outputs = list(tqdm(p.imap(process_func, process_args), total=len(process_args)))
        return outputs

    # Executes a function sequentially with the provided arguments.
    def run_cmd(func, args):
        return func(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data process for Visual Alignment Constraint for Continuous Sign Language Recognition.')

    parser.add_argument('--dataset-prefix', type=str, default='vsl_v0',
                        help='Save prefix for ground truth file')
    parser.add_argument('--processed-feature-root', type=str,
                        default='/home/kafka/Desktop/v-sign/data/processed/VSL_V0',
                        help='Path to save the processed feature')
    parser.add_argument('--dataset-root', type=str,
                        default='/home/kafka/Desktop/v-sign/data/interim/frames/VSL_V0',
                        help='Path to the dataset root (where frame folders are located)')
    parser.add_argument('--annotation-prefix', type=str,
                        default='/home/kafka/Desktop/v-sign/data/splits/VSL_V0/csv/{}_annotations.csv',
                        help='Path template for CSV annotations with mode placeholder (train/test/dev)')
    parser.add_argument('--output-res', type=str, default='256x256px',
                        help='Resize resolution for image sequences, e.g., 256x256px')
    parser.add_argument('--process-image', '-p', action='store_true',
                        help='Enable image resizing')
    parser.add_argument('--multiprocessing', '-m', action='store_true',
                        help='Use multiprocessing to accelerate image preprocessing')

    args = parser.parse_args()

    modes = ["dev", "test", "train"]
    sign_dict = {}

    os.makedirs(args.processed_feature_root, exist_ok=True)

    for mode in modes:
        anno_path = args.annotation_prefix.format(mode)

        # Load annotation info from CSV and convert to dict
        info_dict = Preprocessing.annotation2dict(args.dataset_root, anno_path, split=mode)
        np.save(os.path.join(args.processed_feature_root, f"{mode}_info.npy"), info_dict)

        # Update global gloss dictionary
        Preprocessing.gloss_dict_update(sign_dict, info_dict)

        # Save ground truth STM file
        stm_path = os.path.join(args.processed_feature_root, f"{args.dataset_prefix}-ground-truth-{mode}.stm".lower())
        Preprocessing.generate_stm(info_dict, stm_path)

        # Save ground truth to evaluation folder
        eval_path = "/home/kafka/Desktop/v-sign/src/vsign/evaluation/slr_eval/"
        os.makedirs(eval_path, exist_ok=True)  # Create folder if not exist
        stm_eval_path = os.path.join(eval_path, f"{args.dataset_prefix}-ground-truth-{mode}.stm".lower())
        Preprocessing.generate_stm(info_dict, stm_eval_path)

        # Resize frames if required
        video_indices = np.arange(len(info_dict))
        print(f"Resize image to {args.output_res}")
        if args.process_image:
            resize_fn = partial(Preprocessing.resize_dataset, dsize=args.output_res, info_dict=info_dict)
            if args.multiprocessing:
                Preprocessing.run_mp_cmd(10, resize_fn, video_indices)
            else:
                for idx in tqdm(video_indices):
                    Preprocessing.run_cmd(resize_fn, idx)

    # Save sorted gloss dictionary (sorted by gloss name)
    sorted_gloss = sorted(sign_dict.items(), key=lambda x: x[0])
    save_dict = {k: [i + 1, v[1]] for i, (k, v) in enumerate(sorted_gloss)}
    np.save(os.path.join(args.processed_feature_root, "gloss_dict.npy"), save_dict)

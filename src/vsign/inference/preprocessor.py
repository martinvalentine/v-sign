"""
Image and video preprocessing for sign language recognition.
"""
import os
import cv2
import numpy as np
import torch
import io
from PIL import Image
from vsign.utils import video_augmentation

# Define supported image extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

def is_image_by_extension(file_path):
    """Checks if a file has a common image extension."""
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower() in IMAGE_EXTENSIONS

def load_images_from_folder(image_folder, max_frames_num=360):
    """
    Load images from a folder, sort them, and convert to RGB format.
    
    Args:
        image_folder (str): Path to the folder containing image frames.
        max_frames_num (int, optional): Maximum number of frames to process.
        
    Returns:
        tuple: (img_list, valid_image_paths, error_message)
            - img_list: List of numpy arrays containing the images in RGB format
            - valid_image_paths: List of paths to the successfully loaded images
            - error_message: Error message if any, None otherwise
    """
    if not image_folder or not isinstance(image_folder, str):
        return [], [], "Error: Please provide a valid folder path."

    if not os.path.isdir(image_folder):
        return [], [], f"Error: Input path is not a valid directory: {image_folder}"

    img_list = []
    valid_image_paths = []  # Keep track of images successfully loaded
    
    try:
        all_files = os.listdir(image_folder)
    except OSError as e:
        return [], [], f"Error accessing folder: {e}"

    image_files = sorted([
        os.path.join(image_folder, f) for f in all_files
        if is_image_by_extension(os.path.join(image_folder, f))
    ])

    if not image_files:
        return [], [], f"Error: No supported image files ({', '.join(IMAGE_EXTENSIONS)}) found in directory: {image_folder}"

    print(f"Found {len(image_files)} potential images in {image_folder}.")

    # Apply frame limit if specified
    if len(image_files) > max_frames_num:
        print(f"Limiting to first {max_frames_num} frames.")
        image_files = image_files[:max_frames_num]

    # Load images
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue
        # Convert from BGR (OpenCV default) to RGB
        img_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        valid_image_paths.append(img_path)  # Add path if loaded successfully

    if not img_list:
        return [], [], "Error: Found image files, but none could be loaded successfully. Check file integrity."

    print(f"Processing {len(img_list)} images...")
    return img_list, valid_image_paths, None

def load_images_from_uploads(uploaded_files):
    """
    Load images from uploaded files, sort them by filename, and process.
    
    Args:
        uploaded_files (list): List of uploaded files from Gradio interface.
        
    Returns:
        tuple: (img_list, valid_images, error_message)
            - img_list: List of numpy arrays containing the images in RGB format
            - valid_images: List of file paths or Image objects
            - error_message: Error message if any, None otherwise
    """
    if not uploaded_files or not isinstance(uploaded_files, list):
        return [], [], "Error: No images uploaded."
        
    # Sort files by filename (Gradio File objects have 'name' attribute)
    def get_filename(f):
        if isinstance(f, dict):
            return f.get('name', f.get('orig_name', ''))
        elif hasattr(f, 'name'):
            return os.path.basename(f.name)
        return str(f)
        
    sorted_files = sorted(uploaded_files, key=get_filename)
    img_list = []
    valid_images = []  # For preview gallery
    
    for f in sorted_files:
        try:
            # Handle Gradio cache: files will have a 'name' attribute pointing to the temp file path
            if isinstance(f, dict) and 'name' in f:
                img = Image.open(f['name']).convert('RGB')
                valid_images.append(f['name'])
            elif hasattr(f, 'name'):
                # File-like object with name attribute (path to temp file)
                img = Image.open(f.name).convert('RGB')
                valid_images.append(f.name)
            # Fallbacks for other possible Gradio file formats
            elif hasattr(f, 'read'):
                f.seek(0)
                img = Image.open(f).convert('RGB')
                valid_images.append(img)  # Store the image itself as fallback
            elif isinstance(f, dict) and 'data' in f:
                img = Image.open(io.BytesIO(f['data'])).convert('RGB')
                valid_images.append(img)  # Store the image itself as fallback
            else:
                print(f"Warning: Unrecognized file format: {type(f)}")
                continue
                
            # Convert to numpy array for model processing
            img_np = np.array(img)
            img_list.append(img_np)
        except Exception as e:
            print(f"Warning: Could not load image: {e}")
            continue
            
    if not img_list:
        return [], [], "Error: No valid images for inference."
    
    print(f"Processing {len(img_list)} uploaded images...")
    return img_list, valid_images, None

def preprocess_images(img_list):
    """
    Apply preprocessing transformations to a list of images.
    
    Args:
        img_list (list): List of numpy arrays containing the images in RGB format.
        
    Returns:
        tuple: (vid, label) - Preprocessed video tensor and label
    """
    transform = video_augmentation.Compose([
        video_augmentation.CenterCrop(224),
        video_augmentation.Resize(1.0),
        video_augmentation.ToTensor(),
    ])
    
    vid, label = transform(img_list, None, None)
    vid = vid.float() / 127.5 - 1
    vid = vid.unsqueeze(0)  # Add batch dimension
    
    return vid, label

def apply_padding(vid):
    """
    Apply padding to the video tensor according to model requirements.
    
    Args:
        vid (torch.Tensor): Input video tensor.
        
    Returns:
        tuple: (vid_padded, video_length, padding_info)
            - vid_padded: Padded video tensor
            - video_length: Video length tensor
            - padding_info: Dict with left_pad and right_pad values
    """
    # Padding calculation
    left_pad = 0
    last_stride = 1
    total_stride = 1
    kernel_sizes = ['K5', "P2", 'K5', "P2"]
    
    for ks in kernel_sizes:
        if ks[0] == 'K':
            left_pad = left_pad * last_stride
            left_pad += int((int(ks[1]) - 1) / 2)
        elif ks[0] == 'P':
            last_stride = int(ks[1])
            total_stride = total_stride * last_stride

    max_len_actual = vid.size(1)  # Number of frames
    video_length_calc = np.ceil(max_len_actual / total_stride) * total_stride if max_len_actual > 0 else 0
    video_length = torch.LongTensor([max(1, int(video_length_calc))])  # Model might expect length >= 1

    # Calculate required padding based on model architecture
    right_pad = int(np.ceil(max_len_actual / total_stride)) * total_stride - max_len_actual + left_pad
    right_pad = max(0, right_pad)  # Ensure non-negative padding

    # Apply padding
    if left_pad > 0 and vid.size(1) > 0:
        left_padding = vid[:, 0:1].expand(-1, left_pad, -1, -1, -1)
        vid = torch.cat((left_padding, vid), dim=1)
        
    if right_pad > 0 and vid.size(1) > 0:
        last_frame_index = max_len_actual - 1 + left_pad if left_pad > 0 else max_len_actual - 1
        last_frame_index = min(last_frame_index, vid.size(1) - 1)
        right_padding = vid[:, last_frame_index:last_frame_index+1].expand(-1, right_pad, -1, -1, -1)
        vid = torch.cat((vid, right_padding), dim=1)
    
    padding_info = {
        'left_pad': left_pad,
        'right_pad': right_pad,
    }
    
    return vid, video_length, padding_info

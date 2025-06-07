import os
import cv2
import litellm
import torch
import numpy as np
from collections import OrderedDict
import argparse
import warnings
import gradio as gr
import time  # Added for timing measurements

from vsign.utils import video_augmentation
from vsign.models.slr_network import SLRModel
import vsign.utils as utils

warnings.filterwarnings("ignore")

# Define supported image extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

def is_image_by_extension(file_path):
    """Checks if a file has a common image extension."""
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower() in IMAGE_EXTENSIONS


def llm_rephrase(prompt, input_api_key):
    """Rephrase gloss sentence using Gemini model via LiteLLM."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt}
        ]
    }]

    try:
        llm_start_time = time.time()
        response = litellm.completion(
            model="gemini/gemini-2.0-flash",
            messages=messages,
            api_key=input_api_key
        )
        llm_end_time = time.time()
        llm_time_taken = llm_end_time - llm_start_time
        print(f"LLM rephrasing time: {llm_time_taken:.4f} seconds")
        
        message = response.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "")
        if not content:
            return "LLM response was empty or improperly formatted.", llm_time_taken
        return content.strip(), llm_time_taken
    except Exception as e:
        return f"LLM rephrase sentence failed: {e}", 0.0

def run_inference(image_folder, model, device, gloss_dict, input_api_key, args):
    """
    Run inference on images within a specified folder. Optionally rephrases.

    Args:
        image_folder (str): Path to the folder containing image frames.
        model: The loaded SLRModel.
        device: The utils.GpuDataParallel device handler.
        gloss_dict (dict): The dictionary mapping indices to glosses.
        input_api_key (str): Optional API key for rephrasing.
        args: Command-line arguments.

    Returns:
        tuple: (result_string, list_of_image_paths, rephrased_sentence_or_status, timing_info)
               - result_string: Recognition result or error message
               - list_of_image_paths: List of processed image paths
               - rephrased_sentence_or_status: LLM rephrased output or status message
               - timing_info: Dictionary with 'inference_time', 'llm_time', 'total_time'
    """
    # Start timing the full inference pipeline
    total_start_time = time.time()
    
    # Initialize timing variables
    inference_time_taken = 0.0
    llm_time_taken = 0.0
    
    # Define a default status for the rephrased sentence part
    rephrase_status_default = "Rephrasing not attempted."

    if not image_folder or not isinstance(image_folder, str):
        total_end_time = time.time()
        total_time_taken = total_end_time - total_start_time
        timing_info = {
            'inference_time': inference_time_taken,
            'llm_time': llm_time_taken, 
            'total_time': total_time_taken
        }
        return "Error: Please provide a valid folder path.", [], rephrase_status_default, timing_info

    if not os.path.isdir(image_folder):
        total_end_time = time.time()
        total_time_taken = total_end_time - total_start_time
        timing_info = {
            'inference_time': inference_time_taken,
            'llm_time': llm_time_taken, 
            'total_time': total_time_taken
        }
        return f"Error: Input path is not a valid directory: {image_folder}", [], rephrase_status_default, timing_info

    img_list = []
    valid_image_paths = [] # Keep track of images successfully loaded
    # Get all files, sort them (important for sequence), and filter for images
    try:
        all_files = os.listdir(image_folder)
    except OSError as e:
        total_end_time = time.time()
        total_time_taken = total_end_time - total_start_time
        timing_info = {
            'inference_time': inference_time_taken,
            'llm_time': llm_time_taken, 
            'total_time': total_time_taken
        }
        return f"Error accessing folder: {e}", [], rephrase_status_default, timing_info

    image_files = sorted([
        os.path.join(image_folder, f) for f in all_files
        if is_image_by_extension(os.path.join(image_folder, f))
    ])

    if not image_files:
        total_end_time = time.time()
        total_time_taken = total_end_time - total_start_time
        timing_info = {
            'inference_time': inference_time_taken,
            'llm_time': llm_time_taken, 
            'total_time': total_time_taken
        }
        return f"Error: No supported image files ({', '.join(IMAGE_EXTENSIONS)}) found in directory: {image_folder}", [], rephrase_status_default, timing_info

    print(f"Found {len(image_files)} potential images in {image_folder}.")

    # Apply frame limit if specified
    if len(image_files) > args.max_frames_num:
        print(f"Limiting to first {args.max_frames_num} frames.")
        image_files = image_files[:args.max_frames_num]

    # Load images
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue
        # Convert from BGR (OpenCV default) to RGB
        img_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        valid_image_paths.append(img_path) # Add path if loaded successfully

    if not img_list:
        # This case means images were found but none could be loaded by OpenCV
        total_end_time = time.time()
        total_time_taken = total_end_time - total_start_time
        timing_info = {
            'inference_time': inference_time_taken,
            'llm_time': llm_time_taken, 
            'total_time': total_time_taken
        }
        return "Error: Found image files, but none could be loaded successfully. Check file integrity.", [], rephrase_status_default, timing_info

    print(f"Processing {len(img_list)} images...")

    # --- Apply Transformations ---
    try:
        transform = video_augmentation.Compose([
            video_augmentation.CenterCrop(224),
            video_augmentation.Resize(1.0),
            video_augmentation.ToTensor(),
        ])
        # Add int() conversion for safety if transform outputs float indices somehow
        vid, label = transform(img_list, None, None)
        vid = vid.float() / 127.5 - 1
        vid = vid.unsqueeze(0) # Add batch dimension

        # --- Padding Calculation ---
        left_pad = 0
        last_stride = 1
        total_stride = 1
        kernel_sizes = ['K5', "P2", 'K5', "P2"]
        for layer_idx, ks in enumerate(kernel_sizes):
            if ks[0] == 'K':
                left_pad = left_pad * last_stride
                left_pad += int((int(ks[1]) - 1) / 2)
            elif ks[0] == 'P':
                last_stride = int(ks[1])
                total_stride = total_stride * last_stride

        max_len_actual = vid.size(1) # Number of frames/images
        # Ensure video_length is at least 1 even if max_len_actual is 0
        # Also ensure the result of ceil is cast to int for LongTensor
        video_length_calc = np.ceil(max_len_actual / total_stride) * total_stride if max_len_actual > 0 else 0
        video_length = torch.LongTensor([max(1, int(video_length_calc))]) # Model might expect length >= 1

        # Calculate required padding based on model architecture
        right_pad = int(np.ceil(max_len_actual / total_stride)) * total_stride - max_len_actual + left_pad
        right_pad = max(0, right_pad) # Ensure non-negative padding

        # Apply padding
        if left_pad > 0 and vid.size(1) > 0:
            left_padding = vid[:, 0:1].expand(-1, left_pad, -1, -1, -1)
            vid = torch.cat((left_padding, vid), dim=1)
        if right_pad > 0 and vid.size(1) > 0:
            last_frame_index = max_len_actual -1 + left_pad if left_pad > 0 else max_len_actual -1
            last_frame_index = min(last_frame_index, vid.size(1) - 1)
            right_padding = vid[:, last_frame_index:last_frame_index+1].expand(-1, right_pad, -1, -1, -1)
            vid = torch.cat((vid, right_padding), dim=1)

        # --- Run Model Inference ---
        vid = device.data_to_device(vid)
        vid_lgt = device.data_to_device(video_length)

        # Start timing model inference
        inference_start_time = time.time()
        with torch.no_grad(): # Ensure no gradients are calculated during inference
            ret_dict = model(vid, vid_lgt, label=None, label_lgt=None)
        inference_end_time = time.time()
        inference_time_taken = inference_end_time - inference_start_time
        print(f"Model inference time: {inference_time_taken:.4f} seconds")

        # --- Format Results ---
        rephrased_sentence_result = "Rephrasing skipped: No valid recognition." # Default status
        if 'recognized_sents' in ret_dict and ret_dict['recognized_sents']:
            recognized_glosses = [item[0] for item in ret_dict['recognized_sents'][0]]
            result_string = " ".join(recognized_glosses)

            if not result_string: # Handle case where model outputs empty sequence
                total_end_time = time.time()
                total_time_taken = total_end_time - total_start_time
                timing_info = {
                    'inference_time': inference_time_taken,
                    'llm_time': llm_time_taken, 
                    'total_time': total_time_taken
                }
                return "Recognition returned an empty sequence.", valid_image_paths, "Rephrasing skipped: Empty sequence.", timing_info
            else:
                # --- LLM for rephrase gloss sentence (only if glosses found and API key provided) ---
                if input_api_key and input_api_key.strip():
                    try:
                        if not os.path.exists("/home/martinvalentine/Desktop/v-sign/src/vsign/inference/prompt.txt"):
                             rephrased_sentence_result = "Rephrasing Error: prompt.txt not found."
                             print("Error: prompt.txt not found.") 
                        else:
                            with open("/home/martinvalentine/Desktop/v-sign/src/vsign/inference/prompt.txt", "r") as f:
                                prompt_base = f.read()
                            question = f"VSL Glosses: {result_string}\n    Vietnamese:"
                            prompt = prompt_base + "\n" + question
                            rephrased_sentence_result, llm_time_taken = llm_rephrase(prompt, input_api_key) # llm_rephrase handles internal errors
                    except Exception as e:
                        print(f"Error reading prompt file or preparing prompt: {e}")
                        rephrased_sentence_result = f"Rephrasing Error: Could not prepare prompt ({e})"
                else:
                    rephrased_sentence_result = "Rephrasing skipped: API Key not provided."
            
            # Calculate and print total execution time
            total_end_time = time.time()
            total_time_taken = total_end_time - total_start_time
            print(f"Total inference pipeline execution time: {total_time_taken:.4f} seconds")
            
            # Prepare timing information
            timing_info = {
                'inference_time': inference_time_taken,
                'llm_time': llm_time_taken, 
                'total_time': total_time_taken
            }
            
            return result_string, valid_image_paths, rephrased_sentence_result, timing_info
        else:
            # Model didn't return expected output structure
            total_end_time = time.time()
            total_time_taken = total_end_time - total_start_time
            print(f"Total inference pipeline execution time (no recognition): {total_time_taken:.4f} seconds")
            timing_info = {
                'inference_time': inference_time_taken,
                'llm_time': llm_time_taken, 
                'total_time': total_time_taken
            }
            return "Model did not return recognized sentences.", valid_image_paths, "Rephrasing skipped: Model output error.", timing_info

    except Exception as e:
        # Catch potential errors during transform, padding, or inference
        import traceback
        print("Error during processing or inference:")
        traceback.print_exc() # Print detailed error for debugging server-side
        total_end_time = time.time()
        total_time_taken = total_end_time - total_start_time
        print(f"Total inference pipeline execution time (with error): {total_time_taken:.4f} seconds")
        timing_info = {
            'inference_time': inference_time_taken,
            'llm_time': llm_time_taken, 
            'total_time': total_time_taken
        }
        return f"Error during processing: {e}", valid_image_paths, "Rephrasing skipped: Processing error.", timing_info

def parse_args():
    """Parse command-line arguments needed for model setup (not UI inputs)."""
    parser = argparse.ArgumentParser(description="Run sign language recognition inference using a Gradio UI.")
    # Removed --image_folder argument, will come from UI
    parser.add_argument("--model_path", type=str, required=True,
                        help="The path to the pretrained model weights (.pt file).")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU device ID to use (e.g., 0). Use -1 for CPU.")
    parser.add_argument("--language", type=str, default='vsl_v0', choices=['vsl_v0', 'vsl_v1'],
                        help="Language/Dataset identifier for loading the correct gloss dictionary.")
    parser.add_argument("--max_frames_num", type=int, default=360,
                        help="Maximum number of image frames to process from the folder.")
    parser.add_argument("--dict_path", type=str, default=None,
                        help="Path to the gloss dictionary (.npy file). Overrides default path based on --language.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # --- Setup Device (Done once at startup) ---
    if args.device >= 0 and torch.cuda.is_available():
        print(f"Using GPU device: {args.device}")
        device_id = args.device
        torch_device = torch.device(f"cuda:{device_id}")
    else:
        print("Using CPU")
        device_id = -1 # Indicate CPU for GpuDataParallel utility if needed
        torch_device = torch.device("cpu")

    device_util = utils.GpuDataParallel() # Renamed to avoid conflict with torch_device
    device_util.set_device(device_id)

    # --- Load Gloss Dictionary (Done once at startup) ---
    if args.dict_path:
         dict_path = args.dict_path
    else:
        # Determine default dictionary path based on language
        if args.language == 'vsl_v0':
            dataset = 'VSL_V0'
        elif args.language == 'vsl_v1':
             dataset = 'VSL_V1'
        elif args.language == 'vsl_v2':
             dataset = 'VSL_V2'
        else:
            # Handle unsupported language
            raise ValueError(f"Unsupported language '{args.language}'. Use --dict_path or add support.")
        # Construct default path relative to script or using a base path if needed
        script_dir = os.path.dirname(__file__) if "__file__" in locals() else "."
        dict_path = os.path.join(script_dir, 'data', 'processed', dataset, 'gloss_dict.npy')

    if not os.path.exists(dict_path):
         raise FileNotFoundError(f"Gloss dictionary not found at: {dict_path}. Please check the path or use --dict_path.")

    print(f"Loading gloss dictionary from: {dict_path}")
    gloss_dict = np.load(dict_path, allow_pickle=True).item()
    num_classes = len(gloss_dict) + 1 # Add 1 for blank token used in CTC

    # --- Load Model (Done once at startup) ---
    model_path = os.path.expanduser(args.model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights file not found at: {model_path}")

    print(f"Loading model definition...")
    model = SLRModel(num_classes=num_classes, c2d_type='resnet18', conv_type=2, use_bn=1, gloss_dict=gloss_dict,
                     loss_weights={'ConvCTC': 1.0, 'SeqCTC': 1.0, 'Dist': 25.0})

    print(f"Loading model weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=torch_device)

    if 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
    else: state_dict = checkpoint

    state_dict = OrderedDict([(k.replace('module.', ''), v) for k, v in state_dict.items()])
    model.load_state_dict(state_dict, strict=True)
    model = model.to(torch_device) # Move model to the chosen device
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully. Ready for inference.")

    # --- Define the Gradio Interface Handler ---
    # This function will be called by Gradio button click
    # Needs access to the pre-loaded model, device, gloss_dict, and args
    def gradio_interface_handler(folder_path, input_api_key):
        print(f"\nReceived folder path from UI: {folder_path}")

        # Call inference
        result_string, image_paths, rephrased_sentence, timing_info = run_inference(
            folder_path, model, device_util, gloss_dict, input_api_key, args
        )
        print(f"Inference result: {result_string}")
        print(f"Rephrased sentence (Gemini 2.0): {rephrased_sentence}")
        print(f"Images processed: {len(image_paths)}")

        # Format individual timing displays
        model_time_display = f"{timing_info['inference_time']:.4f} seconds"
        llm_time_display = f"{timing_info['llm_time']:.4f} seconds"
        
        # Add total time to rephrased sentence
        rephrased_with_timing = f"{rephrased_sentence}\n\n Total Pipeline Time: {timing_info['total_time']:.4f} seconds"

        # Return results formatted for Gradio outputs
        return result_string, model_time_display, llm_time_display, rephrased_with_timing, image_paths if image_paths else None


    # --- Create and Launch Gradio Interface ---
    with gr.Blocks(title='Continuous Sign Language Recognition (Folder Input)') as demo:
        gr.Markdown("<center><font size=5>Continuous Sign Language Recognition</font></center>")

        with gr.Tab("Folder path (Local test only)"): # TAB 1: For local test
            # Main row containing the two primary columns
            with gr.Row(equal_height=False):
                # --- Left Column: Contains all controls and text outputs ---
                with gr.Column(scale=2):  # Give this column more relative width
                    # Group the input fields together
                    with gr.Group():
                        folder_path_input = gr.Textbox(
                            label="Image Folder Path",
                            placeholder="/path/to/your/image_folder",
                            info="Enter the full path to the folder containing image frames."
                        )
                        api_key_input = gr.Textbox(
                            label="Gemini API Key (Optional)",
                            placeholder="Enter API Key only if you want to rephrase the gloss sequence",
                            type="password",
                            info="Needed for rephrasing the output glosses into a sentence."
                        )

                    run_button = gr.Button("Run Recognition & Rephrase", variant="primary")

                    # Group the text output fields together
                    with gr.Group():
                        results_output = gr.Textbox(
                            label="Output Gloss Sequence",
                            interactive=False
                        )
                        
                        # Separate timing fields
                        with gr.Row():
                            model_time_display = gr.Textbox(
                                label="Model Inference Time",
                                value="⚡ Waiting...",
                                interactive=False,
                                scale=1
                            )
                            llm_time_display = gr.Textbox(
                                label="LLM Rephrasing Time", 
                                value="🤖 Waiting...",
                                interactive=False,
                                scale=1
                            )
                        
                        rephrased_sentence = gr.Textbox(
                            label="Rephrased Sentence / Status",  # Label reflects it might show status
                            interactive=False
                        )
                # --- Right Column: Contains the visual image output ---
                with gr.Column(scale=1):  # Give this column less relative width
                    image_gallery_tab_1 = gr.Gallery(
                        label="Processed Images",
                        show_label=True,
                        elem_id="gallery",
                        columns=5,
                        height=600,
                        object_fit="contain"  # images fit
                    )
        with gr.Tab("Test with images"): # TAB 2: For online testing deployment 
            # Main row containing the two primary columns
            with gr.Row(equal_height=False):
                # --- Left Column: Contains all controls and text outputs ---
                with gr.Column(scale=2):  # Give this column more relative width   
                    with gr.Row():
                        with gr.Column(scale=1):
                            Multi_image_input = gr.UploadButton(label="Click to upload multiple images", file_types = ['.png','.jpg','.jpeg', '.bmp'], file_count = "multiple")
                            multiple_image_button = gr.Button("Run")  
                        with gr.Column(scale=1):
                            multiple_image_output = gr.Textbox(label="Output")
                # --- Right Column: Contains the visual image output ---
                with gr.Column(scale=1):  # Give this column less relative width
                    image_gallery_tab_2 = gr.Gallery(
                        label="Processed Images",
                        show_label=True,
                        elem_id="gallery",
                        columns=5,
                        height=600,
                        object_fit="contain"  # images fit
                    )
        with gr.Tab("Test with video"): # TAB 3: For online testing deployment 
            # Main row containing the two primary columns
            with gr.Row(equal_height=False):
                # --- Left Column: Contains all controls and text outputs ---
                with gr.Column(scale=2):  # Give this column more relative width
                    with gr.Row():
                        with gr.Column(scale=1):
                            Video_input = gr.Video(sources=["upload"], label="Upload a video file")
                            video_button = gr.Button("Run")  
                        with gr.Column(scale=1):
                            video_output = gr.Textbox(label="Output")
                # --- Right Column: Contains the visual image output ---
                with gr.Column(scale=1):  # Give this column less relative width
                    image_gallery_tab_3 = gr.Gallery(
                        label="Processed Images",
                        show_label=True,
                        elem_id="gallery",
                        columns=5,
                        height=600,
                        object_fit="contain"  # images fit
                    )

        # --- Event Handler Connection ---
        # .click() definition after all components involved are defined.
        run_button.click(
            fn=gradio_interface_handler,  # Backend function
            inputs=[folder_path_input, api_key_input],
            outputs=[results_output, model_time_display, llm_time_display, rephrased_sentence, image_gallery_tab_1]  # Map results to UI
        )

    # Launch app
    print("\nLaunching Gradio Interface...")
    demo.launch(share=False) # share=True generates a public link
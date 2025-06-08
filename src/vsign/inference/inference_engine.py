"""
Core inference engine for sign language recognition.
"""
import torch
import time
import traceback

def run_model_inference(vid, vid_lgt, model):
    """
    Run the model inference on the preprocessed video tensor.
    
    Args:
        vid (torch.Tensor): Preprocessed video tensor.
        vid_lgt (torch.Tensor): Video length tensor.
        model: The loaded SLR model.
        
    Returns:
        tuple: (ret_dict, inference_time)
            - ret_dict: Model output dictionary
            - inference_time: Time taken for inference
    """
    # Start timing model inference
    inference_start_time = time.time()
    with torch.no_grad():  # Ensure no gradients are calculated during inference
        ret_dict = model(vid, vid_lgt, label=None, label_lgt=None)
    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time
    print(f"Model inference time: {inference_time:.4f} seconds")
    
    return ret_dict, inference_time

def process_recognition_output(ret_dict):
    """
    Process model output to extract recognized glosses.
    
    Args:
        ret_dict (dict): Model output dictionary.
        
    Returns:
        tuple: (recognized_glosses, result_string)
            - recognized_glosses: List of recognized glosses
            - result_string: Space-separated string of recognized glosses
    """
    if 'recognized_sents' not in ret_dict or not ret_dict['recognized_sents']:
        return [], ""
        
    recognized_glosses = [item[0] for item in ret_dict['recognized_sents'][0]]
    result_string = " ".join(recognized_glosses)
    
    return recognized_glosses, result_string

def run_inference_pipeline(img_list, valid_images_info, model, device, input_api_key=None, 
                          use_llm_rephrasing=True, llm_handler=None):
    """
    Run the full inference pipeline on a list of images.
    
    Args:
        img_list (list): List of numpy arrays containing the images in RGB format.
        valid_images_info (list): List of image paths or Image objects for visualization.
        model: The loaded SLR model.
        device: Device utility for handling tensors.
        input_api_key (str, optional): API key for LLM rephrasing.
        use_llm_rephrasing (bool): Whether to use LLM for rephrasing.
        llm_handler (module, optional): Module with rephrasing functionality.
        
    Returns:
        tuple: (result_string, valid_images_info, rephrased_text, timing_info)
            - result_string: Space-separated string of recognized glosses or error message
            - valid_images_info: List of valid image paths or objects for visualization
            - rephrased_text: Rephrased text or status message
            - timing_info: Dict with inference_time, llm_time, and total_time
    """
    # Start timing the full inference pipeline
    total_start_time = time.time()
    inference_time_taken = 0.0
    llm_time_taken = 0.0
    
    # Import necessary modules here to avoid circular imports
    from vsign.inference.preprocessor import preprocess_images, apply_padding
    
    try:
        # Preprocess images
        vid, _ = preprocess_images(img_list)
        
        # Apply padding
        vid, vid_lgt, _ = apply_padding(vid)
        
        # Move to device
        vid = device.data_to_device(vid)
        vid_lgt = device.data_to_device(vid_lgt)
        
        # Run model inference
        ret_dict, inference_time_taken = run_model_inference(vid, vid_lgt, model)
        
        # Process recognition output
        recognized_glosses, result_string = process_recognition_output(ret_dict)
        
        # Initialize rephrased_sentence_result with default
        rephrased_sentence_result = "Rephrasing skipped: No valid recognition."
        
        if result_string:
            # Apply LLM rephrasing if applicable
            if use_llm_rephrasing and input_api_key and llm_handler:
                try:
                    rephrased_sentence_result, llm_time_taken = llm_handler.rephrase_glosses(
                        result_string, input_api_key
                    )
                except Exception as e:
                    print(f"Error in LLM rephrasing: {e}")
                    rephrased_sentence_result = f"Rephrasing error: {str(e)}"
            else:
                rephrased_sentence_result = "Rephrasing skipped: API Key not provided."
        else:
            # Empty recognition
            rephrased_sentence_result = "Rephrasing skipped: Empty recognition."
        
        # Calculate total execution time
        total_end_time = time.time()
        total_time_taken = total_end_time - total_start_time
        print(f"Total inference pipeline execution time: {total_time_taken:.4f} seconds")
        
        # Prepare timing information
        timing_info = {
            'inference_time': inference_time_taken,
            'llm_time': llm_time_taken, 
            'total_time': total_time_taken
        }
        
        return result_string, valid_images_info, rephrased_sentence_result, timing_info
        
    except Exception as e:
        # Catch potential errors during transform, padding, or inference
        print("Error during processing or inference:")
        traceback.print_exc()  # Print detailed error for debugging server-side
        
        total_end_time = time.time()
        total_time_taken = total_end_time - total_start_time
        print(f"Total inference pipeline execution time (with error): {total_time_taken:.4f} seconds")
        
        timing_info = {
            'inference_time': inference_time_taken,
            'llm_time': llm_time_taken, 
            'total_time': total_time_taken
        }
        
        return f"Error during processing: {e}", valid_images_info, "Rephrasing skipped: Processing error.", timing_info

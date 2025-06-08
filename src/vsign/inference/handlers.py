"""
Handlers for Gradio UI interface integration.
"""
import os
import sys
import time

# Add path to allow imports from the vsign package
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import modules
from vsign.inference import preprocessor
from vsign.inference import inference_engine
from vsign.inference import llm_handler

def gradio_folder_handler(folder_path, input_api_key, model, device, gloss_dict, args):
    """
    Handler for folder-based inference for Gradio UI.
    
    Args:
        folder_path (str): Path to the folder containing image frames.
        input_api_key (str): Optional API key for rephrasing.
        model: The loaded SLR model.
        device: The device utility for tensor operations.
        gloss_dict (dict): The dictionary mapping indices to glosses.
        args: Command-line arguments.
        
    Returns:
        tuple: (result_string, model_time_display, llm_time_display, rephrased_with_timing, image_paths)
            Formatted results for Gradio UI display
    """
    print(f"\nReceived folder path from UI: {folder_path}")
    
    # Load images from folder
    img_list, valid_image_paths, error = preprocessor.load_images_from_folder(
        folder_path, 
        args.max_frames_num
    )
    
    if error:
        # Return error message with formatted empty timing
        return error, "0.0000 seconds", "0.0000 seconds", error, None
    
    # Run inference pipeline
    result_string, image_paths, rephrased_sentence, timing_info = inference_engine.run_inference_pipeline(
        img_list,
        valid_image_paths,
        model, 
        device, 
        input_api_key,
        use_llm_rephrasing=True,
        llm_handler=llm_handler
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

def gradio_multi_image_inference_handler(uploaded_files, input_api_key, model, device, gloss_dict, args):
    """
    Handler for multi-image upload inference for Gradio UI.
    
    Args:
        uploaded_files (list): List of uploaded image files (Gradio File objects).
        input_api_key (str): Optional API key for rephrasing.
        model: The loaded SLR model.
        device: The device utility for tensor operations.
        gloss_dict (dict): The dictionary mapping indices to glosses.
        args: Command-line arguments.
        
    Returns:
        tuple: (result_string, model_time_display, llm_time_display, rephrased_with_timing, valid_images)
            Formatted results for Gradio UI display
    """
    print(f"\nReceived {len(uploaded_files) if uploaded_files else 0} files for multi-image inference")
    
    # Load images from uploads
    img_list, valid_images, error = preprocessor.load_images_from_uploads(uploaded_files)
    
    if error:
        # Return error message with formatted empty timing
        return error, "0.0000 seconds", "0.0000 seconds", error, None
    
    # Run inference pipeline
    result_string, valid_images, rephrased_sentence_result, timing_info = inference_engine.run_inference_pipeline(
        img_list,
        valid_images,
        model, 
        device, 
        input_api_key,
        use_llm_rephrasing=True,
        llm_handler=llm_handler
    )
    
    # Format outputs for Gradio
    model_time_display = f"{timing_info['inference_time']:.4f} seconds"
    llm_time_display = f"{timing_info['llm_time']:.4f} seconds"
    rephrased_with_timing = f"{rephrased_sentence_result}\n\nTotal Pipeline Time: {timing_info['total_time']:.4f} seconds"
    
    return result_string, model_time_display, llm_time_display, rephrased_with_timing, valid_images

def gradio_video_handler(video_path, input_api_key, model, device, gloss_dict, args):
    """
    Handler for video-based inference for Gradio UI.
    
    Args:
        video_path (str): Path to the video file.
        input_api_key (str): Optional API key for rephrasing.
        model: The loaded SLR model.
        device: The device utility for tensor operations.
        gloss_dict (dict): The dictionary mapping indices to glosses.
        args: Command-line arguments.
        
    Returns:
        tuple: (result_string, extracted_frames)
            Formatted results for Gradio UI display
    """
    # This is a placeholder for the video handler implementation
    # Will need to be implemented for the video tab functionality
    
    # TODO: Implement video frame extraction, possibly using CV2 or decord
    # TODO: Run inference on extracted frames
    
    return "Video processing not yet implemented", None

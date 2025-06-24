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
from vsign.inference.cache_manager import get_global_cache_manager

def gradio_folder_handler(folder_path, input_api_key, model, device, gloss_dict, args):
    """
    Handler for folder-based inference for Gradio UI with caching support.
    
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
    
    # Get cache manager
    cache_manager = get_global_cache_manager()
    
    # Load images from folder
    img_list, valid_image_paths, error = preprocessor.load_images_from_folder(
        folder_path, 
        args.max_frames_num
    )
    
    if error:
        # Return error message with formatted empty timing
        return error, "0.0000 giây", "0.0000 giây", error, None
    
    # Generate cache key for this image sequence
    try:
        cache_key = cache_manager.get_cache_key(file_paths=valid_image_paths)
        print(f"Generated cache key: {cache_key[:8]}...")
        
        # Check if we have an API key for LLM rephrasing
        has_api_key = bool(input_api_key and input_api_key.strip())
        
        # Check if result is cached and whether it needs rephrasing
        needs_rephrasing, cached_result = cache_manager.check_needs_rephrasing(cache_key, has_api_key)
        
        if needs_rephrasing:
            # We have cached model prediction but need to add LLM rephrasing
            print(f"Found cached prediction, running LLM rephrasing only...")
            
            try:
                # Run only the LLM rephrasing step
                rephrased_sentence, llm_time_taken = llm_handler.rephrase_glosses(
                    cached_result.result_string, input_api_key
                )
                
                # Update the cache with new rephrasing
                cache_manager.update_cached_rephrasing(cache_key, rephrased_sentence, llm_time_taken)
                
                # Format displays - model time is cached, LLM time is new
                model_time_display = f"{cached_result.timing_info['inference_time']:.4f} seconds (cached)"
                llm_time_display = f"{llm_time_taken:.4f} seconds"
                cache_info = cache_manager.get_cache_info_string()
                rephrased_with_timing = f"{rephrased_sentence}\n\n🔄 Used cached prediction + new LLM rephrasing! Model time: {cached_result.timing_info['inference_time']:.4f} seconds\n{cache_info}"
                
                return cached_result.result_string, model_time_display, llm_time_display, rephrased_with_timing, valid_image_paths
                
            except Exception as e:
                print(f"Error in LLM rephrasing: {e}")
                # Fall through to regular processing
        
        elif cached_result:
            # Complete cached result (model + LLM both cached)
            print(f"Using complete cached result (hit count: {cached_result.cache_hit_count})")

            # Format displays with cached timing but indicate it's from cache
            model_time_display = f"{cached_result.timing_info['inference_time']:.4f} seconds (cached)"
            llm_time_display = f"{cached_result.timing_info['llm_time']:.4f} seconds (cached)"
            cache_info = cache_manager.get_cache_info_string()
            rephrased_with_timing = f"{cached_result.rephrased_sentence}\n\n⚡ Complete result from cache! Original time: {cached_result.timing_info['total_time']:.4f} seconds\n{cache_info}"

            return cached_result.result_string, model_time_display, llm_time_display, rephrased_with_timing, valid_image_paths
            
    except Exception as e:
        print(f"Lỗi bộ nhớ cache (tiếp tục không sử dụng bộ nhớ cache): {e}")
        cache_key = None
    
    # Run inference pipeline (cache miss or cache error)
    result_string, image_paths, rephrased_sentence, timing_info = inference_engine.run_inference_pipeline(
        img_list,
        valid_image_paths,
        model, 
        device, 
        input_api_key,
        use_llm_rephrasing=True,
        llm_handler=llm_handler
    )
    
    # Store result in cache if cache key was generated successfully
    if cache_key:
        try:
            cache_manager.store_result(cache_key, result_string, rephrased_sentence, timing_info)
            cache_info = cache_manager.get_cache_info_string()
            print(f"Save result to cache.{cache_info}")
        except Exception as e:
            print(f"Failed to save result to cache: {e}")
            cache_info = "Failed to save result to cache"
    else:
        cache_info = "Cache not available"

    print(f"Prediction result: {result_string}")
    print(f"Rephrased sentence: {rephrased_sentence}")
    print(f"Processed images: {len(image_paths)}")

    # Format individual timing displays
    model_time_display = f"{timing_info['inference_time']:.4f} giây"
    llm_time_display = f"{timing_info['llm_time']:.4f} giây"
    
    # Add total time and cache info to rephrased sentence
    rephrased_with_timing = f"{rephrased_sentence}\n\nThời gian tổng cộng: {timing_info['total_time']:.4f} giây\n{cache_info}"

    # Return results formatted for Gradio outputs
    return result_string, model_time_display, llm_time_display, rephrased_with_timing, image_paths if image_paths else None

def gradio_multi_image_inference_handler(uploaded_files, input_api_key, model, device, gloss_dict, args):
    """
    Handler for multi-image upload inference for Gradio UI with caching support.
    
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
    
    # Get cache manager
    cache_manager = get_global_cache_manager()
    
    # Load images from uploads
    img_list, valid_images, error = preprocessor.load_images_from_uploads(uploaded_files)
    
    if error:
        # Return error message with formatted empty timing
        return error, "0.0000 seconds", "0.0000 seconds", error, None
    
    # Generate cache key for this image sequence
    try:
        cache_key = cache_manager.get_cache_key(img_list=img_list)
        print(f"Generated cache key: {cache_key[:8]}...")
        
        # Check if we have an API key for LLM rephrasing
        has_api_key = bool(input_api_key and input_api_key.strip())
        
        # Check if result is cached and whether it needs rephrasing
        needs_rephrasing, cached_result = cache_manager.check_needs_rephrasing(cache_key, has_api_key)
        
        if needs_rephrasing:
            # We have cached model prediction but need to add LLM rephrasing
            print(f"Found cached prediction, running LLM rephrasing only...")
            
            try:
                # Run only the LLM rephrasing step
                rephrased_sentence, llm_time_taken = llm_handler.rephrase_glosses(
                    cached_result.result_string, input_api_key
                )
                
                # Update the cache with new rephrasing
                cache_manager.update_cached_rephrasing(cache_key, rephrased_sentence, llm_time_taken)
                
                # Format displays - model time is cached, LLM time is new
                model_time_display = f"{cached_result.timing_info['inference_time']:.4f} seconds (cached)"
                llm_time_display = f"{llm_time_taken:.4f} seconds"
                cache_info = cache_manager.get_cache_info_string()
                rephrased_with_timing = f"{rephrased_sentence}\n\n🔄 Used cached prediction + new LLM rephrasing! Model time: {cached_result.timing_info['inference_time']:.4f} seconds\n{cache_info}"
                
                return cached_result.result_string, model_time_display, llm_time_display, rephrased_with_timing, valid_images
                
            except Exception as e:
                print(f"Error in LLM rephrasing: {e}")
                # Fall through to regular processing
        
        elif cached_result:
            # Complete cached result (model + LLM both cached)
            print(f"Using complete cached result (hit count: {cached_result.cache_hit_count})")
            
            # Format displays with cached timing but indicate it's from cache
            model_time_display = f"{cached_result.timing_info['inference_time']:.4f} seconds (cached)"
            llm_time_display = f"{cached_result.timing_info['llm_time']:.4f} seconds (cached)"
            cache_info = cache_manager.get_cache_info_string()
            rephrased_with_timing = f"{cached_result.rephrased_sentence}\n\n⚡ Complete result from cache! Original time: {cached_result.timing_info['total_time']:.4f} seconds\n{cache_info}"
            
            return cached_result.result_string, model_time_display, llm_time_display, rephrased_with_timing, valid_images
            
    except Exception as e:
        print(f"Cache error (continuing without cache): {e}")
        cache_key = None
    
    # Run inference pipeline (cache miss or cache error)
    result_string, valid_images, rephrased_sentence_result, timing_info = inference_engine.run_inference_pipeline(
        img_list,
        valid_images,
        model, 
        device, 
        input_api_key,
        use_llm_rephrasing=True,
        llm_handler=llm_handler
    )
    
    # Store result in cache if cache key was generated successfully
    if cache_key:
        try:
            cache_manager.store_result(cache_key, result_string, rephrased_sentence_result, timing_info)
            cache_info = cache_manager.get_cache_info_string()
            print(f"Stored result in cache. {cache_info}")
        except Exception as e:
            print(f"Failed to store result in cache: {e}")
            cache_info = "Cache storage failed"
    else:
        cache_info = "Cache unavailable"
    
    # Format outputs for Gradio
    model_time_display = f"{timing_info['inference_time']:.4f} seconds"
    llm_time_display = f"{timing_info['llm_time']:.4f} seconds"
    rephrased_with_timing = f"{rephrased_sentence_result}\n\nTotal Pipeline Time: {timing_info['total_time']:.4f} seconds\n{cache_info}"
    
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
    # TODO: Add caching support for video processing
    
    return "Video processing not yet implemented", None

# Modular VSL Inference Components

This directory contains the modular components for the VSL (Vietnamese Sign Language) inference system. The code has been structured into separate modules for better maintainability and reusability.

## Module Structure

- `demo.py` - Original monolithic implementation (kept for reference)
- `demo_modular.py` - New modular implementation that uses the component modules
- `model_loader.py` - Handles loading models, devices, and gloss dictionaries
- `preprocessor.py` - Handles image loading and preprocessing tasks
- `inference_engine.py` - Core inference functionality used by all interfaces
- `llm_handler.py` - Handles LLM rephrasing with Gemini API
- `handlers.py` - Integration handlers for Gradio UI components
- `ui_components.py` - Gradio UI component definitions

## Using the Modular Components

The modular structure allows for better code organization, easier maintenance, and simpler extension. Here's how to use it:

1. For the complete Gradio UI application, run `demo_modular.py`:
   ```bash
   python -m vsign.inference.demo_modular --model_path /path/to/model.pt --language vsl_v0
   ```
   For example:
   ```
    python src/vsign/inference/demo.py --model_path _best_model.pt --dict_path data/processed/VSL_V2/gloss_dict.npy
   ```

2. To use only specific components (e.g., for a custom application):
   ```python
   # Example: Just performing inference without UI
   from vsign.inference import model_loader, preprocessor, inference_engine
   
   # Load model and device
   device_util, torch_device = model_loader.setup_device(0)  # Use GPU 0
   model, gloss_dict = model_loader.load_model("path/to/model.pt", "vsl_v0", torch_device)
   
   # Load and preprocess images
   img_list, image_paths, error = preprocessor.load_images_from_folder("path/to/images")
   
   # Run inference
   result, paths, _, _ = inference_engine.run_inference_pipeline(
       img_list, image_paths, model, device_util
   )
   
   print(f"Recognition result: {result}")
   ```

## Module Descriptions

### `model_loader.py`

- `setup_device(device_id)`: Sets up either CPU or GPU device
- `load_gloss_dict(language, dict_path)`: Loads the appropriate gloss dictionary
- `load_model(model_path, language, torch_device, dict_path)`: Loads and initializes the model

### `preprocessor.py`

- `load_images_from_folder(image_folder, max_frames_num)`: Loads images from a directory
- `load_images_from_uploads(uploaded_files)`: Loads images from Gradio uploads
- `preprocess_images(img_list)`: Applies preprocessing transforms
- `apply_padding(vid)`: Applies padding for the model

### `inference_engine.py`

- `run_model_inference(vid, vid_lgt, model)`: Core model inference
- `process_recognition_output(ret_dict)`: Extracts results from model output
- `run_inference_pipeline(img_list, valid_images_info, model, device, ...)`: Complete inference pipeline

### `llm_handler.py`

- `llm_rephrase(prompt, input_api_key)`: Low-level LLM API call
- `prepare_prompt(gloss_string, prompt_file_path)`: Creates LLM prompts
- `rephrase_glosses(gloss_string, input_api_key, prompt_file_path)`: Complete rephrasing pipeline

### `handlers.py`

- `gradio_folder_handler(...)`: Handler for folder-based inference tab
- `gradio_multi_image_inference_handler(...)`: Handler for image upload tab
- `gradio_video_handler(...)`: Handler for video upload tab

### `ui_components.py`

- `create_folder_inference_tab()`: Creates UI for folder inference
- `create_multi_image_inference_tab()`: Creates UI for multi-image inference
- `create_video_inference_tab()`: Creates UI for video inference

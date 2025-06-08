"""
UI components for the Gradio interface.
"""
import gradio as gr

def create_folder_inference_tab():
    """
    Create the folder inference tab UI components.
    
    Returns:
        dict: Dictionary containing UI components
    """
    with gr.Row(equal_height=False) as folder_tab:
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
                        value="Waiting...",
                        interactive=False,
                        scale=1
                    )
                    llm_time_display = gr.Textbox(
                        label="LLM Rephrasing Time", 
                        value="Waiting...",
                        interactive=False,
                        scale=1
                    )
                
                rephrased_sentence = gr.Textbox(
                    label="Rephrased Sentence / Status",  # Label reflects it might show status
                    interactive=False
                )
        # --- Right Column: Contains the visual image output ---
        with gr.Column(scale=1):  # Give this column less relative width
            image_gallery = gr.Gallery(
                label="Processed Images",
                show_label=True,
                elem_id="gallery",
                columns=5,
                height=500,
                object_fit="contain"  # images fit
            )
    
    components = {
        "folder_path_input": folder_path_input,
        "api_key_input": api_key_input,
        "run_button": run_button,
        "results_output": results_output,
        "model_time_display": model_time_display,
        "llm_time_display": llm_time_display,
        "rephrased_sentence": rephrased_sentence,
        "image_gallery": image_gallery
    }
    
    return components

def create_multi_image_inference_tab():
    """
    Create the multi-image upload inference tab UI components.
    
    Returns:
        dict: Dictionary containing UI components
    """
    with gr.Row(equal_height=False) as multi_image_tab:
        with gr.Column(scale=2):
            gr.Markdown("<b>Upload multiple images (frames) for sign language recognition:</b>")
            multi_image_input = gr.Files(
                label="Upload Images",
                file_types=['image'],
                file_count="multiple",
                elem_id="multi_image_upload",
                show_label=True,
                height=200  # Limit height so the file list is scrollable
            )
            multi_api_key_input = gr.Textbox(
                label="Gemini API Key (Optional)",
                placeholder="Enter API Key only if you want to rephrase the gloss sequence",
                type="password",
                info="Needed for rephrasing the output glosses into a sentence."
            )
            with gr.Row():
                preview_button = gr.Button("Preview Images", variant="secondary")
                multi_image_run = gr.Button("Run Inference", variant="primary")
            multi_image_output = gr.Textbox(
                label="Recognized Gloss Sequence",
                interactive=False
            )
            # Separate timing fields
            with gr.Row():
                multi_model_time_display = gr.Textbox(
                    label="Model Inference Time",
                    value="Waiting...",
                    interactive=False,
                    scale=1
                )
                multi_llm_time_display = gr.Textbox(
                    label="LLM Rephrasing Time", 
                    value="Waiting...",
                    interactive=False,
                    scale=1
                )
            multi_rephrased_sentence = gr.Textbox(
                label="Rephrased Sentence / Status",
                interactive=False
            )
        with gr.Column(scale=1):
            multi_image_preview = gr.Gallery(
                label="Preview Uploaded Images",
                show_label=True,
                columns=3,  # Fewer columns for larger images and less clutter
                height=500, # Lower height to show fewer images at once
                object_fit="contain",
                allow_preview=True, # Enable scroll/preview
            )
    
    components = {
        "multi_image_input": multi_image_input,
        "multi_api_key_input": multi_api_key_input,
        "preview_button": preview_button,
        "multi_image_run": multi_image_run,
        "multi_image_output": multi_image_output,
        "multi_model_time_display": multi_model_time_display,
        "multi_llm_time_display": multi_llm_time_display,
        "multi_rephrased_sentence": multi_rephrased_sentence,
        "multi_image_preview": multi_image_preview
    }
    
    return components

def create_video_inference_tab():
    """
    Create the video inference tab UI components.
    
    Returns:
        dict: Dictionary containing UI components
    """
    with gr.Row(equal_height=False) as video_tab:
        with gr.Column(scale=2):
            gr.Markdown("<b>Upload a video file for sign language recognition:</b>")
            video_input = gr.Video(
                sources=["upload"],
                label="Upload Video",
                show_label=True
            )
            video_api_key_input = gr.Textbox(
                label="Gemini API Key (Optional)",
                placeholder="Enter API Key only if you want to rephrase the gloss sequence",
                type="password",
                info="Needed for rephrasing the output glosses into a sentence."
            )
            with gr.Row():
                video_preview_button = gr.Button("Preview Video", variant="secondary")
                video_run = gr.Button("Run Inference", variant="primary")
            video_output = gr.Textbox(
                label="Recognized Gloss Sequence",
                interactive=False
            )
            # Separate timing fields
            with gr.Row():
                video_model_time_display = gr.Textbox(
                    label="Model Inference Time",
                    value="Waiting...",
                    interactive=False,
                    scale=1
                )
                video_llm_time_display = gr.Textbox(
                    label="LLM Rephrasing Time", 
                    value="Waiting...",
                    interactive=False,
                    scale=1
                )
            video_rephrased_sentence = gr.Textbox(
                label="Rephrased Sentence / Status",
                interactive=False
            )
        with gr.Column(scale=1):
            video_gallery = gr.Gallery(
                label="Extracted/Processed Frames",
                show_label=True,
                columns=5,
                height=600,
                object_fit="contain"
            )
    
    components = {
        "video_input": video_input,
        "video_api_key_input": video_api_key_input,
        "video_preview_button": video_preview_button,
        "video_run": video_run,
        "video_output": video_output,
        "video_model_time_display": video_model_time_display,
        "video_llm_time_display": video_llm_time_display,
        "video_rephrased_sentence": video_rephrased_sentence,
        "video_gallery": video_gallery
    }
    
    return components

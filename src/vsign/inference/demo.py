"""
Gradio interface for Sign Language Recognition (Modular Version).
This is the modular version of the demo application.
"""
import os
import argparse
import warnings
import gradio as gr

# Import modules
from vsign.inference import model_loader
from vsign.inference import handlers
from vsign.inference import ui_components

warnings.filterwarnings("ignore")

def parse_args():
    """Parse command-line arguments needed for model setup (not UI inputs)."""
    parser = argparse.ArgumentParser(description="Run sign language recognition inference using a Gradio UI.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="The path to the pretrained model weights (.pt file).")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU device ID to use (e.g., 0). Use -1 for CPU.")
    parser.add_argument("--language", type=str, default='vsl_v0', choices=['vsl_v0', 'vsl_v1', 'vsl_v2'],
                        help="Language/Dataset identifier for loading the correct gloss dictionary.")
    parser.add_argument("--max_frames_num", type=int, default=360,
                        help="Maximum number of image frames to process from the folder.")
    parser.add_argument("--dict_path", type=str, default=None,
                        help="Path to the gloss dictionary (.npy file). Overrides default path based on --language.")

    return parser.parse_args()

def main():
    """Main entry point for the Gradio interface."""
    args = parse_args()

    # --- Setup Device and Load Model ---
    device_util, torch_device = model_loader.setup_device(args.device)
    model, gloss_dict = model_loader.load_model(args.model_path, args.language, torch_device, args.dict_path)

    # --- Create Handler Functions with Bound Parameters ---
    def folder_handler(folder_path, input_api_key):
        return handlers.gradio_folder_handler(folder_path, input_api_key, model, device_util, gloss_dict, args)
    
    def multi_image_handler(uploaded_files, input_api_key):
        return handlers.gradio_multi_image_inference_handler(uploaded_files, input_api_key, model, device_util, gloss_dict, args)
    
    def video_handler(video_path, input_api_key):
        return handlers.gradio_video_handler(video_path, input_api_key, model, device_util, gloss_dict, args)

    # --- Create and Launch Gradio Interface ---
    with gr.Blocks(title='Continuous Sign Language Recognition') as demo:
        gr.Markdown("<center><font size=5>Nhận diện ngôn ngữ ký hiệu Tiếng Việt</font></center>")

        # --- Tab 1: Folder Path Inference ---
        with gr.Tab("Test với thư mục (Chỉ dùng cho test cục bộ)"):
            tab1_components = ui_components.create_folder_inference_tab()
            tab1_components["run_button"].click(
                fn=folder_handler,  # Backend function with bound parameters
                inputs=[
                    tab1_components["folder_path_input"], 
                    tab1_components["api_key_input"]
                ],
                outputs=[
                    tab1_components["results_output"], 
                    tab1_components["model_time_display"],
                    tab1_components["llm_time_display"], 
                    tab1_components["rephrased_sentence"],
                    tab1_components["image_gallery"]
                ]
            )

        # --- Tab 2: Multi-Image Upload Inference ---
        with gr.Tab("Test với nhiều ảnh (dành cho test trực tuyến)"):
            tab2_components = ui_components.create_multi_image_inference_tab()
            # Preview button
            tab2_components["preview_button"].click(
                lambda files: files,
                inputs=tab2_components["multi_image_input"],
                outputs=tab2_components["multi_image_preview"]
            )
            # Run inference button
            tab2_components["multi_image_run"].click(
                fn=multi_image_handler,
                inputs=[
                    tab2_components["multi_image_input"],
                    tab2_components["multi_api_key_input"]
                ],
                outputs=[
                    tab2_components["multi_image_output"],
                    tab2_components["multi_model_time_display"],
                    tab2_components["multi_llm_time_display"],
                    tab2_components["multi_rephrased_sentence"],
                    tab2_components["multi_image_preview"]
                ]
            )

        # --- Tab 3: Video Upload Inference ---
        # with gr.Tab("Test với video (dành cho test trực tuyến)"):
        #     tab3_components = ui_components.create_video_inference_tab()
        #     # Preview button
        #     tab3_components["video_preview_button"].click(
        #         lambda vid: vid,
        #         inputs=tab3_components["video_input"],
        #         outputs=tab3_components["video_gallery"]
        #     )

    # Launch the app
    print("\nLaunching Gradio Interface...")
    demo.launch(share=True)  # share=True generates a public link

if __name__ == "__main__":
    main()

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
                    label="Đường dẫn thư mục ảnh",
                    placeholder="/path/to/your/image_folder",
                    info="Nhập đường dẫn đến thư mục chứa các ảnh khung hình"
                )
                api_key_input = gr.Textbox(
                    label="Gemini API Key (Tùy chọn)",
                    placeholder="Nhập API Key nếu bạn chuyển từ ngôn ngữ ký hiệu sang câu Tiếng Việt",
                    type="password",
                    info="Cần thiết để chuyển từ ngôn ngữ ký hiệu sang câu Tiếng Việt"
                )

            run_button = gr.Button("Chạy mô hình nhận diện & chuyển đổi sang câu Tiếng Việt", variant="primary")

            # Group the text output fields together
            with gr.Group():
                results_output = gr.Textbox(
                    label="Kết quả nhận diện (ngôn ngữ ký hiệu)",
                    interactive=False
                )
                
                # Separate timing fields
                with gr.Row():
                    model_time_display = gr.Textbox(
                        label="Thời gian chạy mô hình nhận diện",
                        value="Waiting...",
                        interactive=False,
                        scale=1
                    )
                    llm_time_display = gr.Textbox(
                        label="Thời gian chuyển đổi sang câu Tiếng Việt", 
                        value="Waiting...",
                        interactive=False,
                        scale=1
                    )
                
                rephrased_sentence = gr.Textbox(
                    label="Câu được chuyển đổi sang Tiếng Việt / Trạng thái",  # Label reflects it might show status
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
            gr.Markdown("<b>Tải lên nhiều ảnh (khung hình) để nhận diện ngôn ngữ ký hiệu Tiếng Việt:</b>")
            multi_image_input = gr.Files(
                label="Tải lên các ảnh",
                file_types=['image'],
                file_count="multiple",
                elem_id="multi_image_upload",
                show_label=True,
                height=200  # Limit height so the file list is scrollable
            )
            multi_api_key_input = gr.Textbox(
                label="Gemini API Key (Tùy chọn)",
                placeholder="Nhập API Key nếu bạn chuyển từ ngôn ngữ ký hiệu sang câu Tiếng Việt",
                type="password",
                info="Cần thiết để chuyển từ ngôn ngữ ký hiệu sang câu Tiếng Việt"
            )
            with gr.Row():
                preview_button = gr.Button("Xem trước các ảnh đã tải lên", variant="secondary")
                multi_image_run = gr.Button("Chạy mô hình nhận diện", variant="primary")
            multi_image_output = gr.Textbox(
                label="Kết quả nhận diện (ngôn ngữ ký hiệu)",
                interactive=False
            )
            # Separate timing fields
            with gr.Row():
                multi_model_time_display = gr.Textbox(
                    label="Thời gian chạy mô hình nhận diện",
                    value="Đang chờ...",
                    interactive=False,
                    scale=1
                )
                multi_llm_time_display = gr.Textbox(
                    label="Thời gian chuyển đổi sang câu Tiếng Việt", 
                    value="Đang chờ...",
                    interactive=False,
                    scale=1
                )
            multi_rephrased_sentence = gr.Textbox(
                label="Câu được chuyển đổi sang Tiếng Việt / Trạng thái",
                interactive=False
            )
        with gr.Column(scale=1):
            multi_image_preview = gr.Gallery(
                label="Xem trước các ảnh đã tải lên",
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
            gr.Markdown("<b>Upload video để nhận diện ngôn ngữ ký hiệu Tiếng Việt:</b>")
            video_input = gr.Video(
                sources=["upload"],
                label="Upload Video",
                show_label=True
            )
            video_api_key_input = gr.Textbox(
                label="Gemini API Key (Tùy chọn)",
                placeholder="Nhập API Key nếu bạn chuyển từ ngôn ngữ ký hiệu sang câu Tiếng Việt",
                type="password",
                info="Cần thiết để chuyển từ ngôn ngữ ký hiệu sang câu Tiếng Việt"
            )
            with gr.Row():
                video_preview_button = gr.Button("Xem trước video", variant="secondary")
                video_run = gr.Button("Chạy mô hình nhận diện", variant="primary")
            video_output = gr.Textbox(
                label="Kết quả nhận diện (ngôn ngữ ký hiệu)",
                interactive=False
            )
            # Separate timing fields
            with gr.Row():
                video_model_time_display = gr.Textbox(
                    label="Thời gian chạy mô hình nhận diện",
                    value="Đang chờ...",
                    interactive=False,
                    scale=1
                )
                video_llm_time_display = gr.Textbox(
                    label="Thời gian chuyển đổi sang câu Tiếng Việt", 
                    value="Đang chờ...",
                    interactive=False,
                    scale=1
                )
            video_rephrased_sentence = gr.Textbox(
                label="Câu được chuyển đổi sang Tiếng Việt / Trạng thái",
                interactive=False
            )
        with gr.Column(scale=1):
            video_gallery = gr.Gallery(
                label="Các khung hình được trích xuất/xử lý",
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

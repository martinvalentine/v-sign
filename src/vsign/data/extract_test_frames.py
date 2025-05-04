import cv2
import os
from pathlib import Path # pathlib is useful for path operations

# Added flip_horizontal parameter with a default of False
def extract_and_resize_frames(video_path, output_folder, target_size=(256, 256), sample_rate=3, flip_horizontal=False):

    print(f"\n--- Processing Video: {Path(video_path).name} ---")
    video_path_obj = Path(video_path)
    output_folder_obj = Path(output_folder)

    # --- Input Validation (for the video file itself) ---
    if not video_path_obj.is_file():
        print(f"Error: Video file not found at '{video_path}'")
        return
    if sample_rate < 1:
        print(f"Error: Sample rate must be 1 or greater. Using 1 for {video_path_obj.name}")
        sample_rate = 1 # Default to 1 if invalid

    # --- Setup ---
    try:
        output_folder_obj.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory '{output_folder_obj}': {e}. Cannot process video.")
        return

    vidcap = cv2.VideoCapture(str(video_path_obj))
    if not vidcap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'.")
        return

    print(f"Outputting frames to: {output_folder_obj}")
    print(f"Target size: {target_size[0]}x{target_size[1]} pixels")
    print(f"Sample rate: {sample_rate}")
    # Print flip status
    print(f"Flip Horizontally: {'Yes' if flip_horizontal else 'No'}")

    # --- Frame Extraction Loop ---
    frame_count = 0
    saved_count = 0
    while True:
        success, frame = vidcap.read()
        if not success:
            break # End of video reached

        # Apply sample rate
        if frame_count % sample_rate == 0:
            # 1. Resize frame
            try:
                interpolation = cv2.INTER_AREA if frame.shape[0] > target_size[1] or frame.shape[1] > target_size[0] else cv2.INTER_LINEAR
                # Start with the resized frame
                processed_frame = cv2.resize(frame, target_size, interpolation=interpolation)
            except Exception as e:
                print(f"Error resizing frame {frame_count} from {video_path_obj.name}: {e}")
                continue # Skip this frame if resizing fails

            # 2. Flip Horizontally (if requested)
            if flip_horizontal:
                try:
                    # flipCode = 1 means flipping around the Y-axis (horizontal)
                    processed_frame = cv2.flip(processed_frame, 1)
                except Exception as e:
                     print(f"Error flipping frame {frame_count} from {video_path_obj.name}: {e}")
                     # Decide if you want to save the unflipped frame or skip
                     continue # Skip if flipping fails

            # 3. Save the processed frame (resized and potentially flipped)
            frame_filename = f"frame_{saved_count:06d}.png"
            frame_filepath = output_folder_obj / frame_filename # Use Path object joining

            try:
                cv2.imwrite(str(frame_filepath), processed_frame) # Use str() for OpenCV path
                saved_count += 1
            except Exception as e:
                 print(f"Error writing frame {frame_count} to {frame_filepath} for {video_path_obj.name}: {e}")

        frame_count += 1

    # --- Cleanup ---
    vidcap.release()
    print(f"\nFinished {video_path_obj.name}. Saved {saved_count} frames from {frame_count} total frames read.")


# --- Main Execution Block ---
if __name__ == "__main__":

    # --- Configuration ---
    # 1. Set the FULL PATH to the single video file you want to process
    INPUT_VIDEO_FILEPATH = Path("/home/martinvalentine/Desktop/khoe_test.mov") # TODO: CHANGE THIS PATH

    # 2. Set the Base directory where the video-specific subfolder will be created
    OUTPUT_BASE_DIR = Path("/home/martinvalentine/Desktop/frame") # TODO: CHANGE THIS PATH

    # 3. Set the desired dimensions for the resized frames
    RESIZE_WIDTH = 256
    RESIZE_HEIGHT = 256

    # 4. Set the Frame sampling rate (1 = save all frames)
    FRAME_SAMPLE_RATE = 1

    # 5. Set whether to flip frames horizontally
    FLIP_FRAMES_HORIZONTALLY = True # TODO: SET TO True TO FLIP, False TO NOT FLIP
    # --- End of Configuration ---


    # --- === No need to change anything below this line === ---

    # --- Prepare Resize Dimensions ---
    target_dimensions = (RESIZE_WIDTH, RESIZE_HEIGHT)

    # --- Validate Input File ---
    if not INPUT_VIDEO_FILEPATH.is_file():
        print(f"Error: Input video file not found: '{INPUT_VIDEO_FILEPATH}'")
        print("Please ensure the INPUT_VIDEO_FILEPATH is correct in the script.")
        exit() # Stop if the source file doesn't exist

    # --- Dynamic Output Path Generation ---
    video_name_stem = INPUT_VIDEO_FILEPATH.stem # Get filename without extension (e.g., "Ban_khoe")
    if not video_name_stem: # Handle edge cases like hidden files ".something.MOV"
        video_name_stem = INPUT_VIDEO_FILEPATH.name # Fallback to full name if stem is empty

    # Create the full path for this video's output folder
    # e.g., /home/martinvalentine/Desktop/frame/Ban_khoe
    output_frames_folder = OUTPUT_BASE_DIR / video_name_stem

    # --- Execute ---
    print("-" * 40)
    print(f"Processing single video: {INPUT_VIDEO_FILEPATH}")
    print(f"Output folder will be: {output_frames_folder}")
    print(f"Target frame size: {RESIZE_WIDTH}x{RESIZE_HEIGHT}")
    print(f"Frame sample rate: {FRAME_SAMPLE_RATE}")
    # Print flip status here as well
    print(f"Flip Horizontally: {FLIP_FRAMES_HORIZONTALLY}")
    print("-" * 40)

    # Call the function to process the single video, passing the flip flag
    extract_and_resize_frames(
        video_path=str(INPUT_VIDEO_FILEPATH),
        output_folder=str(output_frames_folder),
        target_size=target_dimensions,
        sample_rate=FRAME_SAMPLE_RATE,
        flip_horizontal=FLIP_FRAMES_HORIZONTALLY # Pass the flag here
    )

    print("-" * 40)
    print("Processing finished.")
    print("-" * 40)
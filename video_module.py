import cv2
import os
import ffmpeg

def extract_frames(video_path, output_folder="frames", frame_rate=1):
    """
    Extracts frames from a video file.
    frame_rate = how many frames to extract per second
    """
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open video file")
        return []

    
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video loaded successfully")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")

    
    saved_frames = []
    frame_count = 0
    saved_count = 0
    interval = int(fps / frame_rate)

    while True:
        success, frame = video.read()

        if not success:
            break

        
        if frame_count % interval == 0:
            frame_filename = os.path.join(
                output_folder,
                f"frame_{saved_count:04d}.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            saved_frames.append(frame_filename)
            saved_count += 1

        frame_count += 1

    video.release()
    print(f"Extracted {saved_count} frames to '{output_folder}' folder")
    return saved_frames


def extract_audio(video_path, output_audio="audio.wav"):
    """
    Extracts audio from a video file and saves it as WAV.
    WAV format is needed for speech recognition later.
    """
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_audio,
                   acodec='pcm_s16le',
                   ac=1,
                   ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"Audio extracted successfully to '{output_audio}'")
        return output_audio

    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None


def process_video(video_path):
    """
    Main function - runs both frame and audio extraction
    """
    print("="*50)
    print("VIDEO PROCESSING MODULE")
    print("="*50)

    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found")
        return None, None

    print(f"Processing video: {video_path}")
    print()

    
    print("Step 1: Extracting frames...")
    frames = extract_frames(video_path)
    print()

    
    print("Step 2: Extracting audio...")
    audio = extract_audio(video_path)
    print()

    print("="*50)
    print("VIDEO PROCESSING COMPLETE")
    print(f"Frames saved: {len(frames)}")
    print(f"Audio saved: {audio}")
    print("="*50)

    return frames, audio



if __name__ == "__main__":
    test_video = "test_video.mp4"
    frames, audio = process_video(test_video)
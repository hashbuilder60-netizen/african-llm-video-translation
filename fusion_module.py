import json
import os
import cv2
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datetime import timedelta

def load_text_model():
    """
    Loads XLM-R model for text understanding.
    Already downloaded so will load instantly.
    """
    print("Loading language model...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModel.from_pretrained("xlm-roberta-base")
    model.eval()
    print("Language model loaded")
    return tokenizer, model


def get_text_embedding(text, tokenizer, model):
    """
    Gets the meaning vector for a piece of text.
    """
    tokens = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**tokens)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding


def describe_frame(frame_path):
    """
    Extracts basic visual information from a frame.
    Describes colors, brightness and basic scene info.
    """
    
    frame = cv2.imread(frame_path)
    if frame is None:
        return None

    
    height, width = frame.shape[:2]

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))

    
    avg_color = frame.mean(axis=0).mean(axis=0)
    blue = float(avg_color[0])
    green = float(avg_color[1])
    red = float(avg_color[2])

    
    if brightness > 200:
        scene = "bright scene"
    elif brightness > 100:
        scene = "normal lighting scene"
    else:
        scene = "dark scene"

    description = (f"{scene} with RGB values "
                  f"R:{red:.0f} G:{green:.0f} B:{blue:.0f} "
                  f"at resolution {width}x{height}")

    return {
        "description": description,
        "brightness": brightness,
        "color_rgb": [red, green, blue],
        "resolution": [width, height]
    }


def align_frames_with_text(frames, sentences, video_duration):
    """
    Matches video frames with the text that was
    being spoken at that moment in the video.
    This is the fusion - connecting what was seen
    with what was said at the same time.
    """
    aligned_data = []

    
    time_per_sentence = video_duration / len(sentences)

    
    time_per_frame = video_duration / len(frames)

    for i, sentence in enumerate(sentences):
        
        start_time = i * time_per_sentence
        end_time = (i + 1) * time_per_sentence

        
        matching_frames = []
        for j, frame_path in enumerate(frames):
            frame_time = j * time_per_frame
            if start_time <= frame_time < end_time:
                matching_frames.append({
                    "frame_path": frame_path,
                    "timestamp": frame_time
                })

        
        timestamp_str = str(timedelta(seconds=int(start_time)))

        aligned_data.append({
            "sentence_id": i,
            "timestamp": timestamp_str,
            "start_seconds": start_time,
            "text": sentence,
            "matching_frames": matching_frames[:3]
            
        })

    return aligned_data


def fuse_modalities(aligned_data, frames_folder, tokenizer, model):
    """
    The actual fusion step.
    For each moment in the video we combine:
    - What was said (text)
    - What was seen (frames)
    - The meaning of what was said (embeddings)
    """
    print("Fusing video, audio and text...")
    fused_results = []

    for i, item in enumerate(aligned_data[:10]):
        
        print(f"Fusing segment {i+1} at {item['timestamp']}...")

        
        text_embedding = get_text_embedding(
            item['text'],
            tokenizer,
            model
        )

        
        visual_info = []
        for frame_data in item['matching_frames']:
            frame_path = frame_data['frame_path']
            if os.path.exists(frame_path):
                info = describe_frame(frame_path)
                if info:
                    visual_info.append(info)

        
        fused_segment = {
            "segment_id": i,
            "timestamp": item['timestamp'],
            "text": item['text'],
            "text_embedding_shape": list(text_embedding.shape),
            "visual_scenes": visual_info,
            "fusion_summary": {
                "text_length": len(item['text'].split()),
                "frames_analyzed": len(visual_info),
                "has_visual": len(visual_info) > 0,
                "has_text": len(item['text']) > 0
            }
        }

        fused_results.append(fused_segment)

    return fused_results


def run_fusion():
    """
    Main fusion function.
    Brings together all previous modules.
    """
    print("="*50)
    print("MULTIMODAL FUSION MODULE")
    print("="*50)

    
    print("Step 1: Loading transcript data...")
    if not os.path.exists("transcript.json"):
        print("Error: transcript.json not found")
        print("Run audio_module.py first")
        return None

    with open("transcript.json", "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    video_duration = transcript_data["duration_seconds"]
    print(f"Video duration: {video_duration:.2f} seconds")

    
    print("Step 2: Loading processed text...")
    if not os.path.exists("processed_text.json"):
        print("Error: processed_text.json not found")
        print("Run text_module.py first")
        return None

    with open("processed_text.json", "r", encoding="utf-8") as f:
        text_data = json.load(f)

    sentences = text_data["all_sentences"]
    print(f"Total sentences: {len(sentences)}")

    
    print("Step 3: Loading video frames...")
    frames_folder = "frames"
    if not os.path.exists(frames_folder):
        print("Error: frames folder not found")
        print("Run video_module.py first")
        return None

    frames = sorted([
        os.path.join(frames_folder, f)
        for f in os.listdir(frames_folder)
        if f.endswith('.jpg')
    ])
    print(f"Total frames: {len(frames)}")

    
    print("Step 4: Loading language model...")
    tokenizer, model = load_text_model()
    print()

    
    print("Step 5: Aligning frames with text...")
    aligned_data = align_frames_with_text(
        frames,
        sentences,
        video_duration
    )
    print(f"Aligned {len(aligned_data)} segments")
    print()

    
    print("Step 6: Fusing all modalities...")
    fused_results = fuse_modalities(
        aligned_data,
        frames_folder,
        tokenizer,
        model
    )
    print()

    
    output = {
        "video_duration": video_duration,
        "total_segments": len(aligned_data),
        "fused_segments": fused_results
    }

    output_path = "fused_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print("="*50)
    print("FUSION COMPLETE")
    print(f"Total segments fused: {len(fused_results)}")
    print(f"Fused data saved to: {output_path}")
    print()
    print("FUSION PREVIEW:")
    print("-"*50)
    for segment in fused_results[:3]:
        print(f"[{segment['timestamp']}] {segment['text'][:60]}...")
        print(f"  Frames analyzed: "
              f"{segment['fusion_summary']['frames_analyzed']}")
        if segment['visual_scenes']:
            print(f"  Visual: "
                  f"{segment['visual_scenes'][0]['description']}")
        print()
    print("="*50)

    return output



if __name__ == "__main__":
    run_fusion()
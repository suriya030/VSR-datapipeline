import json
import os
import av
import cv2
import numpy as np
from tqdm import tqdm

def load_analysis_json(json_path):
    with open(json_path, 'r') as f:
        analysis_data = json.load(f)
    
    video_path = analysis_data['video_information']['file_path']
    scenes_with_frames = []
    
    for scene in analysis_data['scene_detection']['scenes']:
        if scene['frames_selected'] and scene['selected_frame_range']:
            scenes_with_frames.append({
                'scene_id': scene['scene_id'],
                'selected_frames': analysis_data['quality_analysis']['scene_results'][scene['scene_id']-1]['selected_frames']
            })
    
    return video_path, scenes_with_frames

def get_all_selected_frames(scenes_with_frames):
    all_frames = {}
    for scene_data in scenes_with_frames:
        scene_id = scene_data['scene_id']
        for frame_number in scene_data['selected_frames']:
            all_frames[frame_number] = scene_id
    return all_frames

def read_selected_frames_once(video_path, selected_frames_dict):
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    total_frames = video_stream.frames
    
    if total_frames == 0:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    
    extracted_frames = {}
    current_frame = 0
    
    with tqdm(total=total_frames, desc="Reading video frames", unit="frame") as pbar:
        for frame in container.decode(video_stream):
            current_frame += 1
            pbar.update(1)
            
            if current_frame in selected_frames_dict:
                extracted_frames[current_frame] = frame
    
    container.close()
    return extracted_frames

def save_frames_to_png(extracted_frames, scenes_with_frames, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for scene_data in scenes_with_frames:
        scene_id = scene_data['scene_id']
        selected_frames = scene_data['selected_frames']
        
        scene_folder = os.path.join(output_folder, f"scene_{scene_id:02d}")
        os.makedirs(scene_folder, exist_ok=True)
        
        for frame_number in tqdm(selected_frames, desc=f"Saving Scene {scene_id}"):
            if frame_number in extracted_frames:
                frame = extracted_frames[frame_number]
                output_filename = f"frame_{frame_number:06d}.png"
                output_path = os.path.join(scene_folder, output_filename)
                
                # Convert xyz12le to rgb48le (16-bit RGB) using PyAV
                rgb_frame = frame.reformat(format='rgb48le')
                frame_array = rgb_frame.to_ndarray()
                
                # OpenCV expects BGR format for 16-bit PNG
                bgr_16bit = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                
                # Save as 16-bit PNG using OpenCV
                cv2.imwrite(output_path, bgr_16bit)

def extract_selected_frames(json_path, output_folder):
    video_path, scenes_with_frames = load_analysis_json(json_path)
    selected_frames_dict = get_all_selected_frames(scenes_with_frames)
    
    extracted_frames = read_selected_frames_once(video_path, selected_frames_dict)
    save_frames_to_png(extracted_frames, scenes_with_frames, output_folder)

if __name__ == "__main__":
    json_file = r"analysis_results\Beast (1)_analysis.json"
    output_directory = r"extracted_frames"
    
    extract_selected_frames(json_file, output_directory)
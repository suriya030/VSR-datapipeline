import cv2
import torch
import pyiqa
import numpy as np
from collections import deque
from tqdm import tqdm
from colorama import Fore
from config import QUALITY_ANALYSIS
from utils import get_device, print_processing, print_success, print_warning, print_info

def initialize_quality_metrics(device_preference='auto'):
    """Initialize MUSIQ and NIQE quality metrics"""
    device = get_device(device_preference)
    
    musiq_metric = pyiqa.create_metric('musiq', device=device)
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    print_success("MUSIQ and NIQE models loaded successfully")
    
    return musiq_metric, niqe_metric, device

def find_sequence_per_scene(frames_list, scenes_info, base_video_name, musiq_metric, niqe_metric, device):
    """Find high-quality frame sequences in each detected scene"""
    scene_results = []
    
    print_processing("Starting quality analysis per scene...")
    print(f"{Fore.CYAN}   Thresholds: MUSIQ > {QUALITY_ANALYSIS['musiq_threshold']}, NIQE < {QUALITY_ANALYSIS['niqe_threshold']}")
    print(f"{Fore.CYAN}   Target sequence length: {QUALITY_ANALYSIS['sequence_length']} frames")
    
    # Process each scene individually
    for scene in tqdm(scenes_info, desc="Analyzing scenes", unit="scene", colour="magenta"):
        scene_id = scene['scene_id']
        start_frame = scene['start_frame']
        end_frame = scene['end_frame']
        
        print_info(f"Processing Scene {scene_id} (frames {start_frame}-{end_frame})")
        
        frame_buffer = deque(maxlen=QUALITY_ANALYSIS['sequence_length'])
        selected_frames = []
        
        # Analyze frames within current scene
        for frame_idx in range(start_frame - 1, min(end_frame, len(frames_list))):
            frame_array = frames_list[frame_idx]
            frame_number = frame_idx + 1
            
            # Skip low-variance frames (likely blank/black)
            if np.std(frame_array) < QUALITY_ANALYSIS['min_frame_variance']:
                frame_buffer.clear()
                continue
            
            # Convert frame for quality analysis
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_YUV2BGR_I420)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.tensor(frame_rgb).permute(2, 0, 1).unsqueeze(0) / 255.0
            frame_tensor = frame_tensor.to(device)
            
            # Calculate quality metrics with error handling for NIQE
            with torch.no_grad():
                musiq_score = musiq_metric(frame_tensor).item()
                
                # Try NIQE calculation, skip frame if it fails
                try:
                    niqe_score = niqe_metric(frame_tensor).item()
                except:
                    # Skip this frame if NIQE calculation fails
                    frame_buffer.clear()
                    continue
            
            # Check if frame meets quality thresholds
            is_good_quality = (musiq_score >= QUALITY_ANALYSIS['musiq_threshold']) and (niqe_score <= QUALITY_ANALYSIS['niqe_threshold'])
            
            if is_good_quality:
                frame_buffer.append(frame_number)
            else:
                frame_buffer.clear()
            
            # Check if we found a complete sequence
            if len(frame_buffer) == QUALITY_ANALYSIS['sequence_length']:
                selected_frames = list(frame_buffer)
                break
        
        # Store results for current scene
        success = len(selected_frames) == QUALITY_ANALYSIS['sequence_length']
        scene_result = {
            'scene_id': scene_id,
            'sequence_found': success,
            'selected_frames': selected_frames,
            'total_frames_selected': len(selected_frames)
        }
        
        if success:
            print_success(f"Scene {scene_id}: Found {QUALITY_ANALYSIS['sequence_length']} consecutive high-quality frames")
        else:
            print_warning(f"Scene {scene_id}: Only found {len(selected_frames)} quality frames")
        
        scene_results.append(scene_result)
    
    return scene_results
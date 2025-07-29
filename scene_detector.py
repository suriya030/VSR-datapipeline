from scenedetect import open_video, SceneManager
from scenedetect.detectors import AdaptiveDetector
from colorama import Fore
from config import SCENE_DETECTION
from utils import print_processing, print_success, print_warning

def find_and_split_scenes(video_path, frame_rate):
    """Detect scenes in video using adaptive threshold"""
    print_processing("Starting scene detection...")
    
    # Initialize scene detection
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector(
        adaptive_threshold=SCENE_DETECTION['adaptive_threshold']
    ))
    
    print(f"{Fore.BLUE}🔍 Analyzing video for scene changes...")
    scene_manager.detect_scenes(video=video, show_progress=True)
    
    # Process detected scenes
    scene_list = scene_manager.get_scene_list()
    print_success(f"Found {Fore.YELLOW}{len(scene_list)}{Fore.GREEN} scenes")
    
    scenes_info = []
    if scene_list:
        for i, (start_time, end_time) in enumerate(scene_list):
            start_frame = int(start_time.get_seconds() * frame_rate)
            end_frame = int(end_time.get_seconds() * frame_rate)
            scenes_info.append({
                'scene_id': i + 1,
                'start_time_seconds': round(start_time.get_seconds(), 2),
                'end_time_seconds': round(end_time.get_seconds(), 2),
                'start_frame': start_frame,
                'end_frame': end_frame,
                'frame_count': end_frame - start_frame + 1
            })
            print(f"{Fore.CYAN}  Scene {i+1}: Frames {start_frame}-{end_frame} ({end_frame-start_frame+1} frames)")
    else:
        print_warning("No scenes detected")
    
    return scenes_info
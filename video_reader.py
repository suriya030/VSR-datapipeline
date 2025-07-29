import av
import cv2
import os
from tqdm import tqdm
from colorama import Fore, Style
from config import VIDEO_CONVERSION
from utils import print_processing, print_success

def print_video_info(file_path, container, video_stream, frame_rate, total_frames):
    """Display video file information and metadata"""
    print(f"{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}VIDEO INFORMATION")
    print(f"{Fore.CYAN}{'='*60}")
    print(f"{Fore.WHITE}File: {Fore.YELLOW}{os.path.basename(file_path)}")
    print(f"{Fore.WHITE}Format: {Fore.GREEN}{container.format.name}")
    print(f"{Fore.WHITE}Duration: {Fore.GREEN}{container.duration / av.time_base:.2f} seconds")
    print(f"{Fore.WHITE}Codec: {Fore.GREEN}{video_stream.codec.name}")
    print(f"{Fore.WHITE}Resolution: {Fore.GREEN}{video_stream.width}x{video_stream.height}")
    print(f"{Fore.WHITE}Frame Rate: {Fore.GREEN}{frame_rate} fps")
    print(f"{Fore.WHITE}Pixel Format: {Fore.GREEN}{video_stream.pix_fmt}")
    print(f"{Fore.WHITE}Bitrate: {Fore.GREEN}{container.bit_rate} bps")
    print(f"{Fore.WHITE}Total Frames: {Fore.GREEN}{total_frames}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")

def convert_frame_to_720p(frame):
    """Convert frame to 720p resolution at 8-bit depth"""
    frame_720p = frame.reformat(
        width=VIDEO_CONVERSION['target_width'], 
        height=VIDEO_CONVERSION['target_height'], 
        format=VIDEO_CONVERSION['format']
    )
    return frame_720p.to_ndarray()

def read_mxf_video(file_path):
    """Read MXF video file and convert frames to 720p"""
    print_processing("Opening MXF video file...")
    
    # Open video containers
    container = av.open(file_path)
    video_stream = container.streams.video[0]
    
    # Get accurate frame info using OpenCV
    cap = cv2.VideoCapture(file_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print_video_info(file_path, container, video_stream, frame_rate, total_frames)
    
    # Prepare video metadata
    video_info = {
        'file_path': file_path,
        'format': container.format.name,
        'codec': video_stream.codec.name,
        'original_resolution': f"{video_stream.width}x{video_stream.height}",
        'frame_rate': frame_rate,
        'pixel_format': video_stream.pix_fmt,
        'bitrate': container.bit_rate,
        'total_frames': total_frames
    }
    
    print(f"{Fore.BLUE}🔄 Converting frames to 720p...")
    frame_count = 0
    frames_720p = []
    
    # Process each frame with progress tracking
    with tqdm(total=total_frames, desc="Processing frames", unit="frame", colour="blue") as pbar:
        for frame in container.decode(video_stream):
            frame_count += 1
            converted_frame = convert_frame_to_720p(frame)
            frames_720p.append(converted_frame)
            pbar.update(1)
    
    container.close()
    print_success(f"Successfully processed {frame_count} frames and converted to 720p")
    return frames_720p, video_info
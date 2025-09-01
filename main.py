import os
from colorama import Fore, Style

# Import all modules
from video_reader import read_mxf_video
from scene_detector import find_and_split_scenes
from quality_analyzer import initialize_quality_metrics, find_sequence_per_scene
from data_exporter import save_analysis_json
from utils import print_header, print_step, print_success, get_base_filename
from config import DEVICE

def process_mxf_complete_pipeline(mxf_file_path, output_folder):
    """Complete MXF analysis pipeline: read video, detect scenes, analyze frame quality"""
    
    print_header("🚀 STARTING MXF VIDEO ANALYSIS PIPELINE")
    
    # Step 1: Read and convert MXF video
    print_step(1, "Reading MXF file and converting frames")
    frames_list, video_info = read_mxf_video(mxf_file_path)
    
    # Step 2: Detect video scenes
    print_step(2, "Scene detection")
    scenes_info = find_and_split_scenes(mxf_file_path, video_info['frame_rate'])
    
    # Step 3: Initialize quality analysis models
    print_step(3, "Quality-based frame analysis per scene")
    musiq_metric, niqe_metric, device = initialize_quality_metrics(DEVICE)
    
    # Step 4: Analyze frame quality per scene
    base_video_name = get_base_filename(mxf_file_path)
    scene_results = find_sequence_per_scene(
        frames_list, 
        scenes_info,
        base_video_name,
        musiq_metric, 
        niqe_metric, 
        device
    )
    
    # Step 5: Save results
    print_step(4, "Saving analysis results")
    json_output_path = os.path.join(output_folder, f"{base_video_name}_analysis.json")
    analysis_data = save_analysis_json(video_info, scenes_info, scene_results, json_output_path)
    
    # Final summary
    sequences_found = sum(1 for sr in scene_results if sr['sequence_found'])
    total_scenes = len(scenes_info)
    
    print(f"\n{Fore.GREEN}{'='*80}")
    print(f"{Fore.GREEN}🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{Fore.GREEN}📊 Results: Found quality sequences in {Fore.YELLOW}{sequences_found}/{total_scenes}{Fore.GREEN} scenes")
    print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
    
    return analysis_data

if __name__ == "__main__":
    mxf_folder = "movie"
    output_folder = "analysis_results"
        
    mxf_files_to_process = [f for f in os.listdir(mxf_folder) ]
    
    if not mxf_files_to_process:
        print(f"{Fore.YELLOW}No .mxf files found in the '{mxf_folder}' directory.{Style.RESET_ALL}")
    else:
        print(f"Found {len(mxf_files_to_process)} MXF files to process.\n")
        
        for filename in mxf_files_to_process:
            base_name = os.path.splitext(filename)[0]
            expected_output_file = os.path.join(output_folder, f"{base_name}_analysis.json")
            
            if os.path.exists(expected_output_file):
                print(f"{Fore.YELLOW}⏩ Skipping '{filename}': Analysis file already exists.{Style.RESET_ALL}")
                continue
            
            print(f"{Fore.CYAN}{'-'*80}{Style.RESET_ALL}")
            mxf_file_path = os.path.join(mxf_folder, filename)
            analysis_data = process_mxf_complete_pipeline(mxf_file_path, output_folder)

        print_header("✅ ALL FILES PROCESSED")
# Configuration settings for MXF video analysis

# Video processing settings
VIDEO_CONVERSION = {
    'target_width': 1280,
    'target_height': 720,
    'format': 'yuv420p'
}

# Scene detection settings
SCENE_DETECTION = {
    'adaptive_threshold': 3.0
}

# Quality analysis settings
QUALITY_ANALYSIS = {
    'sequence_length': 15,
    'use_musiq': False,  # Set to False to use NIQE-only
    'musiq_threshold': 35.0,
    'niqe_threshold': 6.0,
    'min_frame_variance': 10.0
}

# Output settings
OUTPUT = {
    'json_indent': 2
}

# Device settings
DEVICE = 'cuda'  # 'auto', 'cuda', or 'cpu'
import os
import torch
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

def get_device(device_preference='auto'):
    """Get the appropriate device for PyTorch operations"""
    if device_preference == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_preference)
    
    print(f"{Fore.BLUE}🔧 Initializing IQA models on device: {Fore.GREEN}{device}")
    return device

def print_header(title):
    """Print a formatted header"""
    print(f"{Fore.MAGENTA}{'='*80}")
    print(f"{Fore.MAGENTA}{title}")
    print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")

def print_step(step_num, description):
    """Print a formatted step header"""
    print(f"\n{Fore.CYAN}📖 STEP {step_num}: {description}")

def print_success(message):
    """Print a success message"""
    print(f"{Fore.GREEN}✅ {message}")

def print_warning(message):
    """Print a warning message"""
    print(f"{Fore.YELLOW}⚠️  {message}")

def print_info(message):
    """Print an info message"""
    print(f"{Fore.BLUE}🔍 {message}")

def print_processing(message):
    """Print a processing message"""
    print(f"{Fore.MAGENTA}🎬 {message}")

def get_base_filename(file_path):
    """Extract base filename without extension"""
    return os.path.basename(file_path).split('.')[0]

def ensure_directory(file_path):
    """Ensure directory exists for given file path"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
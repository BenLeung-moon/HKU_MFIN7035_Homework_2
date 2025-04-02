import subprocess
import sys
import platform

def check_cuda_availability():
    """Check if CUDA is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def check_mps_availability():
    """Check if MPS (Apple Metal) is available"""
    try:
        import torch
        return platform.system() == "Darwin" and torch.backends.mps.is_available()
    except ImportError:
        return False

def install_dependencies():
    """Install dependencies based on system"""
    print("Checking system configuration...")
    
    # Check for CUDA
    has_cuda = check_cuda_availability()
    # Check for MPS (Apple Metal)
    has_mps = check_mps_availability()
    
    if has_cuda:
        print("CUDA is available. Installing CUDA-enabled PyTorch...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
    elif has_mps:
        print("Apple Metal (MPS) is available. Installing MPS-enabled PyTorch...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ])
    else:
        print("No GPU acceleration available. Installing CPU-only PyTorch...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ])
    
    # Install other dependencies
    print("Installing other dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "-r", "requirements.txt"
    ])
    
    print("\nInstallation complete!")
    if has_cuda:
        print("CUDA-enabled PyTorch installed successfully")
    elif has_mps:
        print("Apple Metal (MPS)-enabled PyTorch installed successfully")
    else:
        print("CPU-only PyTorch installed successfully")

if __name__ == "__main__":
    install_dependencies() 
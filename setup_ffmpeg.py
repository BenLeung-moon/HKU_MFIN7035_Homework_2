import os
import requests
import zipfile
import shutil
import winreg
import sys
from pathlib import Path

def download_ffmpeg():
    """Download ffmpeg from gyan.dev"""
    print("Downloading ffmpeg...")
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    response = requests.get(url, stream=True)
    
    # Create tools directory if it doesn't exist
    os.makedirs("tools", exist_ok=True)
    
    # Download the zip file
    zip_path = "tools/ffmpeg.zip"
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    print("Extracting ffmpeg...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("tools")
    
    # Find the extracted ffmpeg directory
    ffmpeg_dir = None
    for item in os.listdir("tools"):
        if item.startswith("ffmpeg"):
            ffmpeg_dir = os.path.join("tools", item)
            break
    
    if not ffmpeg_dir:
        print("Error: Could not find extracted ffmpeg directory")
        return None
    
    # Move bin directory to tools/ffmpeg
    bin_dir = os.path.join(ffmpeg_dir, "bin")
    target_dir = os.path.join("tools", "ffmpeg")
    
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    
    shutil.move(bin_dir, target_dir)
    shutil.rmtree(ffmpeg_dir)  # Clean up the original directory
    os.remove(zip_path)  # Remove the zip file
    
    return os.path.abspath(target_dir)

def add_to_path(new_path):
    """Add a directory to system PATH"""
    try:
        # Open the registry key for the system PATH
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_ALL_ACCESS)
        
        try:
            current_path, _ = winreg.QueryValueEx(key, "Path")
        except WindowsError:
            current_path = ""
        
        # Add new path if it's not already in PATH
        path_list = current_path.split(";")
        if new_path not in path_list:
            new_path_str = current_path + ";" + new_path if current_path else new_path
            winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path_str)
            print(f"Added {new_path} to PATH")
        else:
            print("ffmpeg is already in PATH")
            
    except Exception as e:
        print(f"Error modifying PATH: {e}")
    finally:
        winreg.CloseKey(key)

def main():
    ffmpeg_path = download_ffmpeg()
    if ffmpeg_path:
        print(f"ffmpeg downloaded to: {ffmpeg_path}")
        add_to_path(ffmpeg_path)
        print("\nffmpeg has been installed and added to PATH.")
        print("Please restart your terminal/IDE for the PATH changes to take effect.")
        print("\nYou can test ffmpeg by opening a new terminal and typing: ffmpeg -version")

if __name__ == "__main__":
    main() 
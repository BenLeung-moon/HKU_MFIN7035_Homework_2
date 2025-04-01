# This script is used to download youtube video and convert a video to a script
import os
import json
from pytubefix import YouTube, Channel
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import requests
import time
import re
from urllib.parse import urlparse, parse_qs
from tenacity import retry, stop_after_attempt, wait_exponential
from faster_whisper import WhisperModel
import torch
import platform

class VideoToScript:
    def __init__(self, output_dir="data/videos"):
        """Initialize the VideoToScript class with output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Check for ffmpeg installation
        self._check_ffmpeg()
        
        # Check for GPU availability and optimize settings
        self.device = "cuda" if torch.cuda.is_available() else "mps" if self._is_mac_with_metal() else "cpu"
        
        if self.device == "cuda":
            # Get GPU information
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
            print(f"Using CUDA GPU: {gpu_name} with {gpu_memory:.1f}GB memory")
            
            # Optimize compute type based on GPU memory
            if gpu_memory >= 16:  # High-end GPU
                self.compute_type = "float16"
            else:  # Mid-range GPU
                self.compute_type = "int8_float16"
        elif self.device == "mps":
            print("Using Apple Metal GPU acceleration")
            self.compute_type = "float16"  # Metal works best with float16
        else:
            self.compute_type = "int8"
            print("No GPU available, using CPU")
        
        print(f"Using device: {self.device}")
        print(f"Compute type: {self.compute_type}")
        
        # Initialize whisper model with optimized settings
        print("Loading Whisper model...")
        self.model = WhisperModel(
            "base",  # You can use "tiny", "base", "small", "medium", or "large"
            device=self.device,
            compute_type=self.compute_type,
            download_root="models",  # Save models to a local directory
            num_workers=4 if self.device in ["cuda", "mps"] else 2,  # More workers for GPU
            cpu_threads=6 if self.device in ["cuda", "mps"] else 4  # More CPU threads for GPU
        )
        print("Whisper model loaded successfully")
    
    def _check_ffmpeg(self):
        """Check if ffmpeg is installed and accessible"""
        try:
            import subprocess
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("ffmpeg is not installed or not accessible in PATH")
            print("Please run setup_ffmpeg.py to install ffmpeg")
            print("Or install ffmpeg manually and add it to your system PATH")
            raise RuntimeError("ffmpeg is required but not installed")
    
    def _is_mac_with_metal(self):
        """Check if running on Mac with Metal GPU support"""
        if platform.system() != "Darwin":  # Not macOS
            return False
        try:
            # Check if MPS (Metal Performance Shaders) is available
            return torch.backends.mps.is_available() and torch.backends.mps.is_built()
        except:
            return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _create_youtube_object(self, url):
        """Create YouTube object with retry mechanism"""
        try:
            return YouTube(url)
        except Exception as e:
            print(f"Error creating YouTube object: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _download_stream(self, stream, output_path):
        """Download stream with retry mechanism"""
        try:
            stream.download(output_path=os.path.dirname(output_path), filename=os.path.basename(output_path))
            return output_path
        except Exception as e:
            print(f"Error downloading stream: {e}")
            raise
    
    def download_from_channel(self, channel_url, limit=5):
        """Download videos from a YouTube channel
        
        Args:
            channel_url (str): URL of the YouTube channel
            limit (int): Maximum number of videos to download
            
        Returns:
            list: List of downloaded video paths
        """
        try:
            print(f"Processing channel: {channel_url}")
            try:
                channel = Channel(channel_url)
                print(f"Channel name: {channel.channel_name}")
            except Exception as e:
                print(f"Error creating Channel object: {e}")
                return []
            
            video_paths = []
            count = 0
            
            for video in channel.videos:
                if count >= limit:
                    break
                
                print(f"Downloading: {video.title}")
                video_path = self._download_video(video)
                if video_path:
                    video_paths.append(video_path)
                    count += 1
                    time.sleep(2)  # Add delay between downloads
            
            return video_paths
        except Exception as e:
            print(f"Error downloading from channel: {e}")
            return []
    
    def _normalize_channel_url(self, url):
        """Normalize YouTube channel URL to standard format"""
        try:
            # Remove any trailing slashes
            url = url.rstrip('/')
            
            # Handle different URL formats
            if '/channel/' in url:
                return url
            elif '/c/' in url:
                # Convert /c/ format to /channel/ format
                channel_name = url.split('/c/')[-1]
                # First get the channel ID
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                if response.status_code == 200:
                    channel_id_match = re.search(r'channel/([^/]+)', response.text)
                    if channel_id_match:
                        return f"https://www.youtube.com/channel/{channel_id_match.group(1)}"
                    # Try alternative pattern
                    channel_id_match = re.search(r'data-channel-id="([^"]+)"', response.text)
                    if channel_id_match:
                        return f"https://www.youtube.com/channel/{channel_id_match.group(1)}"
            elif '/user/' in url:
                # Convert /user/ format to /channel/ format
                user_name = url.split('/user/')[-1]
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                if response.status_code == 200:
                    channel_id_match = re.search(r'channel/([^/]+)', response.text)
                    if channel_id_match:
                        return f"https://www.youtube.com/channel/{channel_id_match.group(1)}"
                    # Try alternative pattern
                    channel_id_match = re.search(r'data-channel-id="([^"]+)"', response.text)
                    if channel_id_match:
                        return f"https://www.youtube.com/channel/{channel_id_match.group(1)}"
            elif '@' in url:  # Handle @username format
                username = url.split('@')[-1]
                response = requests.get(f"https://www.youtube.com/@{username}", headers={'User-Agent': 'Mozilla/5.0'})
                if response.status_code == 200:
                    channel_id_match = re.search(r'channel/([^/]+)', response.text)
                    if channel_id_match:
                        return f"https://www.youtube.com/channel/{channel_id_match.group(1)}"
                    # Try alternative pattern
                    channel_id_match = re.search(r'data-channel-id="([^"]+)"', response.text)
                    if channel_id_match:
                        return f"https://www.youtube.com/channel/{channel_id_match.group(1)}"
            
            # If no specific format found, try to get channel ID directly
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code == 200:
                channel_id_match = re.search(r'channel/([^/]+)', response.text)
                if channel_id_match:
                    return f"https://www.youtube.com/channel/{channel_id_match.group(1)}"
                # Try alternative pattern
                channel_id_match = re.search(r'data-channel-id="([^"]+)"', response.text)
                if channel_id_match:
                    return f"https://www.youtube.com/channel/{channel_id_match.group(1)}"
            
            print(f"Could not normalize channel URL: {url}")
            return None
        except Exception as e:
            print(f"Error normalizing channel URL: {e}")
            return None
    
    def _download_video(self, video):
        """Download a single YouTube video
        
        Args:
            video (YouTube): YouTube video object
            
        Returns:
            str: Path to downloaded video or None if failed
        """
        try:
            # Get video ID if it's a URL
            if isinstance(video, str):
                video_id = self._extract_video_id(video)
                if not video_id:
                    print("Invalid video URL")
                    return None
                video = self._create_youtube_object(f"https://www.youtube.com/watch?v={video_id}")
            
            # Get available streams
            streams = video.streams.filter(progressive=True, file_extension='mp4')
            if not streams:
                print("No suitable streams found")
                return None
            
            # Get the highest resolution stream
            stream = streams.order_by('resolution').desc().first()
            if not stream:
                print("No suitable stream found")
                return None
            
            # Create a safe filename using only alphanumeric characters and underscores
            safe_filename = re.sub(r'[^a-zA-Z0-9]', '_', video.title)
            safe_filename = re.sub(r'_+', '_', safe_filename)  # Replace multiple underscores with single
            safe_filename = safe_filename.strip('_')  # Remove leading/trailing underscores
            safe_filename = safe_filename[:100]  # Limit filename length
            
            output_path = os.path.join(self.output_dir, f"{safe_filename}.mp4")
            
            # Download the video with retry mechanism
            try:
                print(f"Downloading stream: {stream.resolution}")
                self._download_stream(stream, output_path)
                print(f"Downloaded: {output_path}")
                return output_path
            except Exception as e:
                print(f"Failed to download video after retries: {e}")
                return None
            
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None
    
    def _extract_video_id(self, url):
        """Extract video ID from URL"""
        try:
            parsed_url = urlparse(url)
            if parsed_url.hostname == 'youtu.be':
                return parsed_url.path[1:]
            if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
                if parsed_url.path == '/watch':
                    return parse_qs(parsed_url.query)['v'][0]
                if parsed_url.path[:7] == '/embed/':
                    return parsed_url.path.split('/')[2]
            return None
        except Exception as e:
            print(f"Error extracting video ID: {e}")
            return None
    
    def download_single_video(self, video_url):
        """Download a single video from URL
        
        Args:
            video_url (str): URL of the YouTube video
            
        Returns:
            str: Path to downloaded video or None if failed
        """
        try:
            # Extract video ID
            video_id = self._extract_video_id(video_url)
            if not video_id:
                print("Invalid video URL")
                return None
            
            # Create YouTube object with retry mechanism
            try:
                print("Creating YouTube object")
                video = self._create_youtube_object(f"https://www.youtube.com/watch?v={video_id}")
                return self._download_video(video)
            except Exception as e:
                print(f"Failed to create YouTube object after retries: {e}")
                return None
                
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None
    
    def _extract_audio(self, video_path):
        """Extract audio from video
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            str: Path to extracted audio file
        """
        try:
            # Create a temporary file for the audio with a safe name
            temp_dir = tempfile.gettempdir()
            safe_filename = os.path.basename(video_path)
            safe_filename = re.sub(r'[^a-zA-Z0-9]', '_', safe_filename)
            safe_filename = re.sub(r'_+', '_', safe_filename)
            safe_filename = safe_filename.strip('_')
            safe_filename = safe_filename[:100]
            
            audio_path = os.path.join(temp_dir, f"{safe_filename}.wav")
            
            # Ensure the video file exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Extract audio using pydub
            print(f"Extracting audio from video: {video_path}")
            video = AudioSegment.from_file(video_path, format="mp4")
            print(f"Exporting audio to: {audio_path}")
            video.export(audio_path, format="wav")
            
            # Verify the audio file was created
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Failed to create audio file: {audio_path}")
            
            print(f"Successfully extracted audio to: {audio_path}")
            return audio_path
        except Exception as e:
            print(f"Error extracting audio: {e}")
            raise
    
    def _transcribe_with_speech_recognition(self, audio_path):
        """Transcribe audio to text using speech_recognition
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            recognizer = sr.Recognizer()
            
            # Load audio file
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
            
            # Transcribe audio
            text = recognizer.recognize_google(audio_data)
            return text
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return f"Transcription failed: {str(e)}"
    
    def _split_audio(self, audio_path, chunk_duration_ms=60000):
        """Split audio file into chunks
        
        Args:
            audio_path (str): Path to the audio file
            chunk_duration_ms (int): Duration of each chunk in milliseconds
            
        Returns:
            list: List of paths to audio chunks
        """
        try:
            # Load the audio file
            audio = AudioSegment.from_wav(audio_path)
            
            # Calculate number of chunks
            num_chunks = len(audio) // chunk_duration_ms + (1 if len(audio) % chunk_duration_ms else 0)
            
            chunk_paths = []
            temp_dir = tempfile.gettempdir()
            base_filename = os.path.splitext(os.path.basename(audio_path))[0]
            
            # Split audio into chunks
            for i in range(num_chunks):
                start = i * chunk_duration_ms
                end = min((i + 1) * chunk_duration_ms, len(audio))
                chunk = audio[start:end]
                
                # Save chunk to temporary file
                chunk_path = os.path.join(temp_dir, f"{base_filename}_chunk{i}.wav")
                chunk.export(chunk_path, format="wav")
                chunk_paths.append(chunk_path)
            
            return chunk_paths
        except Exception as e:
            print(f"Error splitting audio: {e}")
            raise
    
    def _transcribe_with_whisper(self, audio_path):
        """Transcribe audio using Whisper with GPU acceleration
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            print(f"Transcribing audio with Whisper: {audio_path}")
            
            # Optimize transcription parameters based on device
            if self.device in ["cuda", "mps"]:
                segments, _ = self.model.transcribe(
                    audio_path,
                    beam_size=5,
                    vad_filter=True,  # Voice Activity Detection
                    vad_parameters=dict(min_silence_duration_ms=500),  # Adjust silence threshold
                    condition_on_previous_text=True,  # Better context handling
                    no_speech_threshold=0.6,  # Better silence detection
                    compression_ratio_threshold=1.2,  # Better text compression
                    temperature=0.0,  # More deterministic output
                    best_of=5  # Better beam search
                )
            else:
                segments, _ = self.model.transcribe(
                    audio_path,
                    beam_size=3,  # Reduced for CPU
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
            
            text = " ".join([segment.text for segment in segments])
            return text
        except Exception as e:
            print(f"Error with Whisper transcription: {str(e)}")
            return self._transcribe_with_speech_recognition(audio_path)
    
    def video_to_script(self, video_path):
        """Convert video to script
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            dict: JSON with video info and script
        """
        try:
            # Extract video info
            video_filename = os.path.basename(video_path)
            video_name = os.path.splitext(video_filename)[0]
            
            # Convert video to audio
            audio_path = self._extract_audio(video_path)
            
            # Convert audio to text using Whisper
            script = self._transcribe_with_whisper(audio_path)
            
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            # Create result JSON
            result = {
                "video_name": video_name,
                "video_path": video_path,
                "script": script,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return result
        except Exception as e:
            print(f"Error converting video to script: {e}")
            return {
                "video_name": os.path.basename(video_path),
                "video_path": video_path,
                "script": f"Error: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def process_channel(self, channel_url, limit=5):
        """Process videos from a channel
        
        Args:
            channel_url (str): URL of the YouTube channel
            limit (int): Maximum number of videos to process
            
        Returns:
            list: List of results with video info and scripts
        """
        video_paths = self.download_from_channel(channel_url, limit)
        results = []
        
        for video_path in video_paths:
            result = self.video_to_script(video_path)
            results.append(result)
        
        return results
    
    def save_results(self, results, output_file="data/video_scripts.json"):
        """Save results to a JSON file
        
        Args:
            results (list): List of results
            output_file (str): Path to output file
            
        Returns:
            str: Path to output file
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_file}")
        return output_file

# Example usage
if __name__ == "__main__":
    converter = VideoToScript()
    
    # Example 1: Download and process a single video
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    video_path = converter.download_single_video(video_url)
    if video_path:
        result = converter.video_to_script(video_path)
        print(json.dumps(result, indent=2))
    
    # Example 2: Process videos from a channel
    # channel_url = "https://www.youtube.com/c/TradeBrigade"
    # results = converter.process_channel(channel_url, limit=2)
    # converter.save_results(results)

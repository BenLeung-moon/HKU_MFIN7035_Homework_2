# This script is used to download youtube video and convert a video to a script
import os
import json
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import time
import re
from tenacity import retry, stop_after_attempt, wait_exponential
from faster_whisper import WhisperModel
import torch
import platform
import multiprocessing
import subprocess

class VideoToScript:
    def __init__(self, output_dir="data/videos"):
        """Initialize the VideoToScript class with output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Check for ffmpeg installation
        self._check_ffmpeg()
        
        # Get number of CPU cores for fallback
        self.num_cores = multiprocessing.cpu_count()
        print(f"Number of CPU cores available: {self.num_cores}")
        
        # Check for GPU availability and optimize settings
        self.device = "cuda" if torch.cuda.is_available() else "mps" if self._is_mac_with_metal() else "cpu"
        
        if self.device == "cuda":
            # Get GPU information
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
            print(f"Using CUDA GPU: {gpu_name} with {gpu_memory:.1f}GB memory")
            
            # Optimize compute type and model size based on GPU memory
            if gpu_memory >= 16:  # High-end GPU
                self.compute_type = "float16"
                self.model_size = "large"  # Use larger model for high-end GPUs
            else:  # Mid-range GPU
                self.compute_type = "int8_float16"
                self.model_size = "medium"  # Use medium model for mid-range GPUs
            
            # Verify CUDA is working
            try:
                test_tensor = torch.zeros(1).cuda()
                print("CUDA initialization successful")
            except Exception as e:
                print(f"CUDA initialization failed: {e}")
                print("Falling back to CPU")
                self.device = "cpu"
                self.compute_type = "int8"
                self.model_size = "base"
        elif self.device == "mps":
            print("Using Apple Metal GPU acceleration")
            self.compute_type = "float16"  # Metal works best with float16
            self.model_size = "medium"  # Use medium model for Metal GPU
            
            # Verify MPS is working
            try:
                test_tensor = torch.zeros(1).to("mps")
                print("MPS initialization successful")
            except Exception as e:
                print(f"MPS initialization failed: {e}")
                print("Falling back to CPU")
                self.device = "cpu"
                self.compute_type = "int8"
                self.model_size = "base"
        else:
            self.compute_type = "int8"
            self.model_size = "base"  # Use base model for CPU
            print("Using CPU for transcription")
        
        print(f"Using device: {self.device}")
        print(f"Compute type: {self.compute_type}")
        print(f"Model size: {self.model_size}")
        
        # Initialize whisper model with optimized settings
        print("Loading Whisper model...")
        try:
            self.model = WhisperModel(
                self.model_size,  # Use model size based on device
                device=self.device,
                compute_type=self.compute_type,
                download_root="models",  # Save models to a local directory
                num_workers=1 if self.device in ["cuda", "mps"] else self.num_cores,  # Use 1 worker for GPU, all cores for CPU
                cpu_threads=1 if self.device in ["cuda", "mps"] else self.num_cores  # Use 1 thread for GPU, all cores for CPU
            )
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            print("Falling back to CPU with base model")
            self.device = "cpu"
            self.compute_type = "int8"
            self.model_size = "base"
            self.model = WhisperModel(
                "base",
                device="cpu",
                compute_type="int8",
                download_root="models",
                num_workers=self.num_cores,
                cpu_threads=self.num_cores
            )
    
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
                    vad_parameters=dict(min_silence_duration_ms=500),
                    language="en",
                    initial_prompt="This is a financial market analysis video."
                )
            
            text = " ".join([segment.text for segment in segments])
            return text
        except Exception as e:
            print(f"Error with Whisper transcription: {str(e)}")
            return self._transcribe_with_speech_recognition(audio_path)
    
    def video_to_script(self, video_info):
        """Convert video to script
        
        Args:
            video_info (dict): Dictionary containing video path and metadata
            
        Returns:
            dict: JSON with video info and script
        """
        try:
            video_path = video_info["path"]
            video_filename = os.path.basename(video_path)
            video_name = os.path.splitext(video_filename)[0]
            
            # Convert video to audio and transcribe
            print("No captions available, using audio transcription")
            audio_path = self._extract_audio(video_path)
            script = self._transcribe_with_whisper(audio_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            # Create result JSON
            result = {
                "video_name": video_name,
                "video_path": video_path,
                "script": script,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "video_date": video_info["date"],
                "video_title": video_info["title"],
                "video_id": video_info["video_id"],
                "has_captions": video_info.get("has_captions", False)
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

    def process_video(self, downloader, analyzer, video_info):
        """Process a single video: download, transcribe, and analyze"""
        try:
            # Download video and get captions if available
            result = downloader._download_video(video_info['url'])
            if not result:
                print("✗ Download failed, skipping analysis")
                return None
                
            print("✓ Download successful")
            
            # Check if we already have a transcript from captions
            if result.get("transcript"):
                print("✓ Using available captions")
                transcript = result["transcript"]
            else:
                print("No captions available, starting audio transcription...")
                video_path = result['path']
                audio_path = video_path.replace('.mp4', '.wav')
                
                # Convert video to audio with suppressed output
                try:
                    subprocess.run([
                        'ffmpeg', '-i', video_path,
                        '-vn',
                        '-acodec', 'pcm_s16le',
                        '-ar', '44100',
                        '-ac', '2',
                        audio_path
                    ], check=True, capture_output=True)
                    print("✓ Audio extraction successful")
                except subprocess.CalledProcessError as e:
                    print(f"✗ Error extracting audio: {e}")
                    return None
                except FileNotFoundError:
                    print("✗ ffmpeg not found. Please install ffmpeg to use transcription.")
                    return None
                
                # Initialize speech recognizer
                recognizer = sr.Recognizer()
                
                # Transcribe audio with improved error handling
                try:
                    with sr.AudioFile(audio_path) as source:
                        print("Processing audio...")
                        # Adjust for ambient noise
                        recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio = recognizer.record(source)
                        print("Transcribing...")
                        
                        # Try multiple speech recognition services with better error handling
                        transcript = None
                        errors = []
                        
                        # Try Google Speech Recognition first
                        try:
                            transcript = recognizer.recognize_google(audio, language="en-US")
                            print("✓ Google Speech Recognition successful")
                        except sr.UnknownValueError:
                            errors.append("Google Speech Recognition could not understand audio")
                        except sr.RequestError as e:
                            errors.append(f"Google Speech Recognition request failed: {e}")
                        
                        # If Google fails, try Sphinx
                        if not transcript:
                            try:
                                transcript = recognizer.recognize_sphinx(audio)
                                print("✓ Sphinx Speech Recognition successful")
                            except Exception as e:
                                errors.append(f"Sphinx Speech Recognition failed: {e}")
                        
                        # If both fail, try Whisper
                        if not transcript:
                            try:
                                transcript = self._transcribe_with_whisper(audio_path)
                                if transcript:
                                    print("✓ Whisper transcription successful")
                            except Exception as e:
                                errors.append(f"Whisper transcription failed: {e}")
                        
                        if not transcript:
                            print("✗ All transcription methods failed:")
                            for error in errors:
                                print(f"  - {error}")
                            return None
                        
                        print("✓ Transcription successful")
                        
                        # Clean up audio file
                        try:
                            os.remove(audio_path)
                            print("Cleaned up audio file")
                        except Exception as e:
                            print(f"Error cleaning up audio file: {e}")
                
                except Exception as e:
                    print(f"✗ Error during transcription: {e}")
                    return None
            
            # Save transcription to JSON
            transcript_data = {
                "video_id": video_info['video_id'],
                "title": video_info['title'],
                "date": video_info['date'],
                "transcript": transcript,
                "source": result.get("transcript_source", "audio_transcription"),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save to transcripts directory
            os.makedirs("data/transcripts", exist_ok=True)
            transcript_file = f"data/transcripts/{video_info['video_id']}.json"
            with open(transcript_file, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Transcription saved to: {transcript_file}")
            
            # Preprocess transcript
            print("Preprocessing transcript...")
            processed_transcript = analyzer.preprocess_transcript(
                transcript,
                video_info['title'],
                video_info['date']
            )
            
            # Analyze transcript
            print("Analyzing transcript...")
            analysis = analyzer.analyze_transcript(
                processed_transcript,
                video_info['title'],
                video_info['date']
            )
            
            # Combine results
            result.update({
                "transcript": transcript,
                "processed_transcript": processed_transcript,
                "analysis": analysis,
                "transcript_source": result.get("transcript_source", "audio_transcription")
            })
            
            return result
                
        except Exception as e:
            print(f"✗ Error processing video: {str(e)}")
            return None

    def save_progress(self, progress_file, processed_videos, processed_results=None):
        """Save progress to file
        
        Args:
            progress_file (str): Path to progress file
            processed_videos (list): List of processed video info dictionaries
            processed_results (list, optional): List of processed results
        """
        try:
            progress_data = {
                "videos": processed_videos,
                "results": processed_results or []
            }
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
            print(f"Progress saved to {progress_file}")
        except Exception as e:
            print(f"Error saving progress: {e}")

    def load_progress(self, progress_file):
        """Load progress from file
        
        Args:
            progress_file (str): Path to progress file
            
        Returns:
            tuple: (processed_videos, processed_results)
        """
        try:
            if os.path.exists(progress_file):
                with open(progress_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("videos", []), data.get("results", [])
            return [], []
        except Exception as e:
            print(f"Error loading progress: {e}")
            return [], []

# Example usage
if __name__ == "__main__":
    converter = VideoToScript()
    
    # Example: Process a video file
    video_info = {
        "path": "path/to/your/video.mp4",
        "date": "2024-01-01",
        "title": "Example Video",
        "video_id": "example123",
        "has_captions": False
    }
    result = converter.video_to_script(video_info)
    print(json.dumps(result, indent=2))

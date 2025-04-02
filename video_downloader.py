import os
import json
import time
from datetime import datetime
from pytubefix import YouTube, Channel
from tenacity import retry, stop_after_attempt, wait_exponential
import re
from urllib.parse import urlparse, parse_qs
import multiprocessing
from functools import partial
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

class VideoDownloader:
    def __init__(self, output_dir="data/videos"):
        """Initialize the VideoDownloader class with output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Get number of CPU cores for parallel processing
        self.num_cores = multiprocessing.cpu_count()
        print(f"Number of CPU cores available: {self.num_cores}")
        
        # Cache directories
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create visualization directory
        self.viz_dir = "data/visualizations"
        os.makedirs(self.viz_dir, exist_ok=True)
    
    def _get_cache_key(self, channel_url, cache_type="raw"):
        """Generate a unique cache key for the channel"""
        # Create a hash of the channel URL
        url_hash = hashlib.md5(channel_url.encode()).hexdigest()
        return f"{cache_type}_{url_hash}.json"
    
    def _get_cache_path(self, channel_url, cache_type="raw"):
        """Get the path to the cache file"""
        cache_key = self._get_cache_key(channel_url, cache_type)
        return os.path.join(self.cache_dir, cache_key)
    
    def _is_cache_valid(self, channel_url, cache_type="raw"):
        """Check if the cache is valid and not expired"""
        cache_path = self._get_cache_path(channel_url, cache_type)
        if not os.path.exists(cache_path):
            return False
            
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
                
            # Check if cache is expired (24 hours)
            last_updated = datetime.strptime(cache_data.get("last_updated", ""), "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - last_updated).total_seconds() > 24 * 3600:
                return False
                
            # For raw cache, check if video count matches
            if cache_type == "raw":
                try:
                    channel = Channel(channel_url)
                    current_count = len(list(channel.videos))
                    cached_count = len(cache_data.get("videos", []))
                    if current_count != cached_count:
                        print(f"Channel video count changed: {cached_count} -> {current_count}")
                        return False
                except Exception as e:
                    print(f"Error checking channel video count: {e}")
                    return False
                    
            return True
        except json.JSONDecodeError as e:
            print(f"Error reading cache file: {e}")
            return False
        except Exception as e:
            print(f"Error checking cache validity: {e}")
            return False
    
    def _save_cache(self, channel_url, data, cache_type="raw"):
        """Save data to cache"""
        try:
            cache_path = self._get_cache_path(channel_url, cache_type)
            
            # Convert any pandas Timestamp objects to strings
            serializable_data = []
            for video in data:
                video_copy = video.copy()
                if isinstance(video_copy.get('date'), pd.Timestamp):
                    video_copy['date'] = video_copy['date'].strftime("%Y-%m-%d")
                serializable_data.append(video_copy)
            
            cache_data = {
                "channel_url": channel_url,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "videos": serializable_data
            }
            
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            print(f"Cache saved successfully: {cache_type}")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _load_cache(self, channel_url, cache_type="raw"):
        """Load data from cache"""
        try:
            cache_path = self._get_cache_path(channel_url, cache_type)
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                return cache_data.get("videos", [])
            return []
        except Exception as e:
            print(f"Error loading cache: {e}")
            return []
    
    def _create_video_dataframe(self, videos):
        """Create a pandas DataFrame from video list for efficient filtering"""
        try:
            df = pd.DataFrame(videos)
            # Convert date strings to datetime
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            print(f"Error creating video DataFrame: {e}")
            return pd.DataFrame()
    
    def _filter_videos_by_date(self, df, start_date, end_date):
        """Filter videos by date range using pandas"""
        try:
            if start_date:
                start_date = pd.to_datetime(start_date)
                df = df[df['date'] >= start_date]
            
            if end_date:
                end_date = pd.to_datetime(end_date)
                df = df[df['date'] <= end_date]
            
            return df.to_dict('records')
        except Exception as e:
            print(f"Error filtering videos by date: {e}")
            return []
    
    def _download_video_parallel(self, video_info):
        """Download a single video (for parallel processing)"""
        try:
            video = self._create_youtube_object(video_info['url'])
            result = self._download_video(video)
            if result:
                print(f"✓ Downloaded: {result['title']}")
                return result
            else:
                print(f"✗ Failed to download: {video_info['title']}")
                return None
        except Exception as e:
            print(f"✗ Error downloading {video_info['title']}: {e}")
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _create_youtube_object(self, url):
        """Create YouTube object with retry mechanism"""
        try:
            return YouTube(url, use_oauth=False, allow_oauth_cache=True)
        except Exception as e:
            print(f"Error creating YouTube object: {e}")
            try:
                video_id = self._extract_video_id_from_filename(url)
                if video_id:
                    return YouTube(f"https://www.youtube.com/watch?v={video_id}", 
                                 use_oauth=False, 
                                 allow_oauth_cache=True)
            except Exception as e2:
                print(f"Error creating YouTube object from filename: {e2}")
            raise

    def _extract_video_id_from_filename(self, filename):
        """Extract video ID from filename if it contains one"""
        try:
            patterns = [
                r'\[([a-zA-Z0-9_-]{11})\]',  # [VIDEO_ID]
                r'\(([a-zA-Z0-9_-]{11})\)',  # (VIDEO_ID)
                r'([a-zA-Z0-9_-]{11})\.mp4$'  # VIDEO_ID.mp4
            ]
            
            for pattern in patterns:
                match = re.search(pattern, filename)
                if match:
                    return match.group(1)
            return None
        except Exception as e:
            print(f"Error extracting video ID from filename: {e}")
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _download_stream(self, stream, output_path):
        """Download stream with retry mechanism"""
        try:
            stream.download(output_path=os.path.dirname(output_path), filename=os.path.basename(output_path))
            return output_path
        except Exception as e:
            print(f"Error downloading stream: {e}")
            raise

    def _download_captions(self, video):
        """Download captions from YouTube video
        
        Args:
            video (YouTube): YouTube video object
            
        Returns:
            tuple: (bool, str) - (success, transcript or error message)
        """
        try:
            captions = video.captions
            if not captions:
                return False, "No captions available"
            
            # Try different caption types in order of preference
            caption_types = [
                ('a.en', 'English (Auto-generated)'),
                ('en', 'English'),
                ('en-US', 'English (US)')
            ]
            
            caption = None
            for code, name in caption_types:
                try:
                    if code in captions:
                        caption = captions[code]
                        print(f"Found {name} captions")
                        break
                except Exception as e:
                    print(f"Error accessing {name} captions: {e}")
                    continue
            
            if not caption:
                return False, "No suitable captions found"
            
            # Save captions to a temporary file
            temp_caption_file = os.path.join(self.output_dir, f"temp_captions_{video.video_id}.txt")
            caption.save_captions(temp_caption_file)
            
            # Read and process the captions
            with open(temp_caption_file, 'r', encoding='utf-8') as f:
                transcript = f.read()
            
            # Clean up the temporary file
            os.remove(temp_caption_file)
            
            # Convert SRT to plain text
            transcript = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', transcript)
            transcript = re.sub(r'\n+', ' ', transcript).strip()
            
            if not transcript:
                return False, "Empty caption text received"
            
            # Save transcription to JSON
            transcript_data = {
                "video_id": video.video_id,
                "title": video.title,
                "date": video.publish_date.strftime("%Y-%m-%d") if video.publish_date else None,
                "transcript": transcript,
                "source": "youtube_captions",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save to transcripts directory
            os.makedirs("data/transcripts", exist_ok=True)
            transcript_file = f"data/transcripts/{video.video_id}.json"
            with open(transcript_file, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Caption saved to: {transcript_file}")
            
            return True, transcript
            
        except Exception as e:
            return False, f"Error downloading captions: {str(e)}"

    def _download_video(self, video):
        """Download a single YouTube video"""
        try:
            if isinstance(video, str):
                video_id = self._extract_video_id(video)
                if not video_id:
                    print("Invalid video URL")
                    return None
                video = self._create_youtube_object(f"https://www.youtube.com/watch?v={video_id}")
            
            # Get video metadata first
            try:
                video_date = video.publish_date
                if video_date and video_date.tzinfo:
                    video_date = video_date.replace(tzinfo=None)
                video_title = video.title
                video_id = video.video_id
                
                if not video_date or not video_title or not video_id:
                    print("Missing required video metadata")
                    return None
                
                # Try to download captions first
                caption_success, caption_result = self._download_captions(video)
                has_captions = caption_success
                
                # Create safe filename
                safe_title = re.sub(r'[^a-zA-Z0-9\s-]', '', video_title)
                safe_title = re.sub(r'\s+', '_', safe_title)
                safe_title = safe_title.strip('_')
                safe_title = safe_title[:50]
                
                safe_filename = f"{video_date.strftime('%Y-%m-%d')}_{safe_title}.mp4"
                output_path = os.path.join(self.output_dir, safe_filename)
                
                # Get best quality stream
                streams = video.streams.filter(progressive=True, file_extension='mp4')
                if not streams:
                    print("No suitable streams found")
                    return None
                
                stream = streams.order_by('resolution').desc().first()
                if not stream:
                    print("No suitable stream found")
                    return None
                
                # Download the video
                print(f"Downloading stream: {stream.resolution}")
                self._download_stream(stream, output_path)
                print(f"Downloaded: {output_path}")
                
                result = {
                    "path": output_path,
                    "date": video_date.strftime("%Y-%m-%d"),
                    "title": video_title,
                    "video_id": video_id,
                    "has_captions": has_captions,
                    "safe_title": safe_title,
                    "url": f"https://www.youtube.com/watch?v={video_id}"
                }
                
                # Add caption information if available
                if caption_success:
                    result["transcript"] = caption_result
                    result["transcript_source"] = "youtube_captions"
                
                return result
                
            except Exception as e:
                print(f"Error getting video metadata: {e}")
                return None
            
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None

    def _process_video_batch(self, batch, backtest_start_date, backtest_end_date):
        """Process a batch of videos in parallel
        
        Args:
            batch (list): List of (video, publish_date) tuples
            backtest_start_date (datetime): Start date for filtering
            backtest_end_date (datetime): End date for filtering
            
        Returns:
            list: List of video info dictionaries
        """
        batch_results = []
        for video, publish_date in batch:
            try:
                # Skip if outside date range
                if backtest_start_date and publish_date < backtest_start_date:
                    continue
                if backtest_end_date and publish_date > backtest_end_date:
                    continue
                
                # Get video details
                title = video.title
                video_id = video.video_id
                url = f"https://www.youtube.com/watch?v={video_id}"
                
                # Create safe filename
                safe_title = re.sub(r'[^a-zA-Z0-9\s-]', '', title)
                safe_title = re.sub(r'\s+', '_', safe_title)
                safe_title = safe_title.strip('_')
                safe_title = safe_title[:50]
                
                batch_results.append({
                    "video_id": video_id,
                    "title": title,
                    "date": publish_date.strftime("%Y-%m-%d"),
                    "url": url,
                    "safe_title": safe_title
                })
                
            except Exception as e:
                print(f"Error processing video {video.title}: {str(e)}")
                continue
        return batch_results

    def _visualize_filtering_stats(self, stats, channel_name):
        """Visualize video filtering statistics
        
        Args:
            stats (dict): Dictionary containing filtering statistics
            channel_name (str): Name of the YouTube channel
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Video counts
        counts = [stats['total_videos'], stats['valid_dates'], stats['in_date_range']]
        labels = ['Total Videos', 'Valid Dates', 'In Date Range']
        colors = ['#FF9999', '#66B2FF', '#99FF99']
        
        ax1.bar(labels, counts, color=colors)
        ax1.set_title('Video Filtering Process')
        ax1.set_ylabel('Number of Videos')
        
        # Add value labels on top of bars
        for i, v in enumerate(counts):
            ax1.text(i, v, str(v), ha='center', va='bottom')
        
        # Plot 2: Date distribution
        if stats['date_distribution']:
            dates = pd.to_datetime(list(stats['date_distribution'].keys()))
            counts = list(stats['date_distribution'].values())
            
            ax2.plot(dates, counts, marker='o')
            ax2.set_title('Video Distribution by Date')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Number of Videos')
            plt.xticks(rotation=45)
        
        # Add title to the figure
        fig.suptitle(f'Video Filtering Statistics - {channel_name}', fontsize=16)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, f'{channel_name}_filtering_stats.png'))
        plt.close()

    def get_channel_videos(self, channel_url, backtest_start_date=None, backtest_end_date=None):
        """Get list of videos from channel within date range using efficient caching"""
        try:
            print(f"\n=== Fetching Channel Videos ===")
            print(f"Channel URL: {channel_url}")
            print(f"Backtest start date: {backtest_start_date}")
            print(f"Backtest end date: {backtest_end_date}")
            
            # Convert dates to datetime if provided
            if backtest_start_date:
                backtest_start_date = pd.to_datetime(backtest_start_date)
            if backtest_end_date:
                backtest_end_date = pd.to_datetime(backtest_end_date)
            
            # Always try raw cache first
            if self._is_cache_valid(channel_url, "raw"):
                print("Using raw cache...")
                videos = self._load_cache(channel_url, "raw")
            else:
                print("Fetching fresh data from channel...")
                try:
                    channel = Channel(channel_url)
                    videos = []
                    for video in tqdm(channel.videos, desc="Fetching videos"):
                        try:
                            if hasattr(video, 'publish_date') and video.publish_date:
                                publish_date = video.publish_date.replace(tzinfo=None)
                                videos.append({
                                    "video_id": video.video_id,
                                    "title": video.title,
                                    "date": publish_date.strftime("%Y-%m-%d"),
                                    "url": f"https://www.youtube.com/watch?v={video.video_id}",
                                    "safe_title": re.sub(r'[^a-zA-Z0-9\s-]', '', video.title)[:50]
                                })
                        except Exception as e:
                            print(f"Error processing video: {e}")
                            continue
                    
                    # Save raw cache
                    self._save_cache(channel_url, videos, "raw")
                except Exception as e:
                    print(f"Error fetching channel videos: {e}")
                    return []
            
            # Filter videos by date
            filtered_videos = []
            for video in videos:
                video_date = pd.to_datetime(video['date'])
                if (not backtest_start_date or video_date >= backtest_start_date) and \
                   (not backtest_end_date or video_date <= backtest_end_date):
                    filtered_videos.append(video)
            
            print(f"Found {len(filtered_videos)} videos in date range")
            
            # Save filtered cache temporarily
            if filtered_videos:
                self._save_cache(channel_url, filtered_videos, "filtered")
            
            return filtered_videos
            
        except Exception as e:
            print(f"Error getting channel videos: {e}")
            return []

    def download_videos(self, video_list, limit=None):
        """Download videos from the list using parallel processing"""
        try:
            if limit:
                video_list = video_list[:limit]
            
            results = []
            with ThreadPoolExecutor(max_workers=self.num_cores) as executor:
                # Submit all download tasks
                futures = [executor.submit(self._download_video_parallel, video_info) for video_info in video_list]
                
                # Process completed downloads
                for future in tqdm(as_completed(futures), total=len(video_list), desc="Downloading videos"):
                    video_info = future.result()
                    try:
                        if video_info:
                            results.append(video_info)
                    except Exception as e:
                        print(f"Error processing {video_info['title']}: {e}")
            
            return results
            
        except Exception as e:
            print(f"Error downloading videos: {e}")
            return []

    def save_progress(self, progress_file, video_list, downloaded_videos):
        """Save download progress"""
        try:
            progress_data = {
                "video_list": video_list,
                "downloaded_videos": downloaded_videos
            }
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
            print(f"Progress saved to {progress_file}")
        except Exception as e:
            print(f"Error saving progress: {e}")

    def load_progress(self, progress_file):
        """Load download progress"""
        try:
            if os.path.exists(progress_file):
                with open(progress_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("video_list", []), data.get("downloaded_videos", [])
            return [], []
        except Exception as e:
            print(f"Error loading progress: {e}")
            return [], [] 
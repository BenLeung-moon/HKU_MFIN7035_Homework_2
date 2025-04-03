'''
This is the main script that will be used to run the program
Use to evaluate kol prediction on the stock market
use TradeBrigade as an example, and SPY performance as a benchmark for last 1 year
'''

import os
import json
import shutil
from video_downloader import VideoDownloader
from analysis import VideoAnalyzer
from evaluation import PredictionEvaluator
from video_to_script import VideoToScript

# Global configuration variables
CHANNEL_URL = "https://www.youtube.com/c/TradeBrigade"  # TradeBrigade's channel
BACKTEST_START_DATE = "2024-01-01"  
BACKTEST_END_DATE = "2024-12-31"    
TEST_MODE = False  # Set to False to process all videos
MAX_VIDEOS = 10  # Number of videos to analyze
VIDEOS_PER_WEEK = 1  # Maximum number of videos to collect per week
DOWNLOAD_VIDEOS = False  # Set to False to only download captions without the actual videos
USE_PREDOWNLOADED_VIDEOS = False  # Set to True to use pre-downloaded videos for testing
VIDEOS_DIR = "data/videos"  # Directory containing pre-downloaded videos
MODEL = "r1-1776"   # Model to use for analysis
EVALUATE_PREDICTIONS = True  # Whether to evaluate predictions after processing

def main():
    """Main function for processing YouTube videos and analyzing predictions"""
    # Clean up downloaded videos from previous run
    print("\n=== Cleaning Up Previous Videos ===")
    try:
        # Clean up downloaded videos and transcripts
        if os.path.exists("data/videos"):
            shutil.rmtree("data/videos")
            os.makedirs("data/videos")
        print("Cleaned up downloaded videos from previous run")
    except Exception as e:
        print(f"Error during video cleanup: {str(e)}")
    
    # Initialize components
    downloader = VideoDownloader()
    analyzer = VideoAnalyzer()
    evaluator = PredictionEvaluator()
    processor = VideoToScript(load_model=DOWNLOAD_VIDEOS)
    
    results = []
    
    # Process channel videos
    print(f"\n=== Processing Channel Videos ===")
    print(f"Channel: {CHANNEL_URL}")
    print(f"Date range: {BACKTEST_START_DATE} to {BACKTEST_END_DATE}")
    print(f"Maximum videos to process: {MAX_VIDEOS}")
    print(f"Videos per week limit: {VIDEOS_PER_WEEK}")
    print(f"Download videos: {'Yes' if DOWNLOAD_VIDEOS else 'No (captions only)'}")
    
    try:
        videos = downloader.get_channel_videos(
                    CHANNEL_URL,
                    backtest_start_date=BACKTEST_START_DATE,
                    backtest_end_date=BACKTEST_END_DATE,
                    videos_per_week=VIDEOS_PER_WEEK
                )
        
        if videos:
            print(f"\nFound {len(videos)} videos in date range")
            if len(videos) > MAX_VIDEOS:
                print(f"Selecting {MAX_VIDEOS} most recent videos for analysis")
                videos = videos[:MAX_VIDEOS]
            
            print("\nSelected videos for analysis:")
            for video in videos[:5]:  # Show first 5 videos
                print(f"\nTitle: {video['title']}")
                print(f"Date: {video['date']}")
                print(f"URL: {video['url']}")
            if len(videos) > 5:
                print(f"\n... and {len(videos) - 5} more videos")
            
            # Process videos
            print(f"\n=== Processing {len(videos)} Videos ===")
            print(f"Download videos: {'Yes' if DOWNLOAD_VIDEOS else 'No (captions only)'}")
            for i, video in enumerate(videos, 1):
                print(f"\nProcessing video {i}/{len(videos)}")
                try:
                    result = processor.process_video(downloader, analyzer, video, download_video=DOWNLOAD_VIDEOS)
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Error processing video {i}: {str(e)}")
                    continue
        else:
            print("No videos found in date range")
                
    except Exception as e:
        print(f"Error processing channel videos: {str(e)}")
    
    # Save results
    if results:
        output_file = "data/predictions.json"
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Results saved to {output_file}")
            
            # Evaluate predictions if enabled
            if EVALUATE_PREDICTIONS:
                print("\n=== Evaluating Predictions ===")
                try:
                    eval_results = evaluator.evaluate_predictions(output_file)
                    if eval_results:
                        eval_output_file = "data/evaluation_results.json"
                        evaluator.save_evaluation_results(eval_results, eval_output_file)
                        print(f"✓ Evaluation results saved to {eval_output_file}")
                        print("\nEvaluation Metrics:")
                        print(f"Accuracy: {eval_results['metrics']['accuracy']:.2f}")
                        print(f"Precision: {eval_results['metrics']['precision']:.2f}")
                        print(f"Recall: {eval_results['metrics']['recall']:.2f}")
                        print(f"F1 Score: {eval_results['metrics']['f1']:.2f}")
                except Exception as e:
                    print(f"Error during evaluation: {str(e)}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    # Clean up filtered cache only
    print("\n=== Cleaning Up Filtered Cache ===")
    try:
        # Clean up filtered cache
        cache_dir = "data/cache"
        if os.path.exists(cache_dir):
            for file in os.listdir(cache_dir):
                if file.startswith("filtered_"):
                    os.remove(os.path.join(cache_dir, file))
            print("Cleaned up filtered cache")
            
    except Exception as e:
        print(f"Error during cache cleanup: {str(e)}")

if __name__ == "__main__":
    main()





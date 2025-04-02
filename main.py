'''
This is the main script that will be used to run the program
Use to evaluate kol prediction on the stock market
use TradeBrigade as an example, and SPY performance as a benchmark for last 1 year
'''

import os
import json
import pandas as pd
import datetime
import numpy as np
from video_to_script import VideoToScript
import time
import yfinance as yf
import requests
import shutil
from video_downloader import VideoDownloader
from analysis import VideoAnalyzer
from evaluation import PredictionEvaluator

# Global configuration variables
CHANNEL_URL = "https://www.youtube.com/c/TradeBrigade"  # TradeBrigade's channel
BACKTEST_START_DATE = "2024-01-01"  
BACKTEST_END_DATE = "2024-12-31"    
TEST_MODE = False  # Set to False to process all videos
MAX_VIDEOS = 50  # Number of videos to analyze
USE_PREDOWNLOADED_VIDEOS = False  # Set to True to use pre-downloaded videos for testing
VIDEOS_DIR = "data/videos"  # Directory containing pre-downloaded videos
MODEL = "sonar-reasoning"   # Model to use for analysis
EVALUATE_PREDICTIONS = True  # Whether to evaluate predictions after processing

def main():
    """Main function for processing YouTube videos and analyzing predictions"""
    # Initialize components
    downloader = VideoDownloader()
    analyzer = VideoAnalyzer()
    evaluator = PredictionEvaluator()
    processor = VideoToScript()
    
    results = []
    
    # Process channel videos
    print(f"\n=== Processing Channel Videos ===")
    print(f"Channel: {CHANNEL_URL}")
    print(f"Date range: {BACKTEST_START_DATE} to {BACKTEST_END_DATE}")
    print(f"Maximum videos to process: {MAX_VIDEOS}")
    
    try:
        videos = downloader.get_channel_videos(
                    CHANNEL_URL,
                    backtest_start_date=BACKTEST_START_DATE,
                    backtest_end_date=BACKTEST_END_DATE
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
            for i, video in enumerate(videos, 1):
                print(f"\nProcessing video {i}/{len(videos)}")
                result = processor.process_video(downloader, analyzer, video)
                if result:
                    results.append(result)
        else:
            print("No videos found in date range")
                
    except Exception as e:
        print(f"Error processing channel videos: {str(e)}")
        
        # Save results
        if results:
            output_file = "data/predictions.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Results saved to {output_file}")
            
            # Evaluate predictions if enabled
            if EVALUATE_PREDICTIONS:
                print("\n=== Evaluating Predictions ===")
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
            print(f"Error saving results: {e}")
    
    # Clean up
    print("\n=== Cleaning Up ===")
    try:
        # Clean up downloaded videos and transcripts
        if os.path.exists("data/videos"):
            shutil.rmtree("data/videos")
            os.makedirs("data/videos")
        print("Cleaned up downloaded videos")
        
        # Clean up filtered cache
        cache_dir = "data/cache"
        if os.path.exists(cache_dir):
            for file in os.listdir(cache_dir):
                if file.startswith("filtered_"):
                    os.remove(os.path.join(cache_dir, file))
            print("Cleaned up filtered cache")
            
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    main()










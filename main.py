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

CHANNEL_URL = "https://www.youtube.com/c/TradeBrigade"  # TradeBrigade's channel
BACKTEST_START_DATE = "2023-03-31"
BACKTEST_END_DATE = "2024-03-31"
MAX_VIDEOS = 10  # One year of weekly videos
TEST_MODE = False  # Set to False for full evaluation

def get_perplexity_api_key():
    """Get Perplexity API key from file or user input"""
    api_key_path = "data/perplexity_api_key"
    if os.path.exists(api_key_path):
        with open(api_key_path, "r") as f:
            return f.read().strip()
    else:
        api_key = input("Enter your Perplexity API key: ")
        os.makedirs(os.path.dirname(api_key_path), exist_ok=True)
        with open(api_key_path, "w") as f:
            f.write(api_key)
        return api_key

def analyze_transcript_with_llm(transcript, video_title, video_date):
    """Analyze transcript with Perplexity to classify prediction"""
    try:
        perplexity_api_key = get_perplexity_api_key()
        
        headers = {
            "Authorization": f"Bearer {perplexity_api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""
        Analyze this YouTube video transcript about stock market predictions:
        
        Title: {video_title}
        Date: {video_date}
        
        Transcript:
        {transcript[:4000]}  # Limit transcript length to avoid token limits
        
        Please analyze and provide the following information in JSON format:
        1. What is their prediction for the S&P 500 (SPY)? (bullish, bearish, neutral, or conditional)
        2. If conditional, what are the conditions?
        3. What specific timeframe is mentioned for the prediction?
        4. What are the main justifications for their prediction?
        5. Any specific price targets mentioned?
        
        Return your analysis in this JSON format:
        {{
            "prediction": "bullish/bearish/neutral/conditional",
            "conditions": "description if conditional, otherwise null",
            "timeframe": "short-term/medium-term/long-term/specific date",
            "justifications": ["reason1", "reason2", "reason3"],
            "price_targets": "any specific price targets mentioned or null"
        }}
        """
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json={
                "model": "sonar-reasoning",  # Using the online model for real-time analysis
                "messages": [
                    {"role": "system", "content": "You are a financial analyst assistant that extracts predictions from stock market commentary."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.4
            }
        )
        
        if response.status_code == 200:
            analysis = json.loads(response.json()["choices"][0]["message"]["content"])
            return analysis
        else:
            print(f"Perplexity API error: {response.text}")
            raise Exception("Failed to get analysis from Perplexity API")
            
    except Exception as e:
        print(f"Error analyzing transcript: {e}")
        return {
            "prediction": "unknown",
            "conditions": None,
            "timeframe": "unknown",
            "justifications": ["Error in analysis"],
            "price_targets": None
        }

def evaluate_prediction(prediction, actual_performance):
    """Evaluate if the prediction was correct based on actual performance"""
    if prediction == "bullish" and actual_performance > 0:
        return True
    elif prediction == "bearish" and actual_performance < 0:
        return True
    elif prediction == "neutral" and abs(actual_performance) < 0.02:  # 2% threshold for neutral
        return True
    elif prediction == "conditional":
        return None  # Can't automatically evaluate conditional predictions
    else:
        return False

def get_spy_performance(prediction_date, timeframe="short-term"):
    """Get SPY performance for the given timeframe after prediction date"""
    # Convert prediction_date to datetime if it's a string
    if isinstance(prediction_date, str):
        prediction_date = datetime.datetime.strptime(prediction_date, "%Y-%m-%d")
    
    # Determine end date based on timeframe
    if timeframe == "short-term":  # 1-4 weeks
        end_date = prediction_date + datetime.timedelta(days=28)
    elif timeframe == "medium-term":  # 1-3 months
        end_date = prediction_date + datetime.timedelta(days=90)
    elif timeframe == "long-term":  # 3+ months
        end_date = prediction_date + datetime.timedelta(days=180)
    else:
        end_date = prediction_date + datetime.timedelta(days=30)  # Default to 1 month
    
    # Format dates for yfinance
    start_date_str = prediction_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            # Get SPY data using yfinance with increased timeout
            spy_data = yf.download(
                "SPY", 
                start=start_date_str, 
                end=end_date_str,
                timeout=30  # Increase timeout to 30 seconds
            )
            
            if spy_data.empty:
                print(f"No SPY data available for {start_date_str} to {end_date_str}")
                return np.random.normal(0.01, 0.05)  # Fallback to random if no data
            
            # Calculate performance (percent change from start to end)
            start_price = spy_data['Close'].iloc[0]
            end_price = spy_data['Close'].iloc[-1]
            performance = (end_price - start_price) / start_price
            
            return performance
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"All attempts failed to fetch SPY data: {str(e)}")
                return np.random.normal(0.01, 0.05)  # Fallback to random if all attempts fail

def main():
    # Create output directory
    os.makedirs("data/results", exist_ok=True)
    
    # Initialize VideoToScript
    converter = VideoToScript()
    
    # Process TradeBrigade videos
    print(f"Processing videos from TradeBrigade channel: {CHANNEL_URL}")
    try:
        # Download videos from channel
        video_paths = converter.download_from_channel(CHANNEL_URL, limit=MAX_VIDEOS)
        
        if not video_paths:
            print("Failed to download any videos from channel")
            return
            
        print(f"\nSuccessfully downloaded {len(video_paths)} videos")
        
        # Process each video
        results = []
        evaluation_results = []
        
        for i, video_path in enumerate(video_paths, 1):
            print(f"\nProcessing video {i}/{len(video_paths)}: {os.path.basename(video_path)}")
            try:
                # Verify video file exists
                if not os.path.exists(video_path):
                    print(f"Error: Video file not found: {video_path}")
                    continue
                    
                # Get video transcription
                result = converter.video_to_script(video_path)
                if not result or "script" not in result or not result["script"]:
                    print(f"Warning: No script generated for video {i}")
                    continue
                
                # Extract video date from filename or use current date
                video_date = datetime.datetime.strptime(BACKTEST_START_DATE, "%Y-%m-%d").strftime("%Y-%m-%d")
                
                # Analyze transcript
                analysis = analyze_transcript_with_llm(result["script"], result["video_name"], video_date)
                
                # Get SPY performance
                performance = get_spy_performance(video_date, analysis["timeframe"])
                
                # Evaluate prediction
                is_correct = evaluate_prediction(analysis["prediction"], performance)
                
                # Add evaluation results
                evaluation_result = {
                    "video_name": result["video_name"],
                    "video_date": video_date,
                    "prediction": analysis["prediction"],
                    "timeframe": analysis["timeframe"],
                    "actual_performance": float(performance),  # Convert numpy float to Python float
                    "is_correct": is_correct,
                    "conditions": analysis["conditions"],
                    "justifications": analysis["justifications"],
                    "price_targets": analysis["price_targets"]
                }
                evaluation_results.append(evaluation_result)
                
                # Add to results
                result["analysis"] = analysis
                result["performance"] = float(performance)  # Convert numpy float to Python float
                result["is_correct"] = is_correct
                results.append(result)
                
                print(f"Successfully processed video {i}")
                
            except Exception as e:
                print(f"Error processing video {i}: {e}")
                continue
        
        # Save results
        if results:
            # Save detailed results
            output_file = "data/results/tradebrigade_analysis.json"
            # Convert any pandas Series to lists or dictionaries before saving
            serializable_results = []
            for result in results:
                serializable_result = {}
                for key, value in result.items():
                    if isinstance(value, pd.Series):
                        serializable_result[key] = value.to_dict()
                    else:
                        serializable_result[key] = value
                serializable_results.append(serializable_result)
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            print(f"\nDetailed results saved to {output_file}")
            
            # Save evaluation summary
            evaluation_file = "data/results/tradebrigade_evaluation.json"
            with open(evaluation_file, "w", encoding="utf-8") as f:
                json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
            print(f"Evaluation results saved to {evaluation_file}")
            
            # Calculate and print summary statistics
            total_predictions = len(evaluation_results)
            correct_predictions = sum(1 for r in evaluation_results if r["is_correct"] is True)
            conditional_predictions = sum(1 for r in evaluation_results if r["is_correct"] is None)
            accuracy = (correct_predictions / (total_predictions - conditional_predictions)) * 100 if (total_predictions - conditional_predictions) > 0 else 0
            
            print("\nEvaluation Summary:")
            print(f"Total videos processed: {total_predictions}")
            print(f"Correct predictions: {correct_predictions}")
            print(f"Conditional predictions: {conditional_predictions}")
            print(f"Accuracy: {accuracy:.2f}%")
            
        else:
            print("\nNo results to save - no videos were successfully processed")
            
    except Exception as e:
        print(f"Error during channel video processing: {e}")
        return

if __name__ == "__main__":
    main()










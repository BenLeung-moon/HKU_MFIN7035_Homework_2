import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import time
import re

class VideoAnalyzer:
    def __init__(self, model="sonar-reasoning"):
        """Initialize the VideoAnalyzer class"""
        self.model = model
        self.api_key_path = "data/perplexity_api_key"
    
    def get_perplexity_api_key(self):
        """Get Perplexity API key from file or user input"""
        if os.path.exists(self.api_key_path):
            with open(self.api_key_path, "r") as f:
                return f.read().strip()
        else:
            api_key = input("Enter your Perplexity API key: ")
            os.makedirs(os.path.dirname(self.api_key_path), exist_ok=True)
            with open(self.api_key_path, "w") as f:
                f.write(api_key)
            return api_key
    
    def preprocess_transcript(self, transcript, video_title, video_date):
        """Preprocess transcript to extract only SPY-related content using Perplexity"""
        try:
            perplexity_api_key = self.get_perplexity_api_key()
            
            headers = {
                "Authorization": f"Bearer {perplexity_api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""
            Extract and summarize only the content related to S&P 500 (SPY) from this YouTube video transcript:
            
            Title: {video_title}
            Date: {video_date}
            
            Transcript:
            {transcript[:4000]}  # Limit transcript length to avoid token limits
            
            Please:
            1. Extract only the parts that discuss SPY or S&P 500
            2. Remove any unrelated content
            3. Keep the original context and meaning
            4. Maintain chronological order
            5. Keep any specific numbers, percentages, or price targets mentioned
            
            Return only the relevant content, without any additional formatting or explanations.
            """
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a financial analyst assistant that extracts SPY-related content from transcripts. Be precise and maintain the original meaning."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1  # Lower temperature for more focused extraction
                }
            )
            
            if response.status_code == 200:
                try:
                    content = response.json()["choices"][0]["message"]["content"]
                    # Clean up the response
                    content = content.strip()
                    if content.startswith("```"):
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                    
                    if not content:
                        print("Warning: No SPY-related content found in transcript")
                        return transcript[:1000]  # Return first 1000 chars if no relevant content found
                    
                    return content
                except Exception as e:
                    print(f"Error parsing transcript preprocessing response: {e}")
                    return transcript[:1000]  # Fallback to original transcript
            else:
                print(f"Perplexity API error in preprocessing: {response.text}")
                return transcript[:1000]  # Fallback to original transcript
                
        except Exception as e:
            print(f"Error preprocessing transcript: {e}")
            return transcript[:1000]  # Fallback to original transcript
    
    def analyze_transcript(self, transcript, video_title, video_date):
        """Analyze transcript with Perplexity to classify prediction"""
        try:
            perplexity_api_key = self.get_perplexity_api_key()
            
            headers = {
                "Authorization": f"Bearer {perplexity_api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""
            Analyze this YouTube video transcript about stock market predictions and return ONLY a JSON object with the following structure:
            {{
                "prediction": "bullish/bearish/neutral/conditional",
                "conditions": "description if conditional, otherwise null",
                "timeframe": "short-term/medium-term/long-term/specific date",
                "justifications": ["reason1", "reason2", "reason3"],
                "price_targets": "any specific price targets mentioned or null"
            }}

            Title: {video_title}
            Date: {video_date}
            
            Transcript:
            {transcript[:4000]}
            
            Rules:
            1. Return ONLY the JSON object, no other text
            2. Prediction must be one of: bullish, bearish, neutral, conditional
            3. Timeframe must be one of: short-term, medium-term, long-term, or a specific date
            4. Justifications must be a list of strings
            5. Price targets must be a string or null
            """
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a financial analyst assistant that extracts predictions from stock market commentary. Return only valid JSON with the exact format specified."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2
                }
            )
            
            if response.status_code == 200:
                try:
                    content = response.json()["choices"][0]["message"]["content"].strip()
                    
                    # Clean up the response to ensure valid JSON
                    # Remove any markdown code blocks
                    content = re.sub(r'```json\s*', '', content)
                    content = re.sub(r'```\s*', '', content)
                    
                    # Remove any markdown formatting
                    content = re.sub(r'<[^>]+>', '', content)
                    
                    # Remove any leading/trailing whitespace or newlines
                    content = content.strip()
                    
                    # Try to parse JSON with better error handling
                    try:
                        analysis = json.loads(content)
                    except json.JSONDecodeError as e:
                        print(f"✗ JSON parsing error: {str(e)}")
                        print(f"Raw content: {content[:200]}...")  # Print first 200 chars for debugging
                        return {
                            "prediction": "neutral",
                            "conditions": None,
                            "timeframe": "short-term",
                            "justifications": ["Error parsing JSON response"],
                            "price_targets": None
                        }
                    
                    # Validate and fix prediction
                    valid_predictions = ["bullish", "bearish", "neutral", "conditional"]
                    if analysis.get("prediction") not in valid_predictions:
                        print(f"⚠️ Invalid prediction '{analysis.get('prediction')}', defaulting to neutral")
                        analysis["prediction"] = "neutral"
                    
                    # Validate and fix timeframe
                    valid_timeframes = ["short-term", "medium-term", "long-term"]
                    if not any(tf in analysis.get("timeframe", "").lower() for tf in valid_timeframes):
                        print(f"⚠️ Invalid timeframe '{analysis.get('timeframe')}', defaulting to short-term")
                        analysis["timeframe"] = "short-term"
                    
                    # Ensure justifications is a list
                    if not isinstance(analysis.get("justifications"), list):
                        analysis["justifications"] = ["No specific justifications found"]
                    
                    # Ensure conditions is a string or null
                    if analysis.get("conditions") is not None and not isinstance(analysis["conditions"], str):
                        analysis["conditions"] = str(analysis["conditions"])
                    
                    # Ensure price_targets is a string or null
                    if analysis.get("price_targets") is not None and not isinstance(analysis["price_targets"], str):
                        analysis["price_targets"] = str(analysis["price_targets"])
                    
                    print(f"✓ Analysis complete: {analysis['prediction']} ({analysis['timeframe']})")
                    return analysis
                except Exception as e:
                    print(f"✗ Error processing analysis response: {str(e)}")
                    return {
                        "prediction": "neutral",
                        "conditions": None,
                        "timeframe": "short-term",
                        "justifications": ["Error in analysis processing"],
                        "price_targets": None
                    }
            else:
                print(f"✗ API error: {response.status_code}")
                print(f"Response: {response.text[:200]}...")  # Print first 200 chars of error response
                return {
                    "prediction": "neutral",
                    "conditions": None,
                    "timeframe": "short-term",
                    "justifications": ["API error"],
                    "price_targets": None
                }
                
        except Exception as e:
            print(f"✗ Analysis error: {str(e)}")
            return {
                "prediction": "neutral",
                "conditions": None,
                "timeframe": "short-term",
                "justifications": ["Error in analysis"],
                "price_targets": None
            } 
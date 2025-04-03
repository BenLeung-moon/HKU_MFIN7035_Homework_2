import os
import json
import requests
import re

class VideoAnalyzer:
    def __init__(self, model="sonar-reasoning"):
        """Initialize the VideoAnalyzer class"""
        self.model = model
        self.api_key_path = "data/perplexity_api_key"
        self.use_api = True  
    
    def get_perplexity_api_key(self):
        """Get Perplexity API key from file or user input"""
        if not self.use_api:
            return None
            
        if os.path.exists(self.api_key_path):
            with open(self.api_key_path, "r") as f:
                api_key = f.read().strip()
                if api_key:
                    return api_key
                else:
                    print("API key file exists but is empty. Continuing without API.")
                    self.use_api = False
                    return None
        else:
            try:
                # create directory
                os.makedirs(os.path.dirname(self.api_key_path), exist_ok=True)
                
                # ask user if they want to use API
                use_api_input = input("Do you want to use Perplexity API? (y/n, default: n): ").strip().lower()
                if use_api_input == 'y':
                    api_key = input("Enter your Perplexity API key: ")
                    with open(self.api_key_path, "w") as f:
                        f.write(api_key)
                    return api_key
                else:
                    print("Continuing without API.")
                    with open(self.api_key_path, "w") as f:
                        f.write("")  # create empty file
                    self.use_api = False
                    return None
            except Exception as e:
                print(f"Error getting API key: {e}")
                self.use_api = False
                return None
    
    def preprocess_transcript(self, transcript, video_title, video_date):
        """Preprocess transcript to extract only SPY-related content using Perplexity"""
        try:
            # For debugging purposes, use simpler preprocessing without API dependency
            print(f"Preprocessing transcript for video: {video_title}")
            
            # Simple preprocessing without relying on external API
            if not transcript:
                print("Warning: Empty transcript")
                return ""
                
            # Find SPY or S&P 500 related content using regex
            spy_related_parts = []
            lines = transcript.split('\n')
            for line in lines:
                if re.search(r'\b(S&P\s*500|S&P|SPY|S and P|index|market|stock|ETF)\b', line, re.IGNORECASE):
                    spy_related_parts.append(line)
            
            if spy_related_parts:
                processed_text = ' '.join(spy_related_parts)
                # Limit length
                if len(processed_text) > 4000:
                    processed_text = processed_text[:4000]
                return processed_text
            else:
                print("Warning: No SPY-related content found in transcript")
                return transcript[:1000]  # Return first 1000 chars if no relevant content found
            
            # Try using Perplexity API only if available and working
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
                    },
                    timeout=10  # Add timeout to avoid hanging
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
                        return processed_text  # Fallback to our simple preprocessing
                else:
                    print(f"Perplexity API error in preprocessing: {response.text}")
                    return processed_text  # Fallback to our simple preprocessing
            except Exception as e:
                print(f"Error using Perplexity API for preprocessing: {e}")
                return processed_text  # Fallback to our simple preprocessing
                
        except Exception as e:
            print(f"Error preprocessing transcript: {e}")
            return transcript[:1000]  # Fallback to original transcript
    
    def analyze_transcript(self, transcript, video_title, video_date):
        """Analyze transcript with Perplexity to classify prediction"""
        try:
            print(f"Analyzing transcript for video: {video_title}")
            
            # Fallback analysis without API dependency
            default_result = {
                "prediction": "neutral",
                "conditions": None,
                "timeframe": "short-term",
                "justifications": ["Simple analysis without API"],
                "price_targets": None
            }
            
            # Simple rules-based analysis
            if not transcript:
                print("Warning: Empty transcript for analysis")
                return default_result
                
            # Determine sentiment
            bullish_terms = ["bullish", "uptrend", "buy", "growth", "rally", "positive", "upside", "higher"]
            bearish_terms = ["bearish", "downtrend", "sell", "drop", "decline", "negative", "downside", "lower"]
            
            bullish_count = sum(1 for term in bullish_terms if re.search(rf'\b{term}\b', transcript, re.IGNORECASE))
            bearish_count = sum(1 for term in bearish_terms if re.search(rf'\b{term}\b', transcript, re.IGNORECASE))
            
            prediction = "neutral"
            if bullish_count > bearish_count * 1.5:
                prediction = "bullish"
            elif bearish_count > bullish_count * 1.5:
                prediction = "bearish"
            elif abs(bullish_count - bearish_count) <= 1 and (bullish_count > 0 or bearish_count > 0):
                prediction = "conditional"
            
            # Check for timeframe
            timeframe = "short-term"
            if re.search(r'\b(long[- ]term|year|annual|2025|future)\b', transcript, re.IGNORECASE):
                timeframe = "long-term"
            elif re.search(r'\b(mid[- ]term|medium[- ]term|month|quarter|Q[1-4])\b', transcript, re.IGNORECASE):
                timeframe = "medium-term"
            
            # Extract justifications
            justifications = []
            if prediction == "bullish":
                justifications = ["Found bullish sentiment in transcript"]
            elif prediction == "bearish":
                justifications = ["Found bearish sentiment in transcript"]
            elif prediction == "conditional":
                justifications = ["Found mixed sentiment in transcript"]
            else:
                justifications = ["No strong sentiment detected"]
            
            # Extract price targets
            price_target = None
            price_match = re.search(r'(SPY|S&P).{1,30}?(\$\d+|\d+\s*dollars|\d+\s*points)', transcript, re.IGNORECASE)
            if price_match:
                price_target = price_match.group(0)
            
            result = {
                "prediction": prediction,
                "conditions": "Market conditions apply" if prediction == "conditional" else None,
                "timeframe": timeframe,
                "justifications": justifications,
                "price_targets": price_target
            }
            
            # Try using Perplexity API only if available and working
            try:
                perplexity_api_key = self.get_perplexity_api_key()
                
                headers = {
                    "Authorization": f"Bearer {perplexity_api_key}",
                    "Content-Type": "application/json"
                }
                
                prompt = f"""
                Analyze this YouTube video transcript about stock market predictions.
                
                You MUST ONLY return a valid JSON object with exactly this structure, and NO other text or explanations:
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
                1. The output MUST be ONLY the JSON object, with NO other text before or after
                2. Prediction must be one of: bullish, bearish, neutral, conditional
                3. Timeframe must be one of: short-term, medium-term, long-term, or a specific date
                4. Justifications must be a list of strings
                5. Price targets must be a string or null
                6. The JSON must be properly formatted with double quotes for keys and string values
                """
                
                response = requests.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers=headers,
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are a JSON-only financial analyst assistant. Your ONLY job is to extract financial predictions from transcripts and return EXCLUSIVELY valid, properly formatted JSON with no other text or markdown formatting."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1
                    },
                    timeout=10  # Add timeout to avoid hanging
                )
                
                if response.status_code == 200:
                    try:
                        content = response.json()["choices"][0]["message"]["content"].strip()
                        
                        # Extract JSON if it's embedded in other text
                        json_match = re.search(r'({[\s\S]*})', content)
                        if json_match:
                            content = json_match.group(1)
                        
                        # Clean up the response to ensure valid JSON
                        # Remove any markdown code blocks
                        content = re.sub(r'```json\s*', '', content)
                        content = re.sub(r'```\s*', '', content)
                        
                        # Remove any markdown formatting
                        content = re.sub(r'<[^>]+>', '', content)
                        
                        # Remove any leading/trailing whitespace or newlines
                        content = content.strip()
                        
                        # Check if content is empty or invalid
                        if not content:
                            print("Empty response from API, using fallback analysis")
                            return result
                            
                        # Try to parse JSON with better error handling
                        try:
                            # Try to fix common JSON formatting errors
                            # Replace single quotes with double quotes
                            fixed_content = content.replace("'", '"')
                            # Fix unquoted keys
                            fixed_content = re.sub(r'([{,])\s*(\w+):', r'\1"\2":', fixed_content)
                            # Fix null vs None
                            fixed_content = fixed_content.replace(": None", ": null")
                            fixed_content = fixed_content.replace(":None", ":null")
                            
                            try:
                                analysis = json.loads(fixed_content)
                            except json.JSONDecodeError:
                                # If still fails, try the original content
                                analysis = json.loads(content)
                            
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
                        except json.JSONDecodeError as e:
                            print(f"✗ JSON parsing error: {str(e)}")
                            print(f"Raw content: {content[:200]}...")  # Print first 200 chars for debugging
                            return result  # Return fallback result
                    except Exception as e:
                        print(f"✗ Error processing analysis response: {str(e)}")
                        return result  # Return fallback result
                else:
                    print(f"✗ API error: {response.status_code}")
                    print(f"Response: {response.text[:200]}...")  # Print first 200 chars of error response
                    return result  # Return fallback result
            except Exception as e:
                print(f"✗ Error with Perplexity API analysis: {str(e)}")
                return result  # Return fallback result
                
        except Exception as e:
            print(f"✗ Analysis error: {str(e)}")
            return {
                "prediction": "neutral",
                "conditions": None,
                "timeframe": "short-term",
                "justifications": ["Error in analysis"],
                "price_targets": None
            } 
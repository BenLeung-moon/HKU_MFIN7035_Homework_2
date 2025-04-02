import os
import json
import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

class PredictionEvaluator:
    def __init__(self, output_dir="data/evaluation"):
        """Initialize the PredictionEvaluator class"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualization directory
        self.viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
    
    def get_spy_data(self, start_date, end_date):
        """Get SPY price data from Yahoo Finance"""
        try:
            spy = yf.Ticker("SPY")
            df = spy.history(start=start_date, end=end_date)
            return df
        except Exception as e:
            print(f"Error fetching SPY data: {e}")
            return None
    
    def evaluate_prediction(self, prediction, video_date, timeframe):
        """Evaluate a single prediction against actual SPY performance"""
        try:
            # Convert video date to datetime
            video_date = pd.to_datetime(video_date)
            
            # Determine evaluation period based on timeframe
            if timeframe == "short-term":
                eval_days = 5
            elif timeframe == "medium-term":
                eval_days = 20
            elif timeframe == "long-term":
                eval_days = 60
            else:
                # Try to parse specific date
                try:
                    target_date = pd.to_datetime(timeframe)
                    eval_days = (target_date - video_date).days
                except:
                    print(f"Invalid timeframe '{timeframe}', defaulting to short-term")
                    eval_days = 5
            
            # Get SPY data for evaluation period
            end_date = video_date + datetime.timedelta(days=eval_days)
            df = self.get_spy_data(video_date, end_date)
            
            if df is None or df.empty:
                print(f"No SPY data available for evaluation period")
                return None
            
            # Calculate actual performance
            start_price = df['Close'].iloc[0]
            end_price = df['Close'].iloc[-1]
            actual_return = (end_price - start_price) / start_price * 100
            
            # Determine actual direction
            if actual_return > 1.0:  # More than 1% gain
                actual_direction = "bullish"
            elif actual_return < -1.0:  # More than 1% loss
                actual_direction = "bearish"
            else:
                actual_direction = "neutral"
            
            # Handle conditional predictions
            if prediction == "conditional":
                prediction = "neutral"  # Treat conditional as neutral for evaluation
            
            # Compare prediction with actual
            prediction_correct = prediction == actual_direction
            
            return {
                "video_date": video_date.strftime("%Y-%m-%d"),
                "prediction": prediction,
                "actual_direction": actual_direction,
                "actual_return": actual_return,
                "prediction_correct": prediction_correct,
                "timeframe": timeframe,
                "eval_days": eval_days
            }
            
        except Exception as e:
            print(f"Error evaluating prediction: {e}")
            return None
    
    def evaluate_predictions(self, predictions_file):
        """Evaluate all predictions in the file"""
        try:
            # Load predictions
            with open(predictions_file, "r", encoding="utf-8") as f:
                predictions = json.load(f)
            
            results = []
            for pred in predictions:
                # Extract prediction from analysis
                if isinstance(pred, dict) and "analysis" in pred:
                    prediction = pred["analysis"].get("prediction", "neutral")
                    timeframe = pred["analysis"].get("timeframe", "short-term")
                    video_date = pred.get("date")
                else:
                    print(f"Invalid prediction format: {pred}")
                    continue
                
                if not video_date:
                    print(f"Missing video date in prediction: {pred}")
                    continue
                
                eval_result = self.evaluate_prediction(
                    prediction,
                    video_date,
                    timeframe
                )
                if eval_result:
                    results.append(eval_result)
            
            if not results:
                print("No valid predictions to evaluate")
                return None
            
            # Convert results to DataFrame
            df = pd.DataFrame(results)
            
            # Calculate metrics
            metrics = self.calculate_metrics(df)
            
            # Generate visualizations
            self.generate_visualizations(df, metrics)
            
            return {
                "results": results,
                "metrics": metrics
            }
            
        except Exception as e:
            print(f"Error evaluating predictions: {e}")
            return None
    
    def calculate_metrics(self, df):
        """Calculate evaluation metrics"""
        try:
            # Encode labels
            le = LabelEncoder()
            
            # Convert predictions to evaluation format (treat conditional as neutral)
            df['eval_prediction'] = df['prediction'].apply(lambda x: "neutral" if x == "conditional" else x)
            
            # Fit and transform labels
            y_true = le.fit_transform(df['actual_direction'])
            y_pred = le.transform(df['eval_prediction'])
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='weighted'),
                "recall": recall_score(y_true, y_pred, average='weighted'),
                "f1": f1_score(y_true, y_pred, average='weighted'),
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
                "labels": le.classes_.tolist()
            }
            
            # Add prediction counts
            metrics["prediction_counts"] = df['prediction'].value_counts().to_dict()
            metrics["actual_counts"] = df['actual_direction'].value_counts().to_dict()
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return None
    
    def generate_visualizations(self, df, metrics):
        """Generate evaluation visualizations"""
        try:
            # 1. Prediction vs Actual Direction
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df, x='prediction', hue='actual_direction')
            plt.title('Prediction vs Actual Direction')
            plt.savefig(os.path.join(self.viz_dir, 'prediction_vs_actual.png'))
            plt.close()
            
            # 2. Prediction Accuracy by Timeframe
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x='timeframe', y='prediction_correct')
            plt.title('Prediction Accuracy by Timeframe')
            plt.savefig(os.path.join(self.viz_dir, 'accuracy_by_timeframe.png'))
            plt.close()
            
            # 3. Actual Returns Distribution
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x='prediction', y='actual_return')
            plt.title('Actual Returns by Prediction')
            plt.savefig(os.path.join(self.viz_dir, 'returns_by_prediction.png'))
            plt.close()
            
            # 4. Confusion Matrix
            plt.figure(figsize=(8, 6))
            cm = np.array(metrics['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=metrics['labels'],
                       yticklabels=metrics['labels'])
            plt.title('Confusion Matrix')
            plt.savefig(os.path.join(self.viz_dir, 'confusion_matrix.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
    
    def save_evaluation_results(self, results, output_file):
        """Save evaluation results to file"""
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Evaluation results saved to {output_file}")
        except Exception as e:
            print(f"Error saving evaluation results: {e}") 
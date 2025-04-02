# YouTube Video Analysis Tool

This tool downloads YouTube videos, transcribes them, and analyzes market predictions from financial channels.

## Features

- Download YouTube videos with automatic quality selection
- Transcribe videos using Whisper (GPU-accelerated) or Google Speech Recognition
- Analyze transcripts for market predictions
- Evaluate predictions against actual market performance
- Generate visualizations and metrics

## Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system
- GPU (optional, but recommended for faster transcription)

### Installing FFmpeg

#### Windows
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract the archive
3. Add FFmpeg's bin directory to your system PATH

#### Linux
```bash
sudo apt update
sudo apt install ffmpeg
```

#### macOS
```bash
brew install ffmpeg
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Process a Single Video
```bash
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Process Channel Videos
```bash
python main.py --channel "https://www.youtube.com/c/CHANNEL_NAME" --days 30
```

### Evaluate Predictions
Add the `--evaluate` flag to evaluate predictions:
```bash
python main.py --channel "https://www.youtube.com/c/CHANNEL_NAME" --days 30 --evaluate
```

## Project Structure

- `main.py`: Main script for running the analysis
- `video_downloader.py`: Handles YouTube video downloads
- `video_to_script.py`: Converts videos to transcripts
- `analysis.py`: Analyzes transcripts for market predictions
- `evaluation.py`: Evaluates predictions against market data
- `test_downloader.py`: Tests for video downloader functionality

## Output

- Downloaded videos are saved in `data/videos/`
- Transcripts are saved in `data/transcripts/`
- Analysis results are saved in `data/predictions.json`
- Evaluation results are saved in `data/evaluation_results.json`
- Visualizations are saved in `data/visualizations/`

## Error Handling

The tool includes comprehensive error handling for:
- Video download failures
- Transcription errors
- API rate limits
- Cache invalidation
- File system operations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
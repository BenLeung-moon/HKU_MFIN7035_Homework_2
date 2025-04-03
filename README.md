# YouTube Video Analysis Tool

This tool downloads YouTube videos, transcribes them, and analyzes market predictions from financial channels.

## Features

- Download YouTube videos with automatic quality selection
- Transcribe videos using Whisper (GPU-accelerated) or Google Speech Recognition
- Analyze transcripts for market predictions
- Evaluate predictions against actual market performance
- Generate visualizations and metrics
- Control how many videos to analyze per week
- Option to download only captions without video files for faster analysis

## System Requirements

- Python 3.9 or higher
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
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Use the provided shell script to quickly launch the analysis:
```bash
chmod +x run.sh  # Ensure the script is executable
./run.sh
```

## Usage

### Configuration

Modify the following configuration parameters in `main.py`:
- `CHANNEL_URL`: YouTube channel URL to analyze
- `BACKTEST_START_DATE` and `BACKTEST_END_DATE`: Date range for backtesting
- `MAX_VIDEOS`: Maximum number of videos to analyze
- `VIDEOS_PER_WEEK`: Maximum number of videos to analyze per week (e.g., 1 or 2)
- `DOWNLOAD_VIDEOS`: Whether to download actual video files (False = captions only)
- `USE_PREDOWNLOADED_VIDEOS`: Whether to use pre-downloaded videos
- `MODEL`: Model to use for analysis

### Process Videos

After modifying the configuration in the code, run:
```bash
python main.py
```

### Optimizing Performance

For faster analysis, especially during testing or development:

1. Set `DOWNLOAD_VIDEOS = False` to only download captions without video files
2. Set `VIDEOS_PER_WEEK = 1` to limit analysis to 1 video per week
3. Adjust `MAX_VIDEOS` to control the total number of videos processed

This configuration significantly speeds up processing while still providing meaningful results.

## Project Structure

- `main.py`: Main script for running the analysis
- `video_downloader.py`: Handles YouTube video downloads
- `video_to_script.py`: Converts videos to transcripts
- `analysis.py`: Analyzes transcripts for market predictions
- `evaluation.py`: Evaluates predictions against market data
- `setup_gpu.py`: GPU setup utility (optional, not necessary to push to git)
- `setup_ffmpeg.py`: FFmpeg setup utility (optional, not necessary to push to git)
- `run.sh`: Quick start script

### Files Generated During Execution

The following directories and files are created during execution and should typically be excluded from git:
- `data/videos/`: Downloaded video files 
- `data/transcripts/`: Generated transcripts
- `data/cache/`: Cached YouTube API responses
- `data/perplexity_api_key`: API key file (should never be committed to git)
- `models/`: Downloaded AI models
- `.ipynb_checkpoints/`: Jupyter notebook checkpoints

Consider adding these paths to your `.gitignore` file.

## Output

- Downloaded videos are saved in `data/videos/`
- Analysis results are saved in `data/predictions.json`
- Evaluation results are saved in `data/evaluation_results.json`

## Dependencies

Main dependencies include:
- pytubefix: YouTube video downloading
- pandas and numpy: Data processing
- matplotlib and seaborn: Data visualization
- scikit-learn: Prediction evaluation
- yfinance: Financial market data
- faster-whisper: Speech-to-text conversion
- torch: Deep learning framework (for Whisper model)
- SpeechRecognition and pydub: Audio processing

## Troubleshooting

### Common Issues
1. **Video download failures**: Check network connection and YouTube URL validity
2. **Model loading errors**: Ensure correct GPU drivers are installed
3. **Transcription errors**: Check if FFmpeg is correctly installed

### Virtual Environment Setup Issues
If you encounter Python path or virtual environment issues, ensure:
1. Use the `run.sh` script to run the program
2. Or directly use `./venv/bin/python main.py` to specify the Python interpreter in the virtual environment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

# YouTube视频分析工具

该工具用于下载YouTube视频，对其进行转录，并分析金融频道的市场预测。

## 功能特点

- 自动下载YouTube视频并选择最佳质量
- 使用Whisper(GPU加速)或Google语音识别进行视频转录
- 分析转录文本中的市场预测
- 将预测结果与实际市场表现进行对比评估
- 生成可视化图表和指标报告
- 控制每周分析的视频数量
- 选项仅下载字幕而不下载视频文件以加快分析速度

## 系统要求

- Python 3.9或更高版本
- FFmpeg（用于音频处理）
- GPU（可选，但推荐用于更快的转录速度）

### 安装FFmpeg

#### Windows
1. 从 https://ffmpeg.org/download.html 下载FFmpeg
2. 解压压缩包
3. 将FFmpeg的bin目录添加到系统PATH中

#### Linux
```bash
sudo apt update
sudo apt install ffmpeg
```

#### macOS
```bash
brew install ffmpeg
```

## 安装步骤

1. 克隆仓库:
```bash
git clone <repository-url>
cd <repository-name>
```

2. 创建虚拟环境（推荐）:
```bash
python3 -m venv venv
source venv/bin/activate  # Windows上使用: venv\Scripts\activate
```

3. 安装依赖:
```bash
pip install -r requirements.txt
```

## 快速启动

使用提供的shell脚本快速启动分析:
```bash
chmod +x run.sh  # 确保脚本可执行
./run.sh
```

## 使用方法

### 修改配置

在`main.py`中修改以下配置参数：
- `CHANNEL_URL`: 要分析的YouTube频道URL
- `BACKTEST_START_DATE`和`BACKTEST_END_DATE`: 回测日期范围
- `MAX_VIDEOS`: 需要分析的最大视频数量
- `VIDEOS_PER_WEEK`: 每周需要分析的视频数量（例如，1或2）
- `DOWNLOAD_VIDEOS`: 是否下载实际的视频文件（False = 仅下载字幕）
- `USE_PREDOWNLOADED_VIDEOS`: 是否使用预下载的视频
- `MODEL`: 用于分析的模型

### 处理视频
在代码中修改配置后运行:
```bash
python main.py
```

### 优化性能

为了更快地进行分析，特别是在测试或开发期间：

1. 设置 `DOWNLOAD_VIDEOS = False` 仅下载字幕而不下载视频文件
2. 设置 `VIDEOS_PER_WEEK = 1` 限制分析仅限于每周1个视频
3. 调整 `MAX_VIDEOS` 控制处理的视频总数

此配置显著加快了处理速度，同时仍提供有意义的结果。

## 项目结构

- `main.py`: 运行分析的主脚本
- `video_downloader.py`: 处理YouTube视频下载
- `video_to_script.py`: 将视频转换为文本
- `analysis.py`: 分析转录文本中的市场预测
- `evaluation.py`: 评估预测与市场数据的对比
- `setup_gpu.py`: GPU设置工具（可选，不需要推送到git）
- `setup_ffmpeg.py`: FFmpeg设置工具（可选，不需要推送到git）
- `run.sh`: 快速启动脚本

### 执行过程中生成的文件

以下目录和文件在执行过程中创建，通常应该从git中排除：
- `data/videos/`: 下载的视频文件
- `data/transcripts/`: 生成的转录文本
- `data/cache/`: 缓存的YouTube API响应
- `data/perplexity_api_key`: API密钥文件（绝不应提交到git）
- `models/`: 下载的AI模型
- `.ipynb_checkpoints/`: Jupyter笔记本检查点

建议将这些路径添加到您的`.gitignore`文件中。

## 输出内容

- 下载的视频保存在 `data/videos/`
- 分析结果保存在 `data/predictions.json`
- 评估结果保存在 `data/evaluation_results.json`

## 依赖清单

主要依赖包括：
- pytubefix: YouTube视频下载
- pandas 和 numpy: 数据处理
- matplotlib 和 seaborn: 数据可视化
- scikit-learn: 预测评估
- yfinance: 获取金融市场数据
- faster-whisper: 语音转文字
- torch: 深度学习框架（用于Whisper模型）
- SpeechRecognition 和 pydub: 音频处理

## 故障排除

### 常见问题
1. **视频下载失败**: 检查网络连接和YouTube URL有效性
2. **模型加载错误**: 确保安装了正确的GPU驱动程序
3. **转录错误**: 检查FFmpeg是否正确安装

### 虚拟环境设置问题
如果遇到Python路径或虚拟环境问题，请确保:
1. 使用`run.sh`脚本运行程序
2. 或直接使用`./venv/bin/python main.py`指定虚拟环境中的Python解释器

## 贡献

1. Fork 仓库
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用MIT许可证 - 有关详细信息，请参阅LICENSE文件。
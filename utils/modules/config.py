"""
配置模块
包含系统配置和常量
"""
import os
import torch
from pathlib import Path
import logging

# 项目根目录
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# 资源目录
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
VIDEOS_DIR = os.path.join(ASSETS_DIR, "videos")
SUBTITLES_DIR = os.path.join(ASSETS_DIR, "subtitles")
AUDIO_DIR = os.path.join(ASSETS_DIR, "audio")
TMP_DIR = os.path.join(ROOT_DIR, "tmp")

# 头像目录
AVATAR_DIR = os.path.join(ROOT_DIR, "results", "avatars")  # 修正为相对路径
DEFAULT_AVATAR = "avator_10"  # 默认头像ID

# 确保所有目录存在
for dir_path in [ASSETS_DIR, VIDEOS_DIR, SUBTITLES_DIR, AUDIO_DIR, TMP_DIR]:
    os.makedirs(dir_path, exist_ok=True)
    
# 尝试创建头像目录，如果权限允许的话
try:
    if not os.path.exists(AVATAR_DIR):
        os.makedirs(AVATAR_DIR, exist_ok=True)
        logging.info(f"创建头像目录: {AVATAR_DIR}")
except Exception as e:
    logging.warning(f"无法创建头像目录: {AVATAR_DIR}, 错误: {str(e)}")

# 服务配置
DEFAULT_TTS_SERVER = "http://192.168.202.10:9880"
REF_AUDIO_FILE = "/mnt/part2/Dteacher/LiveTalking/models/tts/w1.wav"
REF_TEXT = "在公园的小径上，我静静地观察着四周的一切。"

# 检查参考音频文件是否存在，如果不存在，使用备选路径
if not os.path.exists(REF_AUDIO_FILE):
    logging.warning(f"参考音频文件不存在: {REF_AUDIO_FILE}")
    
    # 尝试查找项目目录下的备选参考音频
    alt_ref_paths = [
        os.path.join(ROOT_DIR, "MuseTalk", "models", "tts", "ref.wav"),
        os.path.join(ROOT_DIR, "models", "tts", "ref.wav"),
        os.path.join(ROOT_DIR, "assets", "audio", "ref.wav")
    ]
    
    for path in alt_ref_paths:
        if os.path.exists(path):
            logging.info(f"使用备选参考音频: {path}")
            REF_AUDIO_FILE = path
            break

# GPU配置
GPU_AVAILABLE = torch.cuda.is_available()
GPU_COUNT = torch.cuda.device_count() if GPU_AVAILABLE else 0

# 初始化GPU信息列表
GPU_INFO = []
if GPU_AVAILABLE:
    for i in range(GPU_COUNT):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        GPU_INFO.append({
            "id": i,
            "name": gpu_name,
            "memory": f"{gpu_memory:.2f}GB"
        })

# 日志配置
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(ROOT_DIR, "service.log")

# 视频配置
DEFAULT_VIDEO_WIDTH = 1280
DEFAULT_VIDEO_HEIGHT = 720
DEFAULT_VIDEO_FPS = 30

# 音频配置
DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_AUDIO_CHANNELS = 1

# 子标题配置
SUBTITLE_FORMATS = ["vtt", "srt"]
DEFAULT_SUBTITLE_FORMAT = "vtt"

# 隐形水印配置
WATERMARK_TEXT = "MuseTalk数字人"
WATERMARK_OPACITY = 0.1

# 内存限制配置
MAX_QUEUE_SIZE = 100  # 最大队列长度
MAX_CONCURRENT_TASKS = 5  # 最大并发任务数
MAX_TASK_HISTORY = 50  # 保存多少条任务历史
MAX_FILE_CACHE_SIZE_MB = 1024  # 最大文件缓存(MB)

# 环境变量覆盖配置
def update_from_env():
    """从环境变量更新配置"""
    global DEFAULT_TTS_SERVER, LOG_LEVEL
    
    if os.environ.get("TTS_SERVER"):
        DEFAULT_TTS_SERVER = os.environ.get("TTS_SERVER")
    
    if os.environ.get("LOG_LEVEL"):
        LOG_LEVEL = os.environ.get("LOG_LEVEL")
        
    # 可以添加更多环境变量覆盖

# 初始化时从环境变量更新
update_from_env()

def get_config():
    """获取所有配置信息"""
    return {
        "gpu": {
            "available": GPU_AVAILABLE,
            "count": GPU_COUNT,
            "info": GPU_INFO
        },
        "paths": {
            "root": ROOT_DIR,
            "assets": ASSETS_DIR,
            "videos": VIDEOS_DIR,
            "subtitles": SUBTITLES_DIR,
            "audio": AUDIO_DIR,
            "tmp": TMP_DIR
        },
        "server": {
            "tts_server": DEFAULT_TTS_SERVER,
            "ref_audio": REF_AUDIO_FILE,
            "ref_text": REF_TEXT
        },
        "video": {
            "width": DEFAULT_VIDEO_WIDTH,
            "height": DEFAULT_VIDEO_HEIGHT,
            "fps": DEFAULT_VIDEO_FPS
        },
        "audio": {
            "sample_rate": DEFAULT_AUDIO_SAMPLE_RATE,
            "channels": DEFAULT_AUDIO_CHANNELS
        },
        "subtitle": {
            "formats": SUBTITLE_FORMATS,
            "default_format": DEFAULT_SUBTITLE_FORMAT
        },
        "limits": {
            "max_queue_size": MAX_QUEUE_SIZE,
            "max_concurrent_tasks": MAX_CONCURRENT_TASKS,
            "max_task_history": MAX_TASK_HISTORY,
            "max_file_cache_size_mb": MAX_FILE_CACHE_SIZE_MB
        }
    } 

import os

# Base application paths
app_root = os.path.dirname(os.path.abspath(__file__))

# Directory paths
output_video_dir = "/mnt/part2/Dteacher/digital-teacher-frontend/resources/videos"
subtitle_dir = "/mnt/part2/Dteacher/digital-teacher-frontend/resources/subtitles"
temp_dir = os.path.join(app_root, "temp")
avatar_dir = "/mnt/part2/Dteacher/MuseTalk/results/v15/avatars"
static_dir = os.path.join(app_root, "static")
template_dir = os.path.join(app_root, "templates")

# GPT-SoVITS TTS configuration
TTS_SERVER = "http://192.168.202.10:9880"

# 新增：TTS 模型配置 (注意：这些通常在 TTS 服务自己的配置文件中管理)
TTS_MODELS = {
    "wang001": {
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "device": "cuda",
        "is_half": True,
        "t2s_weights_path": "/mnt/part2/Dteacher/GPT-SoVITS/GPT_weights_v2/wang005-e15.ckpt", # 示例路径
        "version": "v2",
        "vits_weights_path": "/mnt/part2/Dteacher/GPT-SoVITS/SoVITS_weights_v2/wang005_e8_s328.pth", # 示例路径
        "speaker_name": "王老师（标准）",
        "ref_audio_path": "/mnt/part2/Dteacher/ttt/data/w001.wav",
        "prompt_text": "那个人的行为完全超出了我的忍耐底线",
        "prompt_lang": "zh"
    },
    "5zu": {
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "device": "cuda",
        "is_half": True,
        "t2s_weights_path": "/mnt/part2/Dteacher/GPT-SoVITS/GPT_weights_v2/5zu-e15.ckpt",
        "version": "v2",
        "vits_weights_path": "/mnt/part2/Dteacher/GPT-SoVITS/SoVITS_weights_v2/5zu_e8_s400.pth",
        "speaker_name": "5zu（黄雅雯）",
        "ref_audio_path": "/mnt/part2/Dteacher/ttt/data/audio/5zu.wav",
        "prompt_text": "我的作品被选参加展览，这是一种莫大的肯定",
        "prompt_lang": "zh"
    },
    "6zu": {
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "device": "cuda",
        "is_half": True,
        "t2s_weights_path": "/mnt/part2/Dteacher/GPT-SoVITS/GPT_weights_v2/6zu-e15.ckpt",
        "version": "v2",
        "vits_weights_path": "/mnt/part2/Dteacher/GPT-SoVITS/SoVITS_weights_v2/6zu_e8_s304.pth",
        "speaker_name": "6zu（熊诗宇）",
        "ref_audio_path": "/mnt/part2/Dteacher/ttt/data/audio/6zu.wav",
        "prompt_text": "我用积分换到了一张免费的电影票，太划算了",
        "prompt_lang": "zh"
    }
}

# Audio parameters
AUDIO_SAMPLE_RATE = 32000  # TTS服务返回的音频采样率 (需要确认新API是否固定此值)
AUDIO_CHANNELS = 1         # 单声道
AUDIO_BIT_DEPTH = 16      # 16位PCM (需要确认新API输出格式)

# Default parameters
DEFAULT_TEXT_CUT_METHOD = "cut2"
DEFAULT_FPS = 25
DEFAULT_SUBTITLE_FORMAT = "vtt"
DEFAULT_SUBTITLE_OFFSET = -0.3
DEFAULT_SENTENCE_GAP = 0.5
DEFAULT_CN_SPEAKING_RATE = 4.0  # Chinese characters per second
DEFAULT_EN_WORD_RATE = 3.5      # English words per second

# System parameters
MAX_TASK_AGE = 3600  # Maximum age of completed tasks in seconds (1 hour)
CLEANUP_INTERVAL = 1800  # Task cleanup interval in seconds (30 minutes)
MAX_CACHE_SIZE = 3  # Maximum number of Avatar models to cache per GPU
GPU_IDLE_TIMEOUT = 300  # Seconds of idle time before releasing GPU resources (5 minutes)

# Ensure required directories exist
for directory in [output_video_dir, subtitle_dir, temp_dir, static_dir]:
    os.makedirs(directory, exist_ok=True)
os.makedirs(avatar_dir, exist_ok=True)

# CORS settings
CORS_ORIGINS = ["*"]  # Allow all origins for API access

# Debug mode
DEBUG = os.environ.get("DEBUG", "False").lower() in ["true", "1", "yes"]

def _generate_advanced(self, **kwargs):
    """替换模拟实现，调用真实的MuseTalk功能"""
    # 使用与realtime_inference.py类似的方式加载模型和处理音频
    # 调用Avatar类进行视频生成

"""
MuseTalk Wrapper - 数字人视频生成接口
具有多级降级能力和依赖隔离的MuseTalk封装库
"""

__version__ = "1.0.0"
__author__ = "MuseTalk Team"

from .core.avatar_generator import AvatarGenerator, CapabilityLevel
from .core.dependency_manager import DependencyManager
from .core.fallback_manager import FallbackManager, GenerationMode 
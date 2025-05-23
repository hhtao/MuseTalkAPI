"""
核心模块，包含视频生成、依赖管理和降级策略的实现
"""

from .avatar_generator import AvatarGenerator, CapabilityLevel
from .dependency_manager import DependencyManager
from .fallback_manager import FallbackManager, GenerationMode 

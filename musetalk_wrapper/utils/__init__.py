"""
MuseTalk 工具模块包
包含文本处理、字幕生成等工具函数
"""

# 从迁移的模块中导出关键函数
from .text_cut import get_method, get_method_names
from .subtitle_utils import create_subtitle_for_text

__all__ = [
    'get_method', 
    'get_method_names',
    'create_subtitle_for_text'
] 

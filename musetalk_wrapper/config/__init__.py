"""
配置模块，包含默认配置和环境配置
"""

import os
import yaml

def load_config(config_path=None):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认配置
        
    Returns:
        配置字典
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default.yml")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config 
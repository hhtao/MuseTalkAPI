#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MuseTalk 标准模式测试脚本
测试数字人唇动生成功能
"""

import os
import sys
import logging
import argparse

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入核心模块
from musetalk_wrapper.config import load_config
from musetalk_wrapper.core.avatar_generator import AvatarGenerator, GenerationMode

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Test")

def test_standard_mode(text, avatar_id, output_path, tts_model=None):
    """测试标准模式视频生成，包含唇部动画"""
    print("\n======= 测试标准模式视频生成 =======")
    print("文本内容: " + text)
    print("数字人ID: " + avatar_id)
    print("输出路径: " + output_path)
    if tts_model:
        print("TTS模型: " + tts_model)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载配置
        config = load_config()
        print("配置加载成功")
        
        # 初始化生成器
        generator = AvatarGenerator(config=config)
        print("生成器初始化成功")
        
        # 设置模式为标准模式(唇动)
        mode = GenerationMode.STANDARD
        
        # 开始生成
        print("开始生成视频...")
        result = generator.generate_video(
            text=text,
            avatar_id=avatar_id,
            output_path=output_path,
            mode=mode,
            tts_model=tts_model
        )
        
        # 检查结果
        if result.get("success", False):
            print("\n✓ 视频生成成功:")
            print("  路径: " + result.get("path", "未知"))
            print("  时长: " + str(result.get("duration", 0)) + "秒")
            print("  分辨率: " + result.get("resolution", "未知"))
            print("  生成耗时: " + str(round(result.get("generation_time", 0), 2)) + "秒")
            print("  使用模式: " + result.get("mode", "未知"))
            return True
        else:
            print("\n✗ 视频生成失败:")
            print("  错误: " + result.get("error", "未知错误"))
            return False
            
    except Exception as e:
        print("\n✗ 发生异常:")
        print("  " + str(e))
        return False

def main():
    parser = argparse.ArgumentParser(description="MuseTalk标准模式测试工具")
    parser.add_argument("--text", "-t", default="你好，我是数字人。这是一条测试文本，用于验证唇动功能是否正常。", help="测试文本")
    parser.add_argument("--avatar-id", "-a", default="avator_50", help="数字人ID")
    parser.add_argument("--output", "-o", default="./test_standard_output.mp4", help="输出视频路径")
    parser.add_argument("--tts-model", "-m", default=None, help="TTS模型ID")
    args = parser.parse_args()
    
    test_standard_mode(args.text, args.avatar_id, args.output, args.tts_model)

if __name__ == "__main__":
    main() 

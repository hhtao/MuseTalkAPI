#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
独立的视频生成脚本，完全不依赖于utils中的任何模块
直接复制simple_demo.py的核心逻辑到一个类中
"""

import os
import sys
import glob
import uuid
import shutil
import tempfile
import subprocess
import argparse
import logging
import cv2
import numpy as np

# 添加项目根目录到Python路径
MUSETALK_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(MUSETALK_ROOT)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 配置路径 - 与config.py中相同
avatar_dir = os.path.join(MUSETALK_ROOT, "results", "v15", "avatars")
output_video_dir = os.path.join(MUSETALK_ROOT, "outputs")
subtitle_dir = os.path.join(MUSETALK_ROOT, "subtitles")

class VideoGenerator:
    """
    独立的视频生成器类，不依赖于utils中的任何模块
    直接复制simple_demo.py的核心逻辑
    """
    
    def __init__(
        self,
        avatar_id: str,
        audio_path: str,
        output_path: str = "output.mp4",
        gpu_id: int = 0,
        version: str = "v15",
        bbox_shift: int = 0,
        batch_size: int = 10,
        fps: int = 25,
        parsing_mode: str = "jaw"
    ):
        """初始化视频生成器"""
        self.avatar_id = avatar_id
        self.audio_path = audio_path
        self.output_path = output_path
        self.gpu_id = gpu_id
        self.version = version
        self.bbox_shift = bbox_shift
        self.batch_size = batch_size
        self.fps = fps
        self.parsing_mode = parsing_mode
        
        # 处理输出路径
        self.output_name = os.path.splitext(os.path.basename(output_path))[0]
        
        # 检查音频文件是否存在
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"错误: 音频文件不存在: {audio_path}")
            
        # 设置输出目录
        self.avatar_path = os.path.join(avatar_dir, avatar_id)
        self.video_out_path = os.path.join(self.avatar_path, "vid_output")
        os.makedirs(self.video_out_path, exist_ok=True)
        
        # 初始化实例
        self.avatar = None
        self.generated_video = None
    
    def generate(self):
        """生成视频，完全复制simple_demo.py的逻辑"""
        try:
            logging.info("正在初始化模型...")
            
            # 导入必要的模块 - 直接使用原始的方式
            import scripts.realtime_inference as ri
            from scripts.realtime_inference import Avatar
            
            # 初始化模型
            from musetalk.utils.model_manager import ModelManager
            models = ModelManager.get_models(version=self.version, device_id=self.gpu_id)
            
            # 手动设置模型
            ri.fp = models["fp"]
            ri.vae = models["vae"]
            ri.unet = models["unet"]
            ri.pe = models["pe"]
            ri.whisper = models["whisper"]
            ri.device = models["device"]
            ri.timesteps = models["timesteps"]
            ri.weight_dtype = models["weight_dtype"]
            ri.audio_processor = models["audio_processor"]
            
            # 设置参数
            class Args:
                pass
            
            ri_args = Args()
            ri_args.version = self.version
            ri_args.extra_margin = 10
            ri_args.parsing_mode = self.parsing_mode
            ri_args.skip_save_images = False
            ri_args.audio_padding_length_left = 2
            ri_args.audio_padding_length_right = 2
            
            ri.args = ri_args
            
            logging.info(f"正在初始化数字人: {self.avatar_id}")
            
            # 创建数字人实例
            self.avatar = Avatar(
                avatar_id=self.avatar_id,
                video_path=None,
                bbox_shift=self.bbox_shift,
                batch_size=self.batch_size,
                preparation=False
            )
            
            logging.info(f"开始生成视频，使用音频: {self.audio_path}")
            
            # 生成视频
            success = self.avatar.inference(
                audio_path=self.audio_path,
                out_vid_name=self.output_name,
                fps=self.fps,
                skip_save_images=False
            )
            
            # 获取生成的视频路径
            self.generated_video = os.path.join(self.video_out_path, f"{self.output_name}.mp4")
            
            if success and os.path.exists(self.generated_video):
                # 如果指定了完整输出路径，复制到指定位置
                if self.output_path != "output.mp4":
                    os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
                    shutil.copy(self.generated_video, self.output_path)
                    logging.info(f"视频已生成并复制到: {self.output_path}")
                else:
                    logging.info(f"视频已生成: {self.generated_video}")
                return True
            else:
                logging.error("视频生成失败")
                return False
                
        except Exception as e:
            import traceback
            logging.error(f"发生错误: {e}")
            traceback.print_exc()
            return False
    
    def create_simple_video(self):
        """创建简单视频作为备选"""
        try:
            logging.info("尝试创建简单视频作为备选...")
            
            # 获取数字人图像
            img_files = sorted(glob.glob(os.path.join(self.avatar_path, "full_imgs", "*.[jpJP][pnPN]*[gG]")))
            if not img_files:
                logging.info("未找到全尺寸图片，尝试查找任何可用图片...")
                img_files = sorted(glob.glob(os.path.join(self.avatar_path, "**", "*.[jpJP][pnPN]*[gG]"), recursive=True))
                if not img_files:
                    logging.error("未找到任何图片，无法创建简单视频")
                    return False
                
            logging.info(f"使用图片: {img_files[0]}")
            image_path = img_files[0]
            
            # 输出路径
            output_path = os.path.join(self.video_out_path, f"{self.output_name}_simple.mp4")
            
            # 使用更基本的FFmpeg命令
            ffmpeg_cmd = [
                "ffmpeg", "-y", 
                "-loop", "1",  # 循环播放单个图片
                "-i", image_path,
                "-i", self.audio_path,
                "-c:v", "libx264", 
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",  # 以最短的输入长度为准
                output_path
            ]
            
            logging.info(f"执行命令: {' '.join(ffmpeg_cmd)}")
            subprocess.run(ffmpeg_cmd, check=True)
            
            # 检查生成的视频是否有效
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                # 复制到目标位置
                if self.output_path != "output.mp4":
                    os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
                    shutil.copy(output_path, self.output_path)
                    logging.info(f"简单视频已复制到: {self.output_path}")
                
                logging.info(f"成功创建简单视频: {output_path}")
                return True
            else:
                logging.error(f"视频文件创建失败或大小为零: {output_path}")
                return False
                
        except Exception as e:
            logging.error(f"创建简单视频失败: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MuseTalk 数字人视频生成")
    
    # 基本参数
    parser.add_argument("--avatar_id", type=str, default="avator_50",
                        help="数字人ID")
    parser.add_argument("--audio", type=str, required=True,
                        help="音频文件路径")
    parser.add_argument("--output", type=str, default="output.mp4",
                        help="输出视频路径")
    
    # 高级参数
    parser.add_argument("--gpu", type=int, default=0,
                        help="使用的GPU ID")
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"],
                        help="模型版本")
    parser.add_argument("--bbox_shift", type=int, default=5,
                        help="边界框偏移量")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批处理大小")
    parser.add_argument("--fps", type=int, default=25,
                        help="输出视频帧率")
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    try:
        # 创建视频生成器
        generator = VideoGenerator(
            avatar_id=args.avatar_id,
            audio_path=args.audio,
            output_path=args.output,
            gpu_id=args.gpu,
            version=args.version,
            bbox_shift=args.bbox_shift,
            batch_size=args.batch_size,
            fps=args.fps
        )
        
        # 生成视频
        if generator.generate():
            print("视频生成成功")
            return 0
        else:
            # 尝试创建简单视频
            generator.create_simple_video()
            print("视频生成失败，已尝试创建简单视频")
            return 1
            
    except Exception as e:
        import traceback
        print(f"发生错误: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

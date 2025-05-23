#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
非常简单的独立脚本，只生成静态图片+音频的视频
不依赖任何复杂模型或库，只需要标准库、opencv和ffmpeg
"""

import os
import sys
import glob
import argparse
import subprocess
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SimpleVideo")

def find_image(avatar_id):
    """
    查找指定数字人的图片
    
    Args:
        avatar_id: 数字人ID
    
    Returns:
        str: 图片路径，如果找不到则返回None
    """
    # 定义搜索路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [
        os.path.join(base_dir, "results", "v15", "avatars", avatar_id, "full_imgs"),
        os.path.join(base_dir, "results", "v15", "avatars", avatar_id, "vid_output"),
        os.path.join(base_dir, "results", "v15", "avatars", avatar_id),
        os.path.join(base_dir, "results", "avatars", avatar_id)
    ]
    
    # 搜索可能的图片
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        
        # 查找图片
        patterns = ["*.[jpJP][pnPN]*[gG]", "*.png", "*.jpg", "*.jpeg"]
        for pattern in patterns:
            img_files = sorted(glob.glob(os.path.join(search_dir, pattern)))
            if img_files:
                logger.info(f"找到图片: {img_files[0]}")
                return img_files[0]
    
    # 递归搜索
    avatar_dir = os.path.join(base_dir, "results", "v15", "avatars", avatar_id)
    if os.path.exists(avatar_dir):
        for root, dirs, files in os.walk(avatar_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, file)
                    logger.info(f"通过递归查找到图片: {img_path}")
                    return img_path
    
    logger.error(f"找不到数字人'{avatar_id}'的任何图片")
    return None

def create_video(image_path, audio_path, output_path):
    """
    使用图片和音频创建视频
    
    Args:
        image_path: 图片路径
        audio_path: 音频路径
        output_path: 输出视频路径
    
    Returns:
        bool: 是否成功
    """
    if not os.path.exists(image_path):
        logger.error(f"图片不存在: {image_path}")
        return False
    
    if not os.path.exists(audio_path):
        logger.error(f"音频不存在: {audio_path}")
        return False
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 先获取音频时长
    try:
        duration_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
        if duration_result.returncode == 0:
            duration = float(duration_result.stdout.strip())
            logger.info(f"音频时长: {duration}秒")
        else:
            duration = 10  # 默认10秒
            logger.warning(f"无法获取音频时长，使用默认值: {duration}秒")
    except Exception as e:
        duration = 10  # 默认10秒
        logger.warning(f"获取音频时长失败: {e}，使用默认值: {duration}秒")
    
    try:
        # 创建一个临时目录
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        # 方法1: 最基本的命令，使用mpeg4编码器
        cmd1 = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-framerate", "25",
            "-t", str(duration),  # 设置视频时长
            "-i", image_path, 
            "-i", audio_path,
            "-c:v", "mpeg4",
            "-q:v", "1",  # 最高质量
            "-c:a", "copy",  # 直接复制音频，不重新编码
            "-shortest",
            output_path
        ]
        
        logger.info(f"尝试方法1: {' '.join(cmd1)}")
        process = subprocess.run(cmd1, check=False, capture_output=True, text=True)
        
        if process.returncode != 0:
            logger.warning(f"方法1失败: {process.stderr}")
            
            # 方法2: 尝试一个更简单的命令，使用图片和音频直接生成mp4
            output_path_2 = os.path.join(temp_dir, "output2.mp4")
            cmd2 = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", image_path,
                "-i", audio_path,
                "-vcodec", "mjpeg",  # 使用MJPEG编码器
                "-q:v", "0",  # 最高质量
                "-acodec", "copy",  # 直接复制音频
                "-shortest",
                output_path_2
            ]
            
            logger.info(f"尝试方法2: {' '.join(cmd2)}")
            process = subprocess.run(cmd2, check=False, capture_output=True, text=True)
            
            if process.returncode == 0 and os.path.exists(output_path_2) and os.path.getsize(output_path_2) > 0:
                import shutil
                shutil.copy2(output_path_2, output_path)
                logger.info(f"方法2成功，已复制到: {output_path}")
            else:
                logger.warning(f"方法2失败: {process.stderr}")
                
                # 方法3: 使用更基本的命令，无编码指定
                output_path_3 = os.path.join(temp_dir, "output3.mp4")
                cmd3 = [
                    "ffmpeg", "-y",
                    "-loop", "1",
                    "-i", image_path,
                    "-i", audio_path,
                    "-vframes", "1",  # 只使用一帧
                    "-shortest",
                    output_path_3
                ]
                
                logger.info(f"尝试方法3: {' '.join(cmd3)}")
                process = subprocess.run(cmd3, check=False, capture_output=True, text=True)
                
                if process.returncode == 0 and os.path.exists(output_path_3) and os.path.getsize(output_path_3) > 0:
                    import shutil
                    shutil.copy2(output_path_3, output_path)
                    logger.info(f"方法3成功，已复制到: {output_path}")
                else:
                    logger.error(f"所有方法均失败，最后一次错误: {process.stderr}")
                    return False
        
        # 检查生成的视频是否有效
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"视频创建成功: {output_path}")
            
            # 检查视频是否包含视频轨道
            check_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", output_path]
            check_result = subprocess.run(check_cmd, capture_output=True, text=True)
            
            if check_result.returncode == 0 and check_result.stdout.strip():
                logger.info(f"视频轨道编码: {check_result.stdout.strip()}")
            else:
                logger.warning("视频中可能没有视频轨道")
                return False
            
            return True
        else:
            logger.error(f"输出视频不存在或大小为零: {output_path}")
            return False
            
    except Exception as e:
        import traceback
        logger.error(f"创建视频时发生错误: {e}")
        logger.error(traceback.format_exc())
        return False
    finally:
        # 清理临时目录
        if 'temp_dir' in locals():
            import shutil
            shutil.rmtree(temp_dir)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="简单的静态数字人视频生成器")
    parser.add_argument("--avatar_id", type=str, default="avator_50", help="数字人ID")
    parser.add_argument("--audio", type=str, required=True, help="音频文件路径")
    parser.add_argument("--output", type=str, default="output.mp4", help="输出视频路径")
    
    args = parser.parse_args()
    
    # 查找图片
    image_path = find_image(args.avatar_id)
    if not image_path:
        return 1
    
    # 创建视频
    success = create_video(image_path, args.audio, args.output)
    
    if success:
        print(f"视频已成功生成: {args.output}")
        return 0
    else:
        print("视频生成失败")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 

"""
视频处理工具模块
包含视频生成、合成、处理的功能
"""
import os
import uuid
import logging
import tempfile
import subprocess
import traceback
import cv2
import numpy as np
from pathlib import Path

# 导入自定义模块
from ..modules.audio_utils import get_audio_duration

def create_simple_video(audio_path, output_path, fps=30):
    """创建一个简单的视频，支持处理文件路径或字节数据"""
    temp_audio_file = None
    temp_video = None
    
    try:
        logging.info(f"创建简单视频... 输出路径: {output_path}")
        
        # 如果audio_path是字节数据，先保存到临时文件
        if isinstance(audio_path, bytes):
            temp_audio_file = os.path.join(tempfile.gettempdir(), f"temp_audio_{uuid.uuid4()}.wav")
            with open(temp_audio_file, 'wb') as f:
                f.write(audio_path)
            audio_file_for_processing = temp_audio_file
        else:
            audio_file_for_processing = audio_path
            
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 检查音频文件是否存在
        if not os.path.exists(audio_file_for_processing):
            logging.error(f"音频文件不存在: {audio_file_for_processing}")
            return False
            
        logging.info(f"使用音频文件: {audio_file_for_processing}")
        
        # 获取音频时长
        audio_duration = get_audio_duration(audio_file_for_processing)
        if audio_duration <= 0:
            audio_duration = 5.0  # 默认5秒钟
            
        logging.info(f"音频时长: {audio_duration}秒")
            
        # 创建一个黑屏视频
        video_num = int(audio_duration * fps)
        temp_video = os.path.join(tempfile.gettempdir(), f"temp_video_{uuid.uuid4()}.mp4")
        
        logging.info(f"临时视频路径: {temp_video}")
        
        # 确保可执行文件存在
        ffmpeg_exists = False
        try:
            subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True)
            ffmpeg_exists = True
        except Exception as e:
            logging.warning(f"FFMPEG命令不可用，尝试使用OpenCV: {str(e)}")
        
        success = False
        
        # 方法1：使用FFmpeg创建黑屏视频 (如果可用)
        if ffmpeg_exists:
            try:
                logging.info("方法1: 使用FFMPEG创建视频")
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'lavfi', '-i', f'color=c=black:s=1280x720:r={fps}:d={audio_duration}',
                    '-c:v', 'libx264', '-tune', 'stillimage', '-pix_fmt', 'yuv420p',
                    temp_video
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
                # 添加音频
                logging.info("添加音频...")
                cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_video,
                    '-i', audio_file_for_processing,
                    '-c:v', 'copy', '-c:a', 'aac', '-shortest',
                    output_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                logging.info(f"视频生成成功: {output_path}")
                success = True
            except Exception as e:
                logging.error(f"方法1创建视频失败: {str(e)}")
                traceback.print_exc()
        
        # 方法2：创建静态图像并添加音频
        if not success:
            try:
                logging.info("方法2: 使用OpenCV创建视频")
                # 创建一个临时图像
                img_dir = os.path.join(tempfile.gettempdir(), f"img_temp_{uuid.uuid4()}")
                os.makedirs(img_dir, exist_ok=True)
                temp_img = os.path.join(img_dir, "frame.png")
                
                # 创建一个白色图像
                img = np.ones((720, 1280, 3), dtype=np.uint8) * 255
                
                # 添加文本
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, "数字人视频", (500, 360), font, 1.5, (0, 0, 0), 2)
                
                cv2.imwrite(temp_img, img)
                logging.info(f"已创建临时图像: {temp_img}")
                
                if ffmpeg_exists:
                    # 使用ffmpeg合成
                    temp_video = os.path.join(tempfile.gettempdir(), f"temp_vid_{uuid.uuid4()}.mp4")
                    cmd = [
                        'ffmpeg', '-y',
                        '-loop', '1',
                        '-i', temp_img,
                        '-t', str(audio_duration),
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        temp_video
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    
                    # 添加音频
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', temp_video,
                        '-i', audio_file_for_processing,
                        '-c:v', 'copy', '-c:a', 'aac', '-shortest',
                        output_path
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    success = True
                else:
                    # 使用OpenCV创建视频
                    logging.info("使用OpenCV直接生成视频")
                    writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        (1280, 720)
                    )
                    
                    # 写入帧
                    for _ in range(video_num):
                        writer.write(img)
                    
                    writer.release()
                    logging.info(f"视频写入完成: {output_path}")
                    
                    # 无法直接添加音频，但至少生成了视频
                    success = True
                
                # 清理临时文件
                if os.path.exists(temp_img):
                    os.remove(temp_img)
                if os.path.exists(img_dir):
                    import shutil
                    shutil.rmtree(img_dir, ignore_errors=True)
                    
            except Exception as e:
                logging.error(f"方法2创建视频失败: {str(e)}")
                traceback.print_exc()
                
        # 在方法1和2都失败的情况下，尝试一个非常简单的方法
        if not success:
            try:
                logging.info("方法3: 生成非常简单的视频文件")
                
                # 创建一个帧
                frame = np.ones((360, 640, 3), dtype=np.uint8) * 200
                
                # 写入文本
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, "数字人视频", (200, 180), font, 1, (0, 0, 0), 2)
                
                # 创建视频
                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (640, 360)
                )
                
                # 写入多个相同的帧
                frames_to_write = int(fps * 5)  # 至少5秒
                for _ in range(frames_to_write):
                    writer.write(frame)
                
                writer.release()
                logging.info(f"视频文件已创建: {output_path}, 大小: {os.path.getsize(output_path)}")
                success = True
            except Exception as e:
                logging.error(f"方法3创建视频失败: {str(e)}")
                traceback.print_exc()
        
        # 清理临时文件
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
        if temp_video and os.path.exists(temp_video):
            os.remove(temp_video)
            
        return success
    except Exception as e:
        logging.error(f"创建视频失败: {str(e)}")
        traceback.print_exc()
        
        # 清理临时文件
        if temp_audio_file and os.path.exists(temp_audio_file):
            try:
                os.remove(temp_audio_file)
            except:
                pass
        if temp_video and os.path.exists(temp_video):
            try:
                os.remove(temp_video)
            except:
                pass
                
        return False

def merge_videos(video_paths, output_path):
    """合并多个视频文件"""
    if not video_paths:
        logging.error("没有提供视频路径")
        return False
        
    try:
        # 创建一个临时文件列表
        list_file = os.path.join(tempfile.gettempdir(), f"video_list_{uuid.uuid4()}.txt")
        with open(list_file, 'w') as f:
            for video in video_paths:
                if os.path.exists(video):
                    f.write(f"file '{os.path.abspath(video)}'\n")
        
        # 使用ffmpeg合并视频
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_file,
            '-c', 'copy',
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # 清理临时文件
        os.remove(list_file)
        return True
    except Exception as e:
        logging.error(f"合并视频失败: {str(e)}")
        traceback.print_exc()
        return False

def compress_video(input_path, output_path, crf=23):
    """压缩视频文件，crf值越大压缩率越高，画质越低"""
    try:
        cmd = [
            'ffmpeg', '-y', 
            '-i', input_path,
            '-c:v', 'libx264',
            '-crf', str(crf),
            '-preset', 'medium',
            '-c:a', 'aac',
            '-b:a', '128k',
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        logging.error(f"压缩视频失败: {str(e)}")
        return False

def extract_audio(video_path, output_audio_path):
    """从视频中提取音频"""
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn',  # 不要视频
            '-acodec', 'pcm_s16le',  # 使用原始WAV格式
            '-ar', '16000',  # 16kHz采样率
            '-ac', '1',  # 单声道
            output_audio_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        logging.error(f"提取音频失败: {str(e)}")
        return False

def get_video_duration(video_path):
    """获取视频时长"""
    try:
        # 使用ffprobe获取视频时长
        cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'default=noprint_wrappers=1:nokey=1', 
            video_path
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        logging.error(f"获取视频时长失败: {str(e)}")
        
        # 尝试使用OpenCV
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            return duration
        except Exception as e2:
            logging.error(f"使用OpenCV获取视频时长也失败了: {str(e2)}")
            return 0 

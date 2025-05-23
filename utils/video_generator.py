import os
import glob
import uuid
import shutil
import traceback
import threading
import subprocess
import time
import torch
import tempfile
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

from config import output_video_dir, subtitle_dir, avatar_dir
from utils.audio_utils import gpt_sovits_tts, fallback_tts, get_audio_duration, create_bytes_stream
from utils.subtitle_utils import (
    calculate_mixed_subtitle_timing,
    generate_srt_file,
    generate_vtt_file,
    add_subtitles_to_video,
    split_text_to_sentences,
    validate_text_cut_method,
    create_subtitle_for_text,
    get_video_duration
)
from utils.task_manager import task_progress

# 避免循环导入问题，使用懒加载方式
def get_avatar_class():
    """懒加载Avatar类，避免循环导入"""
    from scripts.realtime_inference import Avatar
    return Avatar

def get_avatar_manager():
    """懒加载AvatarManager类，避免循环导入"""
    from scripts.realtime_inference import AvatarManager
    return AvatarManager

def get_random_avatar():
    """
    Randomly select an available digital human avatar.
    
    Returns:
        str: Avatar ID
    
    Raises:
        Exception: If no avatars are found
    """
    import random
    avatar_dirs = glob.glob(os.path.join(avatar_dir, "*"))
    if not avatar_dirs:
        raise Exception("No available digital human avatars found")
    
    return os.path.basename(random.choice(avatar_dirs))

def get_all_avatars():
    """
    获取所有可用的数字人头像列表
    
    Returns:
        list: 所有可用的数字人ID列表
        
    Raises:
        Exception: 如果没有找到任何数字人
    """
    avatar_dirs = glob.glob(os.path.join(avatar_dir, "*"))
    if not avatar_dirs:
        raise Exception("未找到任何可用的数字人头像")
        
    # 提取所有数字人目录的名称作为ID
    avatars = [os.path.basename(dir_path) for dir_path in avatar_dirs]
    
    return sorted(avatars)  # 排序后返回

def generate_video_from_frames(frames: list, audio_path: str, output_path: str, fps: int = 25) -> bool:
    """从帧列表生成视频"""
    temp_dir = None
    try:
        # 创建临时目录保存帧
        temp_dir = tempfile.mkdtemp()
        frames_dir = os.path.join(temp_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        # 保存帧为图片
        for i, frame in enumerate(frames):
            frame_path = os.path.join(frames_dir, '{:08d}.png'.format(i))
            Image.fromarray(frame).save(frame_path)
            
        # 使用ffmpeg生成视频
        video_path = os.path.join(temp_dir, 'temp.mp4')
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(frames_dir, '%08d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            video_path
        ]
        subprocess.run(cmd, check=True)
        
        # 添加音频
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-strict', 'experimental',
            output_path
        ]
        subprocess.run(cmd, check=True)
        
        return True
    except Exception as e:
        print('Error generating video: {}'.format(e))
        traceback.print_exc()
        return False
    finally:
        # 清理临时文件
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def process_video_task(
    gpu_id: int,
    local_avatar: Optional[get_avatar_class()],
    current_avatar_id: Optional[str],
    task_data: Dict[str, Any],
    task_id: str
) -> Tuple[bool, Dict[str, Any]]:
    """处理视频生成任务"""
    avatar = None  # 定义avatar变量，确保在所有分支中都可访问
    
    try:
        # 导入socketio，用于发送WebSocket通知
        from main import socketio
        
        # 获取任务参数
        avatar_id = task_data.get('avatar_id')
        text = task_data.get('text', '')
        output_path = task_data.get('output_path')
        voice_model_id = task_data.get('voice_model_id', 'wang001')
        speech_rate = task_data.get('speech_rate', 1.0)
        
        # 获取所属的批量任务ID，如果有的话
        batch_task_id = None
        slide_index = None
        # 检查task_id格式，解析出批量任务ID和幻灯片索引
        if task_id and task_id.startswith("slide_"):
            parts = task_id.split("_")
            # 至少需要3个部分: slide_X_Y
            if len(parts) >= 3:
                # 尝试解析最后一部分为整数索引
                try:
                    slide_index = int(parts[-1])  # 最后一部分是索引
                    # 中间部分组合为batch_task_id
                    batch_parts = parts[1:-1]  # 取除第一部分和最后一部分外的所有部分
                    batch_task_id = "_".join(batch_parts)
                    print("解析任务ID成功: {} -> batch_task_id={}, slide_index={}".format(
                        task_id, batch_task_id, slide_index))
                except ValueError as e:
                    print("无法解析幻灯片索引，任务ID格式不符合预期: {} (错误: {})".format(task_id, str(e)))
                    slide_index = None
                    batch_task_id = None
        
        # 获取PPT相关参数
        design_id = task_data.get('design_id')
        ppt_id = task_data.get('ppt_id')
        page_index = task_data.get('page_index')
        
        # 获取字幕相关参数
        add_subtitle = task_data.get('add_subtitle', True)
        subtitle_format = task_data.get('subtitle_format', 'vtt')
        subtitle_encoding = task_data.get('subtitle_encoding', 'utf-8')
        burn_subtitles = task_data.get('burn_subtitles', False)
        subtitle_offset = task_data.get('subtitle_offset', -0.3)
        cn_speaking_rate = task_data.get('cn_speaking_rate', 4.0)
        en_word_rate = task_data.get('en_word_rate', 1.5)
        text_cut_method = task_data.get('text_cut_method', 'cut2')
        
        # 检查参数
        if not all([avatar_id, text, output_path]):
            # 如果是批量任务的一部分，发送失败通知
            if batch_task_id and slide_index is not None:
                try:
                    socketio.emit('slide_status_update', {
                        'task_id': batch_task_id,
                        'slide_index': slide_index,
                        'page_index': page_index,
                        'status': 'failed',
                        'progress': 0,
                        'error': "Missing required parameters"
                    }, room=batch_task_id)
                except Exception as e:
                    print("WebSocket notification failed: {}".format(e))
            
            return False, {"error": "Missing required parameters"}
            
        # 如果是批量任务的一部分，发送开始处理通知
        if batch_task_id and slide_index is not None:
            try:
                socketio.emit('slide_status_update', {
                    'task_id': batch_task_id,
                    'slide_index': slide_index,
                    'page_index': page_index,
                    'status': 'processing',
                    'progress': 10,
                    'message': "开始处理..."
                }, room=batch_task_id)
            except Exception as e:
                print("Failed to send slide status update: {}".format(e))
            
        # 如果有PPT相关参数，重新构建输出路径，使用designID_pptID_pptIndex.mp4的格式
        base_filename = None
        if all([design_id, ppt_id, page_index is not None]):
            # 获取output_path的目录部分
            output_dir = os.path.dirname(output_path)
            # 构建新的文件名
            base_filename = "{}_{}_{}".format(design_id, ppt_id, page_index)
            file_name = "{}.mp4".format(base_filename)
            # 更新output_path
            output_path = os.path.join(output_dir, file_name)
            print("更新输出文件路径为: {}".format(output_path))
            
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 获取Avatar实例，优先使用AvatarManager
        try:
            avatar_manager = get_avatar_manager()
            
            # 打印缓存状态
            cached_instances = list(avatar_manager._instances.keys())
            print("当前AvatarManager中已缓存的实例数量: {}".format(len(cached_instances)))
            print("已缓存的实例: {}".format(cached_instances))
            
            # 检查是否已经有缓存实例
            is_cached = False
            for key in cached_instances:
                cached_avatar_id, cached_gpu_id = key
                if cached_avatar_id == avatar_id:
                    if cached_gpu_id == gpu_id:
                        print("缓存命中! 数字人 {} 在GPU {} 上的实例已缓存".format(avatar_id, gpu_id))
                        is_cached = True
                    else:
                        print("数字人 {} 已在GPU {} 上缓存，但当前请求使用GPU {}".format(
                            avatar_id, cached_gpu_id, gpu_id))
            
            if not is_cached:
                print("数字人 {} 在GPU {} 上未缓存，将创建新实例".format(avatar_id, gpu_id))
            
            # 查找视频路径用于可能的preparation模式
            video_path = None
            avatar_path = os.path.join(avatar_dir, avatar_id)
            latents_path = os.path.join(avatar_path, "latents.pt")
            
            # 如果latents.pt不存在，可能需要准备模式
            if not os.path.exists(latents_path):
                source_video_path = os.path.join(avatar_path, "source.mp4")
                if os.path.exists(source_video_path):
                    video_path = source_video_path
                else:
                    # 尝试查找其他视频文件
                    other_videos = glob.glob(os.path.join(avatar_path, "vid_output", "*.mp4"))
                    if other_videos:
                        video_path = other_videos[0]
            
            # 使用AvatarManager获取实例
            avatar = avatar_manager.get_instance(
                avatar_id=avatar_id,
                video_path=video_path,
                bbox_shift=20,  # 默认bbox_shift
                batch_size=10,  # 批大小
                gpu_id=gpu_id
            )
            
            # 生成实例状态报告
            if hasattr(avatar_manager, 'instance_status_report'):
                avatar_manager.instance_status_report()
            else:
                print("AvatarManager没有instance_status_report方法")
                
            # 检查是否成功获取实例
            if avatar is None:
                print("AvatarManager.get_instance返回None")
                # 如果AvatarManager返回None，检查是否有传入的local_avatar可用
                if local_avatar is not None and current_avatar_id == avatar_id:
                    print("AvatarManager返回None，使用传入的本地Avatar实例")
                    avatar = local_avatar
                else:
                    # 发送失败通知
                    if batch_task_id and slide_index is not None:
                        try:
                            socketio.emit('slide_status_update', {
                                'task_id': batch_task_id,
                                'slide_index': slide_index,
                                'page_index': page_index,
                                'status': 'failed',
                                'progress': 0,
                                'error': "无法获取数字人模型实例"
                            }, room=batch_task_id)
                        except Exception as e:
                            print("Failed to send failure notification: {}".format(e))
                    
                    return False, {"error": "无法获取数字人模型实例"}
            else:
                # 再次检查缓存状态，验证是否成功缓存
                updated_cached_instances = list(avatar_manager._instances.keys())
                print("更新后的缓存实例: {}".format(updated_cached_instances))
        except ImportError:
            # 回退到旧方法
            print("未能导入AvatarManager，使用传统方法加载Avatar")
            # 使用传入的local_avatar或创建新实例
            if local_avatar is not None and current_avatar_id == avatar_id:
                avatar = local_avatar
                print("复用传入的数字人模型: {}".format(avatar_id))
            else:
                # 需要创建新的avatar实例
                try:
                    # 获取源视频路径
                    video_path = os.path.join(avatar_dir, avatar_id, "source.mp4")
                    
                    # 检查基本文件
                    latents_path = os.path.join(avatar_dir, avatar_id, "latents.pt")
                    if not os.path.exists(latents_path) and not os.path.exists(video_path):
                        other_videos = glob.glob(os.path.join(avatar_dir, avatar_id, "vid_output", "*.mp4"))
                        if other_videos:
                            video_path = other_videos[0]
                            print("使用替代视频文件: {}".format(video_path))
                        else:
                            # 发送失败通知
                            if batch_task_id and slide_index is not None:
                                try:
                                    socketio.emit('slide_status_update', {
                                        'task_id': batch_task_id,
                                        'slide_index': slide_index,
                                        'page_index': page_index,
                                        'status': 'failed',
                                        'progress': 0,
                                        'error': "缺少必要的数字人文件"
                                    }, room=batch_task_id)
                                except Exception as e:
                                    print("Failed to send failure notification: {}".format(e))
                            
                            return False, {"error": "缺少必要的数字人文件"}
                    
                    # 创建Avatar实例
                    print("创建新的数字人模型: {}...".format(avatar_id))
                    avatar = get_avatar_class()(
                        avatar_id=avatar_id,
                        video_path=video_path,
                        bbox_shift=20,  # 默认值
                        batch_size=10,   # 默认值
                        preparation=False,
                        gpu_id=gpu_id
                    )
                    print("数字人模型加载成功")
                except Exception as e:
                    print("加载数字人模型失败: {}".format(e))
                    traceback.print_exc()
                    
                    # 发送失败通知
                    if batch_task_id and slide_index is not None:
                        try:
                            socketio.emit('slide_status_update', {
                                'task_id': batch_task_id,
                                'slide_index': slide_index,
                                'page_index': page_index,
                                'status': 'failed',
                                'progress': 0,
                                'error': "加载数字人模型失败: {}".format(e)
                            }, room=batch_task_id)
                        except Exception as e:
                            print("Failed to send failure notification: {}".format(e))
                    
                    return False, {"error": "加载数字人模型失败: {}".format(e)}
        
        # 最终检查avatar是否为None
        if avatar is None:
            # 发送失败通知
            if batch_task_id and slide_index is not None:
                try:
                    socketio.emit('slide_status_update', {
                        'task_id': batch_task_id,
                        'slide_index': slide_index,
                        'page_index': page_index,
                        'status': 'failed',
                        'progress': 0,
                        'error': "无法获取有效的数字人模型实例"
                    }, room=batch_task_id)
                except Exception as e:
                    print("Failed to send failure notification: {}".format(e))
            
            return False, {"error": "无法获取有效的数字人模型实例"}
        
        # 更新进度 - 模型加载完成，开始生成音频
        if batch_task_id and slide_index is not None:
            try:
                socketio.emit('slide_status_update', {
                    'task_id': batch_task_id,
                    'slide_index': slide_index,
                    'page_index': page_index,
                    'status': 'processing',
                    'progress': 20,
                    'message': "生成音频中..."
                }, room=batch_task_id)
            except Exception as e:
                print("Failed to send slide status update: {}".format(e))
            
        # 生成音频
        audio_path = os.path.join(os.path.dirname(output_path), '{}_audio.wav'.format(task_id))
        try:
            # --- 调试：在调用 TTS 前打印最终使用的参数 --- 
            logging.info("即将调用 gpt_sovits_tts，使用的 voice_model_id: %s, speech_rate: %s", voice_model_id, speech_rate)
            # --- 调试结束 ---
            
            generated_audio_path = gpt_sovits_tts(
                text,
                voice_model_id=voice_model_id,
                speech_rate=speech_rate
            )
            if generated_audio_path:
                # 如果成功生成，使用生成的音频路径
                audio_path = generated_audio_path
            else:
                # 如果 gpt_sovits_tts 返回 None，尝试备用方案
                logging.warning("主 TTS (gpt_sovits_tts) 失败，尝试使用 gTTS 备选方案...")
                # === 修改 fallback_tts 调用 ===
                # fallback_tts 函数内部会处理临时文件路径，不需要在这里指定
                # 正确调用：只传递文本，语言默认为 'zh'
                generated_fallback_path = fallback_tts(text)
                # === 修改结束 ===
                if generated_fallback_path:
                    audio_path = generated_fallback_path # 使用 fallback 生成的路径
                    logging.info("备选 TTS 成功，使用音频: %s", audio_path)
                else:
                    # 发送失败通知
                    error_message = "主 TTS 和备选 TTS 均失败"
                    if batch_task_id and slide_index is not None:
                        try:
                            socketio.emit('slide_status_update', {
                                'task_id': batch_task_id,
                                'slide_index': slide_index,
                                'page_index': page_index,
                                'status': 'failed',
                                'progress': 0,
                                'error': error_message
                            }, room=batch_task_id)
                        except Exception as e:
                            print("Failed to send failure notification: {}".format(e))
                    
                    return False, {"error": error_message}
        except Exception as e:
            print('音频生成错误: {}'.format(e))
            traceback.print_exc()
            
            # --- 调试：在调用 TTS 前打印最终使用的参数 --- 
            logging.info("即将调用 gpt_sovits_tts，使用的 voice_model_id: %s, speech_rate: %s", voice_model_id, speech_rate)
            # --- 调试结束 ---
            
            # 发送失败通知
            if batch_task_id and slide_index is not None:
                try:
                    socketio.emit('slide_status_update', {
                        'task_id': batch_task_id,
                        'slide_index': slide_index,
                        'page_index': page_index,
                        'status': 'failed',
                        'progress': 0,
                        'error': "音频生成错误: {}".format(e)
                    }, room=batch_task_id)
                except Exception as e:
                    print("Failed to send failure notification: {}".format(e))
            
            return False, {"error": '音频生成错误: {}'.format(e)}
        
        # 更新进度 - 音频生成完成，开始生成视频
        if batch_task_id and slide_index is not None:
            try:
                socketio.emit('slide_status_update', {
                    'task_id': batch_task_id,
                    'slide_index': slide_index,
                    'page_index': page_index,
                    'status': 'processing',
                    'progress': 40,
                    'message': "生成视频中..."
                }, room=batch_task_id)
            except Exception as e:
                print("Failed to send slide status update: {}".format(e))
            
        # 生成视频帧
        try:
            # 首先检查Avatar实例是否有inference方法
            if not hasattr(avatar, "inference"):
                print("数字人模型实例缺少inference方法，尝试使用create_simple_video")
                if hasattr(avatar, "create_simple_video"):
                    # 使用简单视频生成作为备选
                    final_output_path = output_path
                    output_vid = os.path.join(os.path.dirname(output_path), "{}_simple.mp4".format(task_id))
                    success = avatar.create_simple_video(audio_path, output_vid)
                    if success:
                        # 复制到最终位置
                        shutil.copy(output_vid, final_output_path)
                        print("已生成简单视频: {}".format(final_output_path))
                    else:
                        # 发送失败通知
                        if batch_task_id and slide_index is not None:
                            try:
                                socketio.emit('slide_status_update', {
                                    'task_id': batch_task_id,
                                    'slide_index': slide_index,
                                    'page_index': page_index,
                                    'status': 'failed',
                                    'progress': 0,
                                    'error': "简单视频生成失败"
                                }, room=batch_task_id)
                            except Exception as e:
                                print("Failed to send failure notification: {}".format(e))
                        
                        return False, {"error": "简单视频生成失败"}
                else:
                    # 发送失败通知
                    if batch_task_id and slide_index is not None:
                        try:
                            socketio.emit('slide_status_update', {
                                'task_id': batch_task_id,
                                'slide_index': slide_index,
                                'page_index': page_index,
                                'status': 'failed',
                                'progress': 0,
                                'error': "数字人模型实例无法生成视频"
                            }, room=batch_task_id)
                        except Exception as e:
                            print("Failed to send failure notification: {}".format(e))
                    
                    return False, {"error": "数字人模型实例无法生成视频"}
            else:
                # 常规视频生成
                success = avatar.inference(
                    audio_path=audio_path,
                    out_vid_name=task_id,
                    fps=25,
                    skip_save_images=False
                )
                if not success:
                    # 发送失败通知
                    if batch_task_id and slide_index is not None:
                        try:
                            socketio.emit('slide_status_update', {
                                'task_id': batch_task_id,
                                'slide_index': slide_index,
                                'page_index': page_index,
                                'status': 'failed',
                                'progress': 0,
                                'error': "视频帧生成失败"
                            }, room=batch_task_id)
                        except Exception as e:
                            print("Failed to send slide status update: {}".format(e))
                    
                    return False, {"error": "视频帧生成失败"}
                
                # 获取视频文件
                output_vid = os.path.join(avatar.video_out_path, "{}.mp4".format(task_id))
                if not os.path.exists(output_vid):
                    # 发送失败通知
                    if batch_task_id and slide_index is not None:
                        try:
                            socketio.emit('slide_status_update', {
                                'task_id': batch_task_id,
                                'slide_index': slide_index,
                                'page_index': page_index,
                                'status': 'failed',
                                'progress': 0,
                                'error': "未找到生成的视频文件"
                            }, room=batch_task_id)
                        except Exception as e:
                            print("Failed to send failure notification: {}".format(e))
                    
                    return False, {"error": "未找到生成的视频文件"}
                    
                # 复制生成的视频到指定路径
                try:
                    shutil.copy(output_vid, output_path)
                    print("复制视频从 {} 到 {}".format(output_vid, output_path))
                except Exception as e:
                    print("复制视频错误: {}".format(e))
                    
                    # 发送失败通知
                    if batch_task_id and slide_index is not None:
                        try:
                            socketio.emit('slide_status_update', {
                                'task_id': batch_task_id,
                                'slide_index': slide_index,
                                'page_index': page_index,
                                'status': 'failed',
                                'progress': 0,
                                'error': "复制视频错误: {}".format(e)
                            }, room=batch_task_id)
                        except Exception as e:
                            print("Failed to send failure notification: {}".format(e))
                    
                    return False, {"error": "复制视频错误: {}".format(e)}
            
            # 更新进度 - 视频生成完成，开始生成字幕
            if batch_task_id and slide_index is not None:
                try:
                    socketio.emit('slide_status_update', {
                        'task_id': batch_task_id,
                        'slide_index': slide_index,
                        'page_index': page_index,
                        'status': 'processing',
                        'progress': 80,
                        'message': "生成字幕中..."
                    }, room=batch_task_id)
                except Exception as e:
                    print("Failed to send slide status update: {}".format(e))
                
            # 生成字幕文件
            subtitle_path = None
            if add_subtitle:
                try:
                    # 获取视频时长
                    video_duration = get_video_duration(output_path)
                    
                    # 确定字幕文件名和路径
                    if base_filename:
                        # 使用与视频相同的命名规则
                        subtitle_filename = "{}.{}".format(base_filename, subtitle_format)
                    else:
                        # 使用任务ID作为字幕文件名
                        subtitle_filename = "{}.{}".format(task_id, subtitle_format)
                        
                    subtitle_path = os.path.join(subtitle_dir, subtitle_filename)
                    
                    # 确保subtitle_dir目录存在
                    os.makedirs(subtitle_dir, exist_ok=True)
                    
                    # 创建字幕
                    subtitle_success = create_subtitle_for_text(
                        text=text,
                        output_path=subtitle_path,
                        duration=video_duration,
                        format=subtitle_format,
                        encoding=subtitle_encoding,
                        offset=subtitle_offset,
                        subtitle_speed=1.0,
                        text_cut_method=text_cut_method,
                        sentence_gap=0.4,
                        remove_ending_punct=True
                    )
                    
                    if subtitle_success:
                        print("字幕文件已生成: {}".format(subtitle_path))
                        
                        # 如果需要烧录字幕到视频
                        if burn_subtitles:
                            output_with_sub = output_path.replace('.mp4', '_with_sub.mp4')
                            burn_success = add_subtitles_to_video(
                                input_video=output_path,
                                output_video=output_with_sub,
                                subtitles=subtitle_path,
                                font_scale=task_data.get('subtitle_font_scale', 0.7)
                            )
                            
                            if burn_success:
                                print("带字幕的视频已生成: {}".format(output_with_sub))
                    else:
                        print("生成字幕文件失败")
                        
                except Exception as e:
                    print("生成字幕时出错: {}".format(e))
                    traceback.print_exc()
                    # 不要因为字幕错误而中断视频生成流程
            
        except Exception as e:
            print('视频帧生成错误: {}'.format(e))
            traceback.print_exc()
            
            # 发送失败通知
            if batch_task_id and slide_index is not None:
                try:
                    socketio.emit('slide_status_update', {
                        'task_id': batch_task_id,
                        'slide_index': slide_index,
                        'page_index': page_index,
                        'status': 'failed',
                        'progress': 0,
                        'error': "视频帧生成错误: {}".format(e)
                    }, room=batch_task_id)
                except Exception as e:
                    print("Failed to send failure notification: {}".format(e))
            
            return False, {"error": '视频帧生成错误: {}'.format(e)}
        
        # 发送成功通知
        if batch_task_id and slide_index is not None:
            try:
                # 生成视频URL
                video_url = "/resources/videos/{}".format(os.path.basename(output_path))
                # 生成字幕URL
                subtitle_url = None
                if subtitle_path:
                    subtitle_url = "/resources/subtitles/{}".format(os.path.basename(subtitle_path))
                
                socketio.emit('slide_status_update', {
                    'task_id': batch_task_id,
                    'slide_index': slide_index,
                    'page_index': page_index,
                    'status': 'completed',
                    'progress': 100,
                    'video_url': video_url,
                    'subtitle_url': subtitle_url,
                    'message': "处理完成"
                }, room=batch_task_id)
            except Exception as e:
                print("Failed to send batch completion notification: {}".format(e))
            
        return True, {
            "avatar_id": avatar_id,
            "output_path": output_path,
            "subtitle_path": subtitle_path,
            "design_id": design_id,
            "ppt_id": ppt_id,
            "page_index": page_index,
            "base_filename": base_filename
        }
        
    except Exception as e:
        print('视频任务处理错误: {}'.format(e))
        traceback.print_exc()
        
        # 发送失败通知
        try:
            if batch_task_id and slide_index is not None:
                socketio.emit('slide_status_update', {
                    'task_id': batch_task_id,
                    'slide_index': slide_index,
                    'page_index': page_index if 'page_index' in locals() else None,
                    'status': 'failed',
                    'progress': 0,
                    'error': str(e)
                }, room=batch_task_id)
        except Exception as ex:
            print("Failed to send failure notification: {}".format(ex))
        
        return False, {"error": str(e)}

def process_batch_slide_task(
    gpu_id: int,
    local_avatar: Optional[get_avatar_class()],
    current_avatar_id: Optional[str],
    task_data: Dict[str, Any],
    task_id: str
) -> Tuple[bool, Dict[str, Any]]:
    """处理批量幻灯片任务"""
    try:
        # 导入socketio，用于发送WebSocket通知
        from main import socketio
        
        # 获取任务参数
        avatar_id = task_data.get('avatar_id')
        slides = task_data.get('slides', [])
        output_dir = task_data.get('output_dir')
        
        # 获取批量任务ID - 在这种情况下，task_id就是batch_task_id
        batch_task_id = task_id
        
        # 获取PPT相关的通用参数
        design_id = task_data.get('design_id')
        ppt_id = task_data.get('ppt_id')
        
        # 获取字幕相关参数
        add_subtitle = task_data.get('add_subtitle', True)
        subtitle_format = task_data.get('subtitle_format', 'vtt')
        subtitle_encoding = task_data.get('subtitle_encoding', 'utf-8')
        burn_subtitles = task_data.get('burn_subtitles', False)
        subtitle_offset = task_data.get('subtitle_offset', -0.3)
        subtitle_font_scale = task_data.get('subtitle_font_scale', 0.7)
        text_cut_method = task_data.get('text_cut_method', 'cut2')
        
        # 检查参数
        if not all([avatar_id, slides, output_dir]):
            return False, {"error": "Missing required parameters"}
            
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 发送批处理开始通知
        try:
            socketio.emit('batch_task_update', {
                'task_id': batch_task_id,
                'status': 'processing',
                'total_slides': len(slides),
                'processed_slides': 0,
                'progress': 0,
                'message': "开始处理批量任务..."
            }, room=batch_task_id)
        except Exception as e:
            print("Failed to send batch start notification: {}".format(e))
        
        # 获取Avatar实例，优先使用AvatarManager
        try:
            # 使用懒加载获取AvatarManager
            avatar_manager = get_avatar_manager()
            
            # 打印缓存状态
            cached_instances = list(avatar_manager._instances.keys())
            print("批量任务: 当前AvatarManager中已缓存的实例数量: {}".format(len(cached_instances)))
            print("批量任务: 已缓存的实例: {}".format(cached_instances))
            
            # 检查缓存中是否已有该avatar
            found_cached = False
            for (cached_id, cached_gpu) in cached_instances:
                if cached_id == avatar_id and cached_gpu == gpu_id:
                    print("批量任务: 缓存命中! 数字人 {} 在GPU {} 上已缓存".format(avatar_id, gpu_id))
                    found_cached = True
                    break
            
            if not found_cached:
                print("批量任务: 数字人 {} 在GPU {} 上未缓存，将创建新实例".format(avatar_id, gpu_id))
            
            # 获取avatar实例
            avatar = avatar_manager.get_instance(
                avatar_id=avatar_id,
                video_path=None,  # 自动查找
                bbox_shift=20,
                batch_size=10,
                gpu_id=gpu_id
            )
            
            # 验证实例
            if avatar is None:
                print("批量任务: AvatarManager.get_instance返回None，检查本地Avatar")
                if local_avatar is not None and current_avatar_id == avatar_id:
                    print("批量任务: 使用本地传入的Avatar实例")
                    avatar = local_avatar
                else:
                    # 发送批处理失败通知
                    try:
                        socketio.emit('batch_task_update', {
                            'task_id': batch_task_id,
                            'status': 'failed',
                            'error': "无法获取数字人模型实例"
                        }, room=batch_task_id)
                    except Exception as e:
                        print("Failed to send batch failure notification: {}".format(e))
                    
                    return False, {"error": "无法获取数字人模型实例"}
            else:
                print("批量任务: 成功获取数字人模型实例")
        except Exception as e:
            print("批量任务: 获取Avatar实例时出错: {}".format(e))
            traceback.print_exc()
            
            # 回退到传统方法
            if local_avatar is not None and current_avatar_id == avatar_id:
                print("批量任务: 使用传入的本地Avatar实例")
                avatar = local_avatar
            else:
                try:
                    # 创建新的avatar实例
                    print("批量任务: 创建新的数字人模型")
                    video_path = os.path.join(avatar_dir, avatar_id, "source.mp4")
                    avatar = get_avatar_class()(
                        avatar_id=avatar_id,
                        video_path=video_path,
                        bbox_shift=20,
                        batch_size=10,
                        preparation=False,
                        gpu_id=gpu_id
                    )
                    print("批量任务: 数字人模型创建成功")
                except Exception as e:
                    print("批量任务: 创建数字人模型失败: {}".format(e))
                    traceback.print_exc()
                    
                    # 发送批处理失败通知
                    try:
                        socketio.emit('batch_task_update', {
                            'task_id': batch_task_id,
                            'status': 'failed',
                            'error': "无法创建数字人模型: {}".format(e)
                        }, room=batch_task_id)
                    except Exception as e:
                        print("Failed to send batch failure notification: {}".format(e))
                    
                    return False, {"error": "无法创建数字人模型: {}".format(e)}
        
        # 最终检查avatar是否为None
        if avatar is None:
            # 发送批处理失败通知
            try:
                socketio.emit('batch_task_update', {
                    'task_id': batch_task_id,
                    'status': 'failed',
                    'error': "无法获取有效的数字人模型实例"
                }, room=batch_task_id)
            except Exception as e:
                print("Failed to send batch failure notification: {}".format(e))
                
            return False, {"error": "无法获取有效的数字人模型实例"}
            
        # 处理结果
        results = []
        success_count = 0
        failed_count = 0
        
        # 开始处理每个幻灯片
        for i, slide in enumerate(slides):
            slide_text = slide.get('text', '')
            
            # 获取当前幻灯片的页码索引
            page_index = slide.get('page_index', i)
            slide_index = i  # 使用数组索引作为slide_index
            
            # 发送幻灯片开始处理通知
            try:
                socketio.emit('slide_status_update', {
                    'task_id': batch_task_id,
                    'slide_index': slide_index,
                    'page_index': page_index,
                    'status': 'processing',
                    'progress': 0,
                    'message': "开始处理..."
                }, room=batch_task_id)
            except Exception as e:
                print("Failed to send slide status update: {}".format(e))
            
            # 使用designID_pptID_pptIndex.mp4的格式构建输出文件名
            base_filename = None
            if all([design_id, ppt_id, page_index is not None]):
                base_filename = "{}_{}_{}".format(design_id, ppt_id, page_index)
                output_name = "{}.mp4".format(base_filename)
            else:
                # 如果没有足够的PPT相关参数，使用默认命名
                output_name = slide.get('output_name', 'slide_{}.mp4'.format(i))
                
            output_path = os.path.join(output_dir, output_name)
            print("幻灯片{}的输出文件路径: {}".format(i, output_path))
            
            # 更新进度 - 开始生成音频
            try:
                socketio.emit('slide_status_update', {
                    'task_id': batch_task_id,
                    'slide_index': slide_index,
                    'page_index': page_index,
                    'status': 'processing',
                    'progress': 20,
                    'message': "生成音频中..."
                }, room=batch_task_id)
            except Exception as e:
                print("Failed to send slide status update: {}".format(e))
            
            # 生成音频
            audio_path = os.path.join(output_dir, '{}_slide_{}_audio.wav'.format(task_id, i))
            try:
                generated_audio_path = gpt_sovits_tts(slide_text, text_lang='zh')
                if generated_audio_path:
                    # 如果成功生成，使用生成的音频路径
                    audio_path = generated_audio_path
                else:
                    # 如果 gpt_sovits_tts 返回 None，尝试备用方案
                    logging.warning("主 TTS (gpt_sovits_tts) 失败，尝试使用 gTTS 备选方案...")
                    # === 修改 fallback_tts 调用 ===
                    # fallback_tts 函数内部会处理临时文件路径，不需要在这里指定
                    # 正确调用：只传递文本，语言默认为 'zh'
                    generated_fallback_path = fallback_tts(slide_text)
                    # === 修改结束 ===
                    if generated_fallback_path:
                        audio_path = generated_fallback_path # 使用 fallback 生成的路径
                        logging.info("备选 TTS 成功，使用音频: %s", audio_path)
                    else:
                        # 发送失败通知
                        error_message = "主 TTS 和备选 TTS 均失败"
                        if batch_task_id and slide_index is not None:
                            try:
                                socketio.emit('slide_status_update', {
                                    'task_id': batch_task_id,
                                    'slide_index': slide_index,
                                    'page_index': page_index,
                                    'status': 'failed',
                                    'progress': 0,
                                    'error': error_message
                                }, room=batch_task_id)
                            except Exception as e:
                                print("Failed to send failure notification: {}".format(e))
                        
                        results.append({
                            "slide_index": slide_index,
                            "page_index": page_index,
                            "success": False,
                            "error": error_message
                        })
                        failed_count += 1
                        
                        # 更新总进度
                        processed = i + 1
                        progress = round((processed / len(slides)) * 100, 2)
                        try:
                            socketio.emit('batch_task_update', {
                                'task_id': batch_task_id,
                                'status': 'processing',
                                'total_slides': len(slides),
                                'processed_slides': processed,
                                'progress': progress,
                                'success_count': success_count,
                                'failed_count': failed_count,
                                'current_slide': {
                                    'slide_index': slide_index,
                                    'page_index': page_index,
                                    'status': 'failed'
                                }
                            }, room=batch_task_id)
                        except Exception as e:
                            print("Failed to send batch progress update: {}".format(e))
                        
                        continue
            except Exception as e:
                # 更新失败状态
                try:
                    socketio.emit('slide_status_update', {
                        'task_id': batch_task_id,
                        'slide_index': slide_index,
                        'page_index': page_index,
                        'status': 'failed',
                        'progress': 0,
                        'error': "音频生成错误: {}".format(e)
                    }, room=batch_task_id)
                except Exception as ex:
                    print("Failed to send slide status update: {}".format(ex))
                
                results.append({
                    "slide_index": slide_index,
                    "page_index": page_index,
                    "success": False,
                    "error": 'Audio generation error: {}'.format(e)
                })
                failed_count += 1
                
                # 更新总进度
                processed = i + 1
                progress = round((processed / len(slides)) * 100, 2)
                try:
                    socketio.emit('batch_task_update', {
                        'task_id': batch_task_id,
                        'status': 'processing',
                        'total_slides': len(slides),
                        'processed_slides': processed,
                        'progress': progress,
                        'success_count': success_count,
                        'failed_count': failed_count,
                        'current_slide': {
                            'slide_index': slide_index,
                            'page_index': page_index,
                            'status': 'failed'
                        }
                    }, room=batch_task_id)
                except Exception as e:
                    print("Failed to send batch progress update: {}".format(e))
                
                continue
            
            # 更新进度 - 音频生成完成，开始生成视频
            try:
                socketio.emit('slide_status_update', {
                    'task_id': batch_task_id,
                    'slide_index': slide_index,
                    'page_index': page_index,
                    'status': 'processing',
                    'progress': 40,
                    'message': "生成视频中..."
                }, room=batch_task_id)
            except Exception as e:
                print("Failed to send slide status update: {}".format(e))
                
            # 生成视频帧
            try:
                success = avatar.inference(
                    audio_path=audio_path,
                    out_vid_name=task_id,
                    fps=25,
                    skip_save_images=False
                )
                if not success:
                    # 更新失败状态
                    try:
                        socketio.emit('slide_status_update', {
                            'task_id': batch_task_id,
                            'slide_index': slide_index,
                            'page_index': page_index,
                            'status': 'failed',
                            'progress': 0,
                            'error': "视频帧生成失败"
                        }, room=batch_task_id)
                    except Exception as e:
                        print("Failed to send slide status update: {}".format(e))
                    
                    results.append({
                        "slide_index": slide_index,
                        "page_index": page_index,
                        "success": False,
                        "error": "Frame generation failed"
                    })
                    failed_count += 1
                    
                    # 更新总进度
                    processed = i + 1
                    progress = round((processed / len(slides)) * 100, 2)
                    try:
                        socketio.emit('batch_task_update', {
                            'task_id': batch_task_id,
                            'status': 'processing',
                            'total_slides': len(slides),
                            'processed_slides': processed,
                            'progress': progress,
                            'success_count': success_count,
                            'failed_count': failed_count,
                            'current_slide': {
                                'slide_index': slide_index,
                                'page_index': page_index,
                                'status': 'failed'
                            }
                        }, room=batch_task_id)
                    except Exception as e:
                        print("Failed to send batch progress update: {}".format(e))
                    
                    continue
                
                # 获取视频文件
                output_vid = os.path.join(avatar.video_out_path, "{}.mp4".format(task_id))
                if not os.path.exists(output_vid):
                    # 更新失败状态
                    try:
                        socketio.emit('slide_status_update', {
                            'task_id': batch_task_id,
                            'slide_index': slide_index,
                            'page_index': page_index,
                            'status': 'failed',
                            'progress': 0,
                            'error': "未找到生成的视频文件"
                        }, room=batch_task_id)
                    except Exception as e:
                        print("Failed to send failure notification: {}".format(e))
                    
                    results.append({
                        "slide_index": slide_index,
                        "page_index": page_index,
                        "success": False,
                        "error": "Video file not generated"
                    })
                    failed_count += 1
                    
                    # 更新总进度
                    processed = i + 1
                    progress = round((processed / len(slides)) * 100, 2)
                    try:
                        socketio.emit('batch_task_update', {
                            'task_id': batch_task_id,
                            'status': 'processing',
                            'total_slides': len(slides),
                            'processed_slides': processed,
                            'progress': progress,
                            'success_count': success_count,
                            'failed_count': failed_count,
                            'current_slide': {
                                'slide_index': slide_index,
                                'page_index': page_index,
                                'status': 'failed'
                            }
                        }, room=batch_task_id)
                    except Exception as e:
                        print("Failed to send batch progress update: {}".format(e))
                    
                    continue
                    
                # 复制生成的视频到指定路径
                try:
                    shutil.copy(output_vid, output_path)
                    print("复制视频从 {} 到 {}".format(output_vid, output_path))
                    
                    # 更新进度 - 视频复制完成，开始生成字幕
                    try:
                        socketio.emit('slide_status_update', {
                            'task_id': batch_task_id,
                            'slide_index': slide_index,
                            'page_index': page_index,
                            'status': 'processing',
                            'progress': 80,
                            'message': "生成字幕中..."
                        }, room=batch_task_id)
                    except Exception as e:
                        print("Failed to send slide status update: {}".format(e))
                    
                    # 生成字幕文件
                    subtitle_path = None
                    if add_subtitle:
                        try:
                            # 获取视频时长
                            video_duration = get_video_duration(output_vid)
                            
                            # 确定字幕文件名和路径
                            if base_filename:
                                # 使用与视频相同的命名规则
                                subtitle_filename = "{}.{}".format(base_filename, subtitle_format)
                            else:
                                # 使用幻灯片索引作为字幕文件名
                                subtitle_filename = "slide_{}_{}.{}".format(task_id, i, subtitle_format)
                                
                            subtitle_path = os.path.join(subtitle_dir, subtitle_filename)
                            
                            # 确保subtitle_dir目录存在
                            os.makedirs(subtitle_dir, exist_ok=True)
                            
                            # 创建字幕
                            subtitle_success = create_subtitle_for_text(
                                text=slide_text,
                                output_path=subtitle_path,
                                duration=video_duration,
                                format=subtitle_format,
                                encoding=subtitle_encoding,
                                offset=subtitle_offset,
                                subtitle_speed=1.0,
                                text_cut_method=text_cut_method,
                                sentence_gap=0.4,
                                remove_ending_punct=True
                            )
                            
                            if subtitle_success:
                                print("字幕文件已生成: {}".format(subtitle_path))
                                
                                # 如果需要烧录字幕到视频
                                if burn_subtitles:
                                    output_with_sub = output_path.replace('.mp4', '_with_sub.mp4')
                                    burn_success = add_subtitles_to_video(
                                        input_video=output_path,
                                        output_video=output_with_sub,
                                        subtitles=subtitle_path,
                                        font_scale=subtitle_font_scale
                                    )
                                    
                                    if burn_success:
                                        print("带字幕的视频已生成: {}".format(output_with_sub))
                                        # 可选：替换原始视频
                                        # shutil.move(output_with_sub, output_path)
                            else:
                                print("生成字幕文件失败")
                                
                        except Exception as e:
                            print("生成字幕时出错: {}".format(e))
                            traceback.print_exc()
                            # 不要因为字幕错误而中断视频生成流程
                    
                    # 处理成功，发送完成通知
                    video_url = "/resources/videos/{}".format(os.path.basename(output_path))
                    subtitle_url = None
                    if subtitle_path:
                        subtitle_url = "/resources/subtitles/{}".format(os.path.basename(subtitle_path))
                    
                    try:
                        socketio.emit('slide_status_update', {
                            'task_id': batch_task_id,
                            'slide_index': slide_index,
                            'page_index': page_index,
                            'status': 'completed',
                            'progress': 100,
                            'video_url': video_url,
                            'subtitle_url': subtitle_url,
                            'message': "处理完成"
                        }, room=batch_task_id)
                    except Exception as e:
                        print("Failed to send batch completion notification: {}".format(e))
                    
                    # 添加成功结果
                    results.append({
                        "success": True,
                        "slide_index": slide_index,
                        "page_index": page_index,
                        "output_path": output_path,
                        "subtitle_path": subtitle_path,
                        "video_url": video_url,
                        "subtitle_url": subtitle_url,
                        "status": "completed"
                    })
                    success_count += 1
                    
                except Exception as e:
                    print("复制视频错误: {}".format(e))
                    
                    # 更新失败状态
                    try:
                        socketio.emit('slide_status_update', {
                            'task_id': batch_task_id,
                            'slide_index': slide_index,
                            'page_index': page_index,
                            'status': 'failed',
                            'progress': 0,
                            'error': "复制视频错误: {}".format(e)
                        }, room=batch_task_id)
                    except Exception as ex:
                        print("Failed to send failure notification: {}".format(ex))
                    
                    results.append({
                        "slide_index": slide_index,
                        "page_index": page_index,
                        "success": False,
                        "error": "Error copying video: {}".format(e),
                        "status": "failed"
                    })
                    failed_count += 1
                    
                    # 继续处理下一个幻灯片
                    continue
                
            except Exception as e:
                # 更新失败状态
                try:
                    socketio.emit('slide_status_update', {
                        'task_id': batch_task_id,
                        'slide_index': slide_index,
                        'page_index': page_index,
                        'status': 'failed',
                        'progress': 0,
                        'error': "视频帧生成错误: {}".format(e)
                    }, room=batch_task_id)
                except Exception as ex:
                    print("Failed to send slide status update: {}".format(ex))
                
                results.append({
                    "slide_index": slide_index,
                    "page_index": page_index,
                    "success": False,
                    "error": "Frame generation error: {}".format(e),
                    "status": "failed"
                })
                failed_count += 1
                
                # 继续处理下一个幻灯片
                continue
            
            # 更新总进度
            processed = i + 1
            progress = round((processed / len(slides)) * 100, 2)
            try:
                socketio.emit('batch_task_update', {
                    'task_id': batch_task_id,
                    'status': 'processing',
                    'total_slides': len(slides),
                    'processed_slides': processed,
                    'progress': progress,
                    'success_count': success_count,
                    'failed_count': failed_count,
                    'current_slide': {
                        'slide_index': slide_index,
                        'page_index': page_index,
                        'status': 'completed'
                    }
                }, room=batch_task_id)
            except Exception as e:
                print("Failed to send batch progress update: {}".format(e))
        
        # 所有幻灯片处理完成，发送完成通知
        final_status = 'completed' if failed_count == 0 else 'completed_with_errors'
        try:
            socketio.emit('batch_task_update', {
                'task_id': batch_task_id,
                'status': final_status,
                'total_slides': len(slides),
                'processed_slides': len(slides),
                'progress': 100,
                'success_count': success_count,
                'failed_count': failed_count,
                'results': results,
                'message': "批处理任务完成"
            }, room=batch_task_id)
        except Exception as e:
            print("Failed to send batch completion notification: {}".format(e))
        
        return True, {
            "avatar_id": current_avatar_id,
            "results": results,
            "success_count": success_count,
            "failed_count": failed_count,
            "status": final_status
        }
        
    except Exception as e:
        print('Error processing batch slide task: {}'.format(e))
        traceback.print_exc()
        
        # 发送批处理失败通知
        try:
            socketio.emit('batch_task_update', {
                'task_id': task_id,  # 在这种情况下，task_id就是batch_task_id
                'status': 'failed',
                'error': str(e)
            }, room=task_id)
        except Exception as ex:
            print("Failed to send batch failure notification: {}".format(ex))
        
        return False, {"error": str(e)}

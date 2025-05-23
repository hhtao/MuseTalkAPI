#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
重写的main.py，升级到MuseTalk 1.5版本
保持原有功能和接口，使用新版本的视频生成模型
"""

from gevent import monkey
monkey.patch_all()

import sys
sys.path.append('/mnt/part2/Dteacher/MuseTalk')

import os
import time
import uuid
import glob
import shutil
import copy
import yaml
import logging
import traceback
import threading
from typing import Dict, Any, Optional
import torch
import numpy as np

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room
from datetime import datetime

# 路径配置
temp_dir = "/mnt/part2/Dteacher/MuseTalk/temp"
output_video_dir = "/mnt/part2/Dteacher/MuseTalk/output/videos"
subtitle_dir = "/mnt/part2/Dteacher/MuseTalk/output/subtitles"
static_dir = "/mnt/part2/Dteacher/MuseTalk/static"
template_dir = "/mnt/part2/Dteacher/MuseTalk/templates"
avatar_dir = "/mnt/part2/Dteacher/MuseTalk/results/avatars"

# 其他配置
DEBUG = False
CORS_ORIGINS = ["*"]

# 创建必要的目录
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(output_video_dir, exist_ok=True)
os.makedirs(subtitle_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)
os.makedirs(template_dir, exist_ok=True)
os.makedirs(avatar_dir, exist_ok=True)
os.makedirs('logs', exist_ok=True)

# 导入工具模块
try:
    from utils.gpu_manager import (
        initialize_multi_gpu_system, assign_task_to_gpu,
        active_gpus, gpu_statuses, gpu_locks, gpu_queues, gpu_worker_threads,
        get_gpu_status, clean_gpu_cache
    )
    from utils.task_manager import (
        initialize_task_manager, task_progress, create_task, update_task_progress,
        get_task_status, clean_completed_tasks, get_task_result, task_lock
    )
    from utils.audio_utils import (
        gpt_sovits_tts, fallback_tts, get_audio_duration
    )
    from utils.subtitle_utils import (
        calculate_mixed_subtitle_timing, generate_srt_file, generate_vtt_file,
        add_subtitles_to_video, split_text_to_sentences, validate_text_cut_method
    )
    from utils.video_generator import get_random_avatar, get_all_avatars
except ImportError as e:
    logging.error(f"导入模块失败: {str(e)}")
    sys.exit(1)

# 导入默认和备选的视频生成实现
try:
    from scripts.realtime_inference import Avatar
    USE_SIMPLE_MODE = False
    logging.info("使用realtime_inference的Avatar实现")
except Exception as e:
    from utils.video_generator import SimpleAvatar as Avatar
    USE_SIMPLE_MODE = True
    logging.info("使用SimpleAvatar作为备选实现")

# 简单的视频生成函数
def simple_generate_video(avatar_id: str, audio_path: str, output_path: str, fps: int = 25) -> bool:
    """简单的视频生成函数，在高级方法失败时使用"""
    try:
        import subprocess
        avatar_path = os.path.join(avatar_dir, avatar_id)
        source_video = os.path.join(avatar_path, "source.mp4")
        
        if not os.path.exists(source_video):
            logging.error(f"找不到数字人源视频: {source_video}")
            return False
        
        # 使用ffmpeg复制第一帧循环生成视频
        cmd = [
            'ffmpeg', '-y',
            '-i', source_video,          # 源视频
            '-i', audio_path,            # 音频文件
            '-vf', 'select=eq(n\\,0)',   # 选择第一帧
            '-vframes', '1',             # 只取一帧
            '-loop', '1',                # 循环输入帧
            '-shortest',                 # 以最短输入长度为准
            '-fps_mode', 'vfr',          # 可变帧率
            '-c:v', 'libx264',          # 视频编码器
            '-c:a', 'aac',              # 音频编码器
            '-strict', 'experimental', 
            output_path
        ]
        
        subprocess.run(cmd, check=True)
        return True
        
    except Exception as e:
        logging.error(f"简单视频生成失败: {str(e)}")
        return False

class VideoGeneratorV15:
    """
    视频生成器 V1.5 版本
    使用新版本的Avatar.inference模型生成视频
    """
    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型
        self._init_models()
        
    def _init_models(self):
        """初始化所需模型"""
        try:
            # 我们将缓存Avatar实例
            self.avatars = {}
            self.use_simple_mode = False  # 如果遇到问题将切换到简单模式
            
            logging.info(f"成功在GPU {self.gpu_id} 上初始化模型")
            return True
            
        except Exception as e:
            logging.error(f"初始化模型失败: {str(e)}")
            traceback.print_exc()
            return False
            
    def generate_video(
        self,
        text: str,
        avatar_id: str,
        output_path: str,
        task_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成视频主函数
        
        Args:
            text: 输入文本
            avatar_id: 数字人ID
            output_path: 输出视频路径
            task_id: 可选的任务ID
            **kwargs: 其他参数
        
        Returns:
            Dict包含生成结果
        """
        try:
            # 检查数字人资源
            avatar_res = self._prepare_avatar_resources(avatar_id)
            if not avatar_res['success']:
                return avatar_res
                
            # 生成音频
            audio_path = os.path.join(temp_dir, f"{task_id}_audio.wav")
            audio_res = self._generate_audio(text, audio_path, **kwargs)
            if not audio_res['success']:
                return audio_res
                
            # 尝试生成视频
            if not self.use_simple_mode and not USE_SIMPLE_MODE:
                try:
                    # 尝试使用高级模式
                    if avatar_id not in self.avatars:
                        self.avatars[avatar_id] = Avatar(
                            avatar_id=avatar_id,
                            video_path=avatar_res['video_path'],
                            preparation=False,
                            gpu_id=self.gpu_id
                        )
                        
                    success = self.avatars[avatar_id].inference(
                        audio_path=audio_path,
                        out_vid_name=os.path.basename(output_path),
                        fps=kwargs.get('fps', 25)
                    )
                    
                except Exception as e:
                    logging.warning(f"高级模式失败，切换到简单模式: {str(e)}")
                    self.use_simple_mode = True
                    success = False
                    
            # 如果高级模式失败或已经在简单模式，使用简单模式
            if self.use_simple_mode or not success:
                logging.info("使用简单视频生成模式")
                if avatar_id not in self.avatars:
                    self.avatars[avatar_id] = SimpleAvatar(
                        avatar_id=avatar_id,
                        video_path=avatar_res['video_path']
                    )
            
            success = self.avatars[avatar_id].generate_video(
                audio_path=audio_path,
                output_path=output_path,
                fps=kwargs.get('fps', 25)
            )
            
            if not success:
                return {
                    'success': False,
                    'error': "视频生成失败"
                }
                
            # 如果需要添加字幕
            if kwargs.get('add_subtitle') and kwargs.get('text'):
                subtitle_format = kwargs.get('subtitle_format', 'vtt')
                subtitle_path = os.path.splitext(output_path)[0] + f".{subtitle_format}"
                
                # 分割文本并计算时间轴
                sentences = split_text_to_sentences(kwargs['text'])
                audio_duration = get_audio_duration(audio_path)
                subtitle_timings = calculate_mixed_subtitle_timing(sentences, audio_duration)
                
                # 生成字幕文件
                if subtitle_format == 'srt':
                    generate_srt_file(subtitle_path, sentences, subtitle_timings)
                else:
                    generate_vtt_file(subtitle_path, sentences, subtitle_timings)
                
                # 添加字幕到视频
                add_subtitles_to_video(output_path, subtitle_path)
                
            video_res = {
                'success': True,
                'video_path': output_path,
                'subtitle_path': subtitle_path if kwargs.get('add_subtitle') and kwargs.get('text') else None
            }
            
            # 清理临时文件
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
            return video_res
            
        except Exception as e:
            logging.error(f"生成视频失败: {str(e)}")
            return {
                'success': False,
                'error': f"生成视频失败: {str(e)}"
            }

    def _prepare_avatar_resources(self, avatar_id: str) -> Dict[str, Any]:
        """准备数字人资源"""
        try:
            avatar_path = os.path.join(avatar_dir, avatar_id)
            
            # 检查数字人目录
            if not os.path.exists(avatar_path):
                return {
                    'success': False,
                    'error': f"数字人目录不存在: {avatar_path}"  
                }
                
            # 检查必要的文件
            video_path = self._get_avatar_video_path(avatar_path)
            if not video_path:
                return {
                    'success': False,
                    'error': f"未找到数字人视频文件: {avatar_path}"
                }
                
            return {
                'success': True,
                'avatar_path': avatar_path,
                'video_path': video_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"准备数字人资源失败: {str(e)}"
            }
            
    def _get_avatar_video_path(self, avatar_path: str) -> Optional[str]:
        """获取数字人视频路径"""
        # 优先使用source.mp4
        source_video = os.path.join(avatar_path, "source.mp4")
        if os.path.exists(source_video):
            return source_video
            
        # 查找其他视频文件
        other_videos = glob.glob(os.path.join(avatar_path, "vid_output", "*.mp4"))
        if other_videos:
            return other_videos[0]
            
        return None
        
    def _generate_audio(
        self,
        text: str,
        output_path: str,
        lang: str = 'zh',
        voice_model_id: str = 'default',
        **kwargs
    ) -> Dict[str, Any]:
        """生成音频"""
        try:
            # 使用TTS生成音频
            if voice_model_id == 'gpt_sovits':
                success = gpt_sovits_tts(text, output_path)
            else:
                success = fallback_tts(text, output_path)
                
            if not success:
                return {
                    'success': False,
                    'error': "生成音频失败"
                }
                
            return {
                'success': True,
                'audio_path': output_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"生成音频失败: {str(e)}"
            }
            

# 创建Flask应用
app = Flask(__name__, static_folder=static_dir)
CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}})

# 创建Socket.IO实例
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

# 静态路由
@app.route('/')
def index():
    """主页"""
    return send_from_directory(static_dir, 'index.html')

@app.route('/resources/videos/<path:filename>')
def serve_video(filename):
    """视频文件访问"""
    response = send_from_directory(output_video_dir, filename, mimetype='video/mp4')
    response.headers.extend({
        'Access-Control-Allow-Origin': '*',
        'Accept-Ranges': 'bytes'  # 支持视频断点续传
    })
    return response

@app.route('/resources/subtitles/<path:filename>')
def serve_subtitle(filename):
    """字幕文件访问"""
    mime_type = 'text/vtt; charset=utf-8' if filename.endswith('.vtt') else 'text/plain; charset=utf-8'
    
    response = send_from_directory(subtitle_dir, filename, mimetype=mime_type)
    response.headers.extend({
        'Access-Control-Allow-Origin': '*',
        'Content-Type': mime_type
    })
    return response

@app.route('/list_avatars', methods=['GET'])
def list_avatars():
    """获取所有数字人列表"""
    try:
        avatars = get_all_avatars()
        return jsonify({
            "success": True,
            "avatars": avatars
        })
    except Exception as e:
        logging.error(f"获取数字人列表失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# 创建视频生成器实例
video_generator = None

def initialize_system():
    """初始化系统"""
    try:
        # 设置日志
        setup_logging()
        logging.info("正在初始化系统...")
        
        # 创建必要的目录
        os.makedirs(output_video_dir, exist_ok=True)
        os.makedirs(subtitle_dir, exist_ok=True) 
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(static_dir, exist_ok=True)
        
        # 初始化CUDA设置
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # 检查GPU
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logging.info(f"检测到 {gpu_count} 个可用GPU")
            
            # 初始化多GPU系统
            initialize_multi_gpu_system()
        else:
            logging.warning("未检测到GPU,将使用CPU模式")
            
        # 初始化任务管理器
        initialize_task_manager()
        
        # 初始化视频生成器
        global video_generator
        video_generator = VideoGeneratorV15()
        
        logging.info("系统初始化完成")
        return True
        
    except Exception as e:
        logging.error(f"系统初始化失败: {str(e)}")
        return False

def setup_logging():
    """配置日志"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'smain_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def extract_video_params(data: Dict[str, Any], is_ppt: bool = False) -> Dict[str, Any]:
    """提取并标准化视频参数"""
    params = {
        'text': data.get('text', ''),
        'lang': data.get('lang', 'zh'),
        'design_id': data.get('design_id', 'default'),
        'ppt_id': data.get('ppt_id', '0'),
        'page_index': data.get('page_index', 0),
        'voice_model_id': data.get('voice_model_id', 'wang001'),
        'speech_rate': float(data.get('speech_rate', 1.0)),
        
        # 字幕相关参数
        'add_subtitle': data.get('add_subtitle', True),
        'subtitle_format': data.get('subtitle_format', 'vtt'),
        'subtitle_offset': float(data.get('subtitle_offset', -0.3)),
        'burn_subtitles': data.get('burn_subtitles', False),
        
        # 语音速率参数
        'cn_speaking_rate': float(data.get('cn_speaking_rate', 4.0 if is_ppt else 4.0)),
        'en_word_rate': float(data.get('en_word_rate', 3.5 if is_ppt else 1.5)),
        
        # 其他参数
        'fps': data.get('fps', 25),
        'text_cut_method': validate_text_cut_method(data.get('text_cut_method', 'cut2'))
    }
    
    # 记录所有参数
    logging.info(f"生成参数: {params}")
    return params

@app.route('/generate_video', methods=['POST'])
def generate_video_endpoint():
    """视频生成接口"""
    try:
        data = request.json
        
        # 验证输入
        if not data.get('text'):
            return jsonify({"error": "缺少文本内容"}), 400
            
        # 获取或生成avatar_id
        avatar_id = data.get('avatar_id')
        if not avatar_id:
            avatar_id = get_random_avatar()
            
        # 创建任务ID
        task_id = f"video_{uuid.uuid4().hex[:8]}"
        output_path = os.path.join(output_video_dir, f"{task_id}.mp4")
        
        # 提取标准化的参数
        params = extract_video_params(data)
        params.update({
            'avatar_id': avatar_id,
            'output_path': output_path,
            'task_id': task_id
        })
        
        # 创建任务
        create_task(task_id=task_id, task_type='generate_video', task_data=params)
        
        # 分配到GPU队列
        success = assign_task_to_gpu({
            'task_id': task_id,
            'task_type': 'generate_video',
            'data': params
        })
        
        if success:
            return jsonify({
                "success": True,
                "message": "视频生成任务已提交",
                "task_id": task_id
            })
        else:
            return jsonify({
                "success": False,
                "error": "无法分配GPU资源"
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
        
@app.route('/generate_ppt_videos', methods=['POST'])
def generate_ppt_videos():
    """PPT视频生成接口"""
    try:
        data = request.json
        mode = data.get('mode', 'single')
        
        if mode == 'single':
            # 提取参数
            params = extract_video_params(data, is_ppt=True)
            
            # 验证必需参数
            if not params['text'] or not params['design_id'] or not params['ppt_id']:
                return jsonify({"error": "缺少必需参数：text, design_id, ppt_id"}), 400
                
            # 获取avatar_id
            avatar_id = data.get('avatar_id', get_random_avatar())
            
            # 创建任务
            task_id = f"ppt_{uuid.uuid4().hex[:8]}"
            output_path = os.path.join(output_video_dir, f"{task_id}.mp4")
            
            # 更新参数
            params.update({
                'avatar_id': avatar_id,
                'output_path': output_path,
                'task_id': task_id
            })
            
            # 创建任务
            create_task(task_id=task_id, task_type='generate_ppt_video', task_data=params)
            
            # 分配任务
            success = assign_task_to_gpu({
                'task_id': task_id,
                'task_type': 'generate_ppt_video',
                'data': params
            })
            
            if success:
                return jsonify({
                    "success": True,
                    "message": "PPT视频生成任务已提交",
                    "task_id": task_id
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "无法分配GPU资源"
                }), 500
                
        elif mode == 'batch':
            # 获取任务列表
            slides = data.get('slides', [])
            common_params = data.get('common_params', {})
            
            # 验证数据
            if not slides:
                return jsonify({"error": "未提供幻灯片数据"}), 400
                
            # 设置批处理任务
            batch_task_id = f"batch_{uuid.uuid4().hex[:8]}"
            
            # 提取公共参数
            common_data = extract_video_params(common_params, is_ppt=True)
            common_data['avatar_id'] = common_params.get('avatar_id', get_random_avatar())
            
            # 创建总任务
            create_task(
                task_id=batch_task_id,
                task_type='batch_ppt_videos',
                task_data={
                    'total_slides': len(slides),
                    'common_params': common_data
                },
                total_items=len(slides)
            )
            
            # 启动处理线程
            threading.Thread(
                target=process_batch_slides,
                args=(batch_task_id, slides, common_data),
                daemon=True
            ).start()
            
            return jsonify({
                "success": True,
                "message": "批量PPT视频生成任务已提交",
                "task_id": batch_task_id,
                "total_slides": len(slides)
            })
        else:
            return jsonify({"error": "无效的模式参数"}), 400
            
    except Exception as e:
        logging.error(f"处理PPT视频生成请求失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# 添加PPT视频处理函数
def process_batch_slides(batch_task_id, slides, common_params):
    """处理批量PPT视频生成"""
    try:
        total_slides = len(slides)
        processed = 0
        
        logging.info(f"开始处理批量任务 {batch_task_id}, 共 {total_slides} 个幻灯片")
        
        # 更新任务状态
        update_task_progress(
            task_id=batch_task_id,
            status="processing",
            completed=0,
            message=f"处理第 {processed}/{total_slides} 个幻灯片"
        )
        
        results = []
        for i, slide in enumerate(slides):
            try:
                # 创建单个任务
                slide_task_id = f"slide_{batch_task_id}_{i}"
                
                # 合并参数
                params = copy.deepcopy(common_params)
                params.update(slide)
                params.update({
                    'task_id': slide_task_id,
                    'output_path': os.path.join(output_video_dir, f"{slide_task_id}.mp4")
                })
                
                # 生成视频
                result = video_generator.generate_video(**params)
                results.append({
                    'slide_index': i,
                    'success': result['success'],
                    'output_path': result.get('video_path'),
                    'error': result.get('error')
                })
                
                # 更新进度
                processed += 1
                progress = int((processed / total_slides) * 100)
                update_task_progress(
                    task_id=batch_task_id,
                    status="processing",
                    completed=progress,
                    message=f"处理第 {processed}/{total_slides} 个幻灯片"
                )
                
                # 发送WebSocket更新
                socketio.emit('batch_progress', {
                    'task_id': batch_task_id,
                    'processed': processed,
                    'total': total_slides,
                    'progress': progress,
                    'current_slide': i
                }, room=batch_task_id)
                
            except Exception as e:
                logging.error(f"处理幻灯片 {i} 失败: {str(e)}")
                results.append({
                    'slide_index': i,
                    'success': False,
                    'error': str(e)
                })
        
        # 更新任务完成状态
        update_task_progress(
            task_id=batch_task_id,
            status="completed",
            completed=100,
            message="批量处理完成",
            result={'slides': results}
        )
        
        # 发送完成通知
        socketio.emit('batch_complete', {
            'task_id': batch_task_id,
            'total_processed': processed,
            'results': results
        }, room=batch_task_id)
        
    except Exception as e:
        logging.error(f"批量处理失败: {str(e)}")
        update_task_progress(
            task_id=batch_task_id,
            status="failed",
            error=str(e)
        )

@app.route('/task_progress/<task_id>', methods=['GET'])
def get_task_progress_endpoint(task_id):
    """获取任务进度"""
    try:
        status = get_task_status(task_id)
        if not status:
            return jsonify({"error": "任务不存在"}), 404
            
        # 如果任务已完成，添加视频和字幕URL
        if status.get("status") == "completed":
            task_data = status.get("data", {})
            design_id = task_data.get("design_id")
            ppt_id = task_data.get("ppt_id")
            page_index = task_data.get("page_index")
            
            # 如果是PPT任务，使用设计ID命名
            if all([design_id, ppt_id, page_index is not None]):
                video_filename = f"{design_id}_{ppt_id}_{page_index}.mp4"
                subtitle_filename = f"{design_id}_{ppt_id}_{page_index}.{task_data.get('subtitle_format', 'vtt')}"
            else:
                # 普通视频任务
                output_path = task_data.get("output_path", "")
                video_filename = os.path.basename(output_path)
                subtitle_filename = os.path.splitext(video_filename)[0] + f".{task_data.get('subtitle_format', 'vtt')}"
            
            # 添加URL
            status.update({
                "video_url": f"/resources/videos/{video_filename}",
                "subtitle_url": f"/resources/subtitles/{subtitle_filename}",
                "video_path": os.path.join(output_video_dir, video_filename),
                "subtitle_path": os.path.join(subtitle_dir, subtitle_filename)
            })
            
            # 为批量任务的每个结果添加URL
            for result in status.get("results", []):
                if result.get("success") and "output_path" in result:
                    video_path = result["output_path"]
                    video_filename = os.path.basename(video_path)
                    subtitle_filename = os.path.splitext(video_filename)[0] + f".{task_data.get('subtitle_format', 'vtt')}"
                    
                    result.update({
                        "video_url": f"/resources/videos/{video_filename}",
                        "subtitle_url": f"/resources/subtitles/{subtitle_filename}",
                        "video_path": os.path.abspath(video_path),
                        "subtitle_path": os.path.join(subtitle_dir, subtitle_filename)
                    })
                    
        return jsonify(status)
        
    except Exception as e:
        logging.error(f"获取任务进度失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

# WebSocket事件处理
@socketio.on('connect')
def handle_connect():
    logging.info('客户端已连接')

@socketio.on('disconnect')
def handle_disconnect():
    logging.info('客户端已断开连接')

@socketio.on('subscribe_task')
def handle_subscribe(data):
    """订阅任务进度"""
    task_id = data.get('task_id')
    if not task_id:
        emit('error', {'message': '缺少task_id参数'})
        return
        
    join_room(task_id)
    
    # 发送当前进度
    status = get_task_status(task_id)
    if status:
        emit('task_info', status)
    else:
        emit('error', {'message': '任务不存在'})

if __name__ == "__main__":
    try:
        if initialize_system():
            logging.info("启动服务器...")
            from gevent import pywsgi
            from geventwebsocket.handler import WebSocketHandler
            server = pywsgi.WSGIServer(('0.0.0.0', 5000), app, handler_class=WebSocketHandler)
            server.serve_forever()
        else:
            logging.error("系统初始化失败")
            sys.exit(1)
    except KeyboardInterrupt:
        logging.info("服务已停止")
    except Exception as e:
        logging.error(f"服务启动失败: {str(e)}")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import argparse
import uuid
import threading
import time
import subprocess
import requests
from typing import Dict, Any, Optional
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Main")

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置加载模块
from musetalk_wrapper.config import load_config
from musetalk_wrapper.core.avatar_generator import AvatarGenerator
from musetalk_wrapper.core.dependency_manager import DependencyManager
from musetalk_wrapper.utils.tts_service import TTSService
from musetalk_wrapper.core.fallback_manager import FallbackManager, CapabilityLevel

# 任务管理
task_progress = {}  # 存储任务进度
task_lock = threading.Lock()  # 任务锁

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MuseTalk数字人视频生成工具")
    
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # 生成视频命令
    generate_parser = subparsers.add_parser("generate", help="生成数字人视频")
    generate_parser.add_argument("--text", "-t", required=True, help="要转换的文本")
    generate_parser.add_argument("--avatar-id", "-a", required=True, help="数字人ID")
    generate_parser.add_argument("--output", "-o", required=True, help="输出视频路径")
    generate_parser.add_argument("--audio", help="音频文件路径，如果提供则直接使用该音频")
    generate_parser.add_argument("--subtitle", action="store_true", default=True, help="是否添加字幕（默认添加）")
    generate_parser.add_argument("--no-subtitle", action="store_false", dest="subtitle", help="不添加字幕")
    generate_parser.add_argument("--subtitle-color", default="#FFFFFF", help="字幕颜色")
    generate_parser.add_argument("--background-music", help="背景音乐路径")
    generate_parser.add_argument("--volume-ratio", type=float, default=1.0, help="语音与背景音乐音量比例")
    generate_parser.add_argument("--prefer-mode", choices=["advanced", "standard", "basic"], help="优先使用的模式")
    generate_parser.add_argument("--gpu-id", type=int, default=0, help="使用的GPU ID")
    
    # 列出数字人命令
    list_parser = subparsers.add_parser("list-avatars", help="列出可用的数字人")
    
    # 检查环境命令
    check_parser = subparsers.add_parser("check-env", help="检查环境")
    check_parser.add_argument("--level", choices=["basic", "standard", "advanced"], default="basic", help="检查指定级别所需的环境")
    
    # 启动Web服务命令
    server_parser = subparsers.add_parser("server", help="启动Web服务")
    server_parser.add_argument("--host", default="0.0.0.0", help="服务主机地址")
    server_parser.add_argument("--port", type=int, default=5000, help="服务端口")
    
    # 安装指南命令
    install_parser = subparsers.add_parser("install-guide", help="获取安装指南")
    install_parser.add_argument("--level", choices=["basic", "standard", "advanced"], default="basic", help="获取指定级别的安装指南")
    
    # 生成配置文件命令
    config_parser = subparsers.add_parser("generate-config", help="生成配置文件")
    config_parser.add_argument("--output", "-o", default="./", help="输出目录")
    
    # 测试命令
    test_parser = subparsers.add_parser("test", help="快速测试功能")
    test_parser.add_argument("--text", "-t", default="你好，我是数字人。这是一条测试文本，用于验证功能是否正常。", help="测试文本")
    test_parser.add_argument("--avatar-id", "-a", default="avator_10", help="数字人ID")
    test_parser.add_argument("--output", "-o", default="./test_output.mp4", help="输出视频路径")
    test_parser.add_argument("--subtitle", action="store_true", default=True, help="是否添加字幕（默认添加）")
    test_parser.add_argument("--no-subtitle", action="store_false", dest="subtitle", help="不添加字幕")
    test_parser.add_argument("--mode", choices=["basic", "standard", "advanced"], default="standard", help="生成模式")
    
    args = parser.parse_args()
    
    # 如果没有提供命令，显示帮助
    if not args.command:
        parser.print_help()
        
    return args

def generate_video(args) -> Dict[str, Any]:
    """
    生成数字人视频
    
    Args:
        args: 命令行参数
        
    Returns:
        生成结果
    """
    # 加载配置
    config = load_config()
    
    # 初始化生成器
    generator = AvatarGenerator(config=config)
    
    # 准备选项
    options = {
        "subtitle": args.subtitle,
        "subtitle_color": args.subtitle_color
    }
    
    # 处理音频路径（如果提供）
    if args.audio:
        options["audio_path"] = args.audio
    
    # 处理背景音乐（如果提供）
    if args.background_music:
        options["background_music"] = args.background_music
        options["volume_ratio"] = args.volume_ratio
    
    # 处理模式优先级
    if args.prefer_mode:
        options["prefer_mode"] = args.prefer_mode
    
    # 生成视频
    result = generator.generate_video(
        text=args.text,
        avatar_id=args.avatar_id,
        output_path=args.output,
        **options
    )
    
    return result

def resolve_path(config_path: str, project_root: Path) -> str:
    path = Path(config_path)
    if path.is_absolute():
        return str(path)
    else:
        return str(project_root / path)

def list_avatars():
    """列出可用的数字人"""
    # 加载配置
    config = load_config()
    
    # Resolve avatar_dir relative to project root
    try:
        project_root = Path(__file__).parent.parent.parent.resolve()
        default_avatar_dir = "./results/v15/avatars"
        avatar_dir = resolve_path(
             config.get("paths", {}).get("avatar_dir", default_avatar_dir),
             project_root
         )
        logger.info("Listing avatars from directory: {}".format(avatar_dir))
    except Exception as e:
        logger.error("Could not resolve avatar directory path: {}. Using hardcoded default as last resort.".format(e))
        # Fallback to hardcoded default ONLY if resolution fails catastrophically
        avatar_dir = config.get("paths", {}).get("avatar_dir", "/mnt/part2/Dteacher/MuseTalk/results/v15/avatars")

    if not os.path.exists(avatar_dir):
        logger.warning("数字人目录不存在: " + avatar_dir)
        return []
    
    avatars = []
    for item in os.listdir(avatar_dir):
        if os.path.isdir(os.path.join(avatar_dir, item)):
            avatar_id = item
            avatar_info = {
                "id": avatar_id,
                "name": avatar_id.title(),
                "file": os.path.join(avatar_dir, item)
            }
            avatars.append(avatar_info)
    
    return avatars

def check_environment(level: str = "basic"):
    """
    检查环境
    
    Args:
        level: 功能级别 (basic/standard/advanced)
        
    Returns:
        环境检查结果
    """
    # 初始化依赖管理器
    dependency_manager = DependencyManager()
    
    # 检查环境
    results = dependency_manager.check_all_dependencies(level)
    
    # 添加CUDA详细信息
    if results.get("cuda", False):
        try:
            import torch
            results["cuda_info"] = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
            }
        except Exception as e:
            results["cuda_info"] = {"error": str(e)}
    
    return results

def get_installation_guide(level: str = "basic"):
    """
    获取安装指南
    
    Args:
        level: 功能级别 (basic/standard/advanced)
        
    Returns:
        安装指南
    """
    # 初始化依赖管理器
    dependency_manager = DependencyManager()
    
    # 获取安装指南
    guide = dependency_manager.get_installation_guide(level)
    
    return guide

def generate_config_files(output_dir: str = "./"):
    """
    生成配置文件
    
    Args:
        output_dir: 输出目录
        
    Returns:
        生成的文件列表
    """
    # 初始化依赖管理器
    dependency_manager = DependencyManager()
    
    # 生成conda环境文件
    conda_files = [
        dependency_manager.generate_conda_env_file("basic"),
        dependency_manager.generate_conda_env_file("standard"),
        dependency_manager.generate_conda_env_file("advanced")
    ]
    
    # 生成requirements文件
    requirements_files = dependency_manager.generate_requirements_files()
    
    # 合并结果
    all_files = conda_files + requirements_files
    
    return all_files

# ==================== Web API 服务 ====================

# 创建和更新任务
def create_task(task_id: str, task_type: str, task_data: Dict[str, Any], total_items: int = 1) -> None:
    """创建新任务"""
    with task_lock:
        task_progress[task_id] = {
            "task_id": task_id,
            "type": task_type,
            "status": "created",
            "data": task_data,
            "created_at": time.time(),
            "updated_at": time.time(),
            "completed": 0,
            "total_items": total_items,
            "message": "任务创建成功"
        }

def update_task_progress(task_id: str, status: str = None, completed: int = None, message: str = None, error: str = None, result: Dict[str, Any] = None, extra_data: Dict[str, Any] = None) -> None:
    """更新任务进度
    
    Args:
        task_id: 任务ID
        status: 任务状态
        completed: 已完成的项目数量
        message: 任务消息
        error: 错误消息
        result: 任务结果
        extra_data: 额外的任务数据，可用于更新任何任务字段
    """
    with task_lock:
        if task_id in task_progress:
            if status:
                task_progress[task_id]["status"] = status
            if completed is not None:
                task_progress[task_id]["completed"] = completed
            if message:
                task_progress[task_id]["message"] = message
            if error:
                task_progress[task_id]["error"] = error
            if result:
                task_progress[task_id]["result"] = result
            
            # 更新额外的任务数据
            if extra_data:
                for key, value in extra_data.items():
                    task_progress[task_id][key] = value
                    
            task_progress[task_id]["updated_at"] = time.time()

def get_task_status(task_id: str) -> Dict[str, Any]:
    """获取任务状态"""
    with task_lock:
        if task_id in task_progress:
            return task_progress[task_id].copy()
        return None

def clean_completed_tasks(max_age: int = 3600) -> None:
    """清理已完成的任务"""
    current_time = time.time()
    with task_lock:
        for task_id in list(task_progress.keys()):
            task = task_progress[task_id]
            if task["status"] in ["completed", "failed"] and current_time - task["updated_at"] > max_age:
                del task_progress[task_id]

# 处理数字人视频任务
def process_video_task(task_id: str, data: Dict[str, Any]) -> None:
    """处理视频生成任务"""
    try:
        update_task_progress(task_id, "processing", 10, "初始化生成器")
        
        # 加载配置
        config = load_config()
        
        # 初始化生成器
        generator = AvatarGenerator(config=config)
        
        update_task_progress(task_id, "processing", 20, "开始生成视频")
        
        # 提取参数
        text = data.pop("text", "")  # 从参数字典中取出text并删除，避免重复传入
        avatar_id = data.pop("avatar_id", "")  # 从参数字典中取出avatar_id并删除
        output_path = data.pop("output_path", "")  # 从参数字典中取出output_path并删除
        
        # 处理TTS模型ID - 确保voice_model_id是字符串
        voice_model_id = data.pop("voice_model_id", "wang001")
        # 确保它是字符串，如果是对象则取其id字段
        if isinstance(voice_model_id, dict) and 'id' in voice_model_id:
            voice_model_id = voice_model_id['id']
        
        # 将voice_model_id添加为tts_model参数
        data["tts_model"] = voice_model_id
        
        # 剩余参数作为选项
        options = data
        
        # 生成视频
        result = generator.generate_video(
            text=text,
            avatar_id=avatar_id,
            output_path=output_path,
            **options
        )
        
        if result.get("success", False):
            update_task_progress(task_id, "completed", 100, "视频生成成功", result=result)
        else:
            update_task_progress(task_id, "failed", 100, "视频生成失败", error=result.get("error", "未知错误"))
            
    except Exception as e:
        logger.error("处理任务失败: " + str(e))
        update_task_progress(task_id, "failed", 0, "处理任务失败", error=str(e))

# Flask应用和API接口
def start_server(host: str = "0.0.0.0", port: int = 5000):
    """
    启动Web服务
    
    Args:
        host: 服务主机地址
        port: 服务端口
    """
    try:
        # 检查FFmpeg配置
        check_ffmpeg_config()
        
        # 导入Flask相关库
        from flask import Flask, request, jsonify, send_from_directory
        from flask_cors import CORS
        
        # 创建应用
        app = Flask(__name__)
        CORS(app, resources={r"/*": {"origins": "*"}})
        
        # 路由定义
        @app.route('/api/generate_video', methods=['POST'])
        def generate_video_api():
            """视频生成API"""
            try:
                data = request.json
                
                # 验证输入
                if not data.get('text'):
                    return jsonify({"success": False, "error": "缺少文本内容"}), 400
                    
                # 获取或生成avatar_id，默认使用avator_10
                avatar_id = data.get('avatar_id', 'avator_10')
                if not avatar_id:
                    all_avatars = list_avatars()
                    if all_avatars:
                        avatar_id = 'avator_10'  # 默认使用avator_10
                    else:
                        return jsonify({"success": False, "error": "未找到可用的数字人"}), 400
                
                # 加载配置
                config = load_config()
                output_dir = config.get("paths", {}).get("output_video_dir", "./outputs")
                os.makedirs(output_dir, exist_ok=True)
                
                # 创建任务ID和输出路径
                task_id = "video_" + uuid.uuid4().hex[:8]
                output_path = os.path.join(output_dir, task_id + ".mp4")
                
                # 准备任务数据 - 包含所有可能的参数
                task_data = {
                    "text": data.get("text", ""),
                    "avatar_id": avatar_id,
                    "output_path": output_path,
                    "subtitle": data.get("subtitle", True),  # 默认为True
                    "prefer_mode": data.get("prefer_mode", "standard"),  # 默认使用标准模式，支持唇动
                    "gpu_id": data.get("gpu_id", 0),
                    "subtitle_color": data.get("subtitle_color", "#FFFFFF"),
                    "background_music": data.get("background_music"),
                    "volume_ratio": data.get("volume_ratio", 1.0),
                    "lang": data.get("lang", "zh"),
                    "tts_model": data.get("tts_model")
                }
                
                # 移除所有None值的参数，避免无效参数传递
                task_data = {k: v for k, v in task_data.items() if v is not None}
                
                # 创建任务
                create_task(task_id, "generate_video", task_data)
                
                # 启动处理线程
                threading.Thread(
                    target=process_video_task,
                    args=(task_id, task_data),
                    daemon=True
                ).start()
                
                return jsonify({
                    "success": True,
                    "message": "视频生成任务已提交",
                    "task_id": task_id
                })
                
            except Exception as e:
                logger.error("API处理失败: " + str(e))
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @app.route('/generate_ppt_videos', methods=['POST'])
        def generate_ppt_videos():
            """PPT视频生成API - 与前端PPTEditor.vue匹配"""
            try:
                data = request.json
                
                # 日志记录请求数据，方便调试
                logger.info("收到生成PPT视频请求，数据: " + str(data))
                
                # 验证基本输入
                if data.get('process_all'):
                    # 批量生成模式
                    if not data.get('ppt_id') or not data.get('design_id'):
                        return jsonify({"success": False, "error": "缺少必要参数: ppt_id 和 design_id"}), 400
                    
                    # 检查是否提供了幻灯片数据
                    if not data.get('slides_data'):
                        return jsonify({"success": False, "error": "缺少必要参数: slides_data (幻灯片数据)"}), 400
                else:
                    # 单个视频生成模式
                    if not data.get('text'):
                        return jsonify({"success": False, "error": "缺少文本内容"}), 400
                
                # 获取avatar_id，默认使用avator_10
                avatar_id = data.get('avatar_id', 'avator_10')
                
                # 处理voice_model_id - 检查是否为字符串
                voice_model_id = data.get('voice_model_id', 'wang001')
                # 确保它是字符串，如果是对象则取其id字段
                if isinstance(voice_model_id, dict) and 'id' in voice_model_id:
                    voice_model_id = voice_model_id['id']
                
                # 强制设置为标准模式，忽略前端传来的prefer_mode
                prefer_mode = "standard"
                
                # 加载配置
                config = load_config()
                output_dir_base = config.get("paths", {}).get("output_video_dir", "./outputs")
                os.makedirs(output_dir_base, exist_ok=True)
                
                # 创建任务ID
                task_id = "ppt_video_" + uuid.uuid4().hex[:8]
                
                # 如果是单个视频，构建输出路径
                if not data.get('process_all'):
                    # 为单个PPT幻灯片视频创建特定输出路径
                    design_id = data.get('design_id', '')
                    ppt_id = data.get('ppt_id', '')
                    # 正确获取page_index的方法
                    if data.get('slides_data') and len(data.get('slides_data')) > 0:
                        page_index = data.get('slides_data')[0].get('page_index', 1)
                    else:
                        page_index = data.get('page_index', 1)  # 兼容直接在顶层传递page_index的情况
                    
                    # --- Robust Path Construction --- 
                    # Define expected suffixes
                    video_suffix = os.path.join("resources", "videos")
                    subtitle_suffix = os.path.join("resources", "subtitles")

                    # Determine video output directory
                    if output_dir_base.replace('\\', '/').endswith(video_suffix.replace('\\', '/')):
                        # Config path already includes the suffix
                        videos_dir = output_dir_base
                        # Infer base path for subtitles (go up two levels from video dir)
                        subtitle_base_dir = os.path.dirname(os.path.dirname(videos_dir))
                    else:
                        # Config path is the base, append suffix
                        videos_dir = os.path.join(output_dir_base, video_suffix)
                        subtitle_base_dir = output_dir_base
                    
                    # Determine subtitle directory
                    subtitles_dir = os.path.join(subtitle_base_dir, subtitle_suffix)

                    # Create directories
                    os.makedirs(videos_dir, exist_ok=True)
                    os.makedirs(subtitles_dir, exist_ok=True)
                    
                    # 创建符合前端预期的输出文件名和字幕文件名
                    output_filename = design_id + "_" + ppt_id + "_" + str(page_index) + ".mp4"
                    subtitle_filename = design_id + "_" + ppt_id + "_" + str(page_index) + ".vtt"

                    # Construct final paths
                    output_path = os.path.join(videos_dir, output_filename)
                    subtitle_path = os.path.join(subtitles_dir, subtitle_filename)
                    # --- End Robust Path Construction ---
                    
                    # 准备任务数据
                    task_data = {
                        "text": data.get("text", ""),
                        "avatar_id": avatar_id,
                        "output_path": output_path, # Use corrected path
                        "subtitle_path": subtitle_path, # Use corrected path
                        "design_id": design_id,
                        "ppt_id": ppt_id,
                        "page_index": page_index,
                        "add_subtitle": data.get("add_subtitle", True),
                        "subtitle_format": data.get("subtitle_format", "vtt"),
                        "subtitle_encoding": data.get("subtitle_encoding", "utf-8"),
                        "burn_subtitles": data.get("burn_subtitles", False),
                        "speech_rate": data.get("speech_rate", 0.9),
                        "cn_speaking_rate": data.get("cn_speaking_rate", 4.1),
                        "en_word_rate": data.get("en_word_rate", 1.3),
                        "align_subtitles": data.get("align_subtitles", True),
                        "subtitle_offset": data.get("subtitle_offset", -0.3),
                        "text_cut_method": data.get("text_cut_method", "cut2"),
                        "voice_model_id": voice_model_id,
                        "prefer_mode": prefer_mode,  # 强制使用标准模式
                        "mode": "single"
                    }
                else:
                    # 批量处理模式
                    # --- Robust Batch Path --- 
                    video_suffix = os.path.join("resources", "videos")
                    batch_output_base = output_dir_base
                    if output_dir_base.replace('\\', '/').endswith(video_suffix.replace('\\', '/')):
                        # Config path included video suffix, go up two levels for resource base
                         batch_output_base = os.path.dirname(os.path.dirname(output_dir_base))
                    # --- End Robust Batch Path ---
                    task_data = {
                        "avatar_id": avatar_id,
                        "design_id": data.get("design_id", ""),
                        "ppt_id": data.get("ppt_id", ""),
                        "output_dir": os.path.join(batch_output_base, "resources"), # Corrected base for batch
                        "add_subtitle": data.get("add_subtitle", True),
                        "subtitle_format": data.get("subtitle_format", "vtt"),
                        "subtitle_encoding": data.get("subtitle_encoding", "utf-8"),
                        "burn_subtitles": data.get("burn_subtitles", False),
                        "speech_rate": data.get("speech_rate", 0.9),
                        "align_subtitles": data.get("align_subtitles", True),
                        "subtitle_offset": data.get("subtitle_offset", -0.3),
                        "text_cut_method": data.get("text_cut_method", "cut2"),
                        "voice_model_id": voice_model_id,
                        "prefer_mode": prefer_mode,  # 强制使用标准模式
                        "process_all": True,
                        "mode": "batch",
                        "slides_data": data.get("slides_data", [])  # 接收前端传来的幻灯片数据
                    }
                
                # 移除所有None值的参数，避免无效参数传递
                task_data = {k: v for k, v in task_data.items() if v is not None}
                
                # 创建任务
                create_task(task_id, "generate_ppt_video", task_data, total_items=1)
                
                # 启动处理线程
                threading.Thread(
                    target=process_ppt_video_task,
                    args=(task_id, task_data),
                    daemon=True
                ).start()
                
                return jsonify({
                    "success": True,
                    "message": "PPT视频生成任务已提交",
                    "task_id": task_id
                })
                
            except Exception as e:
                import traceback
                logger.error("PPT视频生成API处理失败: " + str(e))
                logger.error(traceback.format_exc())  # 打印完整的堆栈跟踪
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @app.route('/list_avatars', methods=['GET'])
        def list_avatars_api():
            """获取数字人列表API - 路径与前端匹配"""
            try:
                avatars = list_avatars()
                # 只返回ID列表以匹配前端预期
                avatar_ids = [avatar['id'] for avatar in avatars]
                return jsonify({
                    "success": True,
                    "avatars": avatar_ids
                })
            except Exception as e:
                logger.error("获取数字人列表失败: " + str(e))
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @app.route('/list_voice_models', methods=['GET'])
        def list_voice_models_api():
            """获取语音模型列表API"""
            try:
                # 加载配置
                config = load_config()
                tts_config = config.get("tts", {})
                
                # 从配置文件中读取模型列表
                models_config = tts_config.get("models", {})
                voice_models = []
                
                # 使用配置文件中的模型定义
                for model_id, model_config in models_config.items():
                    voice_models.append({
                        "id": model_id,
                        "name": model_config.get("speaker_name", model_id)  # 使用speaker_name或id
                    })
                
                # 如果配置文件中没有找到模型，使用硬编码的备用模型
                if not voice_models:
                    logger.warning("配置文件中未找到TTS模型，使用默认模型")
                    voice_models = [
                        {"id": "wang001", "name": "王老师（标准）"},
                        {"id": "5zu", "name": "5zu（黄雅雯）"},
                        {"id": "6zu", "name": "6zu（熊诗宇）"}
                    ]
                
                # 尝试从TTS API获取模型列表（保留现有逻辑）
                try:
                    tts_server_url = tts_config.get("server", "http://192.168.202.10:9880")
                    response = requests.get("{}/list_voice_models".format(tts_server_url), timeout=5)
                    if response.status_code == 200:
                        api_models = response.json().get("models", [])
                        if api_models:
                            logger.info("从TTS API获取到{}个模型".format(len(api_models)))
                            return jsonify({
                                "success": True,
                                "models": api_models
                            })
                except Exception as api_e:
                    logger.warning("无法从TTS API获取模型列表: {}".format(str(api_e)))
                
                return jsonify({
                    "success": True,
                    "models": voice_models
                })
            except Exception as e:
                logger.error("获取语音模型列表失败: " + str(e))
                return jsonify({
                    "success": False, 
                    "error": str(e)
                }), 500
                
        @app.route('/task_progress/<task_id>', methods=['GET'])
        def task_progress_api(task_id):
            """获取任务进度API - 路径与前端匹配"""
            try:
                status = get_task_status(task_id)
                if not status:
                    return jsonify({"success": False, "error": "任务不存在"}), 404
                
                # 调整返回格式以匹配前端期望
                response_data = {
                    "success": True,
                    "task_id": task_id,
                    "status": status.get("status", "unknown"),
                    "message": status.get("message", ""),
                    "completed": status.get("completed", 0),
                    "total": status.get("total_items", 1),
                    "results": []
                }
                
                # 如果有结果和错误信息，添加到结果列表
                if status.get("status") == "completed" and status.get("result"):
                    response_data["results"] = [status.get("result")]
                    response_data["failed_count"] = 0
                elif status.get("status") == "failed" and status.get("error"):
                    response_data["results"] = [{"reason": status.get("error")}]
                    response_data["failed_count"] = 1
                
                return jsonify(response_data)
            except Exception as e:
                logger.error("获取任务状态失败: " + str(e))
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @app.route('/api/check_environment', methods=['GET'])
        def check_environment_api():
            """检查环境API"""
            try:
                level = request.args.get('level', 'basic')
                results = check_environment(level)
                return jsonify({
                    "success": True,
                    "environment": results
                })
            except Exception as e:
                logger.error("检查环境失败: " + str(e))
                return jsonify({
                    "success": False, 
                    "error": str(e)
                }), 500
                
        # 定期清理任务
        def cleanup_thread():
            while True:
                try:
                    clean_completed_tasks()
                except Exception as e:
                    logger.error("清理任务失败: " + str(e))
                time.sleep(1800)  # 每30分钟清理一次
                
        threading.Thread(target=cleanup_thread, daemon=True).start()
        
        # 启动服务
        logger.info("启动Web服务: http://" + host + ":" + str(port))
        app.run(host=host, port=port, debug=False, threaded=True)
        
    except ImportError as e:
        logger.error("启动服务失败: 缺少必要的依赖包。错误: " + str(e))
        print("提示: 可以使用以下命令安装必要的依赖包:")
        print("pip install flask flask-cors")

# 修改prepare_params_for_generator函数，确保字幕参数被正确保留。
def prepare_params_for_generator(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    将请求参数转换为AvatarGenerator可接受的格式
    
    Args:
        params: 原始参数字典
        
    Returns:
        处理后的参数字典
    """
    result = params.copy()
    
    # 确保prefer_mode存在且为standard，这是我们目前唯一确定可工作的模式
    # 无论前端传什么，我们都强制使用standard模式
    if "prefer_mode" not in result or result["prefer_mode"] != "standard":
        result["prefer_mode"] = "standard"
    
    # 确保mode参数没有直接被传递，因为它应该是枚举对象
    # mode应该由AvatarGenerator根据prefer_mode自动选择
    if "mode" in result:
        # 删除mode参数，我们只使用prefer_mode
        del result["mode"]
    
    # 处理 voice_model_id，如果存在
    original_voice_model_id = result.get("voice_model_id") # 保存原始值以供日志记录
    logger.info("prepare_params - Received voice_model_id: {}".format(original_voice_model_id))

    if "voice_model_id" in result:
        voice_model_id = result.pop("voice_model_id") # 取出并删除原始键
        processed_tts_model_id = None

        # 检查类型并处理
        if isinstance(voice_model_id, dict) and "id" in voice_model_id:
            processed_tts_model_id = voice_model_id["id"]
        elif isinstance(voice_model_id, str) and voice_model_id.strip(): # 确保是有效字符串
            processed_tts_model_id = voice_model_id.strip()
        elif voice_model_id is not None: # 处理其他非空类型，尝试转为字符串
            try:
                processed_tts_model_id = str(voice_model_id)
                logger.warning("prepare_params - Converted non-string voice_model_id to string: {}".format(processed_tts_model_id))
            except Exception:
                 logger.warning("prepare_params - Could not convert voice_model_id to string: {}".format(voice_model_id))

        # 只有在成功处理得到有效 ID 时才设置 tts_model
        if processed_tts_model_id:
            # 从配置获取模型映射
            config = load_config()
            tts_config = config.get("tts", {})
            model_mapping = tts_config.get("model_mapping", {})
            
            # 在日志中记录所有可用的模型映射关系
            logger.debug("可用的模型映射关系: {}".format(model_mapping))
            
            # 如果ID在映射中，使用映射后的ID
            if processed_tts_model_id in model_mapping:
                mapped_id = model_mapping[processed_tts_model_id]
                logger.info("模型ID映射: {} -> {}".format(processed_tts_model_id, mapped_id))
                processed_tts_model_id = mapped_id
            
            result["tts_model"] = processed_tts_model_id
            result["tts_voice_name"] = processed_tts_model_id # 保持一致性
            logger.info("prepare_params - Set tts_model based on voice_model_id: {}".format(processed_tts_model_id))
            
            # 加载模型特定配置
            models = tts_config.get("models", {})
            
            # 如果模型ID存在于配置中，获取其路径信息
            if processed_tts_model_id in models:
                model_config = models[processed_tts_model_id]
                # 添加所有模型特定参数到结果中
                for key, value in model_config.items():
                    # 只添加非空值
                    if value is not None:
                        result[key] = value
                        if key in ["t2s_weights_path", "vits_weights_path", "cn_speaking_rate", "en_word_rate"]:
                            logger.info("prepare_params - Added {} from model config: {}".format(key, value))

    # 确保TTS参数存在，如果未通过 voice_model_id 设置，则使用默认值
    if "tts_model" not in result or not result["tts_model"]:
        # 从配置中获取默认TTS模型
        config = load_config()
        tts_config = config.get("tts", {})
        default_tts_model = tts_config.get("default_model", "wang001")
        result["tts_model"] = default_tts_model
        result["tts_voice_name"] = default_tts_model
        logger.info("prepare_params - tts_model was missing or invalid, set to default: {}".format(default_tts_model))
        
        # 添加默认模型的配置
        models = tts_config.get("models", {})
        if default_tts_model in models:
            model_config = models[default_tts_model]
            for key, value in model_config.items():
                if value is not None and key not in result:
                    result[key] = value
    
    # 强制添加语音参数 
    result["use_tts"] = True  # 确保使用TTS而不是静音
    
    # 处理数值型参数，确保它们是数字而不是字符串
    numeric_params = ["speech_rate", "cn_speaking_rate", "en_word_rate", "subtitle_offset"]
    for param in numeric_params:
        if param in result and isinstance(result[param], str):
            try:
                result[param] = float(result[param])
            except ValueError:
                # 如果无法转换为浮点数，则保持原样
                pass
    
    # 处理布尔型参数，确保它们是布尔值而不是字符串
    bool_params = ["add_subtitle", "burn_subtitles", "align_subtitles", "subtitle"]
    for param in bool_params:
        if param in result and isinstance(result[param], str):
            result[param] = result[param].lower() in ["true", "yes", "1", "t", "y"]
    
    # 处理文本切分方法参数 - 添加此部分代码 - 关键修复
    if "text_cut_method" in result:
        # 从text_cut_method获取值并添加为text_split_method
        text_cut_method = result["text_cut_method"]
        result["text_split_method"] = text_cut_method
        logger.info("prepare_params - 将text_cut_method映射为text_split_method: {}".format(text_cut_method))
    
    # 合并字幕参数，确保add_subtitle和subtitle参数一致
    if "subtitle" in result and result["subtitle"] and "add_subtitle" not in result:
        result["add_subtitle"] = True
    elif "add_subtitle" in result and result["add_subtitle"] and "subtitle" not in result:
        result["subtitle"] = True
    
    # 特别处理字幕相关参数
    if ("add_subtitle" in result and result["add_subtitle"]) or ("subtitle" in result and result["subtitle"]):
        # 确保add_subtitle是True
        result["add_subtitle"] = True
        result["subtitle"] = True
        
        # 检查是否有subtitle_path参数
        if "subtitle_path" in result and result["subtitle_path"]:
            # 确保subtitle_path是绝对路径
            subtitle_path = result["subtitle_path"]
            if not os.path.isabs(subtitle_path):
                subtitle_path = os.path.abspath(subtitle_path)
                result["subtitle_path"] = subtitle_path
            
            logger.info("prepare_params - 启用字幕生成，字幕路径: {}".format(subtitle_path))
            
            # 确保字幕目录存在
            subtitle_dir = os.path.dirname(subtitle_path)
            os.makedirs(subtitle_dir, exist_ok=True)
        else:
            logger.warning("prepare_params - 启用了字幕生成，但未指定字幕路径")
    
    # 移除冗余的model_id参数，只使用tts_model参数避免混淆
    if "model_id" in result:
        # 检查是否与tts_model一致，不一致时记录警告
        if "tts_model" in result and result["model_id"] != result["tts_model"]:
            logger.warning("prepare_params - 检测到不一致的模型ID: model_id={}, tts_model={}，将使用tts_model".format(
                result["model_id"], result["tts_model"]))
        # 移除model_id，避免API参数冲突
        result.pop("model_id")
    
    # 日志记录最终参数中的关键值
    logger.info("prepare_params - Final prefer_mode: {}, Final tts_model: {}, add_subtitle: {}, subtitle_path: {}, text_split_method: {}".format(
        result.get("prefer_mode"), 
        result.get("tts_model"), 
        result.get("add_subtitle", False),
        result.get("subtitle_path", "None"),
        result.get("text_split_method", "None")  # 添加text_split_method参数到日志
    ))
    
    return result

# 添加一个自定义的函数来处理FFmpeg合并视频和音频
def create_video_from_frames_and_audio(frames_pattern, audio_path, output_path, fps=25):
    """使用兼容参数从图片帧创建视频"""
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用不包含crf选项的命令
        video_cmd = [
            "ffmpeg", "-y", "-v", "warning",
            "-r", str(fps),
            "-f", "image2",
            "-i", frames_pattern,
            "-vcodec", "libx264", 
            "-preset", "medium",  # 使用preset替代crf
            "-pix_fmt", "yuv420p",
            output_path
        ]
        
        logger.info("Executing FFmpeg command: {}".format(" ".join(video_cmd)))
        subprocess.run(video_cmd, check=True)
        
        # 检查视频文件生成
        if not os.path.exists(output_path):
            logger.error("Video file not generated: {}".format(output_path))
            return False
            
        logger.info("Video frames merged successfully: {}".format(output_path))
        
        # 合并视频和音频
        if audio_path and os.path.exists(audio_path):
            final_output = output_path.replace(".temp.mp4", ".mp4")
            merge_cmd = [
                "ffmpeg", "-y", "-v", "warning",
                "-i", audio_path,
                "-i", output_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                final_output
            ]
            
            logger.info("Executing FFmpeg audio merge command: " + " ".join(merge_cmd))
            subprocess.run(merge_cmd, check=True)
            
            if os.path.exists(final_output):
                # logger.info(f"最终视频生成成功: {final_output}")
                logger.info("Final video generated successfully: {}".format(final_output))
                return True
        
        return True
        
    except subprocess.CalledProcessError as e:
        # logger.error(f"FFmpeg命令执行失败: {str(e)}")
        logger.error("FFmpeg command failed: {}".format(str(e)))
        return False
    except Exception as e:
        # logger.error(f"创建视频时出错: {str(e)}")
        logger.error("Error creating video: {}".format(str(e)))
        return False

# 添加一个钩子函数，覆盖AvatarGenerator的视频生成过程
def custom_video_generation(text, avatar_id, output_path, audio_path=None, **kwargs):
    """
    自定义视频生成过程，绕过AvatarGenerator的问题
    直接使用我们的FFmpeg命令
    
    Args:
        text: 文本内容
        avatar_id: 虚拟形象ID
        output_path: 输出路径
        audio_path: 音频路径
        **kwargs: 其他参数
    
    Returns:
        生成结果
    """
    try:
        # 加载配置以获取路径
        config = load_config()
        project_root = Path(__file__).parent.parent.parent.resolve()
        default_avatar_dir = "./results/v15/avatars"
        resolved_avatar_base_dir = resolve_path(
            config.get("paths", {}).get("avatar_dir", default_avatar_dir),
            project_root
        )

        # 假设TTS已经生成了音频
        if not audio_path or not os.path.exists(audio_path):
            logger.error("未找到音频文件: " + str(audio_path))
            return {"success": False, "error": "音频文件不存在"}
            
        # 尝试查找已生成的帧 (using resolved path)
        avatar_dir = os.path.join(resolved_avatar_base_dir, avatar_id) # Construct full path to specific avatar
        # avatar_dir = "./results/v15/avatars/{}".format(avatar_id) # OLD: Hardcoded relative
        tmp_dir = os.path.join(avatar_dir, "tmp") 

        if not os.path.exists(tmp_dir):
            logger.error("Could not find temporary frame directory: {}".format(tmp_dir))
            return {"success": False, "error": "临时帧目录不存在"}
            
        # 构建帧模式
        frames_pattern = os.path.join(tmp_dir, "%08d.png")
        
        # 确保vid_output目录存在 (using resolved path)
        # output_dir = os.path.join("./results/v15/avatars/{}".format(avatar_id), "vid_output") # OLD: Hardcoded relative
        output_dir = os.path.join(avatar_dir, "vid_output")
        os.makedirs(output_dir, exist_ok=True)

        # 使用正确的输出路径 (output_path is the final desired location, name comes from there)
        final_output_filename = os.path.basename(output_path)
        video_gen_path = os.path.join(output_dir, final_output_filename) # Path where ffmpeg initially creates the video
        
        # 直接合成最终视频 (Pass the path where ffmpeg should create it)
        if create_video_from_frames_and_audio(frames_pattern, audio_path, video_gen_path):
            # If successful, the video is at video_gen_path. We want it at output_path.
            # If they are the same, we are done. If different, copy/move.
            # Note: create_video_from_frames_and_audio already handles merging
            # The path returned should ideally be the final merged video path
            # Assuming create_video_from_frames_and_audio puts the final video at video_gen_path
            
            # Ensure the final destination directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Copy or move the generated video to the final requested output path if different
            if os.path.abspath(video_gen_path) != os.path.abspath(output_path):
                import shutil
                try:
                    shutil.copy2(video_gen_path, output_path)
                    logger.info("Custom video copied to final path: {}".format(output_path))
                except Exception as copy_err:
                    logger.error("Failed to copy custom video from {} to {}: {}".format(video_gen_path, output_path, copy_err))
                    # Return failure if copy failed, as user expects file at output_path
                    return {"success": False, "error": "Failed to place video at final destination"}
            else:
                logger.info("Custom video generated successfully at final path: {}".format(output_path))

            # Return success with the *requested* output_path
            return {
                "success": True,
                "path": output_path, # Return the path the user requested
                "mode": "standard",
                "custom_fallback": True
            }
        else:
            return {"success": False, "error": "合成视频失败"}
            
    except Exception as e:
        logger.error("自定义视频生成失败: " + str(e))
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

# 修改process_ppt_video_task函数，添加错误处理和自定义生成逻辑
def process_ppt_video_task(task_id: str, data: Dict[str, Any]) -> None:
    """处理PPT视频生成任务"""
    try:
        update_task_progress(task_id, "processing", 10, "初始化生成器")
        
        # 加载配置
        config = load_config()
        
        # 记录所有关键参数以便于调试
        logger.info("处理PPT视频任务，参数: " + str(data))
        
        # 初始化生成器
        generator = AvatarGenerator(config=config)
        
        # 判断是单个视频生成还是批量生成
        if data.get("mode") == "single":
            update_task_progress(task_id, "processing", 20, "开始生成单个PPT视频")
            
            # 提取必需参数
            text = data.pop("text", "")  # 从参数字典中取出text并删除，避免重复传入
            
            # 打印文本内容，确保不为空
            logger.info("要转换的文本内容: [" + text + "]，长度: " + str(len(text)))
            if not text:
                raise ValueError("文本内容为空，无法生成语音")
            
            # 从参数中获取avatar_id
            avatar_id = data.pop("avatar_id", "avator_10")  # 从参数字典中取出avatar_id并删除
            logger.info("使用数字人模型ID: " + avatar_id)
            
            output_path = data.pop("output_path", "")  # 从参数字典中取出output_path并删除
            
            # 打印输出路径
            logger.info("输出视频路径: " + output_path)
            
            # 强制使用标准模式
            data["prefer_mode"] = "standard"
            
            # 处理文本切分方法 - 不需要传递给generator，但我们保留在日志中
            text_cut_method = data.pop("text_cut_method", "cut2")
            
            # 处理字幕生成 - 确保字幕路径被正确设置
            subtitle_path = data.get("subtitle_path")
            data["add_subtitle"] = data.get("add_subtitle", True)  # 默认添加字幕
            
            if data["add_subtitle"]:
                logger.info("字幕功能已启用，字幕路径: " + str(subtitle_path))
                if not subtitle_path:
                    # 如果未提供字幕路径，生成一个
                    subtitle_format = data.get("subtitle_format", "vtt")
                    subtitle_path = os.path.splitext(output_path)[0] + "." + subtitle_format
                    data["subtitle_path"] = subtitle_path
                    logger.info("自动生成字幕路径: " + subtitle_path)
                
                # 确保字幕目录存在
                subtitle_dir = os.path.dirname(subtitle_path)
                os.makedirs(subtitle_dir, exist_ok=True)
            else:
                logger.info("字幕功能未启用")
            
            # 准备最终参数
            final_params = prepare_params_for_generator(data)
            
            # 确保字幕参数被保留
            if data["add_subtitle"]:
                final_params["add_subtitle"] = True
                final_params["subtitle_path"] = subtitle_path
                final_params["subtitle_format"] = data.get("subtitle_format", "vtt")
                final_params["subtitle_offset"] = data.get("subtitle_offset", -0.3)
                final_params["align_subtitles"] = data.get("align_subtitles", True)
            
            # 打印最终的参数，便于调试
            logger.info("生成视频最终参数: text长度={0}, avatar_id={1}, tts_model={2}, prefer_mode={3}, add_subtitle={4}, subtitle_path={5}".format(
                len(text), avatar_id, final_params.get("tts_model"), final_params.get("prefer_mode"), 
                final_params.get("add_subtitle", False), final_params.get("subtitle_path", "无")))
            
            # 生成视频前检查文件夹权限
            output_dir = os.path.dirname(output_path)
            if not os.access(output_dir, os.W_OK):
                logger.warning("警告：输出目录 " + output_dir + " 可能没有写入权限")
                os.makedirs(output_dir, exist_ok=True)
            
            # 记录当前时间戳，用于后续查找音频文件
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            
            # 生成视频
            logger.info("开始调用AvatarGenerator.generate_video生成视频...")
            result = generator.generate_video(
                text=text,
                avatar_id=avatar_id,
                output_path=output_path,
                **final_params
            )
            
            # 检查结果
            if result.get("success", False):
                # 视频生成成功
                video_url = "resources/videos/" + data.get('design_id') + "_" + data.get('ppt_id') + "_" + str(data.get('page_index')) + ".mp4"
                subtitle_url = "resources/subtitles/" + data.get('design_id') + "_" + data.get('ppt_id') + "_" + str(data.get('page_index')) + ".vtt"
                
                # 如果是标准模式且启用了字幕，但字幕文件不存在或为空，手动生成字幕
                if data["add_subtitle"] and result.get("mode") == "standard":
                    if not os.path.exists(subtitle_path) or os.path.getsize(subtitle_path) == 0:
                        logger.info("标准模式没有生成字幕，尝试手动生成...")
                        # 获取音频路径（从结果中或查找最新生成的）
                        audio_path = result.get("audio_path")
                        if not audio_path:
                            # 尝试查找temp目录中的音频文件
                            temp_dir = "/mnt/part2/Dteacher/MuseTalk/temp"
                            audio_files = [f for f in os.listdir(temp_dir) if f.startswith("tts_") and f.endswith(".wav")]
                            if audio_files:
                                # 按修改时间排序，选择最新的
                                audio_files.sort(key=lambda x: os.path.getmtime(os.path.join(temp_dir, x)), reverse=True)
                                audio_path = os.path.join(temp_dir, audio_files[0])
                                logger.info("找到最新TTS音频文件: " + audio_path)
                        
                        if audio_path and os.path.exists(audio_path):
                            # 生成字幕
                            subtitle_offset = data.get("subtitle_offset", -0.3)
                            align_subtitles = data.get("align_subtitles", True)
                            text_cut_method = data.get("text_cut_method", "cut2")
                            
                            logger.info("开始手动生成字幕，文本: {}，长度: {}，使用切分方法: {}".format(
                                text, len(text), text_cut_method))
                            
                            # 调用TTS服务生成字幕
                            success, sub_result = generator.tts_service.generate_subtitle(
                                text=text,
                                output_path=subtitle_path,
                                format=data.get("subtitle_format", "vtt"),
                                encoding="utf-8",
                                offset=subtitle_offset,
                                subtitle_speed=1.0,
                                text_cut_method=text_cut_method
                            )
                            
                            if success:
                                logger.info("手动生成字幕成功: " + subtitle_path)
                            else:
                                logger.warning("手动生成字幕失败: " + str(sub_result))
                
                result["video_url"] = video_url
                result["subtitle_url"] = subtitle_url
                
                update_task_progress(task_id, "completed", 100, "视频生成成功", result=result)
            else:
                # 视频生成失败，尝试自定义逻辑
                error_msg = result.get("error", "未知错误")
                logger.warning("视频生成失败，尝试自定义逻辑。错误: " + error_msg)
                
                # 尝试查找已生成的音频文件
                temp_dir = "/mnt/part2/Dteacher/MuseTalk/temp"
                # 查找最新的TTS音频文件
                audio_files = [f for f in os.listdir(temp_dir) if f.startswith("tts_") and f.endswith(".wav")]
                if audio_files:
                    # 按修改时间排序，选择最新的
                    audio_files.sort(key=lambda x: os.path.getmtime(os.path.join(temp_dir, x)), reverse=True)
                    audio_path = os.path.join(temp_dir, audio_files[0])
                    logger.info("找到最新TTS音频文件: " + audio_path)
                    
                    # 使用自定义逻辑尝试生成视频
                    custom_result = custom_video_generation(
                        text=text,
                        avatar_id=avatar_id,
                        output_path=output_path,
                        audio_path=audio_path
                    )
                    
                    if custom_result.get("success", False):
                        # 自定义逻辑成功
                        video_url = "resources/videos/" + data.get('design_id') + "_" + data.get('ppt_id') + "_" + str(data.get('page_index')) + ".mp4"
                        subtitle_url = "resources/subtitles/" + data.get('design_id') + "_" + data.get('ppt_id') + "_" + str(data.get('page_index')) + ".vtt"
                        
                        # 尝试生成字幕（如果启用了字幕功能）
                        if data["add_subtitle"]:
                            logger.info("自定义视频生成成功，尝试生成字幕...")
                            subtitle_offset = data.get("subtitle_offset", -0.3)
                            text_cut_method = data.get("text_cut_method", "cut2")
                            
                            # 调用TTS服务生成字幕
                            success, sub_result = generator.tts_service.generate_subtitle(
                                text=text,
                                output_path=subtitle_path,
                                format=data.get("subtitle_format", "vtt"),
                                encoding="utf-8",
                                offset=subtitle_offset,
                                subtitle_speed=1.0,
                                text_cut_method=text_cut_method
                            )
                            
                            if success:
                                logger.info("自定义视频的字幕生成成功: " + subtitle_path)
                            else:
                                logger.warning("自定义视频的字幕生成失败: " + str(sub_result))
                        
                        custom_result["video_url"] = video_url
                        custom_result["subtitle_url"] = subtitle_url
                        custom_result["audio_path"] = audio_path
                        
                        update_task_progress(task_id, "completed", 100, "使用自定义逻辑生成视频成功", result=custom_result)
                    else:
                        # 自定义逻辑也失败
                        update_task_progress(task_id, "failed", 100, "视频生成失败，自定义逻辑也失败", 
                                          error=custom_result.get("error", "未知错误"))
                else:
                    # 没找到音频文件
                    update_task_progress(task_id, "failed", 100, "视频生成失败，未找到TTS音频文件", error=error_msg)
        
        else:
            # 批量生成模式，调用批量处理函数
            process_batch_ppt_videos(task_id, data)
    
    except Exception as e:
        import traceback
        logger.error("处理PPT视频任务失败: " + str(e))
        logger.error(traceback.format_exc())  # 打印完整的堆栈跟踪
        update_task_progress(task_id, "failed", 0, "处理任务失败", error=str(e))

# 修改批量处理PPT视频函数，修复TTS模型处理
def process_batch_ppt_videos(task_id: str, data: Dict[str, Any]) -> None:
    """批量处理PPT视频生成"""
    try:
        # 获取设计ID和PPT ID
        design_id = data.get("design_id", "")
        ppt_id = data.get("ppt_id", "")
        
        # 强制使用标准模式
        data["prefer_mode"] = "standard"
        
        # 确保TTS参数正确
        if "voice_model_id" in data:
            voice_model_id = data["voice_model_id"]
            # 确保它是字符串，如果是对象则取其id字段
            if isinstance(voice_model_id, dict) and 'id' in voice_model_id:
                voice_model_id = voice_model_id['id']
            # 直接使用字符串ID
            data["tts_model"] = voice_model_id
            data["tts_voice_name"] = voice_model_id
        else:
            # 默认使用wang001模型
            data["voice_model_id"] = "wang001"
            data["tts_model"] = "wang001"
            data["tts_voice_name"] = "wang001"
        
        # 确保中英文语速参数存在
        if "cn_speaking_rate" not in data:
            data["cn_speaking_rate"] = 4.1  # 默认中文语速
        
        if "en_word_rate" not in data:
            data["en_word_rate"] = 1.3  # 默认英文语速
            
        # 确保speech_rate参数存在
        if "speech_rate" not in data:
            data["speech_rate"] = 0.9  # 默认语音速度
          
        # 强制使用TTS而不是静音
        data["use_tts"] = True
        
        # 记录批量处理的初始参数
        logger.info("开始批量处理PPT视频，参数: " + str(data))
        
        # 获取前端传递的幻灯片数据
        slides = data.get("slides_data", [])
        
        # 如果没有传递幻灯片数据，使用备用方法获取
        if not slides:
            logger.warning("前端未传递幻灯片数据，尝试使用备用方法获取...")
            slides = get_ppt_slides(ppt_id, design_id)
        
        if not slides:
            raise Exception("没有找到幻灯片数据")
        
        # 更新总任务数量
        total_slides = len(slides)
        update_task_progress(task_id, "processing", 0, "开始处理 " + str(total_slides) + " 个幻灯片", None, None, {"total_items": total_slides})
        
        # 尝试导入缓存管理器
        cache_manager = None
        try:
            from musetalk_wrapper.core.model_cache import ModelCache
            cache_manager = ModelCache.get_instance()
            logger.info("成功导入缓存管理器，将在批量处理中使用缓存加速")
        except Exception as cache_error:
            logger.warning("无法导入缓存管理器，不使用缓存: {}".format(cache_error))
        
        # 加载配置
        config = load_config()
        
        # 初始化生成器
        generator = AvatarGenerator(config=config)
        
        # 创建输出目录
        output_dir = data.get("output_dir", "./outputs/resources")
        videos_dir = os.path.join(output_dir, "videos")
        subtitles_dir = os.path.join(output_dir, "subtitles")
        os.makedirs(videos_dir, exist_ok=True)
        os.makedirs(subtitles_dir, exist_ok=True)
        
        # 确保获取avatar_id，这是一个必需的基本参数
        avatar_id = data.pop("avatar_id", "avator_10")
        if isinstance(avatar_id, dict) and "id" in avatar_id:
            avatar_id = avatar_id["id"]
        
        # 建立基本参数
        base_params = data.copy()
        # 移除不需要传递给生成器的参数
        for key in ["mode", "process_all", "design_id", "ppt_id", "output_dir", "slides_data"]:
            if key in base_params:
                base_params.pop(key)
        
        # 添加avatar_id回来，因为它会单独传递给generate_video函数
        base_params["avatar_id"] = avatar_id
        
        # 确保使用标准模式
        base_params["prefer_mode"] = "standard"
        
        # 确保使用TTS而不是静音
        base_params["use_tts"] = True
        
        # 处理每个幻灯片
        results = []
        failed_count = 0
        
        for idx, slide in enumerate(slides):
            try:
                # 检查是否有脚本
                script = slide.get("script")
                if not script or script.strip() == "":
                    continue  # 跳过没有脚本的幻灯片
                
                # 确保脚本内容不为空
                logger.info("幻灯片 " + str(idx + 1) + " 文本内容: [" + script + "]，长度: " + str(len(script)))
                
                page_index = slide.get("page_index", idx + 1)
                
                # 创建输出文件路径
                output_filename = design_id + "_" + ppt_id + "_" + str(page_index) + ".mp4"
                output_filename = output_filename.replace(" ", "_")
                output_path = os.path.join(videos_dir, output_filename)
                
                # 创建字幕文件路径
                subtitle_filename = design_id + "_" + ppt_id + "_" + str(page_index) + ".vtt"
                subtitle_filename = subtitle_filename.replace(" ", "_")
                subtitle_path = os.path.join(subtitles_dir, subtitle_filename)
                
                # 更新进度
                progress_percent = int((idx / total_slides) * 100)
                progress_message = "正在处理第 " + str(page_index) + " 页幻灯片 (" + str(idx+1) + "/" + str(total_slides) + ")"
                update_task_progress(task_id, "processing", progress_percent, progress_message)
                
                # 准备视频生成参数
                slide_params = base_params.copy()
                slide_params["subtitle_path"] = subtitle_path
                
                # 确保使用标准模式
                slide_params["prefer_mode"] = "standard"
                
                # 使用参数处理函数处理参数
                final_params = prepare_params_for_generator(slide_params)
                
                # 日志记录每个幻灯片的生成参数
                logger.info("生成第 " + str(page_index) + " 页幻灯片视频，参数: " + 
                           "text长度=" + str(len(script)) + 
                           ", avatar_id=" + str(avatar_id) + 
                           ", tts_model=" + str(final_params.get("tts_model")) +
                           ", prefer_mode=" + str(final_params.get("prefer_mode")))
                
                # 生成视频前检查文件夹权限
                if not os.access(os.path.dirname(output_path), os.W_OK):
                    logger.warning("警告：输出目录 " + os.path.dirname(output_path) + " 可能没有写入权限")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # 生成视频
                logger.info("开始为幻灯片 " + str(page_index) + " 生成视频...")
                result = generator.generate_video(
                    text=script,
                    output_path=output_path,
                    **final_params
                )
                
                if result.get("success", False):
                    # 添加到成功结果列表
                    result["page_index"] = page_index
                    result["video_url"] = "resources/videos/" + output_filename
                    result["subtitle_url"] = "resources/subtitles/" + subtitle_filename
                    
                    # 检查字幕是否生成，如果是标准模式且字幕文件不存在或为空，则手动生成字幕
                    add_subtitle = final_params.get("add_subtitle", False)
                    if add_subtitle and result.get("mode") == "standard":
                        if not os.path.exists(subtitle_path) or os.path.getsize(subtitle_path) == 0:
                            logger.info("幻灯片 " + str(page_index) + " 标准模式没有生成字幕，尝试手动生成...")
                            # 获取音频路径（从结果中或查找最新生成的）
                            audio_path = result.get("audio_path")
                            if not audio_path:
                                # 尝试查找temp目录中的音频文件
                                temp_dir = "/mnt/part2/Dteacher/MuseTalk/temp"
                                audio_files = [f for f in os.listdir(temp_dir) if f.startswith("tts_") and f.endswith(".wav")]
                                if audio_files:
                                    # 按修改时间排序，选择最新的
                                    audio_files.sort(key=lambda x: os.path.getmtime(os.path.join(temp_dir, x)), reverse=True)
                                    audio_path = os.path.join(temp_dir, audio_files[0])
                                    logger.info("找到最新TTS音频文件: " + audio_path)
                            
                            if audio_path and os.path.exists(audio_path):
                                # 生成字幕
                                subtitle_offset = slide_params.get("subtitle_offset", -0.3)
                                align_subtitles = slide_params.get("align_subtitles", True)
                                text_cut_method = slide_params.get("text_cut_method", "cut2")
                                
                                logger.info("开始为幻灯片 " + str(page_index) + " 手动生成字幕，文本长度: " + str(len(script)) + 
                                           "，使用切分方法: " + text_cut_method)
                                
                                # 调用TTS服务生成字幕
                                success, sub_result = generator.tts_service.generate_subtitle(
                                    text=script,
                                    output_path=subtitle_path,
                                    format=slide_params.get("subtitle_format", "vtt"),
                                    encoding="utf-8",
                                    offset=subtitle_offset,
                                    subtitle_speed=1.0,
                                    text_cut_method=text_cut_method
                                )
                                
                                if success:
                                    logger.info("幻灯片 " + str(page_index) + " 手动生成字幕成功: " + subtitle_path)
                                else:
                                    logger.warning("幻灯片 " + str(page_index) + " 手动生成字幕失败: " + str(sub_result))
                    
                    results.append(result)
                    logger.info("幻灯片 " + str(page_index) + " 视频生成成功")
                else:
                    # 记录失败
                    error_msg = result.get("error", "未知错误")
                    logger.error("幻灯片 " + str(page_index) + " 视频生成失败: " + error_msg)
                    failed_count += 1
                    results.append({
                        "page_index": page_index,
                        "success": False,
                        "error": error_msg
                    })
            except Exception as e:
                # 处理单个幻灯片失败
                import traceback
                logger.error("处理第 " + str(idx + 1) + " 个幻灯片失败: " + str(e))
                logger.error(traceback.format_exc())  # 打印完整的堆栈跟踪
                failed_count += 1
                results.append({
                    "page_index": idx + 1,
                    "success": False,
                    "error": str(e)
                })
        
        # 更新最终状态
        if failed_count == total_slides:
            error_message = "所有 " + str(total_slides) + " 个幻灯片处理失败"
            update_task_progress(task_id, "failed", 100, error_message, error="所有幻灯片处理失败")
        elif failed_count > 0:
            success_message = "批量处理完成，" + str(total_slides - failed_count) + "/" + str(total_slides) + " 个幻灯片成功，" + str(failed_count) + " 个失败"
            update_task_progress(task_id, "completed", 100, success_message, result={"results": results, "failed_count": failed_count})
        else:
            success_message = "批量处理成功，所有 " + str(total_slides) + " 个幻灯片已生成视频"
            update_task_progress(task_id, "completed", 100, success_message, result={"results": results, "failed_count": 0})
        
        # 批量处理结束时，不清除缓存，以便后续任务复用
        logger.info("批量处理结束，保留模型和图片帧缓存以供后续使用")
        
    except Exception as e:
        import traceback
        logger.error("批量处理任务失败: " + str(e))
        logger.error(traceback.format_exc())  # 打印完整的堆栈跟踪
        update_task_progress(task_id, "failed", 0, "批量处理任务失败", error=str(e))

# 添加获取PPT数据的函数
def get_ppt_slides(ppt_id: str, design_id: str = None) -> list:
    """
    [已弃用] 获取PPT幻灯片数据的函数
    现在直接从前端接收幻灯片数据，不再调用此函数
    仅保留作为备份或调试用途
    
    Args:
        ppt_id: PPT文档ID
        design_id: 设计ID
        
    Returns:
        幻灯片列表，每个元素包含幻灯片ID、内容、脚本等信息
    """
    logger.warning("正在调用已弃用的get_ppt_slides函数。建议直接使用前端传递的幻灯片数据。")
    
    # TODO: 在实际环境中实现真正的API调用
    # 这里返回一些测试数据
    return [
        {
            "page_index": 1,
            "script": "这是第一页幻灯片的脚本内容",
            "image_path": "path/to/image1.jpg"
        },
        {
            "page_index": 2,
            "script": "这是第二页幻灯片的脚本内容",
            "image_path": "path/to/image2.jpg"
        },
        {
            "page_index": 3,
            "script": "这是第三页幻灯片的脚本内容",
            "image_path": "path/to/image3.jpg"
        }
    ]

# 添加一个函数检查FFmpeg版本和配置
def check_ffmpeg_config():
    """检查FFmpeg版本和配置"""
    try:
        # 执行ffmpeg -version命令
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, 
                              text=True, 
                              check=False)
        
        if result.returncode == 0:
            # 获取版本信息
            version_info = result.stdout.split('\n')[0]
            logger.info("FFmpeg版本: " + version_info)
            
            # 检查支持的编码器
            encoders = subprocess.run(["ffmpeg", "-encoders"], 
                                    capture_output=True, 
                                    text=True, 
                                    check=False)
            
            # 检查是否支持libx264
            if "libx264" in encoders.stdout:
                logger.info("FFmpeg支持libx264编码器")
            else:
                logger.warning("FFmpeg可能不支持libx264编码器")
                
            # 检查支持的选项
            help_output = subprocess.run(["ffmpeg", "-h", "encoder=libx264"], 
                                       capture_output=True, 
                                       text=True, 
                                       check=False)
            
            # 检查是否支持crf选项
            if "crf" in help_output.stdout:
                logger.info("FFmpeg支持crf选项")
            else:
                logger.warning("FFmpeg可能不支持crf选项")
        else:
            logger.error("FFmpeg命令执行失败: " + result.stderr)
    except Exception as e:
        logger.error("检查FFmpeg配置时出错: " + str(e))

# 添加一个简单的从音频生成视频的函数作为终极备选方案
def create_simple_video_from_audio(audio_path, image_path, output_path, duration=None):
    """
    使用单一静态图像和音频创建简单视频
    
    Args:
        audio_path: 音频文件路径
        image_path: 图片文件路径 (如果不存在，使用默认黑色图像)
        output_path: 输出视频路径
        duration: 视频持续时间，如果为None则使用音频时长
    
    Returns:
        是否成功
    """
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 如果图片不存在，创建一个黑色图像
        temp_image = None
        if not image_path or not os.path.exists(image_path):
            import numpy as np
            import cv2
            # 创建640x480的黑色图像
            temp_image = os.path.join(os.path.dirname(output_path), "temp_black.jpg")
            img = np.zeros((480, 640, 3), np.uint8)
            cv2.imwrite(temp_image, img)
            image_path = temp_image
            logger.info("创建临时黑色图像: " + image_path)
        
        # 获取音频持续时间
        if duration is None:
            try:
                ffprobe_cmd = [
                    "ffprobe", "-v", "error", 
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    audio_path
                ]
                duration_str = subprocess.check_output(ffprobe_cmd, text=True).strip()
                duration = float(duration_str)
                logger.info("音频持续时间: " + str(duration) + " 秒")
            except Exception as e:
                logger.error("获取音频持续时间失败: " + str(e))
                duration = 10  # 默认10秒
        
        # 使用图片和音频创建视频
        video_cmd = [
            "ffmpeg", "-y", "-v", "warning",
            "-loop", "1", 
            "-framerate", "25",
            "-t", str(duration),
            "-i", image_path,
            "-i", audio_path,
            "-c:v", "libx264",
            "-tune", "stillimage",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            "-shortest",
            output_path
        ]
        
        logger.info("执行简单视频生成FFmpeg命令: " + " ".join(video_cmd))
        subprocess.run(video_cmd, check=True)
        
        # 检查视频是否成功生成
        if not os.path.exists(output_path):
            logger.error("简单视频文件未生成: " + output_path)
            return False
        
        logger.info("简单视频生成成功: " + output_path)
        
        # 清理临时文件
        if temp_image and os.path.exists(temp_image):
            os.remove(temp_image)
            
        return True
        
    except Exception as e:
        logger.error("创建简单视频时出错: " + str(e))
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """主程序入口"""
    args = parse_args()
    
    # 如果没有提供命令，直接返回（帮助信息已在parse_args中显示）
    if not args.command:
        return
    
    if args.command == "generate":
        # 生成视频
        result = generate_video(args)
        
        if result.get("success", False):
            logger.info("视频生成成功: " + str(result.get('path')))
            logger.info("使用模式: " + str(result.get('mode')))
        else:
            logger.error("视频生成失败: " + str(result.get('error', '未知错误')))
        
        # 打印结果
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif args.command == "test":
        # 测试功能
        print("\n===== 开始功能测试 =====")
        print("文本: " + args.text)
        print("数字人: " + args.avatar_id)
        print("输出路径: " + args.output)
        print("生成模式: " + args.mode)
        
        # 确保输出目录存在
        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载配置
        config = load_config()
        
        # 生成视频
        test_args = argparse.Namespace(
            text=args.text,
            avatar_id=args.avatar_id,
            output=args.output,
            subtitle=args.subtitle,
            subtitle_color="#FFFFFF",
            audio=None,
            background_music=None,
            volume_ratio=1.0,
            prefer_mode=args.mode,
            gpu_id=0
        )
        
        print("\n生成视频中，请稍候...")
        start_time = time.time()
        
        # 初始化生成器并传递config
        generator = AvatarGenerator(config=config)
        
        # 提取参数
        options = {
            "subtitle": test_args.subtitle,
            "subtitle_color": test_args.subtitle_color
        }
        
        if test_args.audio:
            options["audio_path"] = test_args.audio
            
        if test_args.prefer_mode:
            options["prefer_mode"] = test_args.prefer_mode
            
        # 生成视频    
        result = generator.generate_video(
            text=test_args.text,
            avatar_id=test_args.avatar_id,
            output_path=test_args.output,
            **options
        )
        
        if result.get("success", False):
            elapsed_time = time.time() - start_time
            print("\n测试成功!")
            print("生成时间: " + str(round(elapsed_time, 2)) + " 秒")
            print("视频文件: " + str(result.get('path')))
            print("使用模式: " + str(result.get('mode')))
        else:
            print("\n测试失败: " + str(result.get('error', '未知错误')))
    
    elif args.command == "list-avatars":
        # 列出数字人
        avatars = list_avatars()
        
        if not avatars:
            logger.warning("未找到可用的数字人")
        else:
            logger.info("找到 " + str(len(avatars)) + " 个数字人")
            for avatar in avatars:
                print("ID: " + avatar['id'] + ", 名称: " + avatar['name'] + ", 文件: " + avatar['file'])
    
    elif args.command == "check-env":
        # 检查环境
        level = args.level
        results = check_environment(level)
        
        print("\n" + level.capitalize() + " 级别环境检查结果:")
        
        # 打印关键组件状态
        print("\n关键组件状态:")
        for name, status in results.items():
            if name != "cuda_info":
                status_text = "✓ 可用" if status else "✗ 不可用"
                print("  " + name + ": " + status_text)
        
        # 打印CUDA信息
        if "cuda_info" in results and results.get("cuda", False):
            print("\nCUDA信息:")
            cuda_info = results["cuda_info"]
            for key, value in cuda_info.items():
                print("  " + key + ": " + str(value))
        
        # 生成安装指南
        print("\n依赖安装指南:")
        all_available = all(status for name, status in results.items() if name != "cuda_info")
        
        if all_available:
            print("  所有依赖项已正确安装")
        else:
            print(get_installation_guide(level))
    
    elif args.command == "install-guide":
        # 获取安装指南
        level = args.level
        guide = get_installation_guide(level)
        
        print("\n" + level.capitalize() + " 级别功能安装指南:\n")
        print(guide)
    
    elif args.command == "generate-config":
        # 生成配置文件
        output_dir = args.output
        files = generate_config_files(output_dir)
        
        print("\n已生成以下配置文件:")
        for file in files:
            if file:
                print("  " + file)
    
    elif args.command == "server":
        # 启动Web服务
        host = args.host
        port = args.port
        start_server(host, port)

if __name__ == "__main__":
    main() 

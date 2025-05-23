#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import logging
import cv2
import numpy as np
import time
import json
import tempfile
import subprocess
import shutil
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import uuid

# 添加FaceParsing导入
try:
    from musetalk.utils.face_parsing import FaceParsing
except ImportError:
    # 记录导入错误但不立即失败，允许基本模式继续工作
    logging.getLogger(__name__).warning("无法导入FaceParsing，标准/高级模式将不可用")

from ..utils.tts_service import TTSService
from .dependency_manager import DependencyManager
from .fallback_manager import FallbackManager, CapabilityLevel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AvatarGenerator")

# Helper function to resolve paths relative to project root
def resolve_path(config_path: str, project_root: Path) -> str:
    path = Path(config_path)
    if path.is_absolute():
        return str(path)
    else:
        return str(project_root / path)

class GenerationMode(Enum):
    """生成模式枚举"""
    AUDIO_ONLY = "audio_only"  # 仅生成音频
    BASIC = "basic"  # 基础模式，静态图像+音频
    STANDARD = "standard"  # 标准模式，2D动画
    ADVANCED = "advanced"  # 高级模式，3D动画

class AvatarGenerator:
    """数字人视频生成器，支持多级降级策略"""
    
    def __init__(self, config: Dict[str, Any], project_root: Optional[Path] = None):
        """
        初始化数字人生成器
        
        Args:
            config: 配置信息
            project_root: 项目根目录
        """
        self.config = config
        self.project_root = project_root or Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).parent
        
        # 设置日志
        self.logger = logging.getLogger("AvatarGenerator")
        
        # 检查TTS配置是否存在
        tts_config = config.get("tts")
        if not tts_config:
            raise ValueError("找不到TTS配置，请确保default.yml文件中包含tts部分")
        
        # 初始化TTS服务 - 传递完整的TTS配置
        self.tts_service = TTSService(tts_config)
        
        # Determine project root (assuming this script is in MuseTalk/musetalk_wrapper/core)
        self.project_root = Path(__file__).parent.parent.parent.parent.resolve()
        logger.info("Project root determined as: {}".format(self.project_root))

        # Resolve output and temp directories relative to project root
        default_output_dir = "./outputs"
        default_temp_dir = "./temp"
        self.output_dir = resolve_path(
            config.get("video", {}).get("output_dir", default_output_dir),
            self.project_root
        )
        self.temp_dir = resolve_path(
            config.get("video", {}).get("temp_dir", default_temp_dir),
            self.project_root
        )

        # Ensure output目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        # Resolve avatar_dir from config relative to project root
        default_avatar_dir = "./results/v15/avatars" # Keep a relative default
        self.avatar_dir = resolve_path(
             config.get("paths", {}).get("avatar_dir", default_avatar_dir),
             self.project_root
         )
        logger.info("Using avatar directory: {}".format(self.avatar_dir))

        # 初始化依赖管理器和降级管理器
        self.dependency_manager = DependencyManager()  # 使用默认路径
        self.fallback_manager = FallbackManager(config.get("fallback", {}))
        
        # 确定当前环境支持的最高能力级别
        self.current_level = self._detect_capability_level()
        logger.info("当前环境支持的最高能力级别: " + self.current_level.name)

        # Models (loaded on demand)
        self.models_loaded = False
        self.device = None
        self.weight_dtype = None
        self.fp = None
        self.audio_processor = None
        self.whisper = None
        self.vae = None
        self.unet = None
        self.pe = None
        self.timesteps = None
        self.realtime_infer_module = None # Store imported module
    
    def _detect_capability_level(self) -> CapabilityLevel:
        """
        检测当前环境支持的能力级别
        
        Returns:
            能力级别枚举
        """
        # 检查高级依赖
        advanced_check = self.dependency_manager.check_all_dependencies("advanced")
        if all(advanced_check.values()):
            return CapabilityLevel.ADVANCED
        
        # 检查标准依赖
        standard_check = self.dependency_manager.check_all_dependencies("standard")
        if all(standard_check.values()):
            return CapabilityLevel.STANDARD
        
        # 检查基础依赖
        basic_check = self.dependency_manager.check_all_dependencies("basic")
        if all(basic_check.values()):
            return CapabilityLevel.BASIC
        
        # 未满足任何级别的依赖
        return CapabilityLevel.AUDIO_ONLY
    
    def get_available_avatars(self) -> List[Dict[str, Any]]:
        """
        获取可用的虚拟形象列表
        
        Returns:
            虚拟形象列表，每个元素包含id、name、preview_image等信息
        """
        result = []
        avatar_dir = self.avatar_dir
        
        try:
            # 遍历虚拟形象目录
            if os.path.exists(avatar_dir):
                for avatar_id in os.listdir(avatar_dir):
                    avatar_path = os.path.join(avatar_dir, avatar_id)
                    
                    # 检查是否为目录
                    if not os.path.isdir(avatar_path):
                        continue
                    
                    # 查找配置文件
                    config_path = os.path.join(avatar_path, "config.json")
                    if not os.path.exists(config_path):
                        continue
                    
                    # 读取配置
                    try:
                        with open(config_path, "r", encoding="utf-8") as f:
                            avatar_config = json.load(f)
                        
                        # 查找预览图
                        preview_image = None
                        for img_name in ["preview.png", "preview.jpg", "thumbnail.png", "thumbnail.jpg"]:
                            img_path = os.path.join(avatar_path, img_name)
                            if os.path.exists(img_path):
                                preview_image = img_path
                                break
                        
                        # 添加到结果
                        result.append({
                            "id": avatar_id,
                            "name": avatar_config.get("name", avatar_id),
                            "description": avatar_config.get("description", ""),
                            "preview_image": preview_image,
                            "type": avatar_config.get("type", "2d"),
                            "capabilities": avatar_config.get("capabilities", [])
                        })
                    except Exception as e:
                        logger.warning("读取虚拟形象配置失败: " + avatar_id + ", 错误: " + str(e))
        
        except Exception as e:
            logger.error("获取虚拟形象列表失败: " + str(e))
        
        return result
    
    def generate_video(
        self, 
        text: str, 
        avatar_id: str, 
        output_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        mode: Optional[GenerationMode] = None,
        tts_model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成数字人视频
        
        Args:
            text: 文本内容
            avatar_id: 虚拟形象ID
            output_path: 输出路径，如果为None则自动生成
            audio_path: 音频路径，如果为None则自动生成
            mode: 生成模式，如果为None则根据环境自动选择
            tts_model: TTS模型ID，如果为None则使用默认模型
            **kwargs: 其他参数
        
        Returns:
            生成结果信息
        """
        start_time = time.time()
        logger.info("开始生成视频，虚拟形象: " + avatar_id + ", 文本长度: " + str(len(text)))
        
        # 确定输出路径
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            output_path = os.path.join(self.output_dir, avatar_id + "_" + timestamp + ".mp4")
        
        # 获取偏好模式
        prefer_mode = kwargs.pop("prefer_mode", None)
        
        # 确定生成模式
        if mode is None:
            mode = self._select_generation_mode(prefer_mode)
        logger.info("选择生成模式: " + mode.name)
        
        # 准备生成参数
        params = {
            "text": text,
            "avatar_id": avatar_id,
            "output_path": output_path,
            "audio_path": audio_path,
            "tts_model": tts_model,
            **kwargs
        }
        
        # 根据模式选择生成方法
        try:
            if mode == GenerationMode.ADVANCED:
                result = self._generate_advanced(**params)
            elif mode == GenerationMode.STANDARD:
                result = self._generate_standard(**params)
            elif mode == GenerationMode.BASIC:
                result = self._generate_basic(**params)
            else:  # AUDIO_ONLY
                result = self._generate_audio_only(**params)
            
            # 添加元数据
            result.update({
                "generation_time": time.time() - start_time,
                "mode": mode.value,
                "output_path": output_path
            })
            
            logger.info("视频生成完成，耗时: " + str(round(result['generation_time'], 2)) + "秒, 输出: " + output_path)
            return result
        
        except Exception as e:
            error_msg = "生成失败: " + str(e)
            logger.error(error_msg)
            
            # 尝试降级处理
            if self.fallback_manager.should_fallback(mode):
                next_mode = self.fallback_manager.get_next_mode(mode)
                if next_mode is not None and next_mode != mode:
                    logger.info("降级处理，从 " + mode.name + " 降级到 " + next_mode.name)
                    params["mode"] = next_mode
                    try:
                        result = self.generate_video(**params)
                        result["fallback_from"] = mode.value
                        return result
                    except Exception as e2:
                        logger.error("降级处理也失败: " + str(e2))
            
            # 所有尝试都失败
            self.fallback_manager.report_failure(mode)
            return {
                "success": False,
                "error": error_msg,
                "mode": mode.value
            }
    
    def _select_generation_mode(self, prefer_mode: Optional[str] = None) -> GenerationMode:
        """
        根据当前环境能力和用户偏好选择合适的生成模式
        
        Args:
            prefer_mode: 偏好的生成模式，如果指定且环境支持则优先使用
            
        Returns:
            生成模式枚举
        """
        # 如果指定了偏好模式，尝试使用该模式（如果环境支持）
        if prefer_mode:
            # 基于能力级别名称（字符串）比较
            current_level_name = self.current_level.name
            
            if prefer_mode == "basic" and current_level_name in ["BASIC", "STANDARD", "ADVANCED"]:
                return GenerationMode.BASIC
            elif prefer_mode == "standard" and current_level_name in ["STANDARD", "ADVANCED"]:
                return GenerationMode.STANDARD  
            elif prefer_mode == "advanced" and current_level_name == "ADVANCED":
                return GenerationMode.ADVANCED
            elif prefer_mode == "audio_only":
                return GenerationMode.AUDIO_ONLY
        
        # 否则根据当前能力级别选择对应的生成模式
        if self.current_level.name == "ADVANCED":
            return GenerationMode.ADVANCED
        elif self.current_level.name == "STANDARD":
            return GenerationMode.STANDARD
        elif self.current_level.name == "BASIC":
            return GenerationMode.BASIC
        else:
            return GenerationMode.AUDIO_ONLY
    
    def _prepare_audio(self, text: str, audio_path: Optional[str], tts_model: Optional[str], text_lang: str = "zh") -> Tuple[bool, str, Optional[str]]:
        """
        Prepare audio file. Returns (success_status, audio_file_path, error_message).
        Error message is None if successful.
        """
        original_error = None # Store original TTS error if fallback occurs
        # 如果已提供音频文件，直接使用
        if audio_path and os.path.exists(audio_path):
            logger.info("使用已提供的音频文件: " + audio_path)
            return True, os.path.abspath(audio_path), None
        
        # 创建临时音频文件
        if audio_path is None:
            # 创建临时目录（确保使用绝对路径）
            temp_dir = self.temp_dir
            os.makedirs(temp_dir, exist_ok=True)
            
            # 生成临时文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            audio_path = os.path.join(temp_dir, "tts_" + timestamp + ".wav")
        
        # 确保音频路径是绝对路径
        audio_path = os.path.abspath(audio_path)
        logger.info("将生成音频文件: " + audio_path)
        
        # 使用TTS服务生成音频
        try:
            success, message = self.tts_service.text_to_speech(text, audio_path, tts_model, text_lang)
            
            # 如果TTS服务失败，尝试离线方式
            if not success:
                original_error = "TTS Service failed: {}".format(message)
                logger.warning("{}. Trying offline generation...".format(original_error))
                success, message = self.tts_service.text_to_speech_offline(text, audio_path, tts_model, text_lang)
            
            # 验证音频文件是否成功生成
            if success and not os.path.exists(audio_path):
                original_error = original_error or "Offline TTS success but file missing"
                logger.error("TTS claimed success, but file missing: {}".format(audio_path))
                success = False
                message = "Audio file not generated successfully"
            
            if not success:
                raise Exception("Failed to prepare audio: {}".format(message))
                
            # 再次验证文件存在
            if not os.path.exists(audio_path):
                raise Exception("Audio file does not exist: {}".format(audio_path))
                
            # 确认文件有内容
            if os.path.getsize(audio_path) == 0:
                raise Exception("Audio file is empty: {}".format(audio_path))
                
            logger.info("Audio file prepared successfully: {}, Size: {} bytes".format(audio_path, os.path.getsize(audio_path)))
            return True, audio_path, None # Success, path, no error
            
        except Exception as e:
            error_msg = "Audio generation exception: {}".format(str(e))
            logger.error(error_msg)
            original_error = original_error or error_msg # Keep first error

            # 尝试使用极简方式生成静音音频文件
            try:
                logger.warning("尝试生成备用静音音频")
                
                # 确保使用绝对路径
                temp_dir = self.temp_dir
                os.makedirs(temp_dir, exist_ok=True)
                
                timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                silent_audio_path = os.path.join(temp_dir, "silent_" + timestamp + ".wav")
                
                # 使用ffmpeg生成1秒静音
                subprocess.run([
                    "ffmpeg", "-y", "-v", "warning",
                    "-f", "lavfi",
                    "-i", "anullsrc=r=44100:cl=mono",
                    "-t", "1",
                    silent_audio_path
                ], check=True, capture_output=True)
                
                if os.path.exists(silent_audio_path):
                    logger.info("已生成静音音频: {}".format(silent_audio_path))
                    # Return success=True but include the original error message
                    return True, silent_audio_path, "Original audio failed ({}), used silent fallback.".format(original_error)
            except Exception as silent_error:
                logger.error("生成静音音频也失败: {}".format(silent_error))
                # Fall through to raise the original exception

            # If silent audio fails, raise the original preparation failure
            raise Exception("Failed to prepare audio and fallback: {}".format(original_error))
    
    def _generate_advanced(self, **kwargs) -> Dict[str, Any]:
        """高级模式：使用3D模型生成视频"""
        # 检查依赖
        advanced_dependencies = self.dependency_manager.check_all_dependencies("advanced")
        if not all(advanced_dependencies.values()):
            missing = [dep for dep, status in advanced_dependencies.items() if not status]
            raise RuntimeError("缺少高级模式所需依赖: " + ", ".join(missing))
        
        # TODO: 实现高级模式生成逻辑
        raise NotImplementedError("高级模式尚未实现")
    
    def _generate_standard(self, **kwargs) -> Dict[str, Any]:
        """标准模式：使用2D动画生成视频"""
        # 检查依赖
        standard_dependencies = self.dependency_manager.check_all_dependencies("standard")
        if not all(standard_dependencies.values()):
            missing = [dep for dep, status in standard_dependencies.items() if not status]
            raise RuntimeError("缺少标准模式所需依赖: " + ", ".join(missing))
        
        # 提取参数
        text = kwargs.get("text", "")
        avatar_id = kwargs.get("avatar_id", "")
        output_path = kwargs.get("output_path", "")
        audio_path = kwargs.get("audio_path")
        tts_model = kwargs.get("tts_model")
        text_lang = kwargs.get("lang", "zh")
        gpu_id = kwargs.get("gpu_id", 0)
        batch_size = kwargs.get("batch_size", 4)  # 减小batch_size以减少内存使用
        fps = self.config.get("video", {}).get("fps", 25)
        
        # 字幕相关参数
        add_subtitle = kwargs.get("add_subtitle", False) or kwargs.get("subtitle", False)
        subtitle_path = kwargs.get("subtitle_path")
        subtitle_format = kwargs.get("subtitle_format", "vtt")
        subtitle_offset = kwargs.get("subtitle_offset", -0.3)
        align_subtitles = kwargs.get("align_subtitles", True)
        text_cut_method = kwargs.get("text_cut_method", "cut2")
        
        # 确保模型已加载
        try:
            self._ensure_models_loaded(gpu_id=gpu_id)
            # 获取Avatar类
            if self.realtime_infer_module:
                Avatar = self.realtime_infer_module.Avatar
            else:
                from scripts.realtime_inference import Avatar
        except Exception as model_load_error:
            logger.error("Failed to load models needed for standard mode: {}".format(model_load_error))
            # 记录失败原因并尝试降级到基本模式
            fallback_reason = "Model loading failed: {}".format(model_load_error)
            logger.info("Attempting fallback to basic mode due to model load failure...")
            return self._generate_basic(fallback_reason=fallback_reason, **kwargs)

        # 准备音频
        audio_prep_success, audio_path, audio_prep_error = self._prepare_audio(text, audio_path, tts_model, text_lang)
        if not audio_prep_success:
            raise RuntimeError("Audio preparation failed: {}".format(audio_prep_error))

        # 处理潜在的静音音频降级 - 如果使用了静音音频则记录警告
        if audio_prep_error:
            logger.warning("Proceeding with standard generation, but audio was problematic: {}".format(audio_prep_error))

        # 确保文件路径都是绝对路径
        audio_path = os.path.abspath(audio_path)
        output_path = os.path.abspath(output_path)
        output_name = os.path.splitext(os.path.basename(output_path))[0]
        
        # 如果需要字幕，先生成字幕
        if add_subtitle:
            logger.info("标准模式中启用字幕生成")
            
            # 如果未提供字幕路径，从输出路径生成一个
            if not subtitle_path:
                subtitle_path = os.path.splitext(output_path)[0] + "." + subtitle_format
                logger.info("自动生成字幕路径: {}".format(subtitle_path))
            
            # 确保字幕路径是绝对路径
            subtitle_path = os.path.abspath(subtitle_path)
            
            # 确保字幕目录存在
            subtitle_dir = os.path.dirname(subtitle_path)
            os.makedirs(subtitle_dir, exist_ok=True)
            
            # 生成字幕
            logger.info("在标准模式中生成字幕, 文件路径: {}".format(subtitle_path))
            
            subtitle_success, subtitle_result = self._generate_subtitle(
                text=text,
                audio_path=audio_path,
                subtitle_path=subtitle_path,
                subtitle_format=subtitle_format,
                subtitle_offset=subtitle_offset,
                align_subtitles=align_subtitles,
                text_cut_method=text_cut_method
            )
            
            if subtitle_success:
                logger.info("标准模式字幕生成成功: {}".format(subtitle_path))
            else:
                logger.warning("标准模式字幕生成失败: {}，将继续视频生成".format(subtitle_result))
        
        logger.info("Starting standard mode lip sync generation, Audio: {}".format(audio_path))
        
        # 获取Avatar路径
        try:
            avatar_path = self._get_avatar_path(avatar_id)
            logger.info("Found avatar path: {}".format(avatar_path))
        except Exception as e:
            logger.error("Could not find avatar path for {}: {}".format(avatar_id, e))
            # 尝试基本降级如果avatar路径失败
            fallback_reason = "Avatar path not found: {}".format(e)
            logger.info("Attempting fallback to basic mode due to avatar path issue...")
            # 将潜在的静音音频路径传递给基本模式
            kwargs['audio_path'] = audio_path
            return self._generate_basic(fallback_reason=fallback_reason, **kwargs)

        # 确保输出目录存在
        video_out_path = os.path.join(avatar_path, "vid_output")
        os.makedirs(video_out_path, exist_ok=True)
        generated_video = os.path.join(video_out_path, "{}.mp4".format(output_name))

        standard_mode_exception = None # Store exception for fallback reason
        try:
            # 尝试从缓存获取Avatar实例
            avatar = None
            try:
                from .model_cache import ModelCache
                cache = ModelCache.get_instance()
                
                # 准备Avatar参数
                avatar_params = {
                    "video_path": None,
                    "bbox_shift": 0,
                    "batch_size": batch_size,
                    "preparation": False
                }
                
                # 从缓存获取或创建Avatar实例
                avatar = cache.get_avatar_instance(
                    avatar_id=avatar_id,
                    avatar_class=Avatar,
                    params=avatar_params
                )
                
                if avatar:
                    logger.info("使用缓存的Avatar实例")
                else:
                    logger.warning("无法从缓存获取Avatar实例，将创建新实例")
            except Exception as e:
                logger.warning("使用Avatar缓存失败，创建新实例: {}".format(e))
            
            # 如果无法从缓存获取，创建新实例
            if avatar is None:
                # 创建Avatar实例
                logger.info("Creating Avatar instance... (batch_size={})".format(batch_size))
                avatar = Avatar(
                    avatar_id=avatar_id,
                    video_path=None,
                    bbox_shift=0,
                    batch_size=batch_size,
                    preparation=False
                )

            # === Custom Inference Wrapper for FFmpeg ===
            original_inference = avatar.inference

            def custom_inference_wrapper(audio_path_arg, out_vid_name_arg, fps_arg, skip_save_images=False):
                logger.info("Using custom FFmpeg video generation method...")
                avatar.skip_video_merge_flag = True
                original_tmp_dir = None
                try:
                    original_tmp_dir = os.path.join(avatar.avatar_path, "tmp")
                    os.makedirs(original_tmp_dir, exist_ok=True)

                    # Setup Args class needed by realtime_inference (Use config values)
                    class Args:
                        pass
                    args = Args()
                    rt_args_config = self.config.get("realtime_inference_args", {})
                    args.version = rt_args_config.get("version", "v15") # Default v15
                    args.extra_margin = rt_args_config.get("extra_margin", 10)
                    args.parsing_mode = rt_args_config.get("parsing_mode", "jaw")
                    args.skip_save_images = rt_args_config.get("skip_save_images", False)
                    args.audio_padding_length_left = rt_args_config.get("audio_padding_length_left", 2)
                    args.audio_padding_length_right = rt_args_config.get("audio_padding_length_right", 2)
                    args.left_cheek_width = rt_args_config.get("left_cheek_width", 90)
                    args.right_cheek_width = rt_args_config.get("right_cheek_width", 90)
                    self.realtime_infer_module.args = args # Set args for the module

                    # 确保全局变量在模块中正确设置
                    # 将所有必要的组件设置为realtime_inference中的全局变量
                    ri_module = self.realtime_infer_module
                    ri_module.audio_processor = self.audio_processor
                    ri_module.whisper = self.whisper
                    ri_module.pe = self.pe
                    ri_module.vae = self.vae
                    ri_module.unet = self.unet
                    ri_module.device = self.device
                    ri_module.weight_dtype = self.weight_dtype
                    ri_module.timesteps = self.timesteps

                    # Initialize FaceParsing (uses args)
                    self.fp = FaceParsing(
                        left_cheek_width=args.left_cheek_width,
                        right_cheek_width=args.right_cheek_width
                    )
                    ri_module.fp = self.fp # Set for the module

                    # 检查缓存中是否已有图片帧
                    frames_loaded_from_cache = False
                    try:
                        from .model_cache import ModelCache
                        cache = ModelCache.get_instance()
                        
                        # 从缓存获取图片帧
                        frames = cache.get_frames(avatar_id, original_tmp_dir)
                        if frames is not None and len(frames) > 0:
                            logger.info("使用缓存的图片帧，跳过图片生成")
                            frames_loaded_from_cache = True
                            # 这里需要对接实际代码，让Avatar跳过图片生成步骤
                            # 由于我们没有完整的realtime_inference代码，这里只是示意
                            try:
                                # 安全设置属性，确保avatar实例有这个属性
                                if hasattr(avatar, "frames_ready"):
                                    avatar.frames_ready = True
                                else:
                                    logger.warning("Avatar实例没有frames_ready属性，缓存图片帧可能无效")
                                    frames_loaded_from_cache = False
                            except Exception as attr_error:
                                logger.warning("设置Avatar属性失败: {}".format(attr_error))
                                frames_loaded_from_cache = False
                    except Exception as cache_error:
                        logger.warning("无法从缓存获取图片帧: {}".format(cache_error))
                    
                    # 如果没有从缓存加载图片帧，则正常生成
                    if not frames_loaded_from_cache:
                        original_inference(audio_path_arg, out_vid_name_arg, fps_arg, skip_save_images)
                        
                        # 缓存生成的图片帧
                        try:
                            from .model_cache import ModelCache
                            cache = ModelCache.get_instance()
                            
                            # 读取生成的图片帧并缓存
                            import glob, cv2
                            frame_files = sorted(glob.glob(os.path.join(original_tmp_dir, "*.png")))
                            if frame_files:
                                frames = []
                                for frame_file in frame_files:
                                    frame = cv2.imread(frame_file)
                                    if frame is not None:
                                        frames.append(frame)
                                
                                # 保存到缓存
                                if frames:
                                    cache.frames_cache[avatar_id] = frames
                                    cache.frames_cache_info[avatar_id] = {
                                        "last_used": time.time(),
                                        "size": len(frames) * frames[0].nbytes
                                    }
                                    logger.info("图片帧已保存到缓存")
                        except Exception as e:
                            logger.warning("保存图片帧到缓存失败: {}".format(e))

                except Exception as e:
                    # Log the actual exception type and message
                    error_type = type(e).__name__
                    error_msg = str(e)
                    logger.warning("Original inference call produced an error (Type: {}, Msg: {}). Checking if it's the expected FFmpeg error...".format(error_type, error_msg))
                    # Still using a placeholder check, needs refinement based on actual errors
                    if "some specific error related to ffmpeg in original code" in error_msg:
                         logger.info("Ignoring expected FFmpeg error in original inference, proceeding...")
                    else:
                         logger.error("Unexpected error during original inference frame generation: {}".format(error_msg))
                         raise # Re-raise unexpected errors
                finally:
                    avatar.skip_video_merge_flag = False

                tmp_dir_path = Path(original_tmp_dir)
                if tmp_dir_path.exists() and any(tmp_dir_path.iterdir()):
                    frames_pattern = os.path.join(str(tmp_dir_path), "%08d.png")
                    output_video_path = os.path.join(avatar.avatar_path, "vid_output", "{}.mp4".format(out_vid_name_arg))
                    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

                    video_cmd = [
                        "ffmpeg", "-y", "-v", "warning",
                        "-r", str(fps_arg),
                        "-f", "image2",
                        "-i", frames_pattern,
                        "-vcodec", "libx264",
                        "-preset", "medium",  # 使用preset替代crf
                        "-pix_fmt", "yuv420p",
                        output_video_path
                    ]
                    logger.info("Executing compatible FFmpeg frame merge command: " + " ".join(video_cmd))
                    try:
                        subprocess.run(video_cmd, check=True, capture_output=True)
                    except subprocess.CalledProcessError as cpe:
                        logger.error("FFmpeg frame merge failed. Return code: {}".format(cpe.returncode))
                        logger.error("FFmpeg stderr: {}".format(cpe.stderr.decode(errors='ignore')))
                        raise Exception("FFmpeg frame merge failed") from cpe

                    if os.path.exists(output_video_path):
                        final_video_path = output_video_path
                        temp_video_path = output_video_path + ".temp.mp4"

                        try:
                            shutil.move(output_video_path, temp_video_path)

                            audio_cmd = [
                                "ffmpeg", "-y", "-v", "warning",
                                "-i", audio_path_arg,
                                "-i", temp_video_path,
                                "-c:v", "copy",
                                "-c:a", "aac",
                                "-shortest",
                                final_video_path
                            ]
                            logger.info("Executing FFmpeg audio merge command: " + " ".join(audio_cmd))
                            subprocess.run(audio_cmd, check=True, capture_output=True)

                            if os.path.exists(final_video_path):
                                os.remove(temp_video_path)
                            else:
                                logger.warning("Audio merge seemed successful but final file {} not found. Kept {}".format(final_video_path, temp_video_path))

                        except subprocess.CalledProcessError as cpe_audio:
                            logger.error("FFmpeg audio merge failed. Return code: {}".format(cpe_audio.returncode))
                            logger.error("FFmpeg stderr: {}".format(cpe_audio.stderr.decode(errors='ignore')))
                            if os.path.exists(temp_video_path) and not os.path.exists(final_video_path):
                                logger.warning("Restoring video without audio due to merge failure.")
                                shutil.move(temp_video_path, final_video_path)
                            raise Exception("FFmpeg audio merge failed") from cpe_audio
                        except Exception as move_err:
                            logger.error("Error during temp file handling in audio merge: {}".format(move_err))
                            if os.path.exists(final_video_path) and os.path.exists(temp_video_path):
                                try: os.remove(temp_video_path)
                                except OSError: pass
                            elif not os.path.exists(final_video_path) and os.path.exists(temp_video_path):
                                logger.warning("Keeping temp video {} due to error.".format(temp_video_path))
                            raise

            avatar.inference = custom_inference_wrapper

            logger.info("Starting wrapped video generation...")
            try:
                avatar.inference(
                    audio_path_arg=audio_path,
                    out_vid_name_arg=output_name,
                    fps_arg=fps
                )
            finally:
                avatar.inference = original_inference
                logger.debug("Restored original inference method.")

            logger.info("Checking generated video file: {}".format(generated_video))
            if os.path.exists(generated_video) and os.path.getsize(generated_video) > 0:
                import shutil
                logger.info("Standard video generation successful. Copying to final output: {}".format(output_path))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.copy2(generated_video, output_path)

                video_info = self._get_video_info(output_path)
                result_data = {
                    "success": True,
                    "path": output_path,
                    "mode": "standard",
                    "duration": video_info.get("duration", 0),
                    "fps": video_info.get("fps", fps),
                    "resolution": video_info.get("resolution", ""),
                    "audio_issue": audio_prep_error,
                    "audio_path": audio_path
                }
                
                # 添加字幕路径到结果中
                if add_subtitle and subtitle_success:
                    result_data["subtitle_path"] = subtitle_path
                    
                return result_data
            else:
                error_msg = "Standard mode ran but failed to produce video file: {}".format(generated_video)
                logger.error(error_msg)
                raise Exception(error_msg)

        except Exception as e:
            standard_mode_exception = e
            logger.error("Standard mode generation failed: {}".format(str(e)))
            import traceback
            logger.error("Error details: {}".format(traceback.format_exc()))

            fallback_reason = "Standard mode failed: {}".format(standard_mode_exception)
            logger.info("Attempting fallback to basic mode. Reason: {}".format(fallback_reason))
            kwargs['audio_path'] = audio_path
            return self._generate_basic(fallback_reason=fallback_reason, **kwargs)
    
    def _get_avatar_path(self, avatar_id: str) -> str:
        """
        获取数字人的路径
        
        Args:
            avatar_id: 数字人ID
            
        Returns:
            数字人路径
        """
        # 检查主配置的avatar目录
        primary_avatar_path = os.path.join(self.avatar_dir, avatar_id)
        if os.path.isdir(primary_avatar_path):
            logger.info("Found avatar path using configured dir: {}".format(primary_avatar_path))
            return primary_avatar_path

        # 降级路径
        fallback_paths_relative = [
            "./results/avatars", # 通用降级
        ]
        possible_paths = [primary_avatar_path] + [resolve_path(p, self.project_root) for p in fallback_paths_relative]

        for path in possible_paths:
            if os.path.isdir(path):
                logger.info("Found avatar path using fallback: {}".format(path))
                return path

        raise FileNotFoundError("Cannot find avatar directory for ID: {} in path: {}".format(avatar_id, possible_paths))
    
    def _generate_basic(self, text: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """
        基础视频生成模式 - 使用静态头像和音频
        
        Args:
            text: 文本内容
            output_path: 输出视频路径
            **kwargs: 其他参数
                audio_path: 预先生成的音频文件路径
                avatar_id: 头像ID
                subtitle: 是否添加字幕
                subtitle_path: 字幕文件路径
                subtitle_color: 字幕颜色
                align_subtitles: 是否对齐字幕
                
        Returns:
            生成结果
        """
        logger.info("使用基础模式生成视频...")
        
        try:
            # 获取头像ID
            avatar_id = kwargs.get("avatar_id")
            if not avatar_id:
                logger.error("未提供头像ID")
                return {"success": False, "error": "未提供头像ID"}
            
            # 获取或生成音频
            audio_path = kwargs.get("audio_path")
            if not audio_path:
                # 生成音频
                audio_path = self._generate_audio(text, **kwargs)
                if not audio_path:
                    logger.error("音频生成失败")
                    return {"success": False, "error": "音频生成失败"}
            
            # 获取头像图像
            avatar_image = self._get_avatar_image(avatar_id)
            if not avatar_image:
                logger.error("无法获取头像图像: {}".format(avatar_id))
                return {"success": False, "error": "无法获取头像图像: {}".format(avatar_id)}
            
            # 字幕处理
            subtitle_path = kwargs.get("subtitle_path")
            add_subtitle = kwargs.get("subtitle", False) or kwargs.get("add_subtitle", False)
            
            # 记录字幕相关参数
            logger.info("字幕参数: add_subtitle={}, subtitle_path={}".format(add_subtitle, subtitle_path))
            
            if add_subtitle:
                # 如果需要字幕但没有提供字幕路径，生成一个
                if not subtitle_path:
                    subtitle_format = kwargs.get("subtitle_format", "vtt")
                    subtitle_path = os.path.splitext(output_path)[0] + "." + subtitle_format
                    logger.info("自动生成字幕路径: {}".format(subtitle_path))
                
                # 确保字幕目录存在
                subtitle_dir = os.path.dirname(subtitle_path)
                os.makedirs(subtitle_dir, exist_ok=True)
                
                # 生成字幕
                logger.info("开始生成字幕，文件路径: {}".format(subtitle_path))
                subtitle_offset = kwargs.get("subtitle_offset", -0.3)
                align_subtitles = kwargs.get("align_subtitles", True)
                
                success, result = self._generate_subtitle(
                    text=text,
                    audio_path=audio_path,
                    subtitle_path=subtitle_path,
                    subtitle_format=kwargs.get("subtitle_format", "vtt"),
                    subtitle_offset=subtitle_offset,
                    align_subtitles=align_subtitles
                )
                
                if success:
                    logger.info("字幕生成成功: {}".format(subtitle_path))
                else:
                    logger.warning("字幕生成失败: {}，将继续视频生成".format(result))
            
            # 创建视频
            logger.info("开始生成基础视频...")
            success, result = self._create_basic_video(
                avatar_image=avatar_image,
                audio_path=audio_path,
                output_path=output_path,
                subtitle_path=subtitle_path if add_subtitle else None,
                subtitle_color=kwargs.get("subtitle_color", "#FFFFFF")
            )
            
            if not success:
                logger.error("基础视频生成失败: {}".format(result))
                return {"success": False, "error": "基础视频生成失败: {}".format(result)}
            
            # 返回结果
            logger.info("基础视频生成成功: {}".format(output_path))
            return {
                "success": True,
                "path": output_path,
                "audio_path": audio_path,
                "subtitle_path": subtitle_path if add_subtitle else None,
                "mode": "basic"
            }
            
        except Exception as e:
            logger.error("基础视频生成异常: {}".format(str(e)))
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": "基础视频生成异常: {}".format(str(e))}
    
    def _generate_audio_only(self, **kwargs) -> Dict[str, Any]:
        """
        仅音频模式：只生成音频文件
        
        Args:
            **kwargs: 生成参数
        
        Returns:
            生成结果
        """
        text = kwargs.get("text", "")
        output_path = kwargs.get("output_path", "")
        audio_path = kwargs.get("audio_path")
        tts_model = kwargs.get("tts_model")
        
        # 获取语言参数，默认为中文
        text_lang = kwargs.get("lang", "zh")
        
        # 修改输出文件扩展名为.wav
        if output_path.lower().endswith((".mp4", ".avi", ".mov")):
            output_path = os.path.splitext(output_path)[0] + ".wav"
        
        # 准备音频
        success, audio_path, audio_prep_error = self._prepare_audio(text, audio_path, tts_model, text_lang)
        
        if success:
            # 如果输出路径不是音频路径，复制音频文件
            if audio_path != output_path:
                import shutil
                shutil.copy2(audio_path, output_path)
            
            # 获取音频信息
            audio_info = self._get_audio_info(output_path)
            
            return {
                "success": True,
                "path": output_path,
                "mode": "audio_only",
                "duration": audio_info.get("duration", 0),
                "sample_rate": audio_info.get("sample_rate", 44100),
                "channels": audio_info.get("channels", 1),
                "audio_issue": audio_prep_error
            }
        else:
            raise Exception("Failed to generate audio only: {}".format(audio_prep_error))
    
    def _get_avatar_image(self, avatar_id: str) -> Optional[str]:
        """
        获取虚拟形象的图片路径
        
        Args:
            avatar_id: 虚拟形象ID
            
        Returns:
            图片路径，如果不存在则返回None
        """
        avatar_base_path = self._get_avatar_path(avatar_id)

        if not avatar_base_path:
            return None

        # 搜索顺序
        search_files = ["portrait.png", "portrait.jpg", "avatar.png", "avatar.jpg", "main.png", "main.jpg", "source.jpg", "source.png"]
        for img_name in search_files:
            img_path = os.path.join(avatar_base_path, img_name)
            if os.path.isfile(img_path):
                logger.info("找到头像图片: {}".format(img_path))
                return img_path

        # 降级：检查vid_output目录下的视频帧
        vid_output_dir = os.path.join(avatar_base_path, "vid_output")
        if os.path.isdir(vid_output_dir):
            try:
                # 找到第一个图像文件（png或jpg）
                for item in sorted(os.listdir(vid_output_dir)):
                    if item.lower().endswith((".png", ".jpg")):
                        img_path = os.path.join(vid_output_dir, item)
                        if os.path.isfile(img_path):
                            logger.info("找到视频帧图片作为头像: {}".format(img_path))
                            return img_path
            except OSError as e:
                logger.warning("无法列出vid_output目录 {}: {}".format(vid_output_dir, e))

        # 降级：检查full_imgs目录
        full_imgs_dir = os.path.join(avatar_base_path, "full_imgs")
        if os.path.isdir(full_imgs_dir):
            try:
                for item in sorted(os.listdir(full_imgs_dir)):
                    if item.lower().endswith((".png", ".jpg")):
                        img_path = os.path.join(full_imgs_dir, item)
                        if os.path.isfile(img_path):
                            logger.info("找到全尺寸图片作为头像: {}".format(img_path))
                            return img_path
            except OSError as e:
                logger.warning("无法列出full_imgs目录 {}: {}".format(full_imgs_dir, e))

        logger.error("找不到任何合适的图像文件用于头像: {} 在路径: {}".format(avatar_id, avatar_base_path))
        return None
    
    def _generate_subtitle(self, text: str, audio_path: str, subtitle_path: str = None, 
                        subtitle_format: str = "vtt", subtitle_offset: float = 0, 
                        align_subtitles: bool = True, text_cut_method: str = "cut2", **kwargs) -> Tuple[bool, str]:
        """
        生成字幕文件
        
        Args:
            text: 文本内容
            audio_path: 音频文件路径
            subtitle_path: 字幕文件路径，如果为None则使用音频路径替换扩展名
            subtitle_format: 字幕格式 (vtt/srt)
            subtitle_offset: 字幕时间偏移量(秒)
            align_subtitles: 是否使用语音识别对齐字幕
            text_cut_method: 文本切分方法
            
        Returns:
            (成功标志, 字幕文件路径或错误信息)
        """
        try:
            # 检查音频文件是否存在
            if not os.path.exists(audio_path):
                error_msg = "Audio file does not exist: {}".format(audio_path)
                logger.error(error_msg)
                return False, error_msg
            
            # 检查TTS服务中字幕模块是否可用
            if not self.tts_service.subtitle_modules_available:
                logger.warning("字幕处理模块不可用，字幕生成可能降级到备用方法")
            
            # 如果未提供字幕路径，使用音频路径的扩展名进行替换
            if not subtitle_path:
                subtitle_path = os.path.splitext(audio_path)[0] + "." + subtitle_format
            
            # 获取音频时长
            audio_info = self._get_audio_info(audio_path)
            audio_duration = audio_info.get("duration", 0)
            
            if audio_duration <= 0:
                error_msg = "Invalid audio duration: {}".format(audio_duration)
                logger.error(error_msg)
                return False, error_msg
            
            logger.info("Generating subtitle for audio: {}, duration: {:.2f}s, format: {}, text_cut_method: {}".format(
                audio_path, audio_duration, subtitle_format, text_cut_method))
            
            # 使用TTS服务生成字幕
            tts_config = self.config.get("tts", {})
            success, result = self.tts_service.generate_subtitle(
                text=text,
                output_path=subtitle_path,
                audio_duration=audio_duration,
                format=subtitle_format,
                encoding="utf-8",
                offset=subtitle_offset,
                subtitle_speed=1.0,
                text_cut_method=text_cut_method
            )
            
            if success:
                logger.info("Subtitle generated successfully: {}".format(subtitle_path))
                return True, subtitle_path
            else:
                error_msg = "Failed to generate subtitle: {}".format(result)
                logger.error(error_msg)
                return False, error_msg
            
        except Exception as e:
            error_msg = "Error generating subtitle: {}".format(str(e))
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            return False, error_msg
    
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        获取视频信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            视频信息字典
        """
        try:
            # 使用ffprobe获取视频信息
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,duration",
                "-of", "json",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            # 提取信息
            stream = info.get("streams", [{}])[0]
            width = stream.get("width", 0)
            height = stream.get("height", 0)
            
            # 处理帧率（格式为"分子/分母"）
            fps = 25.0
            r_frame_rate = stream.get("r_frame_rate", "25/1")
            if "/" in r_frame_rate:
                num, denom = map(int, r_frame_rate.split("/"))
                if denom != 0:
                    fps = num / denom
            
            # 处理时长
            duration = float(stream.get("duration", 0))
            
            return {
                "width": width,
                "height": height,
                "resolution": str(width) + "x" + str(height),
                "fps": fps,
                "duration": duration
            }
            
        except Exception as e:
            logger.warning("获取视频信息失败: {}".format(str(e)))
            return {}
    
    def _get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """
        获取音频信息
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音频信息字典
        """
        try:
            # 使用ffprobe获取音频信息
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate,channels,duration",
                "-of", "json",
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            # 提取信息
            stream = info.get("streams", [{}])[0]
            sample_rate = int(stream.get("sample_rate", 44100))
            channels = int(stream.get("channels", 1))
            
            # 处理时长
            duration = float(stream.get("duration", 0))
            
            return {
                "sample_rate": sample_rate,
                "channels": channels,
                "duration": duration
            }
            
        except Exception as e:
            logger.warning("获取音频信息失败: {}".format(str(e)))
            return {}
    
    def _convert_video_format(self, input_video: str, output_video: str) -> bool:
        """
        将视频转换为标准格式，提高兼容性
        
        Args:
            input_video: 输入视频路径
            output_video: 输出视频路径
            
        Returns:
            是否转换成功
        """
        logger.info("转换视频格式: {} -> {}".format(input_video, output_video))
        
        # 尝试使用mpeg4编码器，这在大多数系统上都支持
        cmd = [
            "ffmpeg", "-y", "-v", "warning",
            "-i", input_video,
            "-c:v", "mpeg4",
            "-q:v", "5",
            "-c:a", "aac",
            "-b:a", "192k",
            output_video
        ]
        logger.info("执行命令: {}".format(" ".join(cmd)))
        
        try:
            process = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8', errors='ignore'
            )
            logger.debug("FFmpeg convert stdout: {}".format(process.stdout))
            logger.debug("FFmpeg convert stderr: {}".format(process.stderr))

            if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
                logger.info("视频转换成功: {}".format(output_video))
                return True
            else:
                logger.error("视频转换命令运行但输出文件缺失或为空: {}".format(output_video))
                return False
                
        except subprocess.CalledProcessError as cpe:
            logger.error("视频转换FFmpeg命令失败。返回码: {}".format(cpe.returncode))
            logger.error("FFmpeg stderr: {}".format(cpe.stderr))
            return False
        except Exception as e:
            logger.error("视频转换失败，异常: {}".format(e))
            return False

    def _ensure_models_loaded(self, gpu_id: int = 0):
        """Load standard/advanced models if not already loaded."""
        if self.models_loaded:
            return

        # 尝试使用缓存
        try:
            # 导入缓存管理器
            from .model_cache import ModelCache
            cache = ModelCache.get_instance()
            global_models = cache.get_global_models()
            
            # 检查缓存是否有效
            if global_models["loaded"]:
                logger.info("使用缓存的全局模型")
                
                # 从缓存中恢复模型
                self.device = global_models["device"]
                self.weight_dtype = global_models["weight_dtype"]
                self.fp = global_models["fp"]
                self.audio_processor = global_models["audio_processor"]
                self.whisper = global_models["whisper"]
                self.vae = global_models["vae"]
                self.unet = global_models["unet"]
                self.pe = global_models["pe"]
                self.timesteps = global_models["timesteps"]
                self.realtime_infer_module = global_models["realtime_infer_module"]
                
                # 更新标志
                self.models_loaded = True
                return
                
            logger.info("缓存中没有找到模型，开始加载...")
        except Exception as e:
            logger.warning("缓存系统不可用或出错，直接加载模型: {}".format(e))

        logger.info("Loading models for standard/advanced generation...")
        try:
            # Add project root to sys.path temporarily and carefully
            # WARNING: Modifying sys.path can have side effects. Consider project structure improvements.
            original_sys_path = sys.path[:]
            if str(self.project_root) not in sys.path:
                sys.path.insert(0, str(self.project_root))
                logger.debug("Temporarily added {} to sys.path for import".format(self.project_root))

            # Import necessary components AFTER potentially modifying sys.path
            # Store the imported module to avoid repeated imports if this method is called again (shouldn't happen with self.models_loaded flag)
            if self.realtime_infer_module is None:
                 # Use importlib for more control if needed, but direct import is simpler here
                 # Assuming scripts directory is directly under project root
                 import scripts.realtime_inference as ri
                 from scripts.realtime_inference import load_all_model as ri_load_all_model
                 self.realtime_infer_module = ri # Store the module
            else:
                 ri = self.realtime_infer_module # Use stored module
                 from scripts.realtime_inference import load_all_model as ri_load_all_model

            from musetalk.utils.face_parsing import FaceParsing
            from musetalk.utils.audio_processor import AudioProcessor
            from transformers import WhisperModel

            # Set device and dtype
            device_str = "cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device_str)

            self.weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            logger.info("Using device: {}, weight_dtype: {}".format(self.device, self.weight_dtype))
            # Make device and dtype available to the realtime_inference module if needed
            ri.device = self.device
            ri.weight_dtype = self.weight_dtype

            # === Correctly read model paths from config['models'] ===
            models_config = self.config.get("models", {})
            models_base_dir_rel = models_config.get("base_dir", "../models") # Relative to config file location? Let's resolve from project root for consistency.
            models_base_path = Path(resolve_path(models_base_dir_rel, self.project_root))
            logger.info("Resolved models base path: {}".format(models_base_path))

            whisper_config = models_config.get("whisper", {})
            whisper_rel_path = whisper_config.get("path", "whisper")
            whisper_model_path = models_base_path / whisper_rel_path

            unet_config = models_config.get("unet", {})
            unet_rel_path = unet_config.get("path", "musetalkV15") # Default from yml
            unet_base_path = models_base_path / unet_rel_path
            unet_model_file = unet_config.get("model", "unet.pth") # Default from yml
            unet_config_file = unet_config.get("config", "musetalk.json") # Default from yml
            unet_path = unet_base_path / unet_model_file
            unet_config_path = unet_base_path / unet_config_file

            vae_config = models_config.get("vae", {})
            vae_rel_path = vae_config.get("path", "sd-vae") # Default from yml
            vae_model_path = models_base_path / vae_rel_path # Path to the VAE directory
            # load_all_model expects the directory path or a known name for vae_type
            vae_type_for_load = str(vae_model_path)
            # Optional: Check if vae_type is a simple name like 'sd-vae', if so, use it directly
            if "/" not in vae_rel_path and "\\" not in vae_rel_path and not vae_rel_path.startswith("."):
                 logger.info("Using VAE type name from config: {}".format(vae_rel_path))
                 vae_type_for_load = vae_rel_path # Use the name if it's not a path

            # === Explicit Path Checks BEFORE loading ===
            if not whisper_model_path.is_dir():
                raise FileNotFoundError("Whisper model directory not found at resolved path: {}. Check config `models.whisper.path` and `models.base_dir`.".format(whisper_model_path))

            if not unet_path.is_file():
                raise FileNotFoundError("MuseTalk UNet model file not found at resolved path: {}. Check config `models.unet.path`, `models.unet.model`, and `models.base_dir`.".format(unet_path))

            if not unet_config_path.is_file():
                 raise FileNotFoundError("MuseTalk UNet config file not found at resolved path: {}. Check config `models.unet.path`, `models.unet.config`, and `models.base_dir`.".format(unet_config_path))

            # Check VAE path (which should be a directory)
            vae_check_path = Path(vae_type_for_load) if "/" in vae_type_for_load or "\\" in vae_type_for_load else models_base_path / vae_type_for_load
            if not vae_check_path.is_dir():
                 raise FileNotFoundError("VAE directory not found at resolved path: {}. Check config `models.vae.path` and `models.base_dir`.".format(vae_check_path))
            # Check for VAE weights within the directory
            vae_weight_file = vae_config.get("model", "diffusion_pytorch_model.bin") # Get specific weight filename
            vae_weight_path = vae_check_path / vae_weight_file
            vae_alt_weight_path = vae_check_path / "diffusion_pytorch_model.safetensors" # Common alternative
            if not vae_weight_path.is_file() and not vae_alt_weight_path.is_file():
                 raise FileNotFoundError("VAE directory found at {}, but missing required weight file ({} or .safetensors). Check config `models.vae.model`.".format(vae_check_path, vae_weight_file))

            # === End Path Checks ===

            # Load models using the resolved paths
            logger.info("Loading VAE, UNet, PE...")
            self.vae, self.unet, self.pe = ri_load_all_model(
                unet_model_path=str(unet_path),
                # Use the resolved path or name for vae_type
                vae_type=vae_type_for_load,
                unet_config=str(unet_config_path),
                device=self.device
            )
            self.timesteps = torch.tensor([0], device=self.device)
            logger.info("VAE, UNet, PE loaded.")

            # 将模型传递给realtime_inference模块
            ri.vae = self.vae
            ri.unet = self.unet
            ri.pe = self.pe
            ri.device = self.device
            ri.weight_dtype = self.weight_dtype
            ri.timesteps = self.timesteps

            # Move models to appropriate device and dtype
            self.pe = self.pe.to(self.device, dtype=self.weight_dtype)
            self.vae.vae = self.vae.vae.to(self.device, dtype=self.weight_dtype)
            self.unet.model = self.unet.model.to(self.device, dtype=self.weight_dtype)

            # Initialize audio processor and Whisper model
            logger.info("Initializing AudioProcessor with whisper path: {}".format(whisper_model_path))
            self.audio_processor = AudioProcessor(feature_extractor_path=str(whisper_model_path))
            # 设置ri.audio_processor确保realtime_inference模块可以访问
            ri.audio_processor = self.audio_processor
            
            logger.info("Loading Whisper model from: {}".format(whisper_model_path))
            self.whisper = WhisperModel.from_pretrained(str(whisper_model_path))
            self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
            self.whisper.requires_grad_(False)
            logger.info("Whisper model loaded.")
            ri.whisper = self.whisper # Set for the module if needed elsewhere

            # Setup Args class needed by realtime_inference (Use config values)
            class Args:
                pass
            args = Args()
            rt_args_config = self.config.get("realtime_inference_args", {})
            args.version = rt_args_config.get("version", "v15") # Default v15
            args.extra_margin = rt_args_config.get("extra_margin", 10)
            args.parsing_mode = rt_args_config.get("parsing_mode", "jaw")
            args.skip_save_images = rt_args_config.get("skip_save_images", False)
            args.audio_padding_length_left = rt_args_config.get("audio_padding_length_left", 2)
            args.audio_padding_length_right = rt_args_config.get("audio_padding_length_right", 2)
            args.left_cheek_width = rt_args_config.get("left_cheek_width", 90)
            args.right_cheek_width = rt_args_config.get("right_cheek_width", 90)
            self.realtime_infer_module.args = args # Set args for the module

            # Initialize FaceParsing (uses args)
            self.fp = FaceParsing(
                left_cheek_width=args.left_cheek_width,
                right_cheek_width=args.right_cheek_width
            )
            self.realtime_infer_module.fp = self.fp # Set for the module
            
            # 将模型保存到缓存
            try:
                from .model_cache import ModelCache
                cache = ModelCache.get_instance()
                cache.set_global_models({
                    "device": self.device,
                    "weight_dtype": self.weight_dtype,
                    "fp": self.fp,
                    "audio_processor": self.audio_processor,
                    "whisper": self.whisper,
                    "vae": self.vae,
                    "unet": self.unet,
                    "pe": self.pe,
                    "timesteps": self.timesteps,
                    "realtime_infer_module": self.realtime_infer_module
                })
                logger.info("模型已保存到缓存")
            except Exception as e:
                logger.warning("保存模型到缓存失败: {}".format(e))
            
            # 标记模型已加载
            self.models_loaded = True
            

        except FileNotFoundError as fnf_error:
            logger.error("Model file/directory not found: {}".format(fnf_error))
            # Don't raise RuntimeError here, let the caller handle the failed loading state
            self.models_loaded = False # Ensure flag is false
            # Re-raise the FileNotFoundError to be caught by the caller
            raise fnf_error
        except ImportError as e:
            logger.error("Failed to import MuseTalk components: {}. Check sys.path and dependencies.".format(e))
            raise RuntimeError("Standard/Advanced mode requires MuseTalk core library and models.") from e
        except Exception as e:
            logger.error("Error loading models: {}".format(e))
            import traceback
            logger.error(traceback.format_exc())
            self.models_loaded = False # Ensure flag is false if loading failed
            raise RuntimeError("Failed to initialize MuseTalk inference environment: {}".format(e)) from e
        finally:
            # Clean up sys.path modification if it was done
            if str(self.project_root) in sys.path and sys.path != original_sys_path:
                 try:
                     sys.path.remove(str(self.project_root))
                     logger.debug("Removed {} from sys.path".format(self.project_root))
                 except ValueError:
                     pass # Ignore if already removed or wasn't there

    def _generate_audio(self, text: str, **kwargs) -> Optional[str]:
        """
        生成音频文件
        
        Args:
            text: 文本内容
            **kwargs: 其他参数
                tts_model: TTS模型ID
                lang: 语言代码
                
        Returns:
            音频文件路径，如果生成失败则返回None
        """
        try:
            # 获取参数
            tts_model = kwargs.get("tts_model")
            lang = kwargs.get("lang", "zh")
            
            # 创建临时音频文件路径
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            audio_path = os.path.join(self.temp_dir, "tts_" + timestamp + ".wav")
            
            # 生成音频
            logger.info("开始生成音频: {}".format(audio_path))
            success, result = self.tts_service.text_to_speech(
                text=text,
                output_path=audio_path,
                model_id=tts_model,
                text_lang=lang
            )
            
            if not success:
                logger.error("音频生成失败: {}".format(result))
                return None
                
            logger.info("音频生成成功: {}".format(audio_path))
            return audio_path
            
        except Exception as e:
            logger.error("生成音频时出错: {}".format(str(e)))
            import traceback
            logger.error(traceback.format_exc())
            return None
            
    def _create_basic_video(self, avatar_image: str, audio_path: str, output_path: str, 
                           subtitle_path: str = None, subtitle_color: str = "#FFFFFF") -> Tuple[bool, str]:
        """
        创建基础视频 - 将静态头像图片和音频合并为视频
        
        Args:
            avatar_image: 头像图片路径
            audio_path: 音频文件路径
            output_path: 输出视频路径
            subtitle_path: 字幕文件路径
            subtitle_color: 字幕颜色（十六进制值，如"#FFFFFF"）
            
        Returns:
            (成功标志, 信息)
        """
        try:
            # 检查输入文件是否存在
            if not os.path.exists(avatar_image):
                return False, "头像图片不存在: {}".format(avatar_image)
                
            if not os.path.exists(audio_path):
                return False, "音频文件不存在: {}".format(audio_path)
                
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取音频信息
            audio_info = self._get_audio_info(audio_path)
            audio_duration = audio_info.get("duration", 0)
            
            if audio_duration <= 0:
                return False, "无法获取音频时长"
                
            # 构建ffmpeg命令
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", avatar_image,
                "-i", audio_path,
                "-c:v", "libx264",
                "-tune", "stillimage",
                "-c:a", "aac",
                "-strict", "experimental",
                "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-shortest"
            ]
            
            # 如果有字幕，添加字幕参数
            if subtitle_path and os.path.exists(subtitle_path):
                logger.info("添加字幕文件: {}".format(subtitle_path))
                
                # 获取字幕格式
                subtitle_ext = os.path.splitext(subtitle_path)[1].lower()
                
                # 添加字幕滤镜
                subtitle_filter = None
                if subtitle_ext == ".srt":
                    subtitle_filter = "subtitles={}:force_style='FontName=Arial,FontSize=24,PrimaryColour=&H{},'".format(
                        subtitle_path.replace(":", "\\:").replace("'", "\\'"),
                        subtitle_color.lstrip("#")
                    )
                elif subtitle_ext == ".vtt":
                    # 对于VTT格式，我们需要先将其转换为ASS格式
                    ass_path = subtitle_path.replace(".vtt", ".ass")
                    convert_cmd = [
                        "ffmpeg", "-y",
                        "-i", subtitle_path,
                        ass_path
                    ]
                    
                    logger.info("转换VTT到ASS格式: {}".format(" ".join(convert_cmd)))
                    try:
                        subprocess.run(convert_cmd, check=True, capture_output=True)
                        if os.path.exists(ass_path):
                            subtitle_filter = "subtitles={}:force_style='FontName=Arial,FontSize=24,PrimaryColour=&H{},'".format(
                                ass_path.replace(":", "\\:").replace("'", "\\'"),
                                subtitle_color.lstrip("#")
                            )
                        else:
                            logger.warning("ASS字幕文件生成失败")
                    except Exception as e:
                        logger.error("转换VTT到ASS格式失败: {}".format(str(e)))
                        
                # 添加字幕滤镜
                if subtitle_filter:
                    cmd.extend(["-vf", subtitle_filter])
                    logger.info("添加字幕滤镜: {}".format(subtitle_filter))
                else:
                    logger.warning("无法添加字幕滤镜，字幕将不会显示")
            
            # 添加输出路径
            cmd.append(output_path)
            
            # 执行命令
            logger.info("执行ffmpeg命令: {}".format(" ".join(cmd)))
            subprocess.run(cmd, check=True, capture_output=True)
            
            # 检查输出文件是否存在
            if not os.path.exists(output_path):
                return False, "视频文件生成失败"
                
            # 检查视频文件大小
            file_size = os.path.getsize(output_path)
            if file_size < 1000:
                return False, "视频文件太小，可能生成失败: {} bytes".format(file_size)
                
            logger.info("基础视频生成成功: {}, 大小: {:.2f} MB".format(output_path, file_size / (1024 * 1024)))
            return True, output_path
            
        except subprocess.CalledProcessError as e:
            error_msg = "执行ffmpeg命令失败: {}".format(str(e))
            if e.stderr:
                error_msg += "\n错误输出: {}".format(e.stderr.decode('utf-8', errors='ignore'))
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            logger.error("创建基础视频时出错: {}".format(str(e)))
            import traceback
            logger.error(traceback.format_exc())
            return False, "创建基础视频时出错: {}".format(str(e))

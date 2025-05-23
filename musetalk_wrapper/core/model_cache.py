#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import time
import gc
import torch
from typing import Dict, Any, Optional, List, Tuple, Union

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelCache")

class ModelCache:
    """
    模型和图片缓存管理器，用于避免重复加载数据
    
    缓存内容:
    1. 图片帧数据
    2. 模型权重
    3. Avatar参数和配置
    """
    
    _instance = None
    
    @staticmethod
    def get_instance():
        """获取单例实例"""
        if ModelCache._instance is None:
            ModelCache._instance = ModelCache()
        return ModelCache._instance
    
    def __init__(self):
        """初始化缓存管理器"""
        # 初始化图片缓存
        self.frames_cache = {}  # {avatar_id: frames}
        self.frames_cache_info = {}  # {avatar_id: {"last_used": timestamp, "size": bytes}}
        
        # 初始化模型缓存
        self.model_cache = {}  # {avatar_id: model_data}
        self.model_cache_info = {}  # {avatar_id: {"last_used": timestamp}}
        
        # 初始化Avatar实例缓存
        self.avatar_cache = {}  # {avatar_id: avatar_instance}
        self.avatar_cache_info = {}  # {avatar_id: {"last_used": timestamp}}
        
        # 缓存配置
        self.max_frames_cache_size = 5000  # 最大图片缓存数量
        self.max_model_cache_size = 2   # 最大模型缓存数量
        self.max_avatar_cache_size = 2  # 最大Avatar实例缓存数量
        
        # 当前加载的模型ID
        self.current_model_id = None
        
        # 全局模型缓存 (不绑定到特定avatar_id)
        self.global_models = {
            "loaded": False,
            "device": None,
            "vae": None,
            "unet": None,
            "pe": None,
            "whisper": None,
            "audio_processor": None,
            "fp": None,
            "timesteps": None,
            "realtime_infer_module": None,
            "last_used": time.time()
        }
    
    def get_frames(self, avatar_id: str, frames_path: str, reload: bool = False) -> Optional[List]:
        """
        获取或加载头像图片帧
        
        Args:
            avatar_id: 头像ID
            frames_path: 图片帧路径
            reload: 是否强制重新加载
            
        Returns:
            图片帧列表
        """
        # 更新缓存使用时间
        if avatar_id in self.frames_cache:
            self.frames_cache_info[avatar_id]["last_used"] = time.time()
            
            # 如果不需要重新加载，直接返回缓存
            if not reload:
                logger.info("使用缓存的图片帧: {}".format(avatar_id))
                return self.frames_cache[avatar_id]
        
        # 加载图片帧
        try:
            logger.info("从路径加载图片帧: {}".format(frames_path))
            import cv2
            import glob
            
            # 获取所有图片文件
            frame_files = sorted(glob.glob(os.path.join(frames_path, "*.png")))
            
            if not frame_files:
                logger.warning("未找到图片帧: {}".format(frames_path))
                return None
            
            # 读取图片
            frames = []
            for frame_file in frame_files:
                frame = cv2.imread(frame_file)
                if frame is not None:
                    frames.append(frame)
            
            # 缓存图片
            self.frames_cache[avatar_id] = frames
            self.frames_cache_info[avatar_id] = {
                "last_used": time.time(),
                "size": len(frames) * frames[0].nbytes if frames else 0
            }
            
            # 清理多余缓存
            self._clean_frames_cache()
            
            return frames
            
        except Exception as e:
            logger.error("加载图片帧出错: {}".format(e))
            return None
    
    def get_avatar_instance(self, avatar_id: str, avatar_class: Any, params: Dict[str, Any] = None, reload: bool = False) -> Any:
        """
        获取或创建头像实例
        
        Args:
            avatar_id: 头像ID
            avatar_class: Avatar类
            params: 创建参数
            reload: 是否强制重新创建
            
        Returns:
            Avatar实例
        """
        # 更新缓存使用时间
        if avatar_id in self.avatar_cache:
            self.avatar_cache_info[avatar_id]["last_used"] = time.time()
            
            # 如果不需要重新创建，直接返回缓存
            if not reload:
                logger.info("使用缓存的Avatar实例: {}".format(avatar_id))
                return self.avatar_cache[avatar_id]
        
        # 创建Avatar实例
        try:
            if params is None:
                params = {}
                
            logger.info("创建新的Avatar实例: {}".format(avatar_id))
            
            # 创建实例
            avatar_instance = avatar_class(
                avatar_id=avatar_id,
                **params
            )
            
            # 缓存实例
            self.avatar_cache[avatar_id] = avatar_instance
            self.avatar_cache_info[avatar_id] = {
                "last_used": time.time()
            }
            
            # 清理多余缓存
            self._clean_avatar_cache()
            
            return avatar_instance
            
        except Exception as e:
            logger.error("创建Avatar实例出错: {}".format(e))
            return None
    
    def get_global_models(self) -> Dict[str, Any]:
        """获取全局模型缓存"""
        self.global_models["last_used"] = time.time()
        return self.global_models
    
    def set_global_models(self, models_data: Dict[str, Any]):
        """设置全局模型缓存"""
        self.global_models.update(models_data)
        self.global_models["last_used"] = time.time()
        self.global_models["loaded"] = True
    
    def clear_global_models(self):
        """清除全局模型缓存"""
        if self.global_models["loaded"]:
            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 重置模型缓存
            self.global_models = {
                "loaded": False,
                "device": None,
                "vae": None,
                "unet": None,
                "pe": None,
                "whisper": None,
                "audio_processor": None,
                "fp": None,
                "timesteps": None,
                "realtime_infer_module": None,
                "last_used": time.time()
            }
            
            # 强制垃圾回收
            gc.collect()
            logger.info("全局模型缓存已清除")
    
    def _clean_frames_cache(self):
        """清理过多的图片帧缓存"""
        if len(self.frames_cache) <= self.max_frames_cache_size:
            return
        
        # 按最后使用时间排序
        sorted_keys = sorted(
            self.frames_cache_info.keys(),
            key=lambda k: self.frames_cache_info[k]["last_used"]
        )
        
        # 移除最早使用的缓存
        while len(self.frames_cache) > self.max_frames_cache_size:
            key_to_remove = sorted_keys.pop(0)
            del self.frames_cache[key_to_remove]
            del self.frames_cache_info[key_to_remove]
            logger.info("清除图片帧缓存: {}".format(key_to_remove))
    
    def _clean_avatar_cache(self):
        """清理过多的Avatar实例缓存"""
        if len(self.avatar_cache) <= self.max_avatar_cache_size:
            return
        
        # 按最后使用时间排序
        sorted_keys = sorted(
            self.avatar_cache_info.keys(),
            key=lambda k: self.avatar_cache_info[k]["last_used"]
        )
        
        # 移除最早使用的缓存
        while len(self.avatar_cache) > self.max_avatar_cache_size:
            key_to_remove = sorted_keys.pop(0)
            del self.avatar_cache[key_to_remove]
            del self.avatar_cache_info[key_to_remove]
            logger.info("清除Avatar实例缓存: {}".format(key_to_remove))
    
    def clear_all_cache(self):
        """清除所有缓存"""
        self.frames_cache.clear()
        self.frames_cache_info.clear()
        self.avatar_cache.clear()
        self.avatar_cache_info.clear()
        self.clear_global_models()
        logger.info("所有缓存已清除") 

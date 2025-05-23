import os
import time
import json
import logging
import torch
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Callable, Union

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FallbackManager")

class CapabilityLevel(Enum):
    """系统可用功能级别"""
    AUDIO_ONLY = "audio_only"  # 仅音频模式
    BASIC = "basic"            # 基础模式：使用ffmpeg静态帧+音频合成
    STANDARD = "standard"      # 标准模式：2D动画生成
    ADVANCED = "advanced"      # 高级模式：3D动画生成

class GenerationMode(Enum):
    """视频生成模式"""
    ADVANCED = auto()  # 高级模式：完整MuseTalk功能
    STANDARD = auto()  # 标准模式：简化的Avatar功能
    BASIC = auto()     # 基础模式：静态图片+音频
    FAILED = auto()    # 生成失败

class FallbackManager:
    """降级管理器，负责决定何时降级及降级策略"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化降级管理器
        
        Args:
            config: 配置信息
        """
        self.config = config
        
        # 配置降级路径
        self.fallback_path = {
            CapabilityLevel.ADVANCED: CapabilityLevel.STANDARD,
            CapabilityLevel.STANDARD: CapabilityLevel.BASIC,
            CapabilityLevel.BASIC: CapabilityLevel.AUDIO_ONLY,
            CapabilityLevel.AUDIO_ONLY: None
        }
        
        # 配置失败计数和阈值
        self.failure_threshold = config.get("failure_threshold", 3)
        self.failure_window = config.get("failure_window", 600)  # 10分钟窗口
        self.failure_records = {
            CapabilityLevel.ADVANCED: [],
            CapabilityLevel.STANDARD: [],
            CapabilityLevel.BASIC: [],
            CapabilityLevel.AUDIO_ONLY: []
        }
        
        # 读取持久化的降级状态（如果有）
        self.state_file = config.get("state_file", "./fallback_state.json")
        self._load_state()
        
        # 降级锁定状态
        self.locked_levels = set()
        
        logger.info("降级管理器初始化完成")
    
    def _load_state(self):
        """加载持久化的降级状态"""
        if not os.path.exists(self.state_file):
            return
        
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # 加载锁定的级别
            self.locked_levels = set()
            for level_name in state.get("locked_levels", []):
                try:
                    self.locked_levels.add(CapabilityLevel(level_name))
                except (ValueError, KeyError):
                    logger.warning(f"无法识别的降级级别: {level_name}")
            
            # 加载失败记录
            for level_name, records in state.get("failure_records", {}).items():
                try:
                    level = CapabilityLevel(level_name)
                    self.failure_records[level] = records
                except (ValueError, KeyError):
                    logger.warning(f"无法识别的降级级别: {level_name}")
            
            logger.info(f"从 {self.state_file} 加载降级状态")
        except Exception as e:
            logger.error(f"加载降级状态文件失败: {str(e)}")
    
    def _save_state(self):
        """持久化当前的降级状态"""
        try:
            state = {
                "locked_levels": [level.value for level in self.locked_levels],
                "failure_records": {
                    level.value: records for level, records in self.failure_records.items()
                },
                "updated_at": time.time()
            }
            
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(self.state_file)), exist_ok=True)
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            
            logger.debug(f"降级状态已保存到 {self.state_file}")
        except Exception as e:
            logger.error(f"保存降级状态文件失败: {str(e)}")
    
    def should_fallback(self, mode: Any) -> bool:
        """
        判断是否应该从当前模式降级
        
        Args:
            mode: 当前模式，可以是字符串、枚举或其他类型，将自动转换为CapabilityLevel
        
        Returns:
            是否应该降级
        """
        # 转换为CapabilityLevel
        level = self._convert_to_capability_level(mode)
        if level is None:
            return False
        
        # 如果级别已被锁定，则应该降级
        if level in self.locked_levels:
            logger.info(f"级别 {level.name} 已被锁定，需要降级")
            return True
        
        # 检查最近的失败记录
        current_time = time.time()
        recent_failures = [
            t for t in self.failure_records[level]
            if current_time - t < self.failure_window
        ]
        
        # 清理旧的失败记录
        self.failure_records[level] = recent_failures
        
        # 如果最近失败次数超过阈值，应该降级
        if len(recent_failures) >= self.failure_threshold:
            logger.warning(f"级别 {level.name} 最近失败次数 ({len(recent_failures)}) 超过阈值 ({self.failure_threshold})，需要降级")
            self.locked_levels.add(level)
            self._save_state()
            return True
        
        return False
    
    def get_next_mode(self, mode: Any) -> Optional[Any]:
        """
        获取当前模式的下一个降级模式
        
        Args:
            mode: 当前模式，可以是字符串、枚举或其他类型，将自动转换为CapabilityLevel
        
        Returns:
            下一个降级模式，如果没有可用的降级模式则返回None
        """
        # 转换为CapabilityLevel
        level = self._convert_to_capability_level(mode)
        if level is None:
            return None
        
        # 获取降级路径中的下一级
        next_level = self.fallback_path.get(level)
        
        # 如果下一级也被锁定，递归查找可用级别
        while next_level and next_level in self.locked_levels:
            next_level = self.fallback_path.get(next_level)
        
        # 如果mode是枚举类型，并且与CapabilityLevel不同，转换回原始枚举类型
        if isinstance(mode, Enum) and mode.__class__ != CapabilityLevel:
            # 尝试找到原始枚举中对应的值
            if next_level:
                for original_mode in mode.__class__:
                    if original_mode.value == next_level.value:
                        return original_mode
            return None
        
        return next_level
    
    def _convert_to_capability_level(self, mode: Any) -> Optional[CapabilityLevel]:
        """
        将任意模式转换为CapabilityLevel
        
        Args:
            mode: 模式，可以是字符串、枚举或其他类型
        
        Returns:
            对应的CapabilityLevel，如果无法转换则返回None
        """
        if isinstance(mode, CapabilityLevel):
            return mode
        
        # 如果是其他枚举类型，使用value进行映射
        if isinstance(mode, Enum):
            try:
                return CapabilityLevel(mode.value)
            except (ValueError, KeyError):
                logger.warning(f"无法将枚举值 {mode.value} 转换为CapabilityLevel")
                return None
        
        # 如果是字符串，直接尝试转换
        if isinstance(mode, str):
            try:
                return CapabilityLevel(mode)
            except (ValueError, KeyError):
                logger.warning(f"无法将字符串 {mode} 转换为CapabilityLevel")
                return None
        
        logger.warning(f"无法将类型 {type(mode)} 转换为CapabilityLevel")
        return None
    
    def report_failure(self, mode: Any):
        """
        报告一次失败，用于追踪失败统计
        
        Args:
            mode: 失败的模式，可以是字符串、枚举或其他类型，将自动转换为CapabilityLevel
        """
        # 转换为CapabilityLevel
        level = self._convert_to_capability_level(mode)
        if level is None:
            return
        
        # 记录失败时间
        self.failure_records[level].append(time.time())
        
        # 持久化状态
        self._save_state()
        
        logger.info(f"记录 {level.name} 模式的一次失败")
    
    def reset_level_lock(self, mode: Any) -> bool:
        """
        重置特定级别的锁定状态
        
        Args:
            mode: 要重置的模式，可以是字符串、枚举或其他类型，将自动转换为CapabilityLevel
        
        Returns:
            是否成功重置
        """
        # 转换为CapabilityLevel
        level = self._convert_to_capability_level(mode)
        if level is None:
            return False
        
        # 清除锁定和失败记录
        if level in self.locked_levels:
            self.locked_levels.remove(level)
            self.failure_records[level] = []
            self._save_state()
            logger.info(f"已重置 {level.name} 模式的锁定状态")
            return True
        
        return False
    
    def reset_all_locks(self):
        """重置所有级别的锁定状态"""
        self.locked_levels.clear()
        for level in self.failure_records.keys():
            self.failure_records[level] = []
        
        self._save_state()
        logger.info("已重置所有模式的锁定状态")
    
    def record_result(self, mode: GenerationMode, success: bool):
        """
        记录生成结果，用于兼容旧代码
        
        Args:
            mode: 生成模式
            success: 是否成功
        """
        if not success:
            self.report_failure(mode)
            logger.info(f"记录 {mode.name} 模式生成失败")
        else:
            logger.info(f"记录 {mode.name} 模式生成成功")
    
    def should_skip_mode(self, mode: GenerationMode) -> bool:
        """
        决定是否应该跳过某种生成模式
        
        Args:
            mode: 生成模式
            
        Returns:
            是否应该跳过该模式
        """
        # 检查是否应该降级
        return self.should_fallback(mode)
    
    def suggest_starting_mode(self, available_modes: List[GenerationMode]) -> GenerationMode:
        """
        根据历史记录推荐起始生成模式
        
        Args:
            available_modes: 当前环境下可用的模式列表
            
        Returns:
            推荐的起始模式
        """
        # 按优先级排序可用模式
        prioritized_modes = [
            mode for mode in [
                GenerationMode.ADVANCED,
                GenerationMode.STANDARD,
                GenerationMode.BASIC
            ] if mode in available_modes
        ]
        
        # 如果没有可用模式，返回FAILED
        if not prioritized_modes:
            return GenerationMode.FAILED
        
        # 检查每种模式是否应该跳过
        for mode in prioritized_modes:
            if not self.should_skip_mode(mode):
                return mode
        
        # 如果所有模式都不建议使用，选择最基础的可用模式
        return prioritized_modes[-1]
    
    def execute_with_fallback(
        self,
        advanced_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        standard_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        basic_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        available_modes: Optional[List[GenerationMode]] = None
    ) -> Dict[str, Any]:
        """
        执行带有多级降级的视频生成
        
        Args:
            advanced_fn: 高级模式生成函数
            standard_fn: 标准模式生成函数
            basic_fn: 基础模式生成函数
            available_modes: 可用的模式列表，如果为None则根据提供的函数自动确定
            
        Returns:
            生成结果
        """
        # 确定可用模式
        if available_modes is None:
            available_modes = []
            if advanced_fn is not None:
                available_modes.append(GenerationMode.ADVANCED)
            if standard_fn is not None:
                available_modes.append(GenerationMode.STANDARD)
            if basic_fn is not None:
                available_modes.append(GenerationMode.BASIC)
        
        # 根据历史记录和可用模式确定起始模式
        start_mode = self.suggest_starting_mode(available_modes)
        
        if start_mode == GenerationMode.FAILED:
            logger.error("没有可用的生成模式")
            return {"success": False, "error": "没有可用的生成模式", "mode": "failed"}
        
        # 尝试高级模式
        if start_mode == GenerationMode.ADVANCED and GenerationMode.ADVANCED in available_modes:
            try:
                logger.info("尝试使用高级模式生成")
                result = advanced_fn()
                success = result.get("success", False)
                self.record_result(GenerationMode.ADVANCED, success)
                
                if success:
                    return result
                
                logger.warning("高级模式生成失败，尝试降级")
            except Exception as e:
                logger.error(f"高级模式生成发生异常: {str(e)}")
                self.record_result(GenerationMode.ADVANCED, False)
        
        # 尝试标准模式
        if GenerationMode.STANDARD in available_modes:
            try:
                logger.info("尝试使用标准模式生成")
                result = standard_fn()
                success = result.get("success", False)
                self.record_result(GenerationMode.STANDARD, success)
                
                if success:
                    return result
                
                logger.warning("标准模式生成失败，尝试降级")
            except Exception as e:
                logger.error(f"标准模式生成发生异常: {str(e)}")
                self.record_result(GenerationMode.STANDARD, False)
        
        # 尝试基础模式
        if GenerationMode.BASIC in available_modes:
            try:
                logger.info("尝试使用基础模式生成")
                result = basic_fn()
                success = result.get("success", False)
                self.record_result(GenerationMode.BASIC, success)
                
                if success:
                    return result
                
                logger.error("基础模式生成失败")
            except Exception as e:
                logger.error(f"基础模式生成发生异常: {str(e)}")
                self.record_result(GenerationMode.BASIC, False)
        
        # 所有模式都失败
        return {"success": False, "error": "所有生成模式均失败", "mode": "failed"}
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取各种模式的成功率统计
        
        Returns:
            统计信息字典
        """
        stats = {}
        
        for mode, history in self.success_history.items():
            if not history:
                stats[mode.name] = {
                    "total": 0,
                    "success": 0,
                    "failure": 0,
                    "success_rate": 0.0
                }
                continue
                
            success_count = sum(1 for result in history if result)
            total_count = len(history)
            
            stats[mode.name] = {
                "total": total_count,
                "success": success_count,
                "failure": total_count - success_count,
                "success_rate": success_count / total_count if total_count > 0 else 0.0
            }
        
        return stats
    
    def get_diagnostic_info(self) -> Dict[str, Any]:
        """
        获取诊断信息
        
        Returns:
            诊断信息字典
        """
        info = {
            "statistics": self.get_statistics(),
            "recent_mode": self.recent_mode.name if self.recent_mode else None,
            "fallback_threshold": self.fallback_threshold,
            "max_history_size": self.max_history_size,
            "cuda_available": torch.cuda.is_available()
        }
        
        # 添加CUDA诊断信息
        if torch.cuda.is_available():
            info["cuda"] = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB",
                "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB"
            }
        
        return info

    def determine_mode(self, hardware_capability: Dict[str, bool]) -> GenerationMode:
        """
        根据硬件能力确定适合的生成模式
        
        Args:
            hardware_capability: 硬件能力字典，包含cuda、mps等键

        Returns:
            推荐的生成模式
        """
        # 首先尝试高级模式 (3D动画生成)
        if (hardware_capability.get("cuda", False) or 
            hardware_capability.get("mps", False)):
            mode = GenerationMode.ADVANCED
            # 检查是否需要降级
            if self.should_fallback(mode):
                next_mode = self.get_next_mode(mode)
                if next_mode:
                    logger.info(f"降级从 {mode.name} 到 {next_mode.name}")
                    mode = next_mode
            return mode
        
        # 然后是标准模式 (2D动画生成)
        mode = GenerationMode.STANDARD
        if self.should_fallback(mode):
            next_mode = self.get_next_mode(mode)
            if next_mode:
                logger.info(f"降级从 {mode.name} 到 {next_mode.name}")
                mode = next_mode
        return mode 
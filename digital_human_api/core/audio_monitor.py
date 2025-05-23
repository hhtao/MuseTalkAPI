import asyncio
import logging
import numpy as np
import queue
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import webrtcvad
from digital_human_api.core.state_manager import StateManager, DigitalHumanState
from digital_human_api.core.external_services import ASRClient

class WakeWordDetector:
    """唤醒词检测器"""
    
    def __init__(self, wake_words: List[str] = None, sensitivity: float = 0.7):
        """
        初始化唤醒词检测器
        
        Args:
            wake_words: 唤醒词列表
            sensitivity: 检测灵敏度(0.0-1.0)
        """
        self.wake_words = wake_words or ["你好数字人", "嘿数字人", "Hello"]
        self.sensitivity = sensitivity
        self.logger = logging.getLogger("WakeWordDetector")
        self.logger.info(f"唤醒词检测器初始化，唤醒词: {self.wake_words}")
        
    def detect(self, text: str) -> bool:
        """
        检测文本中是否包含唤醒词
        
        Args:
            text: 要检测的文本
            
        Returns:
            bool: 是否包含唤醒词
        """
        text = text.lower()
        
        for wake_word in self.wake_words:
            if wake_word.lower() in text:
                self.logger.info(f"检测到唤醒词: {wake_word}")
                return True
                
        return False
        
    def set_sensitivity(self, sensitivity: float) -> None:
        """
        设置检测灵敏度
        
        Args:
            sensitivity: 检测灵敏度(0.0-1.0)
        """
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        
    def add_wake_word(self, wake_word: str) -> None:
        """
        添加唤醒词
        
        Args:
            wake_word: 要添加的唤醒词
        """
        if wake_word and wake_word not in self.wake_words:
            self.wake_words.append(wake_word)
            self.logger.info(f"添加唤醒词: {wake_word}")
            
    def remove_wake_word(self, wake_word: str) -> None:
        """
        移除唤醒词
        
        Args:
            wake_word: 要移除的唤醒词
        """
        if wake_word in self.wake_words and len(self.wake_words) > 1:
            self.wake_words.remove(wake_word)
            self.logger.info(f"移除唤醒词: {wake_word}")


class VADProcessor:
    """语音活动检测处理器"""
    
    def __init__(self, 
                sample_rate: int = 16000, 
                frame_duration_ms: int = 30,
                aggressiveness: int = 2):
        """
        初始化VAD处理器
        
        Args:
            sample_rate: 采样率
            frame_duration_ms: 帧长度(毫秒)
            aggressiveness: 灵敏度(0-3)
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.aggressiveness = min(3, max(0, aggressiveness))
        
        # 初始化WebRTC VAD
        self.vad = webrtcvad.Vad(self.aggressiveness)
        
        # 计算每帧的样本数
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        self.logger = logging.getLogger("VADProcessor")
        self.logger.info(f"VAD处理器初始化: 采样率={sample_rate}, 帧长={frame_duration_ms}ms, 灵敏度={aggressiveness}")
        
    def is_speech(self, audio_frame: bytes) -> bool:
        """
        检测音频帧是否包含语音
        
        Args:
            audio_frame: 音频帧数据
            
        Returns:
            bool: 是否是语音
        """
        # 确保帧大小正确
        if len(audio_frame) != self.frame_size * 2:  # 16位采样，每个样本2字节
            return False
            
        try:
            return self.vad.is_speech(audio_frame, self.sample_rate)
        except Exception as e:
            self.logger.error(f"VAD检测失败: {str(e)}")
            return False
            
    def set_aggressiveness(self, aggressiveness: int) -> None:
        """
        设置VAD灵敏度
        
        Args:
            aggressiveness: 灵敏度(0-3)
        """
        self.aggressiveness = min(3, max(0, aggressiveness))
        self.vad.set_mode(self.aggressiveness)
        self.logger.info(f"VAD灵敏度设置为: {self.aggressiveness}")


class AudioMonitor:
    """音频监听器，处理音频输入并根据状态进行不同处理"""
    
    def __init__(self, 
                state_manager: StateManager,
                asr_client: ASRClient,
                sample_rate: int = 16000,
                frame_duration_ms: int = 30,
                vad_aggressiveness: int = 2,
                min_speech_duration_ms: int = 500,
                min_silence_duration_ms: int = 700):
        """
        初始化音频监听器
        
        Args:
            state_manager: 状态管理器
            asr_client: ASR客户端
            sample_rate: 采样率
            frame_duration_ms: 帧长度(毫秒)
            vad_aggressiveness: VAD灵敏度(0-3)
            min_speech_duration_ms: 最小语音持续时间(毫秒)
            min_silence_duration_ms: 最小静音持续时间(毫秒)
        """
        self.state_manager = state_manager
        self.asr_client = asr_client
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        
        # 初始化VAD处理器
        self.vad_processor = VADProcessor(
            sample_rate=sample_rate,
            frame_duration_ms=frame_duration_ms,
            aggressiveness=vad_aggressiveness
        )
        
        # 初始化唤醒词检测器
        self.wake_word_detector = WakeWordDetector()
        
        # 配置时间参数
        self.min_speech_frames = int(min_speech_duration_ms / frame_duration_ms)
        self.min_silence_frames = int(min_silence_duration_ms / frame_duration_ms)
        
        # 初始化音频处理状态
        self.audio_buffer = bytearray()
        self.is_recording = False
        self.speech_frames_count = 0
        self.silence_frames_count = 0
        
        # 初始化回调函数
        self.interruption_callback = None
        
        # 日志
        self.logger = logging.getLogger("AudioMonitor")
        self.logger.info("音频监听器初始化完成")
        
        # 处理队列和线程
        self.audio_queue = queue.Queue()
        self.running = False
        self.processing_thread = None
        
    async def start(self) -> None:
        """启动音频监听器"""
        if self.running:
            return
            
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_audio_thread, daemon=True)
        self.processing_thread.start()
        self.logger.info("音频监听器启动")
        
    async def stop(self) -> None:
        """停止音频监听器"""
        if not self.running:
            return
            
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        # 清空队列
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
                
        self.logger.info("音频监听器停止")
        
    async def process_audio_frame(self, audio_frame: bytes) -> None:
        """
        处理音频帧
        
        Args:
            audio_frame: 音频帧数据
        """
        if not self.running:
            return
            
        # 将音频帧放入队列
        self.audio_queue.put(audio_frame)
        
    def _process_audio_thread(self) -> None:
        """音频处理线程"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while self.running:
            try:
                # 获取音频帧，超时1秒
                try:
                    audio_frame = self.audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                # 处理音频帧
                loop.run_until_complete(self._process_audio_frame_async(audio_frame))
                
            except Exception as e:
                self.logger.error(f"音频处理线程错误: {str(e)}")
                import traceback
                traceback.print_exc()
                
        loop.close()
        
    async def _process_audio_frame_async(self, audio_frame: bytes) -> None:
        """
        异步处理音频帧
        
        Args:
            audio_frame: 音频帧数据
        """
        # 检测是否是语音
        is_speech = self.vad_processor.is_speech(audio_frame)
        
        # 获取当前状态
        current_state = await self.state_manager.get_current_state()
        
        # 根据当前状态处理音频
        if current_state == DigitalHumanState.IDLE_LISTENING:
            await self._handle_idle_listening(audio_frame, is_speech)
        elif current_state == DigitalHumanState.ACTIVE_LISTENING:
            await self._handle_active_listening(audio_frame, is_speech)
        elif current_state == DigitalHumanState.SPEAKING:
            await self._handle_speaking(audio_frame, is_speech)
            
    async def _handle_idle_listening(self, audio_frame: bytes, is_speech: bool) -> None:
        """
        处理空闲监听状态下的音频
        
        Args:
            audio_frame: 音频帧数据
            is_speech: 是否是语音
        """
        if is_speech:
            self.speech_frames_count += 1
            self.silence_frames_count = 0
            
            # 开始录音
            if not self.is_recording and self.speech_frames_count >= self.min_speech_frames:
                self.is_recording = True
                self.audio_buffer = bytearray()
                self.logger.info("开始录音(空闲状态)")
                
            # 添加到缓冲区
            if self.is_recording:
                self.audio_buffer.extend(audio_frame)
        else:
            if self.is_recording:
                self.silence_frames_count += 1
                self.speech_frames_count = 0
                
                # 添加静音帧到缓冲区
                self.audio_buffer.extend(audio_frame)
                
                # 检测语音结束
                if self.silence_frames_count >= self.min_silence_frames:
                    self.is_recording = False
                    
                    # 检测唤醒词
                    audio_data = bytes(self.audio_buffer)
                    await self._check_wake_word(audio_data)
                    
                    # 重置缓冲区
                    self.audio_buffer = bytearray()
            else:
                # 未录音状态下的静音，重置计数器
                self.speech_frames_count = 0
                self.silence_frames_count = 0
                
    async def _handle_active_listening(self, audio_frame: bytes, is_speech: bool) -> None:
        """
        处理主动倾听状态下的音频
        
        Args:
            audio_frame: 音频帧数据
            is_speech: 是否是语音
        """
        if is_speech:
            self.speech_frames_count += 1
            self.silence_frames_count = 0
            
            # 开始录音
            if not self.is_recording and self.speech_frames_count >= self.min_speech_frames:
                self.is_recording = True
                self.audio_buffer = bytearray()
                self.logger.info("开始录音(倾听状态)")
                
            # 添加到缓冲区
            if self.is_recording:
                self.audio_buffer.extend(audio_frame)
        else:
            if self.is_recording:
                self.silence_frames_count += 1
                self.speech_frames_count = 0
                
                # 添加静音帧到缓冲区
                self.audio_buffer.extend(audio_frame)
                
                # 检测语音结束
                if self.silence_frames_count >= self.min_silence_frames:
                    self.is_recording = False
                    
                    # 处理用户输入
                    audio_data = bytes(self.audio_buffer)
                    await self._process_user_input(audio_data)
                    
                    # 重置缓冲区
                    self.audio_buffer = bytearray()
            else:
                # 未录音状态下的静音，递增静音计数
                self.silence_frames_count += 1
                
                # 长时间静音，返回空闲状态
                if self.silence_frames_count >= 5 * self.min_silence_frames:  # 5倍最小静音时间
                    self.logger.info("长时间静音，返回空闲状态")
                    await self.state_manager.set_state(DigitalHumanState.IDLE_LISTENING)
                    self.speech_frames_count = 0
                    self.silence_frames_count = 0
                    
    async def _handle_speaking(self, audio_frame: bytes, is_speech: bool) -> None:
        """
        处理说话状态下的音频(检测打断)
        
        Args:
            audio_frame: 音频帧数据
            is_speech: 是否是语音
        """
        if is_speech:
            self.speech_frames_count += 1
            self.silence_frames_count = 0
            
            # 检测可能的打断
            if not self.is_recording and self.speech_frames_count >= self.min_speech_frames:
                self.is_recording = True
                self.audio_buffer = bytearray()
                self.logger.info("检测到可能的打断，开始录音")
                
            # 添加到缓冲区
            if self.is_recording:
                self.audio_buffer.extend(audio_frame)
        else:
            if self.is_recording:
                self.silence_frames_count += 1
                self.speech_frames_count = 0
                
                # 添加静音帧到缓冲区
                self.audio_buffer.extend(audio_frame)
                
                # 检测语音结束
                if self.silence_frames_count >= self.min_silence_frames:
                    self.is_recording = False
                    
                    # 处理打断
                    audio_data = bytes(self.audio_buffer)
                    await self._process_interruption(audio_data)
                    
                    # 重置缓冲区
                    self.audio_buffer = bytearray()
            else:
                # 未录音状态下的静音，重置计数器
                self.speech_frames_count = 0
                
    async def _check_wake_word(self, audio_data: bytes) -> None:
        """
        检查是否包含唤醒词
        
        Args:
            audio_data: 音频数据
        """
        try:
            # 将音频转换为文本
            text = await self.asr_client.speech_to_text(audio_data)
            
            if not text:
                return
                
            self.logger.info(f"ASR结果: {text}")
            
            # 检测唤醒词
            if self.wake_word_detector.detect(text):
                self.logger.info("检测到唤醒词，切换到倾听状态")
                
                # 播放唤醒提示音(可选)
                # await self._play_wake_tone()
                
                # 设置状态为倾听
                await self.state_manager.set_state(DigitalHumanState.ACTIVE_LISTENING)
        except Exception as e:
            self.logger.error(f"唤醒词检测错误: {str(e)}")
            
    async def _process_user_input(self, audio_data: bytes) -> None:
        """
        处理用户输入
        
        Args:
            audio_data: 音频数据
        """
        try:
            # 将音频转换为文本
            text = await self.asr_client.speech_to_text(audio_data)
            
            if not text:
                return
                
            self.logger.info(f"用户输入: {text}")
            
            # 保存用户输入到状态上下文
            await self.state_manager.save_state_context({"user_input": text})
            
            # 切换到思考状态
            await self.state_manager.set_state(DigitalHumanState.THINKING)
        except Exception as e:
            self.logger.error(f"处理用户输入错误: {str(e)}")
            await self.state_manager.set_state(DigitalHumanState.IDLE_LISTENING)
            
    async def _process_interruption(self, audio_data: bytes) -> None:
        """
        处理打断
        
        Args:
            audio_data: 音频数据
        """
        try:
            # 将音频转换为文本
            text = await self.asr_client.speech_to_text(audio_data)
            
            if not text:
                return
                
            self.logger.info(f"检测到打断: {text}")
            
            # 保存打断文本到状态上下文
            await self.state_manager.save_state_context({"interruption_text": text})
            
            # 触发打断回调
            if self.interruption_callback:
                await self.interruption_callback(text)
            
            # 设置状态为被打断
            await self.state_manager.set_state(DigitalHumanState.INTERRUPTED)
        except Exception as e:
            self.logger.error(f"处理打断错误: {str(e)}")
            
    async def register_interruption_callback(self, callback: Callable) -> None:
        """
        注册打断回调函数
        
        Args:
            callback: 回调函数
        """
        self.interruption_callback = callback
        self.logger.info("打断回调函数已注册")
        
    def set_vad_aggressiveness(self, aggressiveness: int) -> None:
        """
        设置VAD灵敏度
        
        Args:
            aggressiveness: 灵敏度(0-3)
        """
        self.vad_processor.set_aggressiveness(aggressiveness)
        
    def add_wake_word(self, wake_word: str) -> None:
        """
        添加唤醒词
        
        Args:
            wake_word: 唤醒词
        """
        self.wake_word_detector.add_wake_word(wake_word)
        
    def remove_wake_word(self, wake_word: str) -> None:
        """
        移除唤醒词
        
        Args:
            wake_word: 唤醒词
        """
        self.wake_word_detector.remove_wake_word(wake_word)
        
    def set_wake_word_sensitivity(self, sensitivity: float) -> None:
        """
        设置唤醒词灵敏度
        
        Args:
            sensitivity: 灵敏度(0.0-1.0)
        """
        self.wake_word_detector.set_sensitivity(sensitivity) 
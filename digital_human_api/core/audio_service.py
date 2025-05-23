import asyncio
import logging
import os
import numpy as np
import wave
import webrtcvad
from typing import Callable, Optional, List, Dict, Any
import audioop
import time
import threading
from threading import Lock

from digital_human_api.core.state_manager import StateManager, DigitalHumanState

class VoiceActivityDetector:
    """语音活动检测器，用于检测音频中是否包含语音"""
    
    def __init__(self, sample_rate: int = 16000, frame_duration_ms: int = 30, aggressiveness: int = 3):
        """
        初始化语音活动检测器
        
        Args:
            sample_rate: 音频采样率（Hz）
            frame_duration_ms: 帧长度（毫秒）
            aggressiveness: 灵敏度级别（0-3），3为最灵敏
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.aggressiveness = aggressiveness
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        # 初始化WebRTC VAD
        self.vad = webrtcvad.Vad(self.aggressiveness)
        logging.info(f"Initialized VAD with aggressiveness {aggressiveness}")
    
    def is_speech(self, audio_frame: bytes) -> bool:
        """
        判断音频帧是否包含语音
        
        Args:
            audio_frame: PCM音频帧数据（16位，单声道）
            
        Returns:
            bool: 是否包含语音
        """
        try:
            # 确保帧大小正确
            if len(audio_frame) != self.frame_size * 2:  # *2 因为16位PCM每个样本占2字节
                return False
                
            return self.vad.is_speech(audio_frame, self.sample_rate)
        except Exception as e:
            logging.error(f"VAD error: {str(e)}")
            return False
    
    def set_aggressiveness(self, aggressiveness: int) -> None:
        """设置VAD灵敏度"""
        if 0 <= aggressiveness <= 3:
            self.aggressiveness = aggressiveness
            self.vad.set_mode(aggressiveness)
            logging.info(f"VAD aggressiveness set to {aggressiveness}")
        else:
            logging.error(f"Invalid VAD aggressiveness: {aggressiveness}, must be 0-3")


class WakeWordDetector:
    """唤醒词检测器"""
    
    def __init__(self, wake_words: List[str] = None, threshold: float = 0.7):
        """
        初始化唤醒词检测器
        
        Args:
            wake_words: 唤醒词列表
            threshold: 置信度阈值
        """
        self.wake_words = wake_words or ["你好", "小智", "嗨"]
        self.threshold = threshold
        
        # 这里简化实现，实际项目中应该使用专门的唤醒词检测模型
        logging.info(f"Initialized wake word detector with words: {self.wake_words}")
    
    async def detect_wake_word(self, text: str) -> bool:
        """
        检测文本中是否包含唤醒词
        
        Args:
            text: 转换后的文本
            
        Returns:
            bool: 是否包含唤醒词
        """
        # 简单的字符串匹配，实际项目中应该使用更复杂的算法
        return any(word in text.lower() for word in self.wake_words)


class AudioMonitor:
    """音频监听器，负责处理用户语音输入、打断检测等功能"""
    
    def __init__(self, state_manager: StateManager, asr_client=None):
        """
        初始化音频监听器
        
        Args:
            state_manager: 状态管理器
            asr_client: 语音识别客户端
        """
        self.state_manager = state_manager
        self.asr_client = asr_client
        
        # 初始化VAD和唤醒词检测器
        self.vad = VoiceActivityDetector()
        self.wake_word_detector = WakeWordDetector()
        
        # 音频处理参数
        self.silence_threshold = 500  # 静音阈值（毫秒）
        self.max_speech_duration = 60000  # 最大语音持续时间（毫秒）
        self.speech_buffer = bytearray()  # 语音数据缓冲区
        self.last_speech_time = 0  # 上次检测到语音的时间
        self.is_recording = False  # 是否正在录制
        
        # 打断检测参数
        self.interruption_threshold = 300  # 打断检测阈值（毫秒）
        self.interruption_callback = None  # 打断回调函数
        
        # 线程同步
        self.buffer_lock = Lock()
        self.running = False
        self.processing_thread = None
        
        logging.info("AudioMonitor initialized")
    
    async def start(self) -> None:
        """启动音频监听器"""
        if self.running:
            logging.warning("AudioMonitor already running")
            return
            
        self.running = True
        
        # 注册状态变化回调
        await self.state_manager.register_state_change_callback(
            DigitalHumanState.IDLE_LISTENING, self._on_idle_state
        )
        await self.state_manager.register_state_change_callback(
            DigitalHumanState.ACTIVE_LISTENING, self._on_active_listening_state
        )
        await self.state_manager.register_state_change_callback(
            DigitalHumanState.SPEAKING, self._on_speaking_state
        )
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._audio_processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logging.info("AudioMonitor started")
    
    async def stop(self) -> None:
        """停止音频监听器"""
        if not self.running:
            return
            
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        with self.buffer_lock:
            self.speech_buffer.clear()
            self.is_recording = False
        
        logging.info("AudioMonitor stopped")
    
    async def register_interruption_callback(self, callback: Callable) -> None:
        """注册打断回调函数"""
        self.interruption_callback = callback
        logging.info("Interruption callback registered")
    
    async def process_audio_frame(self, audio_frame: bytes) -> None:
        """
        处理音频帧
        
        Args:
            audio_frame: PCM音频帧数据
        """
        with self.buffer_lock:
            current_state = await self.state_manager.get_current_state()
            is_speech = self.vad.is_speech(audio_frame)
            
            # 根据当前状态处理音频
            if current_state == DigitalHumanState.IDLE_LISTENING:
                await self._handle_idle_listening(audio_frame, is_speech)
            elif current_state == DigitalHumanState.ACTIVE_LISTENING:
                await self._handle_active_listening(audio_frame, is_speech)
            elif current_state == DigitalHumanState.SPEAKING:
                await self._handle_speaking(audio_frame, is_speech)
            
            # 更新最后检测到语音的时间
            if is_speech:
                self.last_speech_time = time.time() * 1000
    
    async def _handle_idle_listening(self, audio_frame: bytes, is_speech: bool) -> None:
        """处理空闲监听状态下的音频"""
        if not is_speech:
            return
            
        # 缓存音频并检测唤醒词
        self.speech_buffer.extend(audio_frame)
        
        # 如果缓冲区足够大，进行唤醒词检测
        if len(self.speech_buffer) > 32000:  # 大约2秒的音频
            if self.asr_client:
                # 使用ASR将语音转为文本
                try:
                    text = await self.asr_client.speech_to_text(bytes(self.speech_buffer))
                    # 检测唤醒词
                    if await self.wake_word_detector.detect_wake_word(text):
                        logging.info(f"Wake word detected: {text}")
                        # 进入主动倾听状态
                        await self.state_manager.set_state(DigitalHumanState.ACTIVE_LISTENING)
                except Exception as e:
                    logging.error(f"ASR error in idle listening: {str(e)}")
            
            # 清空缓冲区
            self.speech_buffer.clear()
    
    async def _handle_active_listening(self, audio_frame: bytes, is_speech: bool) -> None:
        """处理主动倾听状态下的音频"""
        now = time.time() * 1000
        
        # 检查是否已超过最大录音时间
        if self.is_recording and now - self.last_speech_time > self.max_speech_duration:
            logging.info(f"Maximum speech duration reached ({self.max_speech_duration}ms)")
            await self._finalize_speech()
            return
        
        # 如果检测到语音，开始/继续录制
        if is_speech:
            if not self.is_recording:
                logging.info("Speech started")
                self.is_recording = True
                self.speech_buffer.clear()
            
            # 添加音频数据到缓冲区
            self.speech_buffer.extend(audio_frame)
            self.last_speech_time = now
        
        # 如果正在录制但检测到静音，检查是否应该结束录制
        elif self.is_recording and now - self.last_speech_time > self.silence_threshold:
            logging.info(f"Silence detected for {now - self.last_speech_time}ms")
            await self._finalize_speech()
    
    async def _handle_speaking(self, audio_frame: bytes, is_speech: bool) -> None:
        """处理说话状态下的音频（检测打断）"""
        if not is_speech:
            return
            
        # 如果持续检测到语音超过打断阈值，触发打断
        if is_speech:
            self.speech_buffer.extend(audio_frame)
            
            # 如果检测到足够长的语音，可能是打断
            if len(self.speech_buffer) > self.interruption_threshold * 16:  # 16位采样率
                logging.info("Potential interruption detected")
                
                if self.asr_client:
                    try:
                        # 使用ASR获取打断内容
                        text = await self.asr_client.speech_to_text(bytes(self.speech_buffer))
                        if text and len(text.strip()) > 0:
                            logging.info(f"Interruption text: {text}")
                            
                            # 进入被打断状态
                            await self.state_manager.set_state(DigitalHumanState.INTERRUPTED)
                            
                            # 保存打断文本到上下文
                            await self.state_manager.save_state_context({
                                "interruption_text": text
                            })
                            
                            # 调用打断回调函数
                            if self.interruption_callback:
                                await self.interruption_callback(text)
                    except Exception as e:
                        logging.error(f"ASR error in interruption detection: {str(e)}")
                
                # 清空缓冲区
                self.speech_buffer.clear()
    
    async def _finalize_speech(self) -> None:
        """处理并分析完整的语音"""
        if not self.is_recording or len(self.speech_buffer) == 0:
            self.is_recording = False
            return
            
        logging.info(f"Finalizing speech, buffer size: {len(self.speech_buffer)} bytes")
        
        # 转换语音为文本
        if self.asr_client:
            try:
                text = await self.asr_client.speech_to_text(bytes(self.speech_buffer))
                logging.info(f"Speech recognized: {text}")
                
                # 保存识别文本到上下文
                await self.state_manager.save_state_context({
                    "user_input": text
                })
                
                # 转入思考状态
                await self.state_manager.set_state(DigitalHumanState.THINKING)
            except Exception as e:
                logging.error(f"ASR error in speech finalization: {str(e)}")
        
        # 清空缓冲区并重置录制状态
        self.speech_buffer.clear()
        self.is_recording = False
    
    def _audio_processing_loop(self) -> None:
        """音频处理循环（在单独线程中运行）"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while self.running:
            try:
                # 主要逻辑在process_audio_frame中，这里主要是一个空循环
                time.sleep(0.01)
            except Exception as e:
                logging.error(f"Error in audio processing loop: {str(e)}")
        
        loop.close()
    
    async def _on_idle_state(self, previous_state, new_state) -> None:
        """进入空闲状态的回调"""
        with self.buffer_lock:
            self.speech_buffer.clear()
            self.is_recording = False
    
    async def _on_active_listening_state(self, previous_state, new_state) -> None:
        """进入主动倾听状态的回调"""
        with self.buffer_lock:
            self.speech_buffer.clear()
            self.is_recording = False
            self.last_speech_time = time.time() * 1000
    
    async def _on_speaking_state(self, previous_state, new_state) -> None:
        """进入说话状态的回调"""
        with self.buffer_lock:
            self.speech_buffer.clear()
    
    async def save_audio_to_file(self, filename: str) -> None:
        """将当前缓冲区中的音频保存到文件"""
        with self.buffer_lock:
            if len(self.speech_buffer) == 0:
                logging.warning("Empty speech buffer, nothing to save")
                return
                
            try:
                with wave.open(filename, 'wb') as wf:
                    wf.setnchannels(1)  # 单声道
                    wf.setsampwidth(2)  # 16位
                    wf.setframerate(16000)  # 16kHz采样率
                    wf.writeframes(self.speech_buffer)
                
                logging.info(f"Audio saved to {filename}")
            except Exception as e:
                logging.error(f"Failed to save audio: {str(e)}") 
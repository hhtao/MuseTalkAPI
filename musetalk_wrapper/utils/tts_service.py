import os
import json
import time
import logging
import tempfile
import requests
import subprocess
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TTSService")

# 导入字幕处理模块
subtitle_modules_available = False
try:
    from .text_cut import get_method, get_method_names
    from .subtitle_utils import create_subtitle_for_text
    logger.info("成功导入字幕处理模块")
    subtitle_modules_available = True
except ImportError as e:
    logger.error("无法导入字幕处理模块，字幕功能将不可用: {}".format(str(e)))
    subtitle_modules_available = False

class TTSService:
    """TTS服务类，用于将文本转换为语音"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化TTS服务
        
        Args:
            config: TTS配置信息 (来自default.yml的tts部分)
        """
        self.config = config
        
        # 记录字幕模块是否可用
        self.subtitle_modules_available = subtitle_modules_available
        
        # 必需配置项检查
        if not config:
            raise ValueError("TTS配置不能为空，请检查default.yml的tts部分")
            
        # 服务器URL检查
        self.server_url = config.get("server")
        if not self.server_url:
            raise ValueError("TTS服务器URL未配置，请在default.yml的tts.server中设置")
            
        # 文本切分方法检查
        self.default_cut_method = config.get("default_text_cut_method")
        if not self.default_cut_method:
            raise ValueError("默认文本切分方法未配置，请在default.yml的tts.default_text_cut_method中设置")
            
        # 音频参数检查
        self.audio_config = config.get("audio")
        if not self.audio_config:
            raise ValueError("音频参数未配置，请在default.yml的tts.audio中设置")
            
        # TTS模型检查
        self.models = config.get("models")
        if not self.models:
            raise ValueError("TTS模型未配置，请在default.yml的tts.models中设置")
            
        # 默认模型
        self.default_model = next(iter(self.models.keys()), None)
        if not self.default_model:
            raise ValueError("未找到默认TTS模型，请确保tts.models中至少有一个模型配置")
        
        # 模型ID映射，将本地使用的模型ID映射到API支持的模型ID
        self.model_id_mapping = {
            # 默认映射关系，根据实际情况可调整
            "wang001": "default",
            "5zu": "custom",
            "6zu": "default_v2"
        }
        
        # 从配置中加载映射关系（如果存在）
        if "model_mapping" in config:
            self.model_id_mapping.update(config.get("model_mapping", {}))
        
        logger.info("TTS service initialized, server URL: {}".format(self.server_url))
        logger.info("Default TTS model: {}".format(self.default_model))
        logger.info("Available models: {}".format(", ".join(self.models.keys())))
        logger.info("Model ID mapping: {}".format(self.model_id_mapping))
        logger.info("Subtitle modules available: {}".format(self.subtitle_modules_available))
    
    def get_available_models(self) -> List[str]:
        """
        获取可用的TTS模型列表
        
        Returns:
            可用模型列表
        """
        return list(self.models.keys())
    
    def _map_model_id(self, model_id: str) -> str:
        """
        映射模型ID，确保与远程API兼容
        
        Args:
            model_id: 原始模型ID
            
        Returns:
            映射后的模型ID
        """
        if not model_id:
            return self.default_model
        
        # 使用配置文件中的映射
        if model_id in self.model_id_mapping:
            mapped_id = self.model_id_mapping[model_id]
            logger.info("模型ID映射: {} -> {}".format(model_id, mapped_id))
            return mapped_id
        
        return model_id
    
    def text_to_speech(self, text: str, output_path: str, model_id: Optional[str] = None, text_lang: str = "zh", **kwargs) -> Tuple[bool, str]:
        """
        文本转语音
        
        Args:
            text: 文本内容
            output_path: 输出音频路径
            model_id: 模型ID
            text_lang: 文本语言
            **kwargs: 其他参数
            
        Returns:
            (成功标志, 结果消息)
        """
        # 映射模型ID
        model_id = self._map_model_id(model_id)
        
        # 获取模型特定配置
        model_config = self.models.get(model_id, {})
        
        # 整合参数，优先使用传入的参数，其次使用模型配置
        params = {
            "text": text,
            "text_lang": text_lang,
            "voice_model_id": model_id  # 使用voice_model_id参数，与API一致
        }
        
        # 添加模型特定参数
        model_specific_params = {
            "cn_speaking_rate": model_config.get("cn_speaking_rate"),
            "en_word_rate": model_config.get("en_word_rate"),
            "speech_rate": model_config.get("speech_rate"),
            "ref_audio_path": model_config.get("ref_audio_path"),
            "prompt_text": model_config.get("prompt_text"),
            "prompt_lang": model_config.get("prompt_lang")
        }
        
        # 只添加非None的参数
        for key, value in model_specific_params.items():
            if value is not None:
                params[key] = value
        
        # 添加传入的其他参数，覆盖默认值
        for key, value in kwargs.items():
            params[key] = value
        
        # 获取模型路径
        t2s_weights_path = model_config.get("t2s_weights_path")
        vits_weights_path = model_config.get("vits_weights_path")
        
        # 如果配置中有模型路径，添加到请求中
        if t2s_weights_path:
            params["t2s_weights_path"] = t2s_weights_path
            logger.info("添加T2S模型路径到请求: {}".format(t2s_weights_path))
            
        if vits_weights_path:
            params["vits_weights_path"] = vits_weights_path
            logger.info("添加VITS模型路径到请求: {}".format(vits_weights_path))
        
        # 获取本地模型配置，检查是否有参考音频
        ref_audio_path = None
        prompt_text = ""
        prompt_lang = text_lang
        
        if isinstance(model_config, dict):
            ref_audio_path = model_config.get("ref_audio_path")
            prompt_text = model_config.get("prompt_text", "")
            prompt_lang = model_config.get("prompt_lang", text_lang)
        
        # 如果本地有参考音频，添加到请求中
        if ref_audio_path and os.path.exists(ref_audio_path):
            logger.info("使用本地参考音频: {}".format(ref_audio_path))
            params["ref_audio_path"] = ref_audio_path
            params["prompt_text"] = prompt_text
            params["prompt_lang"] = prompt_lang
        
        # 记录请求信息
        logger.info("Sending TTS request, text length: {}, language: {}, model: {} (API model: {})".format(
            len(text), text_lang, model_id, model_id))
        logger.info("Request URL: {}/tts".format(self.server_url))
        log_data = {k: v for k, v in params.items() if k != 'text'}
        logger.info("Request data (partial): {}".format(log_data))
        
        # 发送请求
        start_time = time.time()
        response = requests.post(
            "{}".format(self.server_url) + "/tts",
            json=params,
            timeout=120  # 长文本可能需要较长时间
        )
        
        # 检查响应
        logger.info("Server response status code: {}".format(response.status_code))
        
        if response.status_code != 200:
            logger.error("TTS service request failed: {}".format(response.status_code))
            try:
                error_text = response.text[:1000]  # Log first 1000 chars
                logger.error("Response content: {}".format(error_text))
                
                # 检查特定错误类型，如参考音频缺失
                if "ref_audio_path cannot be empty" in error_text or "参考音频" in error_text:
                    # 尝试调用set_refer_audio接口设置参考音频
                    if ref_audio_path and os.path.exists(ref_audio_path):
                        logger.info("尝试通过set_refer_audio接口设置参考音频: {}".format(ref_audio_path))
                        try:
                            # 构建set_refer_audio请求
                            set_ref_url = "{}".format(self.server_url) + "/set_refer_audio"
                            set_ref_params = {
                                "refer_audio_path": ref_audio_path,
                                "voice_model_id": model_id,
                                "is_url": False
                            }
                            logger.info("发送设置参考音频请求: URL={}, 参数={}".format(set_ref_url, set_ref_params))
                            set_ref_response = requests.get(set_ref_url, params=set_ref_params, timeout=30)
                            
                            if set_ref_response.status_code == 200:
                                logger.info("成功设置参考音频，重新发送TTS请求")
                                # 重新发送原始请求
                                response = requests.post(
                                    "{}".format(self.server_url) + "/tts",
                                    json=params,
                                    timeout=120
                                )
                                # 如果仍然失败，降级到离线模式
                                if response.status_code != 200:
                                    logger.warning("设置参考音频后TTS请求仍然失败，降级到离线模式")
                                    return self.text_to_speech_offline(text, output_path, model_id, text_lang)
                                # 如果成功，继续处理
                                logger.info("设置参考音频后TTS请求成功，状态码: {}".format(response.status_code))
                            else:
                                logger.error("设置参考音频失败: {}, {}".format(
                                    set_ref_response.status_code, set_ref_response.text[:200]))
                                return self.text_to_speech_offline(text, output_path, model_id, text_lang)
                        except Exception as e:
                            logger.error("调用set_refer_audio接口失败: {}".format(str(e)))
                            return self.text_to_speech_offline(text, output_path, model_id, text_lang)
                    else:
                        logger.warning("API需要参考音频但本地没有有效的参考音频文件，直接降级到离线模式")
                        return self.text_to_speech_offline(text, output_path, model_id, text_lang)
            except:
                pass
            return False, "TTS service request failed: {}".format(response.status_code)
        
        # 解析响应
        try:
            result = response.json()
        except Exception as e:
            logger.error("Cannot parse JSON response: {}".format(str(e)))
            logger.error("Response content (first 100 chars): {}".format(response.text[:100]))
            
            # 尝试直接保存响应内容作为音频
            if response.content and len(response.content) > 1000:  # 假设响应大于1KB是二进制音频数据
                logger.info("Attempting to save response content directly as audio data")
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                    duration = self._get_audio_duration(output_path)
                    logger.info("Successfully saved response content as audio file, Size: {:.2f}KB, Duration: {:.2f}s".format(len(response.content)/1024, duration))
                    return True, "Successfully generated audio, Duration: {:.2f}s (saved response directly)".format(duration)
            
            return False, "Cannot parse service response: {}".format(str(e))
        
        if not result.get("success", False):
            error_msg = result.get("message", "Unknown error")
            logger.error("TTS service processing failed: {}".format(error_msg))
            return False, "TTS service processing failed: {}".format(error_msg)
        
        # 获取音频数据 - 可能是URL或直接的base64编码
        audio_url = result.get("audio_url")
        audio_base64 = result.get("audio_data")
        
        if audio_base64:
            # 处理base64编码的音频数据
            import base64
            try:
                audio_data = base64.b64decode(audio_base64)
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                logger.info("已从base64数据保存音频文件")
            except Exception as e:
                logger.error("Failed to decode base64 audio data: {}".format(str(e)))
                return False, "Failed to decode audio data: {}".format(str(e))
        elif audio_url:
            # 处理音频URL
            try:
                logger.info("Downloading audio file: {}".format(audio_url))
                audio_response = requests.get(audio_url, timeout=30)
                if audio_response.status_code != 200:
                    logger.error("Audio download failed: {}".format(audio_response.status_code))
                    return False, "Audio download failed: {}".format(audio_response.status_code)
                
                with open(output_path, "wb") as f:
                    f.write(audio_response.content)
                logger.info("Saved audio file from URL.")
            except Exception as e:
                logger.error("Failed to download audio file: {}".format(str(e)))
                return False, "Failed to download audio file: {}".format(str(e))
        else:
            logger.error("TTS响应中没有音频数据(URL或base64)")
            return False, "TTS response has no audio data"
        
        # 检查音频文件是否存在和有效
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 100:
            file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            logger.error("Generated audio file is invalid: {}, Size: {} bytes".format(output_path, file_size))
            return False, "Generated audio file is invalid: {}".format(output_path)
        
        # 获取音频时长
        duration = self._get_audio_duration(output_path)
        file_size = os.path.getsize(output_path)
        
        elapsed_time = time.time() - start_time
        logger.info("TTS generation complete. Time: {:.2f}s, Duration: {:.2f}s, Size: {:.2f}KB".format(elapsed_time, duration, file_size/1024))
        
        return True, "TTS generation successful, Duration: {:.2f}s".format(duration)
        
    def generate_subtitle(self, text: str, output_path: str, audio_duration: float = None,
                        format: str = "vtt", encoding: str = "utf-8", offset: float = 0,
                        subtitle_speed: float = 1.0, **kwargs) -> Tuple[bool, str]:
        """
        为文本生成字幕文件
        
        Args:
            text: 文本内容
            output_path: 输出字幕文件路径
            audio_duration: 音频时长（秒），如果为None则根据文本长度估算
            format: 字幕格式，支持vtt、srt等
            encoding: 字幕文件编码
            offset: 时间偏移（秒）
            subtitle_speed: 字幕速度倍率
            
        Returns:
            (成功标志, 字幕路径或错误消息)
        """
        # 先检查字幕模块是否可用
        if not self.subtitle_modules_available:
            logger.warning("字幕处理模块不可用，将使用备用方法生成字幕")
            return self._fallback_subtitle_generation(text, output_path, audio_duration, format, encoding, offset)
            
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取文本切分方法
            text_cut_method = kwargs.get("text_cut_method", self.default_cut_method)
            logger.info("字幕生成使用文本切分方法: {}".format(text_cut_method))
            
            # 如果未提供音频时长，根据文本估算
            if audio_duration is None or audio_duration <= 0:
                # 使用默认模型的语速设置
                model_id = kwargs.get("model_id", self.default_model)
                if "zh" in kwargs.get("text_lang", "zh"):
                    # 中文：字符数除以每秒字符数
                    audio_duration = len(text) / self.models[model_id].get("cn_speaking_rate", 4.0)
                else:
                    # 英文：单词数除以每秒单词数
                    word_count = len(text.split())
                    audio_duration = word_count / self.models[model_id].get("en_word_rate", 3.5)
                
                # 加上句子间隔
                sentence_count = text.count("。") + text.count(".") + text.count("!") + text.count("?") + 1
                audio_duration += sentence_count * self.models[model_id].get("sentence_gap", 0.5)
                
                # 确保至少1秒
                audio_duration = max(1.0, audio_duration)
                
                logger.info("估算音频时长: {:.2f}秒".format(audio_duration))
            
            # 从配置中获取额外参数
            sentence_gap = self.models.get(kwargs.get("model_id", self.default_model), {}).get("sentence_gap", 0.4)
            remove_ending_punct = kwargs.get("remove_ending_punct", True)
            
            logger.info("调用create_subtitle_for_text生成字幕, 文件路径: {}".format(output_path))
            logger.info("字幕参数: 格式={}, 时长={:.2f}秒, 速度={}, 偏移={}秒".format(format, audio_duration, subtitle_speed, offset))
            
            # 直接调用subtitle_utils中的create_subtitle_for_text函数
            try:
                success = create_subtitle_for_text(
                    text=text,
                    output_path=output_path,
                    duration=audio_duration,
                    format=format,
                    encoding=encoding,
                    offset=offset,
                    subtitle_speed=subtitle_speed,
                    text_cut_method=text_cut_method,
                    sentence_gap=sentence_gap,
                    remove_ending_punct=remove_ending_punct
                )
                
                if success:
                    # 使用与Avatar生成器相同的英文日志信息，确保日志匹配
                    logger.info("Subtitle generated successfully: {}".format(output_path))
                    return True, output_path
                else:
                    logger.error("Failed to generate subtitle: create_subtitle_for_text returned False")
                    return False, "Failed to generate subtitle"
                    
            except Exception as e:
                logger.error("调用create_subtitle_for_text异常: {}".format(str(e)))
                # 尝试使用备用方式生成基本字幕
                return self._fallback_subtitle_generation(text, output_path, audio_duration, format, encoding, offset)
            
        except Exception as e:
            error_msg = "生成字幕失败: {}".format(str(e))
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            return False, error_msg
        
    def _fallback_subtitle_generation(self, text: str, output_path: str, audio_duration: float,
                                   format: str, encoding: str, offset: float) -> Tuple[bool, str]:
        """当主字幕生成方法失败时的备用方案
        
        使用简单的切分方法生成基本字幕
        """
        try:
            logger.warning("使用备用方法生成字幕")
            
            # 使用简单切分方法
            sentences = text.replace("。", "。\n").replace(".", ".\n").replace("!", "!\n").replace("?", "?\n").split("\n")
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # 确保至少有一个句子
            if not sentences:
                sentences = [text]
            
            # 计算每句的时长
            if audio_duration and len(sentences) > 0:
                avg_duration = audio_duration / len(sentences)
            else:
                avg_duration = 5.0  # 默认每句5秒
            
            # 创建字幕文件
            with open(output_path, "w", encoding=encoding) as f:
                if format.lower() == "vtt":
                    f.write("WEBVTT\n\n")
                
                current_time = offset
                for i, sentence in enumerate(sentences):
                    start_time = current_time
                    end_time = start_time + avg_duration
                    
                    # 格式化时间戳
                    if format.lower() == "vtt":
                        f.write("cue-{}\n".format(i + 1))
                        f.write("{} --> {}\n".format(
                            self._format_timestamp(start_time, "vtt"),
                            self._format_timestamp(end_time, "vtt")
                        ))
                        f.write("{}\n\n".format(sentence))
                    elif format.lower() == "srt":
                        f.write("{}\n".format(i + 1))
                        f.write("{} --> {}\n".format(
                            self._format_timestamp(start_time, "srt"),
                            self._format_timestamp(end_time, "srt")
                        ))
                        f.write("{}\n\n".format(sentence))
                    
                    current_time = end_time
                
            logger.info("使用备用方法生成字幕成功: {}".format(output_path))
            return True, output_path
        except Exception as e:
            logger.error("备用字幕生成失败: {}".format(str(e)))
            return False, "备用字幕生成失败: {}".format(str(e))
    
    def text_to_speech_offline(self, text: str, output_path: str, model_id: Optional[str] = None, text_lang: str = "zh") -> Tuple[bool, str]:
        """
        离线模式下的文本转语音处理（当TTS服务不可用时的备选方案）
        
        Args:
            text: 要转换的文本
            output_path: 输出音频文件路径
            model_id: TTS模型ID，如果为None则使用默认模型
            text_lang: 文本语言，默认为"zh"(中文)
            
        Returns:
            (成功状态, 信息)
        """
        logger.warning("TTS服务不可用，使用离线降级功能生成音频")
        
        try:
            # 使用系统TTS命令（仅适用于有安装相关TTS工具的环境）
            sample_rate = self.audio_config.get("sample_rate", 32000)
            logger.info("离线TTS使用采样率：{}".format(sample_rate))
            
            # 判断系统类型
            if os.name == 'nt':  # Windows
                # 使用Windows内置TTS
                with tempfile.NamedTemporaryFile(suffix=".vbs", delete=False) as f:
                    # 创建VBScript用于TTS
                    script = """
                    Dim speaks, speech
                    speaks="{}"
                    Set speech=CreateObject("sapi.spvoice")
                    speech.Speak speaks
                    Set speech=Nothing
                    """.format(text.replace('"', '""'))
                    f.write(script.encode('utf-8'))
                    script_path = f.name
                
                # 执行脚本
                logger.info("使用Windows SAPI执行TTS")
                subprocess.run(["cscript", "//nologo", script_path], check=True)
                
                # 由于Windows内置TTS不支持直接保存到文件，这里生成一个静音音频
                result = self._generate_silent_audio(output_path, duration=len(text) * 0.1)
                
                # 删除临时脚本
                os.unlink(script_path)
                
                return result, "使用Windows内置TTS生成（无实际音频）"
                
            else:  # Linux/MacOS
                if self._check_command("espeak"):
                    # 使用espeak
                    logger.info("使用espeak生成离线音频")
                    cmd = [
                        "espeak", 
                        "-w", output_path,
                        "-s", str(int(self.models[model_id].get("cn_speaking_rate", 4.0) * 50)),  # 使用配置的语速
                        "-a", "100",  # 音量
                        text
                    ]
                    subprocess.run(cmd, check=True)
                    return True, "使用espeak生成音频"
                    
                elif self._check_command("say") and os.path.exists("/usr/bin/say"):
                    # MacOS say命令
                    logger.info("使用MacOS say生成离线音频")
                    cmd = [
                        "say", 
                        "-o", output_path,
                        "-r", str(self.models[model_id].get("cn_speaking_rate", 4.0) * 50),  # 使用配置的语速
                        text
                    ]
                    subprocess.run(cmd, check=True)
                    return True, "使用MacOS say生成音频"
                    
                else:
                    # 使用静音音频替代（无TTS工具可用）
                    logger.warning("未找到可用的离线TTS工具，生成静音音频")
                    
                    # 计算静音时长，基于字符数和配置的语速
                    if text_lang == "zh":
                        # 中文：字符数除以每秒字符数
                        duration = len(text) / self.models[model_id].get("cn_speaking_rate", 4.0)
                    else:
                        # 英文：单词数除以每秒单词数
                        word_count = len(text.split())
                        duration = word_count / self.models[model_id].get("en_word_rate", 3.5)
                    
                    # 确保至少1秒
                    duration = max(1.0, duration)
                    
                    result = self._generate_silent_audio(output_path, duration=duration)
                    return result, "使用静音音频替代（无TTS工具可用）, 时长: {:.2f}秒".format(duration)
        
        except Exception as e:
            logger.error("离线TTS处理异常: {}".format(str(e)))
            import traceback
            logger.error(traceback.format_exc())
            return False, "离线TTS处理异常: {}".format(str(e))
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """
        获取音频文件时长
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音频时长（秒）
        """
        try:
            # 使用ffprobe获取音频时长
            cmd = [
                "/usr/bin/ffprobe", 
                "-v", "error", 
                "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1", 
                audio_path
            ]
            duration_str = subprocess.check_output(cmd, text=True).strip()
            duration = float(duration_str)
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
            # logger.error(f"获取音频时长失败: {str(e)}")
            logger.error("Failed to get audio duration: {}".format(str(e)))
            duration = 0.0
        
        return duration
    
    def _check_command(self, command: str) -> bool:
        """
        检查系统命令是否可用
        
        Args:
            command: 命令名称
            
        Returns:
            命令是否可用
        """
        try:
            subprocess.run(
                ["which", command] if os.name != 'nt' else ["where", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return True
        except Exception:
            return False
    
    def _generate_silent_audio(self, output_path: str, duration: float = 1.0) -> bool:
        """使用FFmpeg生成静音音频文件
        
        Args:
            output_path: 输出文件路径
            duration: 音频时长(秒)
            
        Returns:
            是否成功
        """
        try:
            # 从配置获取采样率
            sample_rate = self.audio_config.get("sample_rate", 32000)
            channels = self.models[self.default_model].get('channels', 1)
            
            # 构建命令，使用配置的参数
            channel_layout = "mono" if channels == 1 else "stereo"
            cmd = [
                "ffmpeg", "-y", "-v", "warning",
                "-f", "lavfi",
                "-i", "anullsrc=r={}:cl={}".format(sample_rate, channel_layout),
                "-t", str(duration),
                "-acodec", "pcm_s16le",  # 使用无损编码确保质量
                output_path
            ]
            
            logger.info("生成静音音频: {}, 时长: {:.2f}秒, 采样率: {}Hz, 声道: {}".format(
                output_path, duration, sample_rate, channels))
                
            subprocess.run(cmd, check=True, capture_output=True)
            
            # 验证文件是否生成成功
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info("静音音频生成成功: {}".format(output_path))
                return True
            else:
                logger.error("静音音频生成失败: 文件不存在或大小为0")
                return False
                
        except Exception as e:
            logger.error("生成静音音频失败: {}".format(str(e)))
            return False

    def _format_timestamp(self, seconds: float, format_type: str = "vtt") -> str:
        """
        将秒数格式化为字幕时间戳
        
        Args:
            seconds: 秒数
            format_type: 格式类型，支持"vtt"和"srt"
            
        Returns:
            格式化的时间戳字符串
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        
        if format_type.lower() == "vtt":
            return "{:02d}:{:02d}:{:06.3f}".format(hours, minutes, seconds)
        elif format_type.lower() == "srt":
            return "{:02d}:{:02d}:{:02d},{:03d}".format(hours, minutes, int(seconds), int((seconds - int(seconds)) * 1000))
        else:
            return "{:02d}:{:02d}:{:06.3f}".format(hours, minutes, seconds)
            
    def _cut_text(self, text: str, method: str = "cut2") -> List[str]:
        """
        切分文本为段落
        
        Args:
            text: 要切分的文本
            method: 切分方法，支持"cut1"(按标点)、"cut2"(按句子)等
            
        Returns:
            切分后的文本段落列表
        """
        # 根据切分方法选择不同的切分策略
        if method == "cut1":
            # 按标点符号切分
            import re
            segments = re.split(r'([。！？.!?])', text)
            
            # 合并标点和对应句子
            result = []
            for i in range(0, len(segments) - 1, 2):
                if i + 1 < len(segments):
                    result.append(segments[i] + segments[i + 1])
                else:
                    result.append(segments[i])
                    
            # 处理空段落
            result = [seg for seg in result if seg.strip()]
            
            return result if result else [text]
            
        elif method == "cut2":
            # 按句子切分，保留标点
            import re
            pattern = r'(?<=[。！？.!?])'
            segments = re.split(pattern, text)
            
            # 过滤空段落
            result = [seg.strip() for seg in segments if seg.strip()]
            
            return result if result else [text]
            
        else:
            # 默认不切分，返回整段文本
            return [text] 

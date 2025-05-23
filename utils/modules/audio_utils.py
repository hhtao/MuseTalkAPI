"""
音频处理工具模块
包含音频生成、转换、处理的功能
"""
import os
import uuid
import wave
import logging
import tempfile
import numpy as np
from io import BytesIO
import requests
import json
from typing import Optional, Dict, Any, Union, List

# 尝试导入librosa，如果失败则提供备选方案
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    print("警告: librosa 库未安装。将使用备选方法处理音频。")
    print("建议安装 librosa: pip install librosa")
    LIBROSA_AVAILABLE = False

# 音频处理配置
SAMPLE_RATE = 16000

# TTS服务配置
DEFAULT_TTS_SERVER = "http://192.168.202.10:9880"
DEFAULT_REF_AUDIO = "/mnt/part2/Dteacher/ttt/data/w001.wav"
DEFAULT_REF_TEXT = "那个人的行为完全超出了我的忍耐底线"

def generate_silence_audio(duration=1.0):
    """生成指定时长的静音音频，作为最后的备选方案，返回WAV格式字节"""
    # 生成一个静音WAV文件
    sample_rate = SAMPLE_RATE
    num_samples = int(duration * sample_rate)
    
    # 创建静音数据（振幅为0）
    silence_data = np.zeros(num_samples, dtype=np.int16)
    
    # 将NumPy数组转换为字节
    bytes_buffer = BytesIO()
    
    # 创建WAV文件
    with wave.open(bytes_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16位采样
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(silence_data.tobytes())
    
    # 获取WAV文件字节
    bytes_buffer.seek(0)
    return bytes_buffer.read()

def create_bytes_stream(audio_path, target_sample_rate=SAMPLE_RATE):
    """创建音频字节流，支持处理文件路径或字节数据"""
    try:
        # 检查audio_path是文件路径还是字节数据
        if isinstance(audio_path, bytes):
            # 如果是字节数据，需要先保存到临时文件
            # 这是因为whisper模型需要文件路径而不是字节数据
            temp_audio_file = os.path.join(tempfile.gettempdir(), "temp_audio_" + str(uuid.uuid4()) + ".wav")
            with open(temp_audio_file, 'wb') as f:
                f.write(audio_path)
            return temp_audio_file  # 返回临时文件路径而不是字节
        # 如果是文件路径，直接返回
        return audio_path

    except Exception as e:
        logging.error("创建音频字节流失败: {}".format(str(e)))
        # 如果处理失败，返回一个临时静音文件作为备选
        try:
            silence_data = generate_silence_audio(5.0)  # 5秒静音
            temp_file = os.path.join(tempfile.gettempdir(), "silence_" + str(uuid.uuid4()) + ".wav")
            with open(temp_file, 'wb') as f:
                f.write(silence_data)
            return temp_file
        except:
            return None

def get_audio_duration(file_path):
    """获取音频文件的持续时间，支持处理文件路径或字节数据"""
    try:
        # 使用wave模块获取音频时长（作为librosa的备选方案）
        if isinstance(file_path, bytes):
            # 处理字节数据
            temp_file = None
            try:
                temp_file = os.path.join(tempfile.gettempdir(), "temp_duration_" + str(uuid.uuid4()) + ".wav")
                with open(temp_file, 'wb') as f:
                    f.write(file_path)
                with wave.open(temp_file, 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    return frames / float(rate)
            finally:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
        else:
            # 处理文件路径
            try:
                with wave.open(file_path, 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    return frames / float(rate)
            except:
                # 如果wave模块失败，尝试使用librosa
                if LIBROSA_AVAILABLE:
                    audio, sr = librosa.load(file_path, sr=None)
                    return len(audio) / sr
    except Exception as e:
        logging.error("无法读取音频时长: {}".format(str(e)))
    
    # 所有方法都失败，返回默认值
    return 5.0  # 返回默认5秒作为备选

def check_tts_server(server_url: str = DEFAULT_TTS_SERVER) -> Dict[str, Any]:
    """检查TTS服务器状态
    
    Returns:
        Dict[str, Any]: {
            "available": bool,
            "error": Optional[str]
        }
    """
    try:
        # 1. 检查服务器连接
        response = requests.get("{}/health".format(server_url), timeout=5)
        if response.status_code != 200:
            return {
                "available": False,
                "error": "服务器响应异常: {}".format(response.status_code)
            }
            
        return {"available": True, "error": None}
        
    except requests.exceptions.Timeout:
        return {
            "available": False,
            "error": "服务器连接超时"
        }
    except requests.exceptions.ConnectionError:
        return {
            "available": False,
            "error": "无法连接到服务器"
        }
    except Exception as e:
        return {
            "available": False,
            "error": "检查服务器状态时发生错误: {}".format(str(e))
        }

def validate_tts_params(
    text: str,
    ref_audio_path: str = DEFAULT_REF_AUDIO,
    ref_text: str = DEFAULT_REF_TEXT
) -> Dict[str, Any]:
    """验证TTS所需参数
    
    Returns:
        Dict[str, Any]: {
            "valid": bool,
            "error": Optional[str]
        }
    """
    # 检查文本
    if not text or len(text.strip()) == 0:
        return {
            "valid": False,
            "error": "输入文本为空"
        }
        
    # 检查参考音频文件
    if not os.path.exists(ref_audio_path):
        return {
            "valid": False,
            "error": f"参考音频文件不存在: {ref_audio_path}"
        }
        
    # 检查参考文本
    if not ref_text or len(ref_text.strip()) == 0:
        return {
            "valid": False,
            "error": "参考文本为空"
        }
        
    return {"valid": True, "error": None}

def process_tts_audio(audio_data: bytes, target_sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """处理TTS返回的音频数据
    
    Args:
        audio_data (bytes): TTS服务返回的音频数据
        target_sample_rate (int): 目标采样率
        
    Returns:
        np.ndarray: 处理后的音频数据
    """
    try:
        # 1. 将字节数据转换为临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
            
        # 2. 读取音频文件
        audio_array, sample_rate = sf.read(temp_path)
        
        # 3. 删除临时文件
        os.unlink(temp_path)
        
        # 4. 如果需要，重采样到目标采样率
        if sample_rate != target_sample_rate:
            audio_array = resampy.resample(audio_array, sample_rate, target_sample_rate)
            
        # 5. 确保音频是float32类型
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
            
        # 6. 标准化音量
        audio_array = normalize_audio(audio_array)
        
        return audio_array
        
    except Exception as e:
        logging.error("处理TTS音频失败: %s", str(e))
        return None

def save_audio_file(audio_data: Union[bytes, np.ndarray], output_path: str, sample_rate: int = SAMPLE_RATE) -> bool:
    """保存音频数据到文件
    
    Args:
        audio_data: 音频数据（字节流或numpy数组）
        output_path: 输出文件路径
        sample_rate: 采样率
        
    Returns:
        bool: 是否成功
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # 如果是字节流，直接写入
        if isinstance(audio_data, bytes):
            with open(output_path, 'wb') as f:
                f.write(audio_data)
        # 如果是numpy数组，使用soundfile保存
        elif isinstance(audio_data, np.ndarray):
            sf.write(output_path, audio_data, sample_rate)
        else:
            raise ValueError("不支持的音频数据类型")
            
        return True
    except Exception as e:
        logging.error("保存音频文件失败: %s", str(e))
        return False

def preprocess_text_for_tts(text: str) -> List[str]:
    """预处理要发送给TTS的文本
    
    Args:
        text (str): 原始文本
        
    Returns:
        List[str]: 处理后的文本段落列表
    """
    # 1. 清理文本
    text = text.strip()
    
    # 2. 替换换行符为句号（如果句尾没有标点）
    text = text.replace('\n', '。')
    
    # 3. 按标点符号分句
    import re
    sentences = re.split(r'([。！？!?])', text)
    
    # 4. 重新组合句子和标点
    segments = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            segment = sentences[i] + sentences[i+1]
            if segment.strip():
                segments.append(segment.strip())
    
    # 处理最后一个句子
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        segments.append(sentences[-1].strip() + "。")
    
    # 5. 合并过短的句子
    MIN_LENGTH = 5  # 最小句子长度
    merged_segments = []
    current_segment = ""
    
    for segment in segments:
        if len(current_segment) + len(segment) < 50:  # 最大句子长度
            current_segment += segment
        else:
            if current_segment:
                merged_segments.append(current_segment)
            current_segment = segment
    
    if current_segment:
        merged_segments.append(current_segment)
    
    return merged_segments

def gpt_sovits_tts(
    text: str,
    lang: str = 'zh',
    server_url: str = DEFAULT_TTS_SERVER,
    ref_audio_path: str = DEFAULT_REF_AUDIO,
    ref_text: str = DEFAULT_REF_TEXT,
    text_cut_method: str = 'cut2',
    output_path: Optional[str] = None
) -> Union[bytes, str, None]:
    """使用GPT-SoVITS生成TTS音频
    
    Args:
        text (str): 要合成的文本
        lang (str): 语言代码
        server_url (str): TTS服务器地址
        ref_audio_path (str): 参考音频路径
        ref_text (str): 参考文本
        text_cut_method (str): 文本切分方法
        output_path (str, optional): 如果提供，将音频保存到该路径
        
    Returns:
        Union[bytes, str, None]: 
            - 如果提供output_path，返回保存的文件路径
            - 否则返回音频字节流
            - 失败返回None
    """
    try:
        # 1. 验证参数
        params_status = validate_tts_params(text, ref_audio_path, ref_text)
        if not params_status["valid"]:
            logging.error("TTS参数验证失败: {}".format(params_status['error']))
            return None
            
        # 2. 预处理文本
        text_segments = preprocess_text_for_tts(text)
        if not text_segments:
            logging.error("文本预处理后为空")
            return None
            
        # 3. 逐段处理文本
        audio_segments = []
        for segment in text_segments:
            # 准备请求参数
            payload = {
                "text": segment.strip(),  # 确保文本两端没有空白字符
                "text_lang": lang,
                "ref_audio_path": ref_audio_path,
                "prompt_text": ref_text.strip(),  # 确保参考文本两端没有空白字符
                "prompt_lang": lang,
                "text_cut_method": text_cut_method
            }
            
            # 打印请求信息（调试用）
            logging.info("发送TTS请求: %s", json.dumps(payload, ensure_ascii=False))
                
            # 发送请求
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "audio/wav"
                }
                
                response = requests.post(
                    "{}/tts".format(server_url),
                    headers=headers,
                    json=payload,  # 使用json参数而不是data，这样requests会自动处理Content-Type
                    timeout=30
                )
                
                # 检查响应状态码
                if response.status_code == 400:
                    error_msg = response.text if response.text else "未知错误"
                    logging.error("TTS请求参数错误(400): %s", error_msg)
                    logging.error("请求参数: %s", json.dumps(payload, ensure_ascii=False))
                    continue
                elif response.status_code != 200:
                    logging.error("TTS服务错误(%d): %s", response.status_code, response.text)
                    continue
                
                # 检查响应内容类型
                content_type = response.headers.get('content-type', '')
                if 'audio/wav' not in content_type.lower():
                    logging.warning("响应Content-Type不是audio/wav: %s", content_type)
                
                if len(response.content) == 0:
                    logging.error("TTS服务返回空音频数据")
                    continue
                    
                audio_segments.append(response.content)
                
            except requests.exceptions.Timeout:
                logging.error("处理文本段落超时")
                continue
            except requests.exceptions.RequestException as e:
                logging.error("请求异常: %s", str(e))
                continue
            except Exception as e:
                logging.error("处理文本段落失败: %s", str(e))
                continue
        
        if not audio_segments:
            logging.error("没有成功生成的音频段落")
            return None
            
        # 4. 合并音频段落
        if len(audio_segments) == 1:
            final_audio = audio_segments[0]
        else:
            # 将所有音频段落转换为numpy数组并拼接
            audio_arrays = []
            for audio_segment in audio_segments:
                # 创建临时文件
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(audio_segment)
                    temp_path = temp_file.name
                
                try:
                    # 读取音频数据
                    audio_array, _ = sf.read(temp_path)
                    audio_arrays.append(audio_array)
                finally:
                    # 确保删除临时文件
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            
            if not audio_arrays:
                logging.error("无法处理任何音频段落")
                return None
            
            # 拼接音频数组
            final_array = np.concatenate(audio_arrays)
            
            # 转换回WAV格式
            temp_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            try:
                sf.write(temp_output.name, final_array, AUDIO_SAMPLE_RATE)
                
                # 读取合并后的音频
                with open(temp_output.name, 'rb') as f:
                    final_audio = f.read()
            finally:
                # 确保删除临时文件
                if os.path.exists(temp_output.name):
                    os.unlink(temp_output.name)
        
        # 5. 保存或返回结果
        if output_path:
            if save_audio_file(final_audio, output_path):
                return output_path
            return None
            
        return final_audio
            
    except Exception as e:
        logging.error("TTS生成失败: %s", str(e))
        return None

def fallback_tts(text, lang='zh'):
    """当主要TTS失败时的备选方案"""
    try:
        # 尝试使用系统命令 espeak 生成语音
        try:
            temp_file = os.path.join(tempfile.gettempdir(), 'fallback_audio_{}.wav'.format(uuid.uuid4()))
            
            # 设置语言映射
            lang_map = {
                'zh': 'zh',
                'en': 'en',
                'ja': 'ja',
                'ko': 'ko'
            }
            espeak_lang = lang_map.get(lang, 'en')
            
            # 生成命令
            cmd = [
                'espeak', 
                '-v', espeak_lang,
                '-w', temp_file,
                text
            ]
            
            # 执行命令
            import subprocess
            subprocess.run(cmd, check=True, capture_output=True)
            
            # 读取生成的文件
            with open(temp_file, 'rb') as f:
                audio_data = f.read()
                
            # 清理临时文件
            os.remove(temp_file)
            
            return audio_data
        except Exception as espeak_error:
            logging.warning("espeak 生成失败: {}，使用内置音频生成".format(str(espeak_error)))
            
        # 如果 espeak 失败，使用简单的音频生成方法
        logging.info("使用内置音频生成方法")
        
        # 确保导入wave模块
        import wave
        
        # 根据文本长度生成音频时长
        text_length = len(text)
        duration = max(2.0, text_length * 0.2)  # 每个字符约0.2秒，最短2秒
        
        # 确保SAMPLE_RATE已经定义
        try:
            sample_rate = SAMPLE_RATE
        except NameError:
            logging.warning("SAMPLE_RATE未定义，使用默认值16000")
            sample_rate = 16000
        
        logging.info("生成音频，文本长度: {}，音频时长: {}秒，采样率: {}".format(text_length, duration, sample_rate))
        
        # 生成一系列不同频率的音调来模拟语音
        total_samples = int(duration * sample_rate)
        audio_data = np.zeros(total_samples, dtype=np.float32)
        
        # 创建一个简单的音调序列
        frequencies = [262, 294, 330, 349, 392, 440, 494, 523]  # C4-C5音阶
        char_index = 0
        samples_per_char = int(0.2 * sample_rate)  # 每个字符0.2秒
        
        for i in range(min(text_length, int(duration / 0.2))):
            start_idx = i * samples_per_char
            end_idx = min(start_idx + samples_per_char, total_samples)
            
            # 选择一个频率
            freq = frequencies[char_index % len(frequencies)]
            char_index += 1
            
            # 生成该频率的音调
            t = np.linspace(0, (end_idx - start_idx) / sample_rate, end_idx - start_idx, False)
            tone = 0.3 * np.sin(2 * np.pi * freq * t)
            
            # 添加淡入淡出效果
            fade_samples = min(int(0.05 * sample_rate), len(tone) // 4)
            if fade_samples > 0:
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                tone[:fade_samples] *= fade_in
                tone[-fade_samples:] *= fade_out
            
            # 合并到主音频流
            audio_data[start_idx:end_idx] = tone
        
        # 转换为16位PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # 保存为WAV
        temp_file = os.path.join(tempfile.gettempdir(), "synth_audio_{}.wav".format(uuid.uuid4()))
        logging.info("保存合成的音频到: {}".format(temp_file))
        
        try:
            with wave.open(temp_file, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
                
            # 检查文件是否生成并有效
            if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                logging.info("成功创建音频文件，大小: {} 字节".format(os.path.getsize(temp_file)))
            else:
                logging.error("音频文件生成失败或为空: {}".format(temp_file))
                return generate_silence_audio(5.0)  # 生成静音
                
            # 读取文件并返回字节
            with open(temp_file, 'rb') as f:
                result = f.read()
                
            # 清理临时文件
            try:
                os.remove(temp_file)
            except Exception as e:
                logging.warning("清理临时文件失败: {}".format(str(e)))
                
            return result
            
        except Exception as wave_error:
            logging.error("保存WAV文件失败: {}".format(str(wave_error)))
            # 尝试更简单的方式生成音频
            return generate_silence_audio(5.0)  # 生成静音
            
    except Exception as e:
        logging.error("备选TTS也失败了: {}".format(str(e)))
        import traceback
        traceback.print_exc()
        # 最后的备选方案：生成静音
        return generate_silence_audio(5.0)  # 5秒静音

def create_reference_audio_if_missing(ref_path):
    """如果参考音频不存在，创建一个默认的参考音频文件"""
    if os.path.exists(ref_path):
        return ref_path
        
    # 参考音频不存在，创建一个默认的参考音频
    logging.warning("参考音频不存在，将创建默认参考音频: {}".format(ref_path))
    
    # 创建参考音频存储目录
    os.makedirs(os.path.dirname(ref_path), exist_ok=True)
    
    # 生成一段参考音频（例如1秒的440Hz正弦波）
    import numpy as np
    
    duration = 1.0  # 1秒
    sample_rate = SAMPLE_RATE
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # 生成440Hz的正弦波
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # 添加淡入淡出效果
    fade_samples = int(0.1 * sample_rate)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    # 应用淡入淡出
    tone[:fade_samples] *= fade_in
    tone[-fade_samples:] *= fade_out
    
    # 将浮点数转换为16位整数
    audio_data = (tone * 32767).astype(np.int16)
    
    # 保存为WAV文件
    with wave.open(ref_path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16位
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    return ref_path 

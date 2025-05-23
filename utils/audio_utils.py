import os
import uuid
import numpy as np
import soundfile as sf
import resampy
import requests
from io import BytesIO
import traceback
import json
import wave
import re
from typing import List, Optional, Dict, Any
import logging

# 音频参数配置
SAMPLE_RATE = 16000  # 采样率
CHANNELS = 1  # 单声道
SAMPLE_WIDTH = 2  # 16位采样
MAX_AUDIO_LENGTH = 60  # 最大音频长度(秒)

# TTS相关配置
# 修正服务器URL和端口
API2_SERVER_URL = "http://192.168.202.10:9880" # 指向正确的TTS服务

# 导入配置
# from config import REF_FILE, REF_TEXT # No longer needed

def normalize_audio(audio_data, target_db=-20):
    """标准化音频音量"""
    if len(audio_data) == 0:
        return audio_data
    # 计算当前RMS
    rms = np.sqrt(np.mean(audio_data**2))
    if rms > 0:
        target_rms = 10**(target_db/20)
        gain = target_rms / rms
        normalized = audio_data * gain
        # 防止截幅
        if np.max(np.abs(normalized)) > 1:
            normalized = normalized / np.max(np.abs(normalized))
        return normalized
    return audio_data

def process_audio(audio_data, sample_rate):
    """处理音频数据，确保格式正确"""
    # 转换为float32
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # 如果是多通道，转换为单声道
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # 重采样
    if sample_rate != SAMPLE_RATE:
        audio_data = resampy.resample(audio_data, sample_rate, SAMPLE_RATE)
    
    # 标准化音量
    audio_data = normalize_audio(audio_data)
    
    return audio_data

def save_wav(audio_data, file_path):
    """保存音频为WAV格式"""
    # 确保音频数据是float32类型
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # 将float32转换为16位整数
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # 保存为WAV文件
    with wave.open(file_path, 'wb') as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_int16.tobytes())

def preprocess_text(text: str) -> str:
    """预处理文本，处理特殊字符和长度"""
    # 替换特殊引号
    text = text.replace('"', '"').replace('"', '"')
    # 替换度数符号
    text = text.replace('℃', '度')
    # 移除其他特殊字符，保留中文、英文字母、数字和常用标点符号
    text = re.sub(r'[^\u4e00-\u9fff\u3002\uff0c\uff01\uff1f\u2026\u201c\u201d\u2018\u2019\uff08\uff09\u3001\uff1b\uff1a\u300a\u300b\sa-zA-Z0-9.,!?;:()\[\]\'\"_-]+', '', text)
    return text

def gpt_sovits_tts(
    text: str,
    voice_model_id: str = 'wang001',
    speech_rate: float = 1.0,
    server_url: str = None,
    text_lang: str = 'zh',
    text_cut_method: str = 'cut2'
) -> Optional[str]:
    """
    使用 GPT-SoVITS 进行文本到语音转换 (调用 TTS服务)

    Args:
        text: 输入文本
        voice_model_id: 使用的语音模型 ID
        speech_rate: 语音速率
        server_url: TTS 服务器 URL
        text_lang: 文本语言
        text_cut_method: 文本切分方法

    Returns:
        str: 生成的音频文件绝对路径，失败返回None
    """
    try:
        # 预处理文本
        processed_text = preprocess_text(text)
        if not processed_text:
            logging.warning("预处理后的文本为空，无法生成 TTS")
            return None

        # 使用默认服务器URL
        if not server_url:
            server_url = API2_SERVER_URL

        # 准备请求数据 - 更新为TTS API所需的格式
        request_data = {
            "text": processed_text,
            "text_lang": text_lang,
            "prompt_lang": text_lang,
            "text_cut_method": text_cut_method,
            "voice_model_id": voice_model_id,
            "speech_rate": speech_rate
        }

        logging.info("发送 TTS 请求到 api2: {}".format(json.dumps(request_data, ensure_ascii=False)))

        # 发送请求到 TTS 服务器的 /tts - 添加适当的 headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "audio/wav"  # 指定希望接收音频数据
        }
        
        response = requests.post(
            "{}/tts".format(server_url), 
            json=request_data, 
            headers=headers,
            timeout=120
        )

        # 检查响应状态码
        if response.status_code == 200:
            # 处理二进制音频数据
            if len(response.content) > 0:
                # 创建临时文件保存音频
                temp_audio_path = "./temp/" + str(uuid.uuid4()) + ".wav"
                os.makedirs("./temp", exist_ok=True)
                
                # 保存音频文件
                with open(temp_audio_path, 'wb') as f:
                    f.write(response.content)
                
                logging.info("TTS 成功，音频保存到临时文件: {}".format(temp_audio_path))
                return temp_audio_path
            else:
                logging.error("TTS 响应成功，但返回了空的音频数据")
                return None
        else:
            try:
                # 尝试获取错误信息
                error_content = response.text
                logging.error("TTS 请求失败，状态码: {}, 响应: {}".format(response.status_code, error_content))
            except Exception:
                logging.error("TTS 请求失败，状态码: {}, 无法读取响应内容".format(response.status_code))
            
            logging.error("请求数据: {}".format(json.dumps(request_data, ensure_ascii=False)))
            return None

    except requests.exceptions.RequestException as req_err:
        logging.error("TTS 请求网络错误: {}".format(str(req_err)))
        traceback.print_exc()
        return None
    except Exception as e:
        logging.error("TTS 处理失败: {}".format(str(e)))
        traceback.print_exc() # 打印详细堆栈信息
        return None

def fallback_tts(text, lang='zh'):
    """
    如果GPT-SoVITS失败，使用gTTS作为备选方案
    
    Args:
        text (str): 要合成的文本
        lang (str): 语言代码，默认为'zh'
        
    Returns:
        str: 生成的临时音频文件路径
    """
    from gtts import gTTS
    
    try:
        # 确保临时目录存在
        os.makedirs("./temp", exist_ok=True)
        
        temp_audio_path = "./temp/" + str(uuid.uuid4()) + ".wav"
        
        # 使用gTTS生成音频
        tts = gTTS(text=text, lang=lang)
        temp_mp3_path = temp_audio_path.replace('.wav', '.mp3')
        tts.save(temp_mp3_path)
        
        # 读取MP3并转换为WAV格式
        audio_data, sample_rate = sf.read(temp_mp3_path)
        
        # 处理音频数据
        audio_data = process_audio(audio_data, sample_rate)
        
        # 保存为WAV
        save_wav(audio_data, temp_audio_path)
        
        # 删除临时MP3文件
        os.remove(temp_mp3_path)
        
        return temp_audio_path
        
    except Exception as e:
        print("备选TTS失败:", str(e))
        traceback.print_exc()
        return None

def get_audio_duration(file_path):
    """
    获取音频文件的时长（秒）
    
    Args:
        file_path (str): 音频文件路径
        
    Returns:
        float: 音频时长（秒）
    """
    try:
        with wave.open(file_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print("获取音频时长出错:", str(e))
        traceback.print_exc()
        
        # 如果无法获取，返回一个估计值（假设每个字符需要0.3秒）
        try:
            with open(file_path, 'rb') as f:
                size_kb = len(f.read()) / 1024
                return max(size_kb * 0.5, 3.0)  # 保证至少3秒
        except:
            return 5.0  # 默认返回5秒

def create_bytes_stream(audio_path, target_sample_rate=SAMPLE_RATE):
    """
    处理音频文件，转换为特定采样率的numpy数组
    
    Args:
        audio_path (str): 音频文件路径
        target_sample_rate (int): 目标采样率
        
    Returns:
        numpy.ndarray: 音频数据
    """
    try:
        # 读取音频文件
        audio_data, sample_rate = sf.read(audio_path)
        print("音频文件采样率:", sample_rate, "形状:", audio_data.shape)
        
        # 处理音频数据
        audio_data = process_audio(audio_data, sample_rate)
        
        return audio_data
        
    except Exception as e:
        print("处理音频文件错误:", str(e))
        traceback.print_exc()
        raise

def generate_audio(
    text: str,
    voice_model_id: str = 'wang001',
    speech_rate: float = 1.0,
    text_lang: str = 'zh',
    text_cut_method: str = 'cut2',
    use_fallback: bool = False
) -> Optional[str]:
    """
    生成音频，综合主方法和备用方法

    Args:
        text (str): 需要合成的文本
        voice_model_id (str): 语音模型 ID
        speech_rate (float): 语音速率
        text_lang (str): 文本语言
        text_cut_method (str): 文本切分方法
        use_fallback (bool): 是否直接使用备用方法

    Returns:
        str: 生成的音频文件路径
    """
    try:
        if use_fallback:
            # 注意：fallback_tts 可能也需要更新以接受新参数或有自己的配置
            return fallback_tts(text, lang=text_lang) # 保持 fallback 简单
        else:
            try:
                # 调用更新后的 gpt_sovits_tts
                audio_path = gpt_sovits_tts(
                    text, 
                    voice_model_id=voice_model_id, 
                    speech_rate=speech_rate,
                    text_lang=text_lang,
                    text_cut_method=text_cut_method
                )
                if audio_path:
                    return audio_path
                else:
                    logging.warning("主 TTS 方法失败，使用备选方案")
                    return fallback_tts(text, lang=text_lang)
            except Exception as e:
                logging.error("主 TTS 方法调用出错: {}，使用备选方案".format(e))
                return fallback_tts(text, lang=text_lang)
    except Exception as e:
        logging.error("音频生成失败: {}".format(e))
        traceback.print_exc()
        return None # 返回 None 而不是抛出异常

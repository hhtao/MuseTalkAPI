"""
字幕工具模块
包含字幕生成、格式转换和添加功能
"""
import os
import re
import math
import logging
import subprocess
import tempfile
import uuid
import shutil
from typing import List, Dict, Tuple, Optional

def split_text_to_sentences(text):
    """将文本分割为句子列表"""
    # 使用正则表达式匹配句子分隔符
    # 中文句号，英文句号，问号，感叹号等
    pattern = r'[。.!?！？；;，,]+'
    sentences = re.split(pattern, text)
    
    # 过滤空句子并去除前后空白
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def calculate_mixed_subtitle_timing(text, total_duration, subtitle_speed=1.0, offset=0.0):
    """计算字幕显示时间，根据文本长度和总时长"""
    sentences = split_text_to_sentences(text)
    if not sentences:
        return []
        
    # 简单按字符数分配时间
    total_chars = sum(len(s) for s in sentences)
    
    subtitle_timing = []
    current_time = offset
    
    for sentence in sentences:
        if total_chars == 0:
            duration = total_duration / len(sentences)
        else:
            # 根据句子长度占比分配时间
            duration = (len(sentence) / total_chars) * total_duration
            
        # 应用速度调整
        adjusted_duration = duration / subtitle_speed
        
        # 确保最短时间不小于0.5秒
        if adjusted_duration < 0.5:
            adjusted_duration = 0.5
            
        subtitle_timing.append({
            "text": sentence,
            "start": current_time,
            "end": current_time + adjusted_duration
        })
        
        current_time += adjusted_duration
    
    return subtitle_timing

def format_time_vtt(seconds):
    """将秒数格式化为VTT时间格式 (HH:MM:SS.mmm)"""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds_remainder = seconds % 60
    
    return f"{hours:02d}:{minutes:02d}:{seconds_remainder:06.3f}".replace(".", ",")

def format_time_srt(seconds):
    """将秒数格式化为SRT时间格式 (HH:MM:SS,mmm)"""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds_remainder = seconds % 60
    milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d},{milliseconds:03d}"

def generate_vtt_file(subtitle_timing, output_path, encoding="utf-8"):
    """生成WebVTT格式字幕文件"""
    with open(output_path, "w", encoding=encoding) as f:
        f.write("WEBVTT\n\n")
        
        for i, subtitle in enumerate(subtitle_timing):
            start_time = format_time_vtt(subtitle["start"])
            end_time = format_time_vtt(subtitle["end"])
            
            f.write(f"{i+1}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{subtitle['text']}\n\n")
    
    return True

def generate_srt_file(subtitle_timing, output_path, encoding="utf-8"):
    """生成SRT格式字幕文件"""
    with open(output_path, "w", encoding=encoding) as f:
        for i, subtitle in enumerate(subtitle_timing):
            start_time = format_time_srt(subtitle["start"])
            end_time = format_time_srt(subtitle["end"])
            
            f.write(f"{i+1}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{subtitle['text']}\n\n")
    
    return True

def create_subtitle_for_text(text, output_path, duration, format='vtt', encoding='utf-8', 
                             offset=0.0, subtitle_speed=1.0, text_cut_method='cut3'):
    """从文本创建字幕文件"""
    try:
        # 导入text_cut模块中的方法
        from MuseTalk.utils.text_cut import get_method
        
        # 使用指定的文本切分方法
        try:
            # 获取文本切分方法
            cut_method = get_method(text_cut_method)
            # 处理文本并分割成句子
            processed_text = cut_method(text)
            # 如果返回的是带换行符的字符串，按换行拆分成句子
            if isinstance(processed_text, str) and '\n' in processed_text:
                sentences = processed_text.split('\n')
                # 过滤空行
                sentences = [s.strip() for s in sentences if s.strip()]
            else:
                # 如果没有换行，则视为单个句子
                sentences = [processed_text]
        except Exception as e:
            logging.warning(f"应用文本切分方法失败: {str(e)}，使用默认句子切分")
            sentences = split_text_to_sentences(text)
            
        # 如果没有有效句子，使用原始文本
        if not sentences:
            logging.warning("文本切分后没有有效句子，使用原始文本")
            sentences = [text]
            
        # 计算每个句子的时间点
        subtitle_timing = []
        total_chars = sum(len(s) for s in sentences)
        current_time = offset
        
        for sentence in sentences:
            if total_chars == 0:
                duration_per_sentence = duration / len(sentences)
            else:
                # 根据句子长度占比分配时间
                duration_per_sentence = (len(sentence) / total_chars) * duration
                
            # 应用速度调整
            adjusted_duration = duration_per_sentence / subtitle_speed
            
            # 确保最短时间不小于0.5秒
            if adjusted_duration < 0.5:
                adjusted_duration = 0.5
                
            end_time = current_time + adjusted_duration
            
            subtitle_timing.append({
                'start_time': format_time_vtt(current_time) if format.lower() == 'vtt' else format_time_srt(current_time),
                'end_time': format_time_vtt(end_time) if format.lower() == 'vtt' else format_time_srt(end_time),
                'text': sentence
            })
            
            current_time = end_time
        
        # 生成字幕文件
        if format.lower() == 'vtt':
            return generate_vtt_file(subtitle_timing, output_path, encoding)
        elif format.lower() == 'srt':
            return generate_srt_file(subtitle_timing, output_path, encoding)
        else:
            logging.error(f"不支持的字幕格式: {format}")
            return False
    except Exception as e:
        logging.error(f"创建字幕失败: {str(e)}")
        return False

def add_subtitles_to_video(video_path, subtitle_path, output_path, font_size=24, font_color="white", 
                           position="bottom", outline=True):
    """将字幕添加到视频中"""
    try:
        # 确保输入文件存在
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        if not os.path.exists(subtitle_path):
            raise FileNotFoundError(f"字幕文件不存在: {subtitle_path}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 判断字幕格式
        subtitle_ext = os.path.splitext(subtitle_path)[1].lower()
        
        # 设置字幕样式
        font_params = f"fontsize={font_size}:fontcolor={font_color}"
        if outline:
            font_params += ":borderw=1.5:bordercolor=black"
            
        # 设置字幕位置
        if position == "top":
            vertical_pos = "(h/10)"
        elif position == "middle":
            vertical_pos = "(h/2)"
        else:  # bottom (default)
            vertical_pos = "(h-h/10)"
            
        # 处理路径中的特殊字符
        video_path_escaped = video_path.replace("'", "\\'").replace(":", "\\:")
        subtitle_path_escaped = subtitle_path.replace("'", "\\'").replace(":", "\\:")
        output_path_escaped = output_path.replace("'", "\\'").replace(":", "\\:")
        
        # 创建临时字幕文件 (复制一份，避免路径问题)
        temp_subtitle = os.path.join(tempfile.gettempdir(), f"temp_subtitle_{uuid.uuid4()}{subtitle_ext}")
        shutil.copy(subtitle_path, temp_subtitle)
        
        # 使用FFmpeg添加字幕 (两种不同的方法)
        try:
            # 方法1: 使用 subtitles 滤镜
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vf', f"subtitles='{temp_subtitle}':force_style='{font_params},y={vertical_pos}'",
                '-c:a', 'copy',
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
        except Exception as e:
            logging.warning(f"方法1添加字幕失败: {str(e)}，尝试方法2")
            
            # 方法2: 将字幕转换为 ASS 格式，使用 ass 滤镜
            try:
                # 创建临时 ASS 文件
                temp_ass = os.path.join(tempfile.gettempdir(), f"temp_subtitle_{uuid.uuid4()}.ass")
                
                # 将字幕转换为 ASS 格式
                convert_cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_subtitle,
                    temp_ass
                ]
                subprocess.run(convert_cmd, check=True, capture_output=True)
                
                # 使用 ASS 滤镜添加字幕
                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-vf', f"ass='{temp_ass}'",
                    '-c:a', 'copy',
                    output_path
                ]
                
                subprocess.run(cmd, check=True, capture_output=True)
                
                # 清理临时 ASS 文件
                if os.path.exists(temp_ass):
                    os.remove(temp_ass)
                    
            except Exception as e2:
                logging.error(f"方法2添加字幕也失败了: {str(e2)}")
                
                # 方法3: 如果字幕添加失败，只复制视频
                logging.warning("所有字幕添加方法失败，将创建不包含字幕的视频")
                
                # 使用简单的复制
                copy_cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-c', 'copy',
                    output_path
                ]
                subprocess.run(copy_cmd, check=True, capture_output=True)
        
        # 清理临时字幕文件
        if os.path.exists(temp_subtitle):
            os.remove(temp_subtitle)
            
        return True
    except Exception as e:
        logging.error(f"添加字幕失败: {str(e)}")
        
        # 最终备选方案: 直接复制视频文件
        try:
            shutil.copy(video_path, output_path)
            logging.warning(f"字幕添加完全失败，已复制原始视频: {video_path} -> {output_path}")
            return True
        except Exception as copy_e:
            logging.error(f"复制视频也失败了: {str(copy_e)}")
            return False

def get_video_duration(video_path):
    """获取视频时长"""
    try:
        cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'default=noprint_wrappers=1:nokey=1', 
            video_path
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        logging.error(f"获取视频时长失败: {str(e)}")
        return 0.0

def adjust_subtitle_timing(subtitle_path, output_path, offset, 
                          scale=1.0, from_format=None, to_format=None):
    """调整字幕时间"""
    if from_format is None:
        from_format = os.path.splitext(subtitle_path)[1].lstrip('.')
    
    if to_format is None:
        to_format = os.path.splitext(output_path)[1].lstrip('.')
    
    try:
        # 读取原字幕文件
        with open(subtitle_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 处理VTT格式
        if from_format.lower() == 'vtt':
            # 处理WebVTT时间戳格式 00:00:00.000 --> 00:00:05.000
            pattern = r'(\d{2}:\d{2}:\d{2}[,\.]\d{3}) --> (\d{2}:\d{2}:\d{2}[,\.]\d{3})'
            
            def adjust_time(match):
                start_time = parse_time_vtt(match.group(1))
                end_time = parse_time_vtt(match.group(2))
                
                # 应用偏移和缩放
                new_start = start_time * scale + offset
                new_end = end_time * scale + offset
                
                # 确保时间不为负
                if new_start < 0:
                    new_start = 0
                if new_end < 0:
                    new_end = 0
                
                # 格式化为目标格式
                if to_format.lower() == 'vtt':
                    return f"{format_time_vtt(new_start)} --> {format_time_vtt(new_end)}"
                else:  # srt
                    return f"{format_time_srt(new_start)} --> {format_time_srt(new_end)}"
            
            adjusted_content = re.sub(pattern, adjust_time, content)
            
        # 处理SRT格式
        elif from_format.lower() == 'srt':
            # 处理SRT时间戳格式 00:00:00,000 --> 00:00:05,000
            pattern = r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})'
            
            def adjust_time(match):
                start_time = parse_time_srt(match.group(1))
                end_time = parse_time_srt(match.group(2))
                
                # 应用偏移和缩放
                new_start = start_time * scale + offset
                new_end = end_time * scale + offset
                
                # 确保时间不为负
                if new_start < 0:
                    new_start = 0
                if new_end < 0:
                    new_end = 0
                
                # 格式化为目标格式
                if to_format.lower() == 'vtt':
                    return f"{format_time_vtt(new_start)} --> {format_time_vtt(new_end)}"
                else:  # srt
                    return f"{format_time_srt(new_start)} --> {format_time_srt(new_end)}"
            
            adjusted_content = re.sub(pattern, adjust_time, content)
            
        else:
            logging.error(f"不支持的字幕格式: {from_format}")
            return False
        
        # 如果转换格式为VTT但原格式为SRT，添加VTT头
        if from_format.lower() == 'srt' and to_format.lower() == 'vtt':
            adjusted_content = "WEBVTT\n\n" + adjusted_content
        
        # 保存调整后的字幕
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(adjusted_content)
            
        return True
        
    except Exception as e:
        logging.error(f"调整字幕时间失败: {str(e)}")
        return False

def parse_time_vtt(time_str):
    """解析VTT时间格式为秒"""
    # 处理逗号或点作为毫秒分隔符
    time_str = time_str.replace(',', '.')
    
    hours, minutes, seconds = time_str.split(':')
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return total_seconds

def parse_time_srt(time_str):
    """解析SRT时间格式为秒"""
    hours, minutes, rest = time_str.split(':')
    seconds, milliseconds = rest.split(',')
    
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    total_seconds += int(milliseconds) / 1000
    
    return total_seconds 
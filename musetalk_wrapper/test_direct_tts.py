#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MuseTalk Wrapper 直接TTS测试脚本 
使用直接HTTP请求测试TTS服务，跳过所有复杂组件
"""

import os
import sys
import time
import json
import argparse
import requests
import subprocess
from pathlib import Path

def direct_tts_test(text, output_path, model_id="wang001", server_url="http://192.168.202.10:9880"):
    """直接使用HTTP请求测试TTS服务"""
    print("\n======= 直接TTS测试 =======")
    print(f"服务器: {server_url}")
    print(f"模型ID: {model_id}")
    print(f"文本内容: {text}")
    print(f"输出路径: {output_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 准备数据
    payload = {
        "text": text,
        "text_lang": "zh",
        "model_id": model_id,
        "sample_rate": 44100
    }
    
    print(f"请求数据: {json.dumps(payload, ensure_ascii=False)}")
    print("发送请求中...")
    
    try:
        # 发送请求
        start_time = time.time()
        response = requests.post(
            f"{server_url}/tts",
            json=payload,
            timeout=60
        )
        elapsed = time.time() - start_time
        
        print(f"请求耗时: {elapsed:.2f}秒")
        print(f"状态码: {response.status_code}")
        print(f"响应头: {response.headers}")
        
        content_type = response.headers.get('Content-Type', '')
        print(f"内容类型: {content_type}")
        
        # 保存响应内容
        if 'application/json' in content_type.lower():
            # JSON响应
            try:
                json_data = response.json()
                print(f"JSON响应: {json.dumps(json_data, indent=2, ensure_ascii=False)}")
                
                # 检查是否有音频URL
                audio_url = json_data.get('audio_url')
                if audio_url:
                    print(f"发现音频URL: {audio_url}")
                    print("下载音频...")
                    audio_response = requests.get(audio_url, timeout=30)
                    with open(output_path, 'wb') as f:
                        f.write(audio_response.content)
                    print(f"音频保存到: {output_path}")
                
                # 检查是否有base64音频数据
                audio_data = json_data.get('audio_data')
                if audio_data:
                    print("发现Base64音频数据")
                    import base64
                    binary_data = base64.b64decode(audio_data)
                    with open(output_path, 'wb') as f:
                        f.write(binary_data)
                    print(f"音频保存到: {output_path}")
            except Exception as e:
                print(f"解析JSON失败: {str(e)}")
                # 保存原始文本以便检查
                with open(output_path + '.json', 'w', encoding='utf-8') as f:
                    f.write(response.text[:2000])  # 只保存前2000字符
                print(f"原始响应保存到: {output_path}.json")
        
        elif 'audio/' in content_type.lower() or 'application/octet-stream' in content_type.lower():
            # 二进制音频数据
            print("收到二进制音频数据")
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"音频保存到: {output_path}")
        
        else:
            # 未知响应类型，保存为文本
            print("收到未知类型响应")
            with open(output_path + '.unknown', 'wb') as f:
                f.write(response.content)
            print(f"原始响应保存到: {output_path}.unknown")
            
            # 尝试也保存为音频
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"同时尝试保存为音频: {output_path}")
        
        # 检查生成的文件
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"生成的文件大小: {size/1024:.2f} KB")
            
            if size > 1000:  # 超过1KB可能是有效音频
                # 尝试获取音频信息
                try:
                    cmd = [
                        "/usr/bin/ffprobe", 
                        "-v", "error", 
                        "-show_entries", "format=duration,bit_rate", 
                        "-of", "json", 
                        output_path
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    info = json.loads(result.stdout)
                    print(f"音频信息: {json.dumps(info, indent=2)}")
                    
                    # 检测音量
                    silence_cmd = [
                        "/usr/bin/ffmpeg", 
                        "-i", output_path, 
                        "-af", "volumedetect", 
                        "-f", "null", 
                        "/dev/null"
                    ]
                    result = subprocess.run(silence_cmd, capture_output=True, text=True)
                    if "mean_volume:" in result.stderr:
                        volume_line = [l for l in result.stderr.split('\n') if "mean_volume:" in l][0]
                        print(f"音量信息: {volume_line}")
                except Exception as e:
                    print(f"无法获取音频信息: {str(e)}")
                
                # 尝试播放音频
                try:
                    print("\n尝试播放音频...")
                    subprocess.run(["/usr/bin/ffplay", "-nodisp", "-autoexit", output_path], 
                                  stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                    print("音频播放完成")
                except Exception as e:
                    print(f"无法播放音频: {str(e)}")
                    
        return True
                
    except Exception as e:
        print(f"请求失败: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="直接TTS测试脚本")
    parser.add_argument("--text", default="你好，我是数字人。这是一条测试文本，用于验证TTS服务是否正常。", help="测试文本")
    parser.add_argument("--output", default=None, help="输出音频路径")
    parser.add_argument("--model", default="wang001", help="TTS模型ID")
    parser.add_argument("--server", default="http://192.168.202.10:9880", help="TTS服务器URL")
    args = parser.parse_args()
    
    # 如果未指定输出路径，使用temp目录
    if args.output is None:
        # 获取脚本路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 定位MuseTalk根目录
        musetalk_root = os.path.dirname(script_dir)
        temp_dir = os.path.join(musetalk_root, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 生成时间戳文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        args.output = os.path.join(temp_dir, f"direct_tts_{timestamp}.wav")
    
    # 执行测试
    direct_tts_test(args.text, args.output, args.model, args.server)

if __name__ == "__main__":
    main() 

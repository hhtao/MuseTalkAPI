#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MuseTalk Wrapper API测试脚本
测试文字转视频功能和TTS服务
"""

import os
import sys
import json
import time
import requests
import argparse
import subprocess

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_list_avatars(api_url):
    """测试获取数字人列表"""
    print("\n======= 测试获取数字人列表 =======")
    response = requests.get(api_url + "/api/list_avatars")
    
    if response.status_code != 200:
        print("请求失败: " + str(response.status_code))
        return False
        
    result = response.json()
    if not result.get("success"):
        print("获取数字人列表失败: " + str(result.get('error')))
        return False
        
    avatars = result.get("avatars", [])
    print("找到 " + str(len(avatars)) + " 个数字人:")
    for avatar in avatars:
        print("  ID: " + avatar['id'] + ", 名称: " + avatar['name'])
        
    return True

def test_tts(text, output_path, model_id="wang001"):
    """测试文本转语音功能"""
    # TTS服务使用不同的服务器和端点
    tts_server_url = "http://192.168.202.10:9880"
    
    print("\n======= 测试TTS服务 =======")
    print("文本内容: " + text)
    print("输出路径: " + output_path)
    print("模型ID: " + model_id)
    print("TTS服务器: " + tts_server_url)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 准备请求数据
    payload = {
        "text": text,
        "model_id": model_id,
        "text_lang": "zh",
        "sample_rate": 44100
    }
    
    print("发送TTS请求...")
    try:
        # 发送请求
        start_time = time.time()
        response = requests.post(
            tts_server_url + "/tts",
            json=payload,
            timeout=60
        )
        elapsed = time.time() - start_time
        
        print("请求耗时: " + str(elapsed) + "秒")
        print("状态码: " + str(response.status_code))
        
        content_type = response.headers.get('Content-Type', '')
        print("内容类型: " + content_type)
        
        if response.status_code != 200:
            print("请求失败: " + str(response.status_code))
            return False
        
        # 处理响应内容
        if 'application/json' in content_type.lower():
            # JSON响应
            try:
                result = response.json()
                print("JSON响应: " + json.dumps(result, indent=2, ensure_ascii=False))
                
                # 检查是否有音频URL或音频数据
                audio_url = result.get('audio_url')
                audio_data = result.get('audio_data')
                
                if audio_url:
                    print("发现音频URL，下载中...")
                    audio_response = requests.get(audio_url, timeout=30)
                    with open(output_path, 'wb') as f:
                        f.write(audio_response.content)
                elif audio_data:
                    print("发现Base64音频数据")
                    import base64
                    binary_data = base64.b64decode(audio_data)
                    with open(output_path, 'wb') as f:
                        f.write(binary_data)
                else:
                    print("响应中没有音频数据")
                    return False
            except Exception as e:
                print("解析JSON失败: " + str(e))
                return False
                
        elif 'audio/' in content_type.lower() or 'application/octet-stream' in content_type.lower():
            # 二进制音频数据
            print("收到二进制音频数据")
            with open(output_path, 'wb') as f:
                f.write(response.content)
        else:
            # 未知类型，尝试保存为音频
            print("收到未知类型响应，尝试保存为音频")
            with open(output_path, 'wb') as f:
                f.write(response.content)
    except Exception as e:
        print("请求异常: " + str(e))
        return False
    
    # 检查音频文件是否存在
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print("音频文件生成成功: " + output_path + ", 大小: " + str(size/1024) + " KB")
        
        # 获取音频信息
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
            print("音频信息: " + json.dumps(info, indent=2))
            
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
                print("音量信息: " + volume_line)
                
            return True
        except Exception as e:
            print("获取音频信息失败: " + str(e))
            return size > 1000  # 如果文件大于1KB，认为可能是有效音频
    else:
        print("错误: 音频文件不存在: " + output_path)
        return False

def test_generate_video(api_url, text, avatar_id='avator_50'):
    """测试生成视频"""
    print("\n======= 测试生成视频 (数字人: " + avatar_id + ") =======")
    print("文本内容: " + text)
    
    payload = {
        "text": text,
        "avatar_id": avatar_id,
        "subtitle": True,
        "prefer_mode": "standard"  # 使用标准模式，支持唇动动画
    }
    
    response = requests.post(api_url + "/api/generate_video", json=payload)
    
    if response.status_code != 200:
        print("请求失败: " + str(response.status_code))
        return False
        
    result = response.json()
    if not result.get("success"):
        print("提交任务失败: " + str(result.get('error')))
        return False
        
    task_id = result.get("task_id")
    print("任务ID: " + task_id)
    
    # 轮询任务状态
    print("等待任务完成...")
    start_time = time.time()
    timeout = 300  # 5分钟超时
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(api_url + "/api/task_status/" + task_id)
            if response.status_code != 200:
                print("获取任务状态失败: " + str(response.status_code))
                time.sleep(2)
                continue
                
            status_data = response.json()
            if not status_data.get("success"):
                print("获取任务状态错误: " + str(status_data.get('error')))
                time.sleep(2)
                continue
                
            task_info = status_data.get("task_info", {})
            status = task_info.get("status")
            completed = task_info.get("completed", 0)
            
            print("任务状态: " + status + ", 进度: " + str(completed) + "%, 消息: " + str(task_info.get('message', '')))
            
            if status == "completed":
                print("任务完成!")
                if "result" in task_info:
                    result = task_info['result']
                    print("结果: " + json.dumps(result, indent=2, ensure_ascii=False))
                    
                    # 检查视频文件是否存在
                    if "output_path" in result:
                        video_path = result["output_path"]
                        if os.path.exists(video_path):
                            print("成功: 视频文件已生成: " + video_path + ", 大小: " + str(os.path.getsize(video_path)/1024) + " KB")
                        else:
                            print("失败: 视频文件不存在: " + video_path)
                            
                        # 如果有临时路径，也检查一下
                        if "temp_path" in result:
                            temp_path = result["temp_path"]
                            if os.path.exists(temp_path):
                                print("成功: 临时视频文件存在: " + temp_path + ", 大小: " + str(os.path.getsize(temp_path)/1024) + " KB")
                            else:
                                print("失败: 临时视频文件不存在: " + temp_path)
                return True
                
            if status == "failed":
                error = task_info.get('error', '未知错误')
                print("任务失败: " + error)
                
                # 检查失败原因是否包含FFmpeg命令
                if "FFmpeg命令失败" in error:
                    print("\n检查临时目录中是否有音频文件:")
                    if os.path.exists("./temp"):
                        for file in os.listdir("./temp"):
                            if file.endswith((".wav", ".mp3")):
                                file_path = os.path.join("./temp", file)
                                print("  - " + file + ": " + str(os.path.getsize(file_path)/1024) + " KB")
                    else:
                        print("  临时目录不存在")
                return False
        
        except Exception as e:
            print("获取任务状态时出错: " + str(e))
            
        time.sleep(2)
    
    print("等待超时!")
    return False

def test_check_environment(api_url):
    """测试环境检查"""
    print("\n======= 测试环境检查 =======")
    response = requests.get(api_url + "/api/check_environment")
    
    if response.status_code != 200:
        print("请求失败: " + str(response.status_code))
        return False
        
    result = response.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result.get("success", False)

def main():
    parser = argparse.ArgumentParser(description="MuseTalk API测试脚本")
    parser.add_argument("--url", default="http://localhost:5000", help="API基础URL")
    parser.add_argument("--text", default="你好，我是数字人。这是一条测试文本，用于验证文字转视频功能是否正常。", help="测试文本")
    parser.add_argument("--avatar", default="avator_50", help="数字人ID")
    parser.add_argument("--tts-model", default="wang001", help="TTS模型ID")
    parser.add_argument("--test-tts-only", action="store_true", help="仅测试TTS功能")
    args = parser.parse_args()
    
    api_url = args.url.rstrip('/')
    
    if args.test_tts_only:
        # 生成临时路径
        temp_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp"))
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, "tts_test_" + str(int(time.time())) + ".wav")
        
        # 仅测试TTS
        test_tts(args.text, output_path, args.tts_model)
    else:
        # 测试环境
        test_check_environment(api_url)
        
        # 测试获取数字人列表
        test_list_avatars(api_url)
        
        # 测试TTS功能
        temp_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp"))
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, "tts_test_" + str(int(time.time())) + ".wav")
        test_tts(args.text, output_path, args.tts_model)
        
        # 测试生成视频
        test_generate_video(api_url, args.text, args.avatar)

if __name__ == "__main__":
    main() 

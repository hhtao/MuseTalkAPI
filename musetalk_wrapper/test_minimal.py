#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MuseTalk 极简测试脚本
仅使用TTS和Avatar类直接测试
"""

import os
import sys
import logging
import argparse
import time
import subprocess
from types import SimpleNamespace
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MinimalTest")

def test_direct_integration(text, avatar_id, output_path):
    """直接测试TTS和Avatar集成"""
    print("\n======= 极简模式测试 =======")
    print("文本内容: " + text)
    print("数字人ID: " + avatar_id)
    print("输出路径: " + output_path)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建临时目录
    temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 第1步: 使用TTS生成音频
    from musetalk_wrapper.utils.tts_service import TTSService
    tts_config = {
        "server": "http://192.168.202.10:9880",
        "models": {
            "wang001": {"name": "王力宏"}
        }
    }
    
    tts_service = TTSService(tts_config)
    print("TTS服务初始化完成")
    
    # 生成音频文件
    audio_path = os.path.join(temp_dir, "tts_test_" + str(int(time.time())) + ".wav")
    print("音频文件路径: " + audio_path)
    
    success, message = tts_service.text_to_speech(text, audio_path, "wang001", "zh")
    if not success:
        print("TTS生成失败: " + message)
        return False
    
    print("音频生成成功: " + audio_path)
    
    # 第2步: 准备Avatar类所需的参数
    # 设置args参数
    args = SimpleNamespace()
    args.version = "v15"
    args.parsing_mode = "jaw"
    args.extra_margin = 10
    args.skip_save_images = False
    args.gpu_id = 0
    args.audio_padding_length_left = 2
    args.audio_padding_length_right = 2
    args.vae_type = "sd-vae"
    args.unet_config = "./models/musetalk/musetalk.json"
    args.unet_model_path = "./models/musetalk/pytorch_model.bin"
    args.whisper_dir = "./models/whisper"
    args.left_cheek_width = 90
    args.right_cheek_width = 90
    
    try:
        # 第3步: 动态导入Avatar类
        print("导入Avatar类...")
        # 设置CUDA设备
        import torch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("使用设备: " + str(device))
        
        # 先导入必要的模块
        try:
            from scripts.realtime_inference import Avatar, AudioProcessor, load_all_model
            print("成功导入Avatar类")
        except ImportError as e:
            print("直接导入失败: " + str(e))
            import importlib.util
            script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "scripts/realtime_inference.py")
            print("尝试从文件导入: " + script_path)
            
            spec = importlib.util.spec_from_file_location("realtime_inference", script_path)
            realtime_inference = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(realtime_inference)
            Avatar = realtime_inference.Avatar
            AudioProcessor = realtime_inference.AudioProcessor
            load_all_model = realtime_inference.load_all_model
            print("从文件路径导入成功")
        
        # 第4步: 初始化模型
        print("加载模型...")
        import scripts.realtime_inference
        scripts.realtime_inference.args = args
        scripts.realtime_inference.device = device
        
        # 确保模型目录存在
        models_dir = "./models"
        if not os.path.exists(os.path.join(models_dir, "musetalk")):
            print("警告: 模型目录不存在: " + os.path.join(models_dir, "musetalk"))
            print("尝试使用绝对路径...")
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            args.unet_config = os.path.join(models_dir, "musetalk/musetalk.json")
            args.unet_model_path = os.path.join(models_dir, "musetalk/pytorch_model.bin")
            args.whisper_dir = os.path.join(models_dir, "whisper")
            
        print("使用配置:")
        print("  UNet配置: " + args.unet_config)
        print("  UNet模型: " + args.unet_model_path)
        print("  Whisper目录: " + args.whisper_dir)
            
        # 初始化模型
        vae, unet, pe = load_all_model(
            unet_model_path=args.unet_model_path,
            vae_type=args.vae_type,
            unet_config=args.unet_config,
            device=device
        )
        
        # 设置全局变量
        scripts.realtime_inference.vae = vae
        scripts.realtime_inference.unet = unet
        scripts.realtime_inference.pe = pe
        scripts.realtime_inference.timesteps = torch.tensor([0], device=device)
        
        # 模型转换为半精度
        weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        scripts.realtime_inference.weight_dtype = weight_dtype
        
        # 转换模型精度
        if torch.cuda.is_available():
            pe = pe.half().to(device)
            vae.vae = vae.vae.half().to(device)
            unet.model = unet.model.half().to(device)
        else:
            pe = pe.to(device)
            vae.vae = vae.vae.to(device)
            unet.model = unet.model.to(device)
        
        # 初始化AudioProcessor
        from transformers import WhisperModel
        audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
        scripts.realtime_inference.audio_processor = audio_processor
        
        # 初始化Whisper模型
        whisper = WhisperModel.from_pretrained(args.whisper_dir)
        whisper = whisper.to(device=device, dtype=weight_dtype).eval()
        whisper.requires_grad_(False)
        scripts.realtime_inference.whisper = whisper
        
        # 初始化FaceParsing
        from musetalk.utils.face_parsing import FaceParsing
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        )
        scripts.realtime_inference.fp = fp
        
        print("模型加载完成")
        
        # 第5步: 获取Avatar路径
        avatar_dir = "./results/v15/avatars"
        if not os.path.exists(os.path.join(avatar_dir, avatar_id)):
            print("警告: 数字人目录不存在: " + os.path.join(avatar_dir, avatar_id))
            print("尝试使用绝对路径...")
            avatar_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results/v15/avatars")
        
        avatar_path = os.path.join(avatar_dir, avatar_id)
        print("数字人路径: " + avatar_path)
        
        if not os.path.exists(avatar_path):
            print("错误: 数字人路径不存在: " + avatar_path)
            return False
        
        # 第6步: 创建Avatar实例
        print("创建Avatar实例...")
        avatar_instance = Avatar(
            avatar_id=avatar_id,
            video_path=avatar_path,
            bbox_shift=0,
            batch_size=4,
            preparation=False
        )
        
        # 第7步: 生成视频
        out_vid_name = os.path.splitext(os.path.basename(output_path))[0]
        print("开始生成视频...")
        print("  音频文件: " + audio_path)
        print("  输出文件: " + out_vid_name)
        
        avatar_instance.inference(
            audio_path=audio_path,
            out_vid_name=out_vid_name,
            fps=25,
            skip_save_images=False
        )
        
        # 第8步: 复制最终视频
        # 查找生成的视频
        generated_video = os.path.join(avatar_path, "vid_output", out_vid_name + ".mp4")
        print("查找生成的视频: " + generated_video)
        
        if os.path.exists(generated_video):
            # 转换视频格式后复制到最终输出路径
            temp_output = os.path.join(temp_dir, "temp_output_" + str(int(time.time())) + ".mp4")
            convert_success = convert_video_format(generated_video, temp_output)
            
            if convert_success:
                # 复制转换后的视频
                import shutil
                shutil.copy2(temp_output, output_path)
                print("视频已转换并复制到: " + output_path)
                
                # 清理临时文件
                try:
                    os.remove(temp_output)
                except:
                    pass
                    
                return True
            else:
                # 如果转换失败，直接复制原始视频
                import shutil
                shutil.copy2(generated_video, output_path)
                print("无法转换视频格式，直接复制原始视频到: " + output_path)
                return True
        else:
            print("错误: 未找到生成的视频: " + generated_video)
            # 尝试查找其他可能的位置
            possible_paths = [
                os.path.join(avatar_path, "vid_output", out_vid_name + ".mp4"),
                os.path.join(avatar_path, out_vid_name + ".mp4"),
                os.path.join(os.path.dirname(avatar_path), out_vid_name + ".mp4")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    # 转换视频格式后复制
                    temp_output = os.path.join(temp_dir, "temp_output_" + str(int(time.time())) + ".mp4")
                    convert_success = convert_video_format(path, temp_output)
                    
                    if convert_success:
                        import shutil
                        shutil.copy2(temp_output, output_path)
                        print("视频已转换并复制到: " + output_path)
                        
                        # 清理临时文件
                        try:
                            os.remove(temp_output)
                        except:
                            pass
                            
                        return True
                    else:
                        # 如果转换失败，直接复制原始视频
                        import shutil
                        shutil.copy2(path, output_path)
                        print("无法转换视频格式，直接复制原始视频到: " + output_path)
                        return True
                    
            return False
            
    except Exception as e:
        import traceback
        print("发生异常: " + str(e))
        print("详细错误: ")
        print(traceback.format_exc())
        return False

# 添加视频格式转换函数
def convert_video_format(input_video, output_video):
    """将视频转换为更兼容的格式"""
    print(f"转换视频格式: {input_video} -> {output_video}")
    
    # 使用libx264编码器和高兼容性参数
    cmd = f"ffmpeg -y -v warning -i {input_video} -c:v libx264 -preset medium -crf 23 -c:a aac -b:a 192k -pix_fmt yuv420p {output_video}"
    print(f"执行命令: {cmd}")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
            print(f"视频格式转换成功: {output_video}")
            return True
        else:
            print(f"视频格式转换失败: 输出文件不存在或为空")
            return False
    except subprocess.CalledProcessError as e:
        print(f"视频格式转换失败: {e}")
        
        # 尝试使用mpeg4编码器
        try:
            cmd = f"ffmpeg -y -v warning -i {input_video} -c:v mpeg4 -q:v 5 -c:a aac -b:a 192k {output_video}"
            print(f"尝试备用编码器: {cmd}")
            subprocess.run(cmd, shell=True, check=True)
            
            if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
                print(f"使用备用编码器转换成功: {output_video}")
                return True
            else:
                return False
        except:
            return False

def main():
    parser = argparse.ArgumentParser(description="MuseTalk极简测试工具")
    parser.add_argument("--text", "-t", default="你好，我是数字人。这是一条极简模式的测试文本。", help="测试文本")
    parser.add_argument("--avatar-id", "-a", default="avator_50", help="数字人ID")
    parser.add_argument("--output", "-o", default="./minimal_test_output.mp4", help="输出视频路径")
    args = parser.parse_args()
    
    test_direct_integration(args.text, args.avatar_id, args.output)

if __name__ == "__main__":
    main() 

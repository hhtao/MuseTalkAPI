"""
任务管理模块
负责任务队列、进度跟踪和状态管理
"""
import os
import time
import uuid
import queue
import threading
import logging
import traceback
import tempfile
import shutil
from typing import Dict, List, Any, Optional

# 尝试导入可能缺失的库
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    logging.warning("未安装 imageio 库，高级视频生成功能将不可用")
    logging.warning("如需使用全部功能，请安装: pip install imageio imageio-ffmpeg")
    IMAGEIO_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    logging.warning("未安装 opencv-python 库，视频处理功能将受限")
    logging.warning("如需使用全部功能，请安装: pip install opencv-python")
    CV2_AVAILABLE = False

try:
    from moviepy.editor import VideoFileClip, AudioFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    logging.warning("未安装 moviepy 库，高级视频编辑功能将不可用")
    logging.warning("如需使用全部功能，请安装: pip install moviepy")
    MOVIEPY_AVAILABLE = False

# 创建全局任务队列和状态
task_queue = queue.Queue()
task_progress = {}
task_cancel_flags = {}
gpu_statuses = {}
is_processing_tasks = False

class Task:
    """表示一个待处理任务的类"""
    def __init__(self, task_id=None, task_type=None, data=None):
        self.task_id = task_id or str(uuid.uuid4())
        self.task_type = task_type
        self.data = data or {}
        self.created_at = time.time()
        
    def to_dict(self):
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "created_at": self.created_at,
            "data": self.data
        }

def init_task_system(gpu_count=0):
    """初始化任务系统"""
    global gpu_statuses
    
    # 初始化GPU状态
    for i in range(gpu_count):
        gpu_statuses[i] = {
            "status": "idle",  # idle, busy
            "task_id": None,
            "last_activity": time.time(),
            "processed_count": 0
        }
    
    # 启动任务处理线程
    if not is_processing_tasks:
        threading.Thread(target=process_task_queue, daemon=True).start()
    
    return True

def add_task(task_type, data, task_id=None):
    """添加新任务到队列"""
    global task_queue, task_progress, is_processing_tasks
    
    task_id = task_id or str(uuid.uuid4())
    task = Task(task_id=task_id, task_type=task_type, data=data)
    
    # 初始化任务进度
    with threading.Lock():
        task_progress[task_id] = {
            "status": "pending",  # pending, processing, completed, error
            "progress": 0,
            "total": 100,  # 默认总进度为100
            "created_at": time.time(),
            "audio_status": "等待处理",
            "video_status": "等待处理",
            "error": None
        }
    
    # 添加到队列
    task_queue.put(task)
    
    # 确保处理线程已启动
    if not is_processing_tasks:
        threading.Thread(target=process_task_queue, daemon=True).start()
    
    return task_id

def get_task_progress(task_id):
    """获取任务进度"""
    with threading.Lock():
        if task_id in task_progress:
            return task_progress[task_id]
    return None

def cancel_task(task_id):
    """取消任务"""
    global task_queue
    
    with threading.Lock():
        task_cancel_flags[task_id] = True
        
        # 如果任务还在队列中且未开始处理，从队列中移除
        new_queue = queue.Queue()
        while not task_queue.empty():
            task = task_queue.get()
            if task.task_id != task_id:
                new_queue.put(task)
                
        # 替换原有队列
        task_queue = new_queue
        
        # 更新任务状态
        if task_id in task_progress:
            task_progress[task_id]["status"] = "canceled"
            task_progress[task_id]["error"] = "任务已取消"
    
    return True

def clear_tasks():
    """清除所有任务"""
    global task_queue, task_progress, task_cancel_flags
    
    with threading.Lock():
        # 清空队列
        while not task_queue.empty():
            try:
                task_queue.get_nowait()
            except:
                pass
        
        # 标记所有任务为取消
        for task_id in task_progress:
            task_cancel_flags[task_id] = True
            task_progress[task_id]["status"] = "canceled"
            task_progress[task_id]["error"] = "任务已清除"
    
    return True

def process_task_queue():
    """任务处理线程，不断从队列获取任务并处理"""
    global is_processing_tasks, task_queue, task_cancel_flags, task_progress
    
    # 导入必要的视频生成模块
    from MuseTalk.utils.modules.audio_utils import gpt_sovits_tts, fallback_tts
    from MuseTalk.utils.modules.video_utils import create_simple_video
    from MuseTalk.utils.modules.subtitle_utils import create_subtitle_for_text, add_subtitles_to_video
    from MuseTalk.utils.modules.config import VIDEOS_DIR, SUBTITLES_DIR, AUDIO_DIR, DEFAULT_TTS_SERVER, REF_AUDIO_FILE, REF_TEXT
    
    # 导入 MuseTalk 的音频到视频转换功能
    try:
        import sys
        import torch
        
        # 检查是否已安装必要的库
        if not IMAGEIO_AVAILABLE or not CV2_AVAILABLE or not MOVIEPY_AVAILABLE:
            logging.warning("由于缺少必要的库，高级视频生成功能不可用")
            musetalk_available = False
        else:
            from MuseTalk.musetalk.utils.utils import load_all_model, get_file_type, get_video_fps, datagen
            from MuseTalk.musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder, get_bbox_range
            from MuseTalk.musetalk.utils.blending import get_image
            
            # 尝试加载 MuseTalk 模型
            audio_processor, vae, unet, pe = load_all_model()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            timesteps = torch.tensor([0], device=device)
            musetalk_available = True
            logging.info("成功加载 MuseTalk 模型，可以使用高级视频生成功能")
    except Exception as e:
        logging.error(f"加载 MuseTalk 模型失败: {str(e)}")
        musetalk_available = False
        logging.warning("将使用简单视频生成作为替代")
    
    is_processing_tasks = True
    
    try:
        while True:
            try:
                task = task_queue.get(timeout=1)  # 等待1秒
                
                # 检查任务是否已取消
                if task.task_id in task_cancel_flags and task_cancel_flags[task.task_id]:
                    logging.info(f"任务 {task.task_id} 已取消，跳过处理")
                    continue
                
                # 更新任务状态
                with threading.Lock():
                    if task.task_id in task_progress:
                        task_progress[task.task_id]["status"] = "processing"
                
                # 根据任务类型调用不同的处理函数
                if task.task_type == "generate_video":
                    # 实际处理视频生成任务
                    logging.info(f"处理生成视频任务: {task.task_id}")
                    
                    try:
                        # 获取任务数据
                        text = task.data.get('text', '')
                        lang = task.data.get('lang', 'zh')
                        avatar_id = task.data.get('avatar_id', None)
                        
                        # 更新音频状态
                        with threading.Lock():
                            if task.task_id in task_progress:
                                task_progress[task.task_id]["audio_status"] = "正在生成音频..."
                        
                        # 确保目录存在
                        os.makedirs(AUDIO_DIR, exist_ok=True)
                        os.makedirs(VIDEOS_DIR, exist_ok=True)
                        os.makedirs(SUBTITLES_DIR, exist_ok=True)
                        
                        # 1. 生成音频
                        logging.info(f"开始为任务 {task.task_id} 生成音频")
                        audio_bytes = gpt_sovits_tts(
                            text=text,
                            lang=lang,
                            server_url=DEFAULT_TTS_SERVER,
                            ref_audio_path=REF_AUDIO_FILE,
                            ref_text=REF_TEXT
                        )
                        
                        # 如果主要TTS失败，使用备选方法
                        if not audio_bytes:
                            logging.warning(f"主要TTS失败，使用备选方法: {task.task_id}")
                            audio_bytes = fallback_tts(text, lang)
                            
                        if not audio_bytes:
                            raise Exception("无法生成音频")
                            
                        # 保存音频
                        audio_filename = f"audio_{task.task_id}.wav"
                        audio_path = os.path.join(AUDIO_DIR, audio_filename)
                        
                        logging.info(f"保存音频到: {audio_path}")
                        with open(audio_path, 'wb') as f:
                            f.write(audio_bytes)
                        
                        # 检查音频文件是否生成成功
                        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                            logging.error(f"音频文件生成失败: {audio_path}")
                            raise Exception("音频生成失败")
                        else:
                            logging.info(f"音频生成成功，大小: {os.path.getsize(audio_path)} 字节")
                            
                        # 更新音频状态
                        with threading.Lock():
                            if task.task_id in task_progress:
                                task_progress[task.task_id]["audio_status"] = "音频生成完成"
                                task_progress[task.task_id]["progress"] = 30
                                task_progress[task.task_id]["video_status"] = "正在生成视频..."
                        
                        # 2. 选择视频生成方法
                        final_video_filename = None
                        final_video_path = None
                        
                        # 视频生成逻辑
                        if musetalk_available and avatar_id and IMAGEIO_AVAILABLE and CV2_AVAILABLE and MOVIEPY_AVAILABLE:
                            # 使用 MuseTalk 生成高质量视频
                            with threading.Lock():
                                if task.task_id in task_progress:
                                    task_progress[task.task_id]["video_status"] = "使用MuseTalk生成高质量视频..."
                            
                            # 查找选定的头像视频模板
                            from MuseTalk.utils.modules.config import AVATAR_DIR
                            video_template = None
                            
                            # 首先检查头像目录本身是否有视频文件
                            avatar_dir = os.path.join(AVATAR_DIR, avatar_id)
                            if os.path.exists(avatar_dir):
                                logging.info(f"查找头像目录中的视频文件: {avatar_dir}")
                                
                                # 首先检查vid_output目录
                                vid_output_dir = os.path.join(avatar_dir, "vid_output")
                                if os.path.exists(vid_output_dir):
                                    mp4_files = [f for f in os.listdir(vid_output_dir) if f.endswith('.mp4')]
                                    if mp4_files:
                                        video_template = os.path.join(vid_output_dir, mp4_files[0])
                                        logging.info(f"找到vid_output目录中的视频模板: {video_template}")
                                
                                # 如果vid_output中没有找到，检查主目录
                                if not video_template:
                                    for video_name in ['temp.mp4', 'template.mp4', 'video.mp4', 'output.mp4']:
                                        test_path = os.path.join(avatar_dir, video_name)
                                        if os.path.exists(test_path):
                                            video_template = test_path
                                            logging.info(f"找到头像视频模板: {video_template}")
                                            break
                                        
                                    # 如果没找到特定名称的视频，尝试查找任何mp4文件
                                    if not video_template:
                                        mp4_files = [f for f in os.listdir(avatar_dir) if f.endswith('.mp4')]
                                        if mp4_files:
                                            video_template = os.path.join(avatar_dir, mp4_files[0])
                                            logging.info(f"找到头像目录中的MP4文件: {video_template}")
                            
                            # 如果没有在头像目录找到，检查模板目录
                            if not video_template:
                                template_dir = os.path.join("MuseTalk", "models", "templates")
                                if os.path.exists(template_dir):
                                    logging.info(f"在模板目录中查找: {template_dir}")
                                    for file in os.listdir(template_dir):
                                        if file.startswith(avatar_id) and file.endswith((".mp4", ".avi")):
                                            video_template = os.path.join(template_dir, file)
                                            logging.info(f"在模板目录中找到: {video_template}")
                                            break
                            
                            # 如果仍未找到，使用默认模板
                            if not video_template:
                                # 尝试使用默认模板
                                default_template = os.path.join("MuseTalk", "models", "templates", "default.mp4")
                                if os.path.exists(default_template):
                                    video_template = default_template
                                    logging.info(f"使用默认视频模板: {default_template}")
                                else:
                                    # 如果默认模板也不存在，尝试找任何一个头像的视频作为模板
                                    for avatar_folder in os.listdir(AVATAR_DIR):
                                        avatar_folder_path = os.path.join(AVATAR_DIR, avatar_folder)
                                        if os.path.isdir(avatar_folder_path):
                                            vid_output_dir = os.path.join(avatar_folder_path, "vid_output")
                                            if os.path.exists(vid_output_dir):
                                                mp4_files = [f for f in os.listdir(vid_output_dir) if f.endswith('.mp4')]
                                                if mp4_files:
                                                    video_template = os.path.join(vid_output_dir, mp4_files[0])
                                                    logging.info(f"使用其他头像的视频作为模板: {video_template}")
                                                    break
                                            
                                            if not video_template:
                                                for video_name in ['temp.mp4', 'template.mp4', 'video.mp4', 'output.mp4']:
                                                    test_path = os.path.join(avatar_folder_path, video_name)
                                                    if os.path.exists(test_path):
                                                        video_template = test_path
                                                        logging.info(f"使用其他头像的视频作为模板: {video_template}")
                                                        break
                                        if video_template:
                                            break
                            
                            # 最终检查
                            if not video_template:
                                # 如果所有方法都失败，回退到简单视频生成
                                raise Exception("找不到任何可用的视频模板，将使用简单视频生成")
                            
                            # 使用 MuseTalk 的 inference 函数生成视频
                            # 参考 app.py 中的实现
                            try:
                                import torch
                                from moviepy.editor import VideoFileClip, AudioFileClip
                                
                                # 定义输出文件名
                                temp_video = os.path.join(VIDEOS_DIR, f"temp_{task.task_id}.mp4")
                                final_video_filename = f"video_{task.task_id}.mp4"
                                final_video_path = os.path.join(VIDEOS_DIR, final_video_filename)
                                
                                logging.info(f"开始高质量视频生成: 任务ID={task.task_id}, 模板={video_template}")
                                
                                # 从模板创建临时目录
                                temp_dir = os.path.join(tempfile.gettempdir(), f"musetalk_temp_{task.task_id}")
                                os.makedirs(temp_dir, exist_ok=True)
                                logging.info(f"创建临时目录: {temp_dir}")
                                
                                try:
                                    # 1. 提取人脸边界框
                                    with threading.Lock():
                                        if task.task_id in task_progress:
                                            task_progress[task.task_id]["video_status"] = "分析视频模板..."
                                    
                                    # 检查视频模板文件
                                    if not os.path.exists(video_template):
                                        raise Exception(f"视频模板文件不存在: {video_template}")
                                    
                                    logging.info(f"开始提取视频帧: {video_template}")
                                    
                                    # 从模板提取坐标和边界框
                                    input_img_list = []
                                    save_dir_full = os.path.join(temp_dir, "frames")
                                    os.makedirs(save_dir_full, exist_ok=True)
                                    
                                    # 使用 moviepy 提取帧
                                    try:
                                        video_clip = VideoFileClip(video_template)
                                        fps = video_clip.fps
                                        logging.info(f"视频FPS: {fps}, 时长: {video_clip.duration}秒")
                                        
                                        # 提取帧
                                        frame_count = 0
                                        for i, frame in enumerate(video_clip.iter_frames()):
                                            frame_path = os.path.join(save_dir_full, f"{i:08d}.png")
                                            imageio.imsave(frame_path, frame)
                                            input_img_list.append(frame_path)
                                            frame_count += 1
                                            
                                            # 为了防止处理太多帧导致卡住，限制帧数
                                            if frame_count >= 300:  # 最多处理10秒左右的视频
                                                logging.warning(f"视频帧数过多，仅处理前{frame_count}帧")
                                                break
                                                
                                        logging.info(f"成功提取{frame_count}帧")
                                    except Exception as e:
                                        logging.error(f"提取视频帧失败: {str(e)}")
                                        traceback.print_exc()
                                        raise Exception(f"视频帧提取失败: {str(e)}")
                                    
                                    # 默认的边界框偏移 (可以根据需要调整)
                                    bbox_shift = [0, 0, 0, 0]
                                    
                                    # 2. 处理音频
                                    with threading.Lock():
                                        if task.task_id in task_progress:
                                            task_progress[task.task_id]["video_status"] = "处理音频特征..."
                                            task_progress[task.task_id]["progress"] = 50
                                    
                                    logging.info("开始处理音频特征")
                                    try:
                                        whisper_feature = audio_processor.audio2feat(audio_path)
                                        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
                                        logging.info(f"音频特征处理完成，chunks数量: {len(whisper_chunks)}")
                                    except Exception as e:
                                        logging.error(f"音频特征处理失败: {str(e)}")
                                        traceback.print_exc()
                                        raise Exception(f"音频特征处理失败: {str(e)}")
                                    
                                    # 3. 提取图像特征
                                    with threading.Lock():
                                        if task.task_id in task_progress:
                                            task_progress[task.task_id]["video_status"] = "处理视频特征..."
                                            task_progress[task.task_id]["progress"] = 60
                                    
                                    logging.info("开始提取图像特征和边界框")
                                    try:
                                        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
                                        logging.info(f"图像特征提取完成，帧数量: {len(frame_list)}")
                                    except Exception as e:
                                        logging.error(f"图像特征提取失败: {str(e)}")
                                        traceback.print_exc()
                                        raise Exception(f"图像特征提取失败: {str(e)}")
                                    
                                    # 4. 准备输入特征
                                    logging.info("准备VAE输入特征")
                                    try:
                                        input_latent_list = []
                                        valid_frames = 0
                                        for bbox, frame in zip(coord_list, frame_list):
                                            if bbox == coord_placeholder:
                                                continue
                                            try:
                                                x1, y1, x2, y2 = bbox
                                                crop_frame = frame[y1:y2, x1:x2]
                                                crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                                                latents = vae.get_latents_for_unet(crop_frame)
                                                input_latent_list.append(latents)
                                                valid_frames += 1
                                            except Exception as frame_error:
                                                logging.warning(f"处理单个帧时出错，跳过: {str(frame_error)}")
                                                continue
                                        
                                        logging.info(f"有效输入特征数量: {valid_frames}")
                                        
                                        if len(input_latent_list) == 0:
                                            raise Exception("没有有效的输入特征，无法生成视频")
                                            
                                        # 平滑首尾帧
                                        frame_list_cycle = frame_list + frame_list[::-1]
                                        coord_list_cycle = coord_list + coord_list[::-1]
                                        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
                                        logging.info("特征准备完成")
                                    except Exception as e:
                                        logging.error(f"准备输入特征失败: {str(e)}")
                                        traceback.print_exc()
                                        raise Exception(f"准备输入特征失败: {str(e)}")
                                    
                                    # 5. 逐批次推理
                                    with threading.Lock():
                                        if task.task_id in task_progress:
                                            task_progress[task.task_id]["video_status"] = "生成唇形同步视频..."
                                            task_progress[task.task_id]["progress"] = 70
                                    
                                    logging.info("开始生成唇形同步视频")
                                    try:
                                        video_num = len(whisper_chunks)
                                        batch_size = 10  # 可以调整
                                        gen = datagen(whisper_chunks, input_latent_list_cycle, batch_size)
                                        res_frame_list = []
                                        
                                        # 逐批次处理
                                        batch_count = 0
                                        for i, (whisper_batch, latent_batch) in enumerate(gen):
                                            try:
                                                tensor_list = [torch.FloatTensor(arr) for arr in whisper_batch]
                                                audio_feature_batch = torch.stack(tensor_list).to(unet.device)
                                                audio_feature_batch = pe(audio_feature_batch)
                                                
                                                pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                                                recon = vae.decode_latents(pred_latents)
                                                for res_frame in recon:
                                                    res_frame_list.append(res_frame)
                                                
                                                batch_count += 1
                                                if batch_count % 5 == 0:
                                                    logging.info(f"已处理{batch_count}批次，生成{len(res_frame_list)}帧")
                                            except Exception as batch_error:
                                                logging.warning(f"处理批次{i}失败，跳过: {str(batch_error)}")
                                                continue
                                        
                                        logging.info(f"生成完成，共{len(res_frame_list)}帧")
                                    except Exception as e:
                                        logging.error(f"唇形同步视频生成失败: {str(e)}")
                                        traceback.print_exc()
                                        raise Exception(f"唇形同步视频生成失败: {str(e)}")
                                    
                                    # 6. 合成完整画面
                                    with threading.Lock():
                                        if task.task_id in task_progress:
                                            task_progress[task.task_id]["video_status"] = "合成最终视频..."
                                            task_progress[task.task_id]["progress"] = 80
                                    
                                    logging.info("开始合成完整视频帧")
                                    try:
                                        result_img_save_path = os.path.join(temp_dir, "results")
                                        os.makedirs(result_img_save_path, exist_ok=True)
                                        
                                        success_frame_count = 0
                                        for i, res_frame in enumerate(res_frame_list):
                                            try:
                                                bbox = coord_list_cycle[i % len(coord_list_cycle)]
                                                ori_frame = frame_list_cycle[i % len(frame_list_cycle)].copy()
                                                
                                                x1, y1, x2, y2 = bbox
                                                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                                                combine_frame = get_image(ori_frame, res_frame, bbox)
                                                cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)
                                                success_frame_count += 1
                                            except Exception as frame_error:
                                                logging.warning(f"合成帧{i}失败: {str(frame_error)}")
                                                continue
                                        
                                        logging.info(f"成功合成{success_frame_count}帧")
                                        
                                        if success_frame_count == 0:
                                            raise Exception("没有成功合成的帧，无法生成视频")
                                    except Exception as e:
                                        logging.error(f"合成视频帧失败: {str(e)}")
                                        traceback.print_exc()
                                        raise Exception(f"合成视频帧失败: {str(e)}")
                                    
                                    # 7. 生成视频文件
                                    with threading.Lock():
                                        if task.task_id in task_progress:
                                            task_progress[task.task_id]["video_status"] = "生成最终视频文件..."
                                            task_progress[task.task_id]["progress"] = 90
                                    
                                    logging.info("开始生成最终视频文件")
                                    try:
                                        # 读取生成的帧
                                        images = []
                                        files = sorted(os.listdir(result_img_save_path), key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 0)
                                        for file in files:
                                            if file.endswith('.png'):
                                                filename = os.path.join(result_img_save_path, file)
                                                images.append(imageio.imread(filename))
                                        
                                        logging.info(f"准备生成视频，帧数: {len(images)}")
                                        
                                        # 生成临时视频
                                        imageio.mimwrite(temp_video, images, 'FFMPEG', fps=fps, codec='libx264', pixelformat='yuv420p')
                                        logging.info(f"临时视频生成完成: {temp_video}")
                                        
                                        # 添加音频
                                        video_clip = VideoFileClip(temp_video)
                                        audio_clip = AudioFileClip(audio_path)
                                        video_clip = video_clip.set_audio(audio_clip)
                                        video_clip.write_videofile(final_video_path, codec='libx264', audio_codec='aac', fps=fps)
                                        logging.info(f"完整视频生成成功: {final_video_path}")
                                    except Exception as e:
                                        logging.error(f"生成最终视频文件失败: {str(e)}")
                                        traceback.print_exc()
                                        raise Exception(f"生成最终视频文件失败: {str(e)}")
                                    
                                    # 清理临时文件
                                    try:
                                        if os.path.exists(temp_video):
                                            os.remove(temp_video)
                                        shutil.rmtree(temp_dir, ignore_errors=True)
                                        logging.info("临时文件清理完成")
                                    except Exception as cleanup_error:
                                        logging.warning(f"清理临时文件失败: {str(cleanup_error)}")
                                    
                                except Exception as inner_e:
                                    logging.error(f"高质量视频生成内部错误: {str(inner_e)}")
                                    traceback.print_exc()
                                    
                                    # 清理临时文件
                                    try:
                                        if os.path.exists(temp_video):
                                            os.remove(temp_video)
                                        shutil.rmtree(temp_dir, ignore_errors=True)
                                    except:
                                        pass
                                    
                                    # 内部错误，回退到简单视频生成
                                    logging.info("回退到简单视频生成")
                                    from MuseTalk.utils.modules.video_utils import create_simple_video
                                    success = create_simple_video(audio_path, final_video_path)
                                    if not success:
                                        raise Exception("所有视频生成方法都失败了")
                                    
                            except Exception as e:
                                logging.error(f"MuseTalk视频生成失败，将使用简单方法: {str(e)}")
                                traceback.print_exc()
                                
                                # 如果MuseTalk失败，回退到简单视频生成
                                logging.info(f"尝试使用简单方法生成视频: {task.task_id}")
                                from MuseTalk.utils.modules.video_utils import create_simple_video
                                success = create_simple_video(audio_path, final_video_path)
                                if not success:
                                    raise Exception("视频生成失败")
                        else:
                            # 使用简单方法生成视频
                            logging.info(f"使用简单方法生成视频: {task.task_id}")
                            video_filename = f"video_{task.task_id}.mp4"
                            video_path = os.path.join(VIDEOS_DIR, video_filename)
                            
                            from MuseTalk.utils.modules.video_utils import create_simple_video
                            logging.info(f"调用create_simple_video，音频: {audio_path}，输出: {video_path}")
                            success = create_simple_video(audio_path, video_path)
                            
                            if not success:
                                logging.error(f"视频生成失败: {video_path}")
                                raise Exception("视频生成失败")
                                
                            # 检查生成的视频是否存在
                            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                                logging.error(f"视频文件不存在或为空: {video_path}")
                                raise Exception("视频生成失败：文件不存在或为空")
                            else:
                                logging.info(f"视频生成成功，大小: {os.path.getsize(video_path)} 字节")
                            
                            final_video_filename = video_filename
                            final_video_path = video_path
                        
                        # 更新视频状态
                        with threading.Lock():
                            if task.task_id in task_progress:
                                task_progress[task.task_id]["video_status"] = "视频生成完成"
                                task_progress[task.task_id]["progress"] = 80
                                task_progress[task.task_id]["subtitle_status"] = "正在生成字幕..."
                        
                        # 3. 生成字幕
                        subtitle_filename = f"subtitle_{task.task_id}.vtt"
                        subtitle_path = os.path.join(SUBTITLES_DIR, subtitle_filename)
                        
                        # 估算时长
                        if audio_duration:
                            duration = audio_duration
                        else:
                            duration = len(text) * 0.2  # 简单估计: 每个字符0.2秒
                            
                        # 创建字幕，指定text_cut_method
                        # 如果任务数据中有text_cut_method，使用它，否则使用默认的cut3
                        text_cut_method = task.data.get('text_cut_method', 'cut3')
                        speech_rate = task.data.get('speech_rate', 1.0)
                        subtitle_offset = task.data.get('subtitle_offset', -0.3)
                        
                        create_subtitle_for_text(
                            text, 
                            subtitle_path,
                            duration,
                            format='vtt',
                            encoding='utf-8',
                            offset=subtitle_offset,
                            subtitle_speed=speech_rate,
                            text_cut_method=text_cut_method
                        )
                        
                        # 添加字幕到视频
                        subtitle_video_filename = f"final_{task.task_id}.mp4"
                        subtitle_video_path = os.path.join(VIDEOS_DIR, subtitle_video_filename)
                        
                        # 尝试添加字幕，如果失败则使用原始视频
                        try:
                            add_subtitles_to_video(final_video_path, subtitle_path, subtitle_video_path)
                            final_video_filename = subtitle_video_filename
                        except:
                            import shutil
                            shutil.copy(final_video_path, subtitle_video_path)
                            final_video_filename = subtitle_video_filename
                            
                        # 更新任务完成
                        with threading.Lock():
                            if task.task_id in task_progress:
                                task_progress[task.task_id]["status"] = "completed"
                                task_progress[task.task_id]["progress"] = 100
                                task_progress[task.task_id]["subtitle_status"] = "字幕生成完成"
                                task_progress[task.task_id]["video_url"] = f"/assets/videos/{final_video_filename}"
                                task_progress[task.task_id]["subtitle_url"] = f"/assets/subtitles/{subtitle_filename}"
                                task_progress[task.task_id]["audio_url"] = f"/assets/audio/{audio_filename}"
                                
                    except Exception as e:
                        logging.error(f"生成视频任务失败: {task.task_id}, 错误: {str(e)}")
                        traceback.print_exc()
                        
                        # 更新任务失败状态
                        with threading.Lock():
                            if task.task_id in task_progress:
                                task_progress[task.task_id]["status"] = "error"
                                task_progress[task.task_id]["error"] = f"生成视频失败: {str(e)}"
                        
                elif task.task_type == "generate_ppt_video":
                    logging.info(f"处理生成PPT视频任务: {task.task_id}")
                    # PPT视频生成逻辑将在后续实现
                    
                elif task.task_type == "generate_all_ppt_videos":
                    logging.info(f"处理生成所有PPT视频任务: {task.task_id}")
                    # 批量PPT视频生成逻辑将在后续实现
                    
                else:
                    logging.warning(f"未知任务类型: {task.task_type}")
                    
                # 标记任务完成
                task_queue.task_done()
                
            except queue.Empty:
                # 队列为空，等待新任务
                pass
            except Exception as e:
                logging.error(f"处理任务出错: {str(e)}")
                traceback.print_exc()
    except Exception as e:
        logging.error(f"任务处理线程崩溃: {str(e)}")
    finally:
        is_processing_tasks = False

def assign_gpu_for_task(task_id):
    """为任务分配一个可用的GPU"""
    with threading.Lock():
        # 查找空闲的GPU
        for gpu_id, status in gpu_statuses.items():
            if status["status"] == "idle":
                # 分配GPU
                gpu_statuses[gpu_id]["status"] = "busy"
                gpu_statuses[gpu_id]["task_id"] = task_id
                gpu_statuses[gpu_id]["last_activity"] = time.time()
                return gpu_id
                
    # 没有可用GPU
    return None

def release_gpu(gpu_id):
    """释放GPU资源"""
    with threading.Lock():
        if gpu_id in gpu_statuses:
            gpu_statuses[gpu_id]["status"] = "idle"
            gpu_statuses[gpu_id]["task_id"] = None
            gpu_statuses[gpu_id]["last_activity"] = time.time()
            gpu_statuses[gpu_id]["processed_count"] += 1
            
    return True

def get_system_status():
    """获取系统状态信息"""
    with threading.Lock():
        # 统计任务情况
        pending_count = 0
        processing_count = 0
        completed_count = 0
        error_count = 0
        
        for task_id, progress in task_progress.items():
            if progress["status"] == "pending":
                pending_count += 1
            elif progress["status"] == "processing":
                processing_count += 1
            elif progress["status"] == "completed":
                completed_count += 1
            elif progress["status"] == "error":
                error_count += 1
        
        # 获取GPU状态
        gpu_info = []
        for gpu_id, status in gpu_statuses.items():
            gpu_info.append({
                "gpu_id": gpu_id,
                "status": status["status"],
                "task_id": status["task_id"],
                "last_activity": status["last_activity"],
                "processed_count": status["processed_count"]
            })
            
        return {
            "queue_size": task_queue.qsize(),
            "pending_tasks": pending_count,
            "processing_tasks": processing_count,
            "completed_tasks": completed_count,
            "error_tasks": error_count,
            "gpu_status": gpu_info,
            "is_processing": is_processing_tasks
        } 

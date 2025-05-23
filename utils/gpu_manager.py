import threading
import queue
import time
import traceback
import torch
import os
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import logging

# 导入 process_video_task 函数
from utils.video_generator import process_video_task
# 导入 task_progress 变量
from utils.task_manager import task_progress

# 添加懒加载函数，避免循环导入
def get_avatar_manager():
    """懒加载AvatarManager类，避免循环导入"""
    from scripts.realtime_inference import AvatarManager
    return AvatarManager

logger = logging.getLogger(__name__)

# 全局GPU变量
gpu_queues = []
gpu_statuses = []
gpu_locks = []
gpu_worker_threads = []
active_gpus = 0
avatar_cache = {}


def gpu_worker_process(gpu_id, gpu_queue, gpu_status, gpu_lock):
    """
    GPU工作线程，从队列中获取任务并在特定GPU上执行
    
    Args:
        gpu_id: GPU ID
        gpu_queue: 该GPU对应的任务队列
        gpu_status: 该GPU的状态字典
        gpu_lock: 该GPU的互斥锁
    """
    # 设置GPU环境
    try:
        # 使用torch.cuda.set_device来设置当前线程使用的GPU
        torch.cuda.set_device(gpu_id)
        thread_id = threading.get_ident()
        print("GPU #{} worker thread successfully set CUDA device".format(gpu_id))
        print("GPU #{} worker thread started: GPU-Worker-{}, thread ID: {}".format(gpu_id, gpu_id, thread_id))
    except Exception as e:
        print("GPU #{} failed to set CUDA device: {}".format(gpu_id, e))
    
    # 记录上次活动时间和资源监控
    last_heartbeat = time.time()
    last_resource_check = time.time()
    
    print("GPU #{} worker thread started".format(gpu_id))
    
    # 定义释放资源的辅助函数
    def clean_resources():
        try:
            print("GPU #{} 开始清理资源...".format(gpu_id))
            
            # 尝试从scripts.realtime_inference导入AvatarManager
            try:
                AvatarManager = get_avatar_manager()
                
                # 检查AvatarManager是否有_instances属性
                if not hasattr(AvatarManager, '_instances'):
                    print("AvatarManager没有_instances属性，可能未正确初始化")
                    raise ImportError("AvatarManager未正确初始化")
                    
                # 获取与当前GPU相关的所有实例
                instances_count = 0
                for (avatar_id, instance_gpu_id) in list(AvatarManager._instances.keys()):
                    if instance_gpu_id == gpu_id:
                        try:
                            print("尝试释放数字人 {} 的资源 (GPU: {})".format(avatar_id, gpu_id))
                            AvatarManager.release_instance(avatar_id, gpu_id)
                            instances_count += 1
                        except Exception as e:
                            print("释放数字人 {} 资源时出错: {}".format(avatar_id, e))
                            traceback.print_exc()
                
                if instances_count > 0:
                    print("GPU #{} 成功释放 {} 个数字人实例资源".format(gpu_id, instances_count))
                else:
                    print("GPU #{} 没有需要释放的数字人实例".format(gpu_id))
                    
            except (ImportError, AttributeError) as e:
                print("导入或使用AvatarManager时出错: {}".format(e))
                
                # 如果无法导入AvatarManager，尝试导入main中的current_avatar（向后兼容）
                try:
                    from main import current_avatar
                    
                    # 检查是否是此GPU上的Avatar实例
                    if current_avatar.get("gpu_id") == gpu_id and current_avatar.get("instance") is not None:
                        print("GPU #{} 正在释放current_avatar实例资源".format(gpu_id))
                        try:
                            # 调用显式释放方法
                            if hasattr(current_avatar["instance"], "release_resources"):
                                current_avatar["instance"].release_resources()
                                print("成功调用release_resources方法")
                            else:
                                print("current_avatar实例没有release_resources方法")
                        except Exception as e:
                            print("GPU #{} 在调用release_resources时出错: {}".format(gpu_id, e))
                            traceback.print_exc()
                            
                        # 删除实例引用
                        avatar_id = current_avatar.get("id", "unknown")
                        print("删除数字人 {} 的current_avatar引用".format(avatar_id))
                        current_avatar["id"] = None
                        current_avatar["instance"] = None
                        current_avatar["gpu_id"] = None
                except ImportError as e:
                    print("无法导入main.current_avatar: {}".format(e))
            except Exception as e:
                print("处理AvatarManager时出现未知错误: {}".format(e))
                traceback.print_exc()
                
            # 强制清理GPU缓存
            try:
                torch.cuda.empty_cache()
                memory_stats = {}
                if torch.cuda.is_available():
                    memory_stats["allocated"] = torch.cuda.memory_allocated(gpu_id) / (1024 * 1024)
                    memory_stats["reserved"] = torch.cuda.memory_reserved(gpu_id) / (1024 * 1024)
                    memory_stats["max_allocated"] = torch.cuda.max_memory_allocated(gpu_id) / (1024 * 1024)
                print("GPU #{} 缓存已清理, 当前内存状态: 已分配={:.2f}MB, 预留={:.2f}MB, 峰值={:.2f}MB".format(
                    gpu_id, memory_stats.get("allocated", 0), memory_stats.get("reserved", 0), 
                    memory_stats.get("max_allocated", 0)))
                
                # 重置峰值内存统计
                torch.cuda.reset_peak_memory_stats(gpu_id)
            except Exception as e:
                print("GPU #{} 清理缓存时出错: {}".format(gpu_id, e))
                
            print("GPU #{} 资源清理完成".format(gpu_id))
        except Exception as e:
            print("GPU #{} 清理资源过程中发生错误: {}".format(gpu_id, e))
            traceback.print_exc()
    
    while True:
        try:
            # 记录心跳和检查资源状态
            current_time = time.time()
            
            # 每分钟记录一次心跳
            if current_time - last_heartbeat > 60:
                print("GPU #{} worker thread heartbeat: active, processed tasks: {}".format(
                    gpu_id, gpu_status.get('processed_tasks', 0)))
                last_heartbeat = current_time
            
            # 每5分钟检查一次是否需要释放资源
            if current_time - last_resource_check > 300:
                print("GPU #{} 定时检查资源状态".format(gpu_id))
                # 如果过去5分钟没有活动
                if current_time - gpu_status.get("last_active", 0) > 300:
                    print("GPU #{} 空闲超过5分钟，释放资源".format(gpu_id))
                    clean_resources()
                last_resource_check = current_time
            
            # 更新GPU状态为空闲
            with gpu_lock:
                gpu_status["status"] = "idle"
                gpu_status["task_id"] = None
                gpu_status["task_type"] = None
                gpu_status["started_at"] = None
                gpu_status["last_active"] = time.time()
            
            # 从队列获取任务 (非阻塞，如果队列为空则抛出queue.Empty异常)
            try:
                # 有30秒超时，避免永久阻塞
                task = gpu_queue.get(block=True, timeout=30)
                print("GPU #{} got task from queue".format(gpu_id))
            except queue.Empty:
                # 队列为空，检查是否需要释放资源
                if current_time - gpu_status.get("last_active", 0) > 300:  # 5分钟无活动
                    print("GPU #{} 队列空闲超过5分钟，释放资源".format(gpu_id))
                    clean_resources()
                continue
            
            # 解析任务信息
            task_id = task.get('task_id', 'unknown')
            task_type = task.get('task_type', 'unknown')
            task_data = task.get('data', {})
            
            # 更新GPU状态为忙碌
            with gpu_lock:
                gpu_status["status"] = "busy"
                gpu_status["task_id"] = task_id
                gpu_status["task_type"] = task_type
                gpu_status["started_at"] = time.time()
                gpu_status["last_active"] = time.time()
            
            print("GPU #{} starting to process task {} (type: {})".format(gpu_id, task_id, task_type))
            
            # 根据任务类型处理任务
            try:
                if task_type == "generate_video":
                    # 单个视频生成任务
                    from utils.video_generator import process_video_task
                    success, result = process_video_task(gpu_id, None, None, task_data, task_id)
                    
                    # 确保结果包含 page_index 字段
                    if success and 'page_index' not in result:
                        result['page_index'] = task_data.get('page_index', 0)
                    
                    # 更新任务状态
                    with threading.Lock():
                        if task_id in task_progress:
                            if success:
                                task_progress[task_id]["success_count"] += 1
                            else:
                                task_progress[task_id]["failed_count"] += 1
                            task_progress[task_id]["completed"] += 1
                            
                            # 删除不可序列化的对象
                            serializable_result = result.copy() if isinstance(result, dict) else {"success": success}
                            if isinstance(serializable_result, dict):
                                # 移除Avatar对象和其他不可序列化的对象
                                if 'local_avatar' in serializable_result:
                                    del serializable_result['local_avatar']
                            
                            task_progress[task_id]["results"].append(serializable_result)
                            
                            # 检查是否所有任务都已完成
                            if task_progress[task_id]["completed"] >= task_progress[task_id]["total"]:
                                task_progress[task_id]["status"] = "completed"
                    
                    # 更新任务数据中的结果
                    task_data['result'] = result
                
                elif task_type == "batch_video":
                    from utils.video_generator import process_batch_slide_task
                    process_batch_slide_task(gpu_id, None, None, task_data, task_id)
                else:
                    print("GPU #{} received unknown task type: {}".format(gpu_id, task_type))
            except Exception as e:
                print("GPU #{} error processing task {}: {}".format(gpu_id, task_id, e))
                traceback.print_exc()
                continue
            finally:
                # 任务完成或失败后，更新最后活动时间
                gpu_status["last_active"] = time.time()
            
            # 更新GPU状态
            with gpu_lock:
                gpu_status["status"] = "idle"
                gpu_status["task_id"] = None
                gpu_status["task_type"] = None
                gpu_status["processed_tasks"] = gpu_status.get("processed_tasks", 0) + 1
            
            # 标记任务完成
            gpu_queue.task_done()
            print("GPU #{} completed task {}".format(gpu_id, task_id))
            
        except Exception as e:
            # 发生错误
            print("GPU #{} error processing task: {}".format(gpu_id, e))
            traceback.print_exc()
            
            # 尝试恢复并继续
            try:
                # 更新GPU状态
                with gpu_lock:
                    gpu_status["status"] = "error" 
                    gpu_status["last_active"] = time.time()
                
                # 清理CUDA缓存以防万一
                clean_resources()
                
                # 如果是批量任务，更新任务状态
                if task_type == "batch_video" and task_data:
                    parent_task_id = task_data.get('parent_task_id')
                    page_index = task_data.get('page_index', 0)
                    if parent_task_id:
                        with threading.Lock():
                            if parent_task_id in task_progress:
                                task_progress[parent_task_id]["completed"] += 1
                                task_progress[parent_task_id]["failed_count"] += 1
                                task_progress[parent_task_id]["results"].append({
                                    "page_index": page_index,
                                    "success": False,
                                    "reason": "处理失败: {}".format(str(e))
                                })
            except:
                # 如果恢复过程也出错，只能继续
                pass

def assign_task_to_gpu(task):
    """
    将任务分配给最空闲的GPU
    
    Args:
        task: 任务字典
        
    Returns:
        bool: 是否成功分配
    """
    try:
        # 检查是否有可用的GPU
        if len(gpu_queues) == 0 or active_gpus == 0:
            print("没有可用的GPU，无法分配任务")
            return False
            
        # 找到队列最短的GPU
        best_gpu_id = None
        min_queue_size = float('inf')
        
        for gpu_id in range(active_gpus):
            try:
                # 检查GPU状态
                with gpu_locks[gpu_id]:
                    status = gpu_statuses[gpu_id].get("status", "unknown")
                    last_active = gpu_statuses[gpu_id].get("last_active", 0)
                    
                # 检查GPU是否存活
                gpu_alive = gpu_id < len(gpu_worker_threads) and gpu_worker_threads[gpu_id].is_alive()
                if not gpu_alive:
                    print("GPU #{} worker thread not alive, skipping".format(gpu_id))
                    continue
                
                # 获取队列长度
                queue_size = gpu_queues[gpu_id].qsize()
                
                # 计算空闲时间（秒）
                idle_time = time.time() - last_active
                
                # 优先选择空闲时间更长的GPU
                effective_size = queue_size
                if idle_time > 5 and status != "busy":  # 如果空闲超过5秒且状态不是忙碌
                    effective_size -= 1  # 给空闲GPU更高优先级
                
                if effective_size < min_queue_size:
                    min_queue_size = effective_size
                    best_gpu_id = gpu_id
            except Exception as e:
                print("检查GPU #{}状态时出错: {}".format(gpu_id, e))
                continue
        
        # 如果找不到合适的GPU，返回失败
        if best_gpu_id is None:
            print("找不到可用的GPU来处理任务")
            return False
        
        # 将任务分配给选定的GPU
        print("将任务 {} 分配给 GPU #{}".format(task.get('task_id', 'unknown'), best_gpu_id))
        gpu_queues[best_gpu_id].put(task)
        return True
    except Exception as e:
        print("分配任务到GPU时出错: {}".format(e))
        traceback.print_exc()
        return False

def get_gpu_status():
    """
    获取所有GPU的状态信息
    
    Returns:
        list: GPU状态信息列表
    """
    status_info = []
    
    for idx in range(len(gpu_statuses)):
        try:
            with gpu_locks[idx]:
                status = gpu_statuses[idx].copy()
            
            # 添加队列信息
            if idx < len(gpu_queues):
                status["queue_size"] = gpu_queues[idx].qsize()
                status["queue_empty"] = gpu_queues[idx].empty()
            
            # 添加线程信息
            if idx < len(gpu_worker_threads):
                status["thread_alive"] = gpu_worker_threads[idx].is_alive()
                status["thread_name"] = gpu_worker_threads[idx].name
                status["thread_id"] = gpu_worker_threads[idx].native_id
            
            status_info.append(status)
        except:
            status_info.append({"error": "获取GPU #{}状态失败".format(idx)})
    
    return status_info

def clean_gpu_cache():
    """清理所有GPU的缓存"""
    # 尝试从AvatarManager释放所有实例
    try:
        AvatarManager = get_avatar_manager()
        AvatarManager.release_all_instances()
        print("通过AvatarManager释放所有数字人实例资源")
    except ImportError:
        print("无法导入AvatarManager，将只清理GPU缓存")
    
    # 清理各GPU缓存
    for gpu_id in range(active_gpus):
        try:
            # 设置当前设备
            torch.cuda.set_device(gpu_id)
            # 清理缓存
            torch.cuda.empty_cache()
            print("GPU #{} cache cleared".format(gpu_id))
        except Exception as e:
            print("Failed to clean GPU #{} cache: {}".format(gpu_id, e))

def process_batch_slides(task_id: str, slides: List[Dict[str, Any]], common_params: Dict[str, Any]):
    """
    处理批量幻灯片任务，将任务分配给可用的GPU并行处理
    
    Args:
        task_id (str): 批量任务ID
        slides (list): 幻灯片数据列表
        common_params (dict): 所有幻灯片共享的参数
    """
    total_slides = len(slides)
    print("开始处理批量任务 {}，共 {} 张幻灯片".format(task_id, total_slides))
    
    # 更新任务状态
    with threading.Lock():
        if task_id in task_progress:
            task_progress[task_id]["total"] = total_slides
            task_progress[task_id]["completed"] = 0
            task_progress[task_id]["current_page"] = -1
            task_progress[task_id]["success_count"] = 0
            task_progress[task_id]["failed_count"] = 0
            task_progress[task_id]["status"] = "processing"
            task_progress[task_id]["results"] = []
    
    # 检查活跃GPU数量
    available_gpus = len(gpu_worker_threads)
    if available_gpus == 0:
        print("警告: 没有可用的GPU工作线程，尝试重新初始化多GPU系统")
        gpu_count = initialize_multi_gpu_system()
        if gpu_count > 0:
            available_gpus = gpu_count
            print("成功初始化 {} 个GPU工作线程".format(available_gpus))
        else:
            print("错误: 无法初始化GPU系统，批量任务无法执行")
            with threading.Lock():
                if task_id in task_progress:
                    task_progress[task_id]["status"] = "error"
                    task_progress[task_id]["error"] = "无法初始化GPU系统"
            return
    
    print("可用GPU工作线程数量: {}".format(available_gpus))
    
    # 处理无脚本内容的幻灯片，并创建有脚本内容的幻灯片任务列表
    valid_tasks = []
    for slide in slides:
        page_index = slide.get('page_index', 0)
        text = slide.get('script', '')
        
        # 如果没有脚本内容，直接标记为失败
        if not text or text.strip() == '':
            print("幻灯片 {} 没有脚本，跳过".format(page_index))
            result = {
                "page_index": page_index,
                "success": False,
                "reason": "没有脚本内容"
            }
            
            # 更新任务状态
            with threading.Lock():
                if task_id in task_progress:
                    task_progress[task_id]["failed_count"] += 1
                    task_progress[task_id]["completed"] += 1
                    task_progress[task_id]["results"].append(result)
        else:
            # 创建任务数据
            task_data = {
                'parent_task_id': task_id,
                'slide': slide,
                'page_index': page_index,
                'text': text,
                'common_params': common_params
            }
            
            # 添加到有效任务列表
            valid_tasks.append({
                'task_type': 'batch_video',
                'task_id': "{}_{}".format(task_id, page_index),
                'data': task_data
            })
    
    # 如果没有有效任务，直接完成
    if not valid_tasks:
        print("批量任务 {} 没有有效的幻灯片任务".format(task_id))
        with threading.Lock():
            if task_id in task_progress:
                task_progress[task_id]["status"] = "completed"
        return
    
    # 实现真正的并行处理：按GPU数量均分任务
    print("开始智能分配 {} 个有效任务到 {} 个GPU并行处理".format(len(valid_tasks), available_gpus))
    
    # 1. 根据页码排序任务，确保相邻页面可能分配到不同GPU，提高并行性
    valid_tasks.sort(key=lambda x: x['data']['page_index'])
    
    # 2. 创建每个GPU的任务列表 - 按照轮询方式分配，实现最佳负载均衡
    gpu_task_assignments = [[] for _ in range(available_gpus)]
    
    # 3. 按照"轮询"方式分配任务到各个GPU
    for i, task in enumerate(valid_tasks):
        gpu_id = i % available_gpus  # 轮询分配
        gpu_task_assignments[gpu_id].append(task)
        
    # 4. 显示分配结果
    for gpu_id in range(available_gpus):
        task_count = len(gpu_task_assignments[gpu_id])
        if task_count > 0:
            page_indices = [task['data']['page_index'] for task in gpu_task_assignments[gpu_id]]
            print("GPU #{} assigned {} tasks, pages: {}".format(gpu_id, task_count, page_indices))
    
    # 5. 将所有任务立即提交到各个GPU的队列，让它们并行处理
    assigned_count = 0
    for gpu_id, tasks in enumerate(gpu_task_assignments):
        for task in tasks:
            page_index = task['data']['page_index']
            print("将幻灯片 {} 任务分配给 GPU #{}".format(page_index, gpu_id))
            try:
                # 直接放入对应GPU的队列，不使用assign_task_to_gpu函数
                gpu_queues[gpu_id].put(task)
                assigned_count += 1
                print("成功分配幻灯片 {} 任务到 GPU #{}".format(page_index, gpu_id))
            except Exception as e:
                print("分配幻灯片 {} 任务到 GPU #{} 失败: {}".format(page_index, gpu_id, e))
    
    print("批量任务 {} 已分配 {} / {} 个任务到 {} 个GPU并行处理".format(task_id, assigned_count, len(valid_tasks), available_gpus))
    
    # 所有任务已分配，后台线程会处理并更新进度
    with threading.Lock():
        if task_id in task_progress and task_progress[task_id]["completed"] == total_slides:
            task_progress[task_id]["status"] = "completed"            


# 多GPU处理系统初始化
def initialize_multi_gpu_system():
    """
    初始化多GPU处理系统
    返回可用的GPU数量，如果没有可用GPU则返回0
    """
    global active_gpus, gpu_queues, gpu_statuses, gpu_locks, gpu_worker_threads
    
    # 重置全局变量，避免重复初始化
    gpu_queues = []
    gpu_statuses = []
    gpu_locks = []
    gpu_worker_threads = []
    
    try:
        # 检测可用GPU
        active_gpus = torch.cuda.device_count()
        print("发现 {} 个可用GPU".format(active_gpus))
        
        if active_gpus == 0:
            print("警告: 没有发现可用的GPU，将使用CPU模式")
            return 0
            
        # 初始化GPU状态
        for gpu_id in range(active_gpus):
            gpu_name = torch.cuda.get_device_name(gpu_id)
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            print("GPU #{}: {}, memory: {:.2f} GB".format(gpu_id, gpu_name, total_memory/1024/1024/1024))
            
            # 创建队列和状态
            gpu_queues.append(queue.Queue())
            gpu_statuses.append({
                "status": "idle",
                "task_id": None,
                "task_type": None,
                "started_at": None,
                "processed_tasks": 0,
                "last_active": time.time(),
                "device_name": gpu_name,
                "total_memory": total_memory
            })
            gpu_locks.append(threading.Lock())
        
        # 启动GPU工作线程
        for gpu_id in range(active_gpus):
            thread = threading.Thread(
                target=gpu_worker_process,
                args=(gpu_id, gpu_queues[gpu_id], gpu_statuses[gpu_id], gpu_locks[gpu_id]),
                daemon=True
            )
            thread.name = "GPU-Worker-{}".format(gpu_id)
            thread.start()
            gpu_worker_threads.append(thread)
            print("GPU #{} worker thread started: {}, thread ID: {}".format(gpu_id, thread.name, thread.native_id))
            
            # 确认线程已启动
            time.sleep(0.5)
            if not thread.is_alive():
                print("Error: GPU #{} worker thread failed to start".format(gpu_id))
                return 0
        
        print("成功初始化多GPU系统: {} 个GPU工作线程就绪".format(active_gpus))
        return active_gpus
    except Exception as e:
        print("初始化多GPU系统出错: {}".format(e))
        traceback.print_exc()
        return 0

class GPUManager:
    def __init__(self):
        self.gpu_states = {}
        self.task_queues = {}
        self.processing_threads = {}
        self.stop_flags = {}
        self.avatar_cache = {}  # 每个GPU的Avatar模型缓存
        
    def process_task_on_gpu(self, gpu_id: int):
        """在指定GPU上处理任务"""
        local_avatar = None  # 当前加载的Avatar模型
        current_avatar_id = None  # 当前Avatar模型的ID
        
        while not self.stop_flags.get(gpu_id, False):
            try:
                if gpu_id not in self.task_queues:
                    time.sleep(1)
                    continue
                    
                task = self.task_queues[gpu_id].get(timeout=1)
                if task is None:
                    continue
                    
                task_id = task.get('task_id')
                task_type = task.get('task_type', 'video')
                task_data = task.get('task_data', {})
                
                print("GPU #{} 开始处理任务 {}".format(gpu_id, task_id))
                
                try:
                    if task_type == 'video':
                        from utils.video_generator import process_video_task
                        success, result = process_video_task(
                            gpu_id=gpu_id,
                            local_avatar=local_avatar,
                            current_avatar_id=current_avatar_id,
                            task_data=task_data,
                            task_id=task_id
                        )
                        # 更新当前Avatar模型ID
                        if success:
                            current_avatar_id = result.get('avatar_id')
                            
                    elif task_type == 'batch_slide':
                        from utils.video_generator import process_batch_slide_task
                        success, result = process_batch_slide_task(
                            gpu_id=gpu_id,
                            local_avatar=local_avatar,
                            current_avatar_id=current_avatar_id,
                            task_data=task_data,
                            task_id=task_id
                        )
                        # 更新当前Avatar模型ID
                        if success:
                            current_avatar_id = result.get('avatar_id')
                    else:
                        print("GPU #{} 未知任务类型: {}".format(gpu_id, task_type))
                        continue
                        
                    # 更新任务状态
                    if task_id in task_progress:
                        with threading.Lock():
                            if success:
                                task_progress[task_id]["success_count"] += 1
                            else:
                                task_progress[task_id]["failed_count"] += 1
                            task_progress[task_id]["completed"] += 1
                            
                            # 删除不可序列化的对象
                            serializable_result = result.copy() if isinstance(result, dict) else {"success": success}
                            if isinstance(serializable_result, dict):
                                # 移除Avatar对象和其他不可序列化的对象
                                if 'local_avatar' in serializable_result:
                                    del serializable_result['local_avatar']
                            
                            task_progress[task_id]["results"].append(serializable_result)
                            
                            # 检查是否所有任务都已完成
                            if task_progress[task_id]["completed"] >= task_progress[task_id]["total"]:
                                task_progress[task_id]["status"] = "completed"
                                
                except Exception as e:
                    print("GPU #{} 处理任务 {} 时出错: {}".format(gpu_id, task_id, e))
                    traceback.print_exc()
                    
                    # 更新任务状态
                    if task_id in task_progress:
                        with threading.Lock():
                            task_progress[task_id]["failed_count"] += 1
                            task_progress[task_id]["completed"] += 1
                            task_progress[task_id]["results"].append({
                                "success": False,
                                "error": str(e)
                            })
                            
                            # 检查是否所有任务都已完成
                            if task_progress[task_id]["completed"] >= task_progress[task_id]["total"]:
                                task_progress[task_id]["status"] = "completed"
                                
            except queue.Empty:
                continue
            except Exception as e:
                print("GPU #{} 处理队列时出错: {}".format(gpu_id, e))
                traceback.print_exc()
                time.sleep(1)
                
        print("GPU #{} 处理线程退出".format(gpu_id))
        
    def cleanup_gpu(self, gpu_id: int):
        """清理GPU资源"""
        try:
            # 清理Avatar缓存
            if gpu_id in self.avatar_cache:
                self.avatar_cache[gpu_id].clear()
            # 清理CUDA缓存
            import torch
            torch.cuda.empty_cache()
        except Exception as e:
            print("清理GPU #{} 资源时出错: {}".format(gpu_id, e))
            
    def stop_gpu(self, gpu_id: int):
        """停止指定GPU的处理"""
        self.stop_flags[gpu_id] = True
        if gpu_id in self.processing_threads:
            self.processing_threads[gpu_id].join()
            del self.processing_threads[gpu_id]
        self.cleanup_gpu(gpu_id)
        
    def stop_all(self):
        """停止所有GPU的处理"""
        for gpu_id in list(self.processing_threads.keys()):
            self.stop_gpu(gpu_id)
            
# 全局GPU管理器实例
gpu_manager = GPUManager()

class GPUMemoryManager:
    def __init__(self, threshold=0.8, check_interval=30):
        """
        GPU内存管理器
        
        Args:
            threshold: 内存使用阈值，超过此值将触发警告
            check_interval: 检查间隔(秒)
        """
        self.threshold = threshold
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """启动GPU内存监控"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                self.check_memory()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error("内存监控出错: " + str(e))
                
    def check_memory(self):
        """检查当前GPU内存使用情况"""
        if not torch.cuda.is_available():
            return
            
        # 检查所有可用的GPU
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_reserved = torch.cuda.memory_reserved(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory
            
            usage_ratio = memory_allocated / memory_total
            reserved_ratio = memory_reserved / memory_total
            
            logger.info("GPU #{}: 使用率 {:.2%}, 预留率 {:.2%}".format(i, usage_ratio, reserved_ratio))
            
            if usage_ratio > self.threshold:
                logger.warning("GPU #{} 内存使用率高: {:.2%}".format(i, usage_ratio))
                
    def get_available_gpu(self):
        """获取内存使用率最低的GPU ID"""
        if not torch.cuda.is_available():
            return None
            
        if torch.cuda.device_count() == 1:
            return 0
            
        min_usage = float('inf')
        selected_gpu = 0
        
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory
            usage_ratio = memory_allocated / memory_total
            
            if usage_ratio < min_usage:
                min_usage = usage_ratio
                selected_gpu = i
                
        logger.info("选择GPU #{}, 当前使用率: {:.2%}".format(selected_gpu, min_usage))
        return selected_gpu

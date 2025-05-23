import os
import time
import uuid
import threading
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union

# Global task tracking dictionary
task_progress = {}
task_lock = threading.Lock()

def initialize_task_manager():
    """初始化任务管理器"""
    global task_progress
    with task_lock:
        task_progress.clear()
    return True

def create_task(
    task_id: Optional[str] = None,
    task_type: str = 'video',
    task_data: Optional[Dict[str, Any]] = None,
    total_items: int = 1
) -> Dict[str, Any]:
    """创建新任务"""
    if not task_id:
        task_id = str(uuid.uuid4())
        
    with task_lock:
        task_progress[task_id] = {
            "task_id": task_id,
            "type": task_type,
            "status": "pending",
            "completed": 0,
            "total": total_items,
            "success_count": 0,
            "failed_count": 0,
            "results": [],
            "start_time": time.time(),
            "data": task_data or {}
        }
        
    return {
        "task_id": task_id,
        "task_type": task_type,
        "data": task_data
    }

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """获取任务状态"""
    with task_lock:
        if task_id not in task_progress:
            return None
        return dict(task_progress[task_id])

def get_task_result(task_id: str) -> Optional[Dict[str, Any]]:
    """获取任务结果"""
    with task_lock:
        if task_id not in task_progress:
            return None
        task = task_progress[task_id]
        if task["status"] != "completed":
            return None
        return {
            "success": task["failed_count"] == 0,
            "results": task["results"]
        }

def update_task_progress(
    task_id: str,
    status: Optional[str] = None,
    completed: Optional[int] = None,
    success: Optional[bool] = None,
    result: Optional[Dict[str, Any]] = None
):
    """更新任务进度"""
    with task_lock:
        if task_id not in task_progress:
            return False
            
        task = task_progress[task_id]
        
        if status:
            task["status"] = status
            
        if completed is not None:
            task["completed"] = completed
            if completed >= task["total"]:
                task["status"] = "completed"
                
        if success is not None:
            if success:
                task["success_count"] += 1
            else:
                task["failed_count"] += 1
                
        if result:
            task["results"].append(result)
            
        return True

def clean_completed_tasks(max_age: int = 3600):
    """清理已完成的任务"""
    current_time = time.time()
    with task_lock:
        for task_id in list(task_progress.keys()):
            task = task_progress[task_id]
            if task["status"] == "completed":
                if current_time - task["start_time"] > max_age:
                    del task_progress[task_id]

def get_active_tasks() -> List[str]:
    """
    Get a list of active task IDs.
    
    Returns:
        list: List of active task IDs
    """
    with threading.Lock():
        return [task_id for task_id, task in task_progress.items() if task["status"] == "processing"]

def get_all_tasks_status() -> Dict[str, Dict[str, Any]]:
    """
    Get the status of all tasks.
    
    Returns:
        dict: Dictionary mapping task IDs to task status information
    """
    with threading.Lock():
        return {task_id: dict(task) for task_id, task in task_progress.items()}

def mark_task_error(task_id: str, error_message: str) -> bool:
    """
    Mark a task as errored.
    
    Args:
        task_id (str): Task ID
        error_message (str): Error message
        
    Returns:
        bool: Whether the update was successful
    """
    with threading.Lock():
        if task_id in task_progress:
            task_progress[task_id]["status"] = "error"
            task_progress[task_id]["error"] = error_message
            task_progress[task_id]["updated_at"] = time.time()
            return True
        return False

import os
import sys
import json
import time
import uuid
import logging
import threading
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# 添加上级目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.avatar_generator import AvatarGenerator
from core.dependency_manager import DependencyManager
from api.socket_manager import SocketManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("API")

# 创建应用
app = FastAPI(
    title="MuseTalk API",
    description="数字人视频生成API，支持多级降级能力",
    version="1.0.0"
)

# WebSocket管理器
socket_manager = SocketManager()

# 任务管理
tasks = {}
task_results = {}

# 请求模型
class GenerationRequest(BaseModel):
    """视频生成请求"""
    text: str = Field(..., description="要转换的文本")
    avatar_id: str = Field(..., description="数字人ID")
    audio_path: Optional[str] = Field(None, description="音频文件路径，如果提供则直接使用该音频")
    subtitle: bool = Field(False, description="是否添加字幕")
    subtitle_color: str = Field("#FFFFFF", description="字幕颜色")
    prefer_mode: Optional[str] = Field(None, description="优先使用的模式: advanced, standard, basic")
    output_path: Optional[str] = Field(None, description="输出视频路径，不提供则自动生成")
    background_music: Optional[str] = Field(None, description="背景音乐路径")
    volume_ratio: float = Field(1.0, description="语音与背景音乐音量比例")

class StatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态: pending, processing, completed, failed")
    progress: float = Field(0.0, description="进度百分比")
    result: Optional[Dict[str, Any]] = Field(None, description="任务结果")
    message: Optional[str] = Field(None, description="状态消息")
    created_at: float = Field(..., description="创建时间戳")
    updated_at: float = Field(..., description="更新时间戳")

class AvatarInfo(BaseModel):
    """数字人信息"""
    id: str = Field(..., description="数字人ID")
    name: str = Field(..., description="数字人名称")
    preview_url: str = Field(..., description="预览图URL")
    description: Optional[str] = Field(None, description="描述")

# 初始化AvatarGenerator (延迟初始化)
avatar_generator = None
dependency_manager = DependencyManager()

def get_avatar_generator():
    """获取或初始化AvatarGenerator实例"""
    global avatar_generator
    if avatar_generator is None:
        avatar_generator = AvatarGenerator()
    return avatar_generator

def update_task_status(task_id: str, status: str, progress: float = 0.0, message: Optional[str] = None, result: Optional[Dict[str, Any]] = None):
    """更新任务状态并通知WebSocket客户端"""
    if task_id not in tasks:
        tasks[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0.0,
            "result": None,
            "message": "任务初始化中",
            "created_at": time.time(),
            "updated_at": time.time()
        }
    
    tasks[task_id]["status"] = status
    tasks[task_id]["progress"] = progress
    tasks[task_id]["updated_at"] = time.time()
    
    if message:
        tasks[task_id]["message"] = message
    
    if result:
        tasks[task_id]["result"] = result
        task_results[task_id] = result
    
    # 通知WebSocket客户端
    socket_manager.broadcast_status_update(tasks[task_id])

def process_video_generation(task_id: str, request: GenerationRequest):
    """后台处理视频生成任务"""
    try:
        # 更新任务状态为处理中
        update_task_status(
            task_id=task_id,
            status="processing",
            progress=10.0,
            message="正在初始化视频生成器"
        )
        
        # 获取生成器
        generator = get_avatar_generator()
        
        # 设置输出路径
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = request.output_path
        if not output_path:
            # 生成唯一的输出路径
            timestamp = int(time.time())
            output_filename = f"{request.avatar_id}_{timestamp}.mp4"
            output_path = os.path.join(output_dir, output_filename)
        
        # 更新任务状态
        update_task_status(
            task_id=task_id,
            status="processing",
            progress=20.0,
            message="正在生成视频"
        )
        
        # 准备选项
        options = {
            "subtitle": request.subtitle,
            "subtitle_color": request.subtitle_color,
            "background_music": request.background_music,
            "volume_ratio": request.volume_ratio
        }
        
        # 处理音频路径（如果提供）
        if request.audio_path:
            options["audio_path"] = request.audio_path
        
        # 处理模式优先级
        prefer_mode = request.prefer_mode
        if prefer_mode:
            options["prefer_mode"] = prefer_mode
        
        # 生成视频
        result = generator.generate_video(
            text=request.text,
            avatar_id=request.avatar_id,
            output_path=output_path,
            **options
        )
        
        # 更新任务状态
        if result.get("success", False):
            # 成功
            update_task_status(
                task_id=task_id,
                status="completed",
                progress=100.0,
                message="视频生成完成",
                result=result
            )
        else:
            # 失败
            update_task_status(
                task_id=task_id,
                status="failed",
                progress=100.0,
                message=f"视频生成失败: {result.get('error', '未知错误')}",
                result=result
            )
    except Exception as e:
        logger.error(f"视频生成任务异常: {str(e)}")
        # 更新任务状态为失败
        update_task_status(
            task_id=task_id,
            status="failed",
            progress=100.0,
            message=f"视频生成异常: {str(e)}",
            result={"success": False, "error": str(e), "mode": "error"}
        )

@app.post("/api/generate", response_model=StatusResponse)
async def generate_video(request: GenerationRequest, background_tasks: BackgroundTasks):
    """
    生成数字人视频
    
    根据提供的文本和数字人ID，生成一个数字人视频。
    会自动使用最适合的生成模式，并在需要时进行降级。
    
    Returns:
        任务状态信息，包含任务ID用于后续查询
    """
    # 生成唯一任务ID
    task_id = str(uuid.uuid4())
    
    # 初始化任务状态
    update_task_status(
        task_id=task_id,
        status="pending",
        progress=0.0,
        message="任务已创建，等待处理"
    )
    
    # 添加到后台任务
    background_tasks.add_task(process_video_generation, task_id, request)
    
    # 返回任务状态
    return tasks[task_id]

@app.get("/api/check_status/{task_id}", response_model=StatusResponse)
async def check_status(task_id: str):
    """
    查询任务状态
    
    根据任务ID查询视频生成任务的状态
    
    Args:
        task_id: 任务ID
        
    Returns:
        任务状态信息
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return tasks[task_id]

@app.get("/api/avatars", response_model=List[AvatarInfo])
async def get_avatars():
    """
    获取可用数字人列表
    
    Returns:
        可用数字人信息列表
    """
    # 这里应该从实际存储的位置获取数字人列表
    # 示例实现
    avatar_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "assets", "avatars")
    
    if not os.path.exists(avatar_dir):
        return []
    
    avatars = []
    for item in os.listdir(avatar_dir):
        if os.path.isfile(os.path.join(avatar_dir, item)) and item.lower().endswith(('.png', '.jpg', '.jpeg')):
            avatar_id = os.path.splitext(item)[0]
            avatar_info = {
                "id": avatar_id,
                "name": avatar_id.title(),
                "preview_url": f"/static/avatars/{item}",
                "description": f"{avatar_id.title()} 数字人"
            }
            avatars.append(AvatarInfo(**avatar_info))
    
    return avatars

@app.get("/api/videos/{video_filename}")
async def get_video(video_filename: str):
    """
    获取生成的视频文件
    
    Args:
        video_filename: 视频文件名
        
    Returns:
        视频文件
    """
    video_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", video_filename)
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="视频文件不存在")
    
    return FileResponse(video_path)

@app.get("/api/check_environment")
async def check_environment():
    """
    检查系统环境
    
    检查当前系统环境，返回各种依赖和功能的可用状态
    
    Returns:
        环境检查结果
    """
    generator = get_avatar_generator()
    
    return {
        "capability_level": generator.capability_level.name,
        "environment_status": generator.env_status,
        "cuda_available": generator.env_status["cuda_available"]
    }

@app.get("/api/installation_guide/{level}")
async def get_installation_guide(level: str):
    """
    获取安装指南
    
    根据指定级别获取依赖安装指南
    
    Args:
        level: 功能级别 (basic/standard/advanced)
        
    Returns:
        安装指南
    """
    if level not in ["basic", "standard", "advanced"]:
        raise HTTPException(status_code=400, detail="无效的功能级别")
    
    guide = dependency_manager.get_installation_guide(level)
    return {"guide": guide}

@app.websocket("/ws/status")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket状态更新接口
    """
    await socket_manager.connect(websocket)
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            
            # 解析消息（可以实现更复杂的协议）
            try:
                message = json.loads(data)
                # 处理订阅请求
                if message.get("action") == "subscribe" and "task_id" in message:
                    task_id = message["task_id"]
                    # 立即发送最新状态
                    if task_id in tasks:
                        await websocket.send_json(tasks[task_id])
            except Exception as e:
                logger.error(f"处理WebSocket消息时出错: {str(e)}")
    except WebSocketDisconnect:
        socket_manager.disconnect(websocket)

# 设置静态文件服务
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 预初始化AvatarGenerator
    threading.Thread(target=get_avatar_generator).start()
    
    # 生成依赖配置文件
    dependency_manager.generate_conda_env_file("basic")
    dependency_manager.generate_conda_env_file("standard")
    dependency_manager.generate_conda_env_file("advanced")
    dependency_manager.generate_requirements_files()

# 挂载静态文件
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
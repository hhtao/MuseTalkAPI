import json
import logging
import asyncio
from typing import Dict, Set, Any, List
from fastapi import WebSocket

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SocketManager")

class SocketManager:
    """WebSocket连接管理器，用于实时进度通知"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.task_subscriptions: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket):
        """
        处理新的WebSocket连接
        
        Args:
            websocket: WebSocket连接对象
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket连接已建立，当前活跃连接数: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """
        处理WebSocket断开连接
        
        Args:
            websocket: WebSocket连接对象
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # 清理订阅记录
        for task_id, subscribers in list(self.task_subscriptions.items()):
            if websocket in subscribers:
                subscribers.remove(websocket)
                # 如果没有订阅者，清理该任务的订阅记录
                if not subscribers:
                    del self.task_subscriptions[task_id]
        
        logger.info(f"WebSocket连接已断开，当前活跃连接数: {len(self.active_connections)}")
    
    def subscribe_to_task(self, websocket: WebSocket, task_id: str):
        """
        订阅任务状态更新
        
        Args:
            websocket: WebSocket连接对象
            task_id: 任务ID
        """
        if task_id not in self.task_subscriptions:
            self.task_subscriptions[task_id] = set()
        
        self.task_subscriptions[task_id].add(websocket)
        logger.info(f"WebSocket客户端订阅了任务 {task_id}，当前订阅数: {len(self.task_subscriptions[task_id])}")
    
    def unsubscribe_from_task(self, websocket: WebSocket, task_id: str):
        """
        取消订阅任务状态更新
        
        Args:
            websocket: WebSocket连接对象
            task_id: 任务ID
        """
        if task_id in self.task_subscriptions and websocket in self.task_subscriptions[task_id]:
            self.task_subscriptions[task_id].remove(websocket)
            
            # 如果没有订阅者，清理该任务的订阅记录
            if not self.task_subscriptions[task_id]:
                del self.task_subscriptions[task_id]
            
            logger.info(f"WebSocket客户端取消订阅了任务 {task_id}")
    
    async def broadcast_status_update(self, status_data: Dict[str, Any]):
        """
        广播任务状态更新
        
        Args:
            status_data: 状态数据
        """
        task_id = status_data.get("task_id")
        
        if not task_id:
            return
        
        # 发送给所有订阅该任务的客户端
        if task_id in self.task_subscriptions:
            subscribers = self.task_subscriptions[task_id].copy()
            
            for websocket in subscribers:
                try:
                    await websocket.send_json(status_data)
                except Exception as e:
                    logger.error(f"发送状态更新失败: {str(e)}")
                    # 移除失败的连接
                    self.disconnect(websocket)
        
        # 也发送给所有活跃连接（可选，取决于业务需求）
        # 对于大量连接的情况，可以只发送给订阅者
        for websocket in self.active_connections:
            try:
                # 检查是否已经发送过
                if task_id in self.task_subscriptions and websocket in self.task_subscriptions[task_id]:
                    continue
                
                await websocket.send_json(status_data)
            except Exception as e:
                logger.warning(f"向未订阅客户端发送状态更新失败: {str(e)}")
                # 不在此处移除连接，避免修改迭代中的列表
    
    async def send_message_to_client(self, websocket: WebSocket, message: Dict[str, Any]):
        """
        向特定客户端发送消息
        
        Args:
            websocket: WebSocket连接对象
            message: 消息数据
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"向客户端发送消息失败: {str(e)}")
            # 移除失败的连接
            self.disconnect(websocket)
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """
        向所有连接的客户端广播消息
        
        Args:
            message: 消息数据
        """
        disconnected = []
        
        for websocket in self.active_connections:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"广播消息失败: {str(e)}")
                disconnected.append(websocket)
        
        # 移除断开的连接
        for websocket in disconnected:
            self.disconnect(websocket) 
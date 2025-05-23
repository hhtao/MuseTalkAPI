import asyncio
import enum
import logging
from typing import Callable, Dict, List, Optional, Any
import time
import json
import os

class DigitalHumanState(enum.Enum):
    IDLE_LISTENING = "idle_listening"
    ACTIVE_LISTENING = "active_listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"

# 定义状态转换有效性检查字典
VALID_TRANSITIONS = {
    DigitalHumanState.IDLE_LISTENING: [
        DigitalHumanState.ACTIVE_LISTENING
    ],
    DigitalHumanState.ACTIVE_LISTENING: [
        DigitalHumanState.THINKING,
        DigitalHumanState.IDLE_LISTENING  # 超时或取消
    ],
    DigitalHumanState.THINKING: [
        DigitalHumanState.SPEAKING,
        DigitalHumanState.IDLE_LISTENING  # 超时或错误
    ],
    DigitalHumanState.SPEAKING: [
        DigitalHumanState.INTERRUPTED,
        DigitalHumanState.IDLE_LISTENING  # 说话结束
    ],
    DigitalHumanState.INTERRUPTED: [
        DigitalHumanState.SPEAKING,       # 确认性打断后恢复
        DigitalHumanState.ACTIVE_LISTENING,  # 纠正性打断
        DigitalHumanState.IDLE_LISTENING  # 结束对话
    ]
}

class StateManager:
    """数字人状态管理器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化状态管理器
        
        Args:
            config_path: 配置文件路径，用于加载状态超时设置等
        """
        self._current_state = DigitalHumanState.IDLE_LISTENING
        self._previous_state = None
        self._state_context = {}
        self._state_callbacks = {state: [] for state in DigitalHumanState}
        self._state_start_time = time.time()
        self._timeout_tasks = {}
        
        # 状态超时配置（单位：秒）
        self._state_timeouts = {
            DigitalHumanState.IDLE_LISTENING: None,  # 无超时
            DigitalHumanState.ACTIVE_LISTENING: 10,  # 10秒无输入超时
            DigitalHumanState.THINKING: 15,          # 15秒思考超时
            DigitalHumanState.SPEAKING: None,        # 无超时
            DigitalHumanState.INTERRUPTED: 5         # 5秒打断超时
        }
        
        # 从配置文件加载自定义设置
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
            
        # 状态持久化路径
        self._state_persist_path = "state_data.json"
        
        logging.info("StateManager initialized with state: %s", self._current_state)
    
    def _load_config(self, config_path: str) -> None:
        """从配置文件加载状态超时设置"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if 'state_timeouts' in config:
                for state_name, timeout in config['state_timeouts'].items():
                    try:
                        state = DigitalHumanState(state_name)
                        self._state_timeouts[state] = timeout
                    except ValueError:
                        logging.warning(f"Unknown state in config: {state_name}")
            
            if 'state_persist_path' in config:
                self._state_persist_path = config['state_persist_path']
                
            logging.info("Loaded state configuration from %s", config_path)
        except Exception as e:
            logging.error("Failed to load config: %s", str(e))
    
    async def set_state(self, new_state: DigitalHumanState) -> bool:
        """
        设置当前状态并触发相关事件
        
        Args:
            new_state: 新状态
            
        Returns:
            bool: 状态转换是否成功
        """
        # 检查状态转换是否有效
        if new_state not in VALID_TRANSITIONS.get(self._current_state, []):
            logging.warning(
                f"Invalid state transition: {self._current_state} -> {new_state}"
            )
            return False
        
        # 取消当前状态的超时任务
        if self._current_state in self._timeout_tasks and self._timeout_tasks[self._current_state]:
            self._timeout_tasks[self._current_state].cancel()
            self._timeout_tasks[self._current_state] = None
            
        # 记录先前状态
        self._previous_state = self._current_state
        
        # 更新状态
        self._current_state = new_state
        self._state_start_time = time.time()
        
        logging.info(f"State changed: {self._previous_state} -> {new_state}")
        
        # 触发回调
        for callback in self._state_callbacks[new_state]:
            try:
                await callback(self._previous_state, new_state)
            except Exception as e:
                logging.error(f"Error in state callback: {e}")
        
        # 设置新状态的超时任务
        timeout = self._state_timeouts.get(new_state)
        if timeout:
            self._timeout_tasks[new_state] = asyncio.create_task(
                self._handle_state_timeout(new_state, timeout)
            )
        
        # 保存状态上下文
        await self._persist_state()
        
        return True
    
    async def _handle_state_timeout(self, state: DigitalHumanState, timeout: int) -> None:
        """处理状态超时"""
        await asyncio.sleep(timeout)
        
        # 如果当前状态仍然是超时的状态，则转换到IDLE_LISTENING
        if self._current_state == state:
            logging.info(f"State {state} timed out after {timeout}s, returning to IDLE_LISTENING")
            await self.set_state(DigitalHumanState.IDLE_LISTENING)
    
    async def register_state_change_callback(self, state: DigitalHumanState, callback: Callable) -> None:
        """
        注册状态变化回调函数
        
        Args:
            state: 要监听的状态
            callback: 回调函数，接收参数 (previous_state, new_state)
        """
        self._state_callbacks[state].append(callback)
        logging.info(f"Registered callback for state: {state}")
    
    async def get_current_state(self) -> DigitalHumanState:
        """获取当前状态"""
        return self._current_state
    
    async def get_previous_state(self) -> Optional[DigitalHumanState]:
        """获取前一个状态"""
        return self._previous_state
    
    async def save_state_context(self, context_data: dict) -> None:
        """
        保存状态上下文信息
        
        Args:
            context_data: 上下文数据字典
        """
        self._state_context.update(context_data)
        await self._persist_state()
        logging.debug(f"Updated state context: {context_data.keys()}")
    
    async def get_state_context(self) -> dict:
        """获取状态上下文信息"""
        return self._state_context.copy()
    
    async def get_state_duration(self) -> float:
        """获取当前状态持续时间（秒）"""
        return time.time() - self._state_start_time
    
    async def _persist_state(self) -> None:
        """持久化状态信息到文件"""
        try:
            state_data = {
                "current_state": self._current_state.value,
                "previous_state": self._previous_state.value if self._previous_state else None,
                "context": self._state_context,
                "timestamp": time.time()
            }
            
            with open(self._state_persist_path, 'w') as f:
                json.dump(state_data, f)
                
            logging.debug("State persisted to %s", self._state_persist_path)
        except Exception as e:
            logging.error("Failed to persist state: %s", str(e))
    
    async def load_persisted_state(self) -> bool:
        """
        从文件加载持久化的状态信息
        
        Returns:
            bool: 是否成功加载
        """
        if not os.path.exists(self._state_persist_path):
            logging.warning("No persisted state found at %s", self._state_persist_path)
            return False
            
        try:
            with open(self._state_persist_path, 'r') as f:
                state_data = json.load(f)
            
            # 恢复状态
            try:
                self._current_state = DigitalHumanState(state_data["current_state"])
                self._previous_state = DigitalHumanState(state_data["previous_state"]) if state_data["previous_state"] else None
                self._state_context = state_data["context"]
                self._state_start_time = state_data["timestamp"]
                
                logging.info("Loaded persisted state: %s", self._current_state)
                return True
            except (ValueError, KeyError) as e:
                logging.error("Invalid state data: %s", str(e))
                return False
                
        except Exception as e:
            logging.error("Failed to load persisted state: %s", str(e))
            return False
    
    async def reset_state(self) -> None:
        """重置到初始状态（IDLE_LISTENING）"""
        # 取消所有超时任务
        for state, task in self._timeout_tasks.items():
            if task:
                task.cancel()
        
        self._timeout_tasks = {}
        self._current_state = DigitalHumanState.IDLE_LISTENING
        self._previous_state = None
        self._state_context = {}
        self._state_start_time = time.time()
        
        # 持久化重置后的状态
        await self._persist_state()
        
        logging.info("State reset to IDLE_LISTENING") 
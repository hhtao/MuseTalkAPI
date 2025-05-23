import asyncio
import aiohttp
import logging
import json
import os
import time
import base64
from typing import List, Dict, Any, Optional, Tuple, Union

class ExternalServiceClient:
    """外部服务客户端基类"""
    
    def __init__(self, base_url: str):
        """
        初始化外部服务客户端
        
        Args:
            base_url: 服务基础URL
        """
        self.base_url = base_url
        self.session = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_retries = 3
        self.retry_delay = 1.0  # 重试延迟(秒)
        
    async def _ensure_session(self) -> None:
        """确保aiohttp会话已创建"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
    async def _request(self, 
                      method: str, 
                      endpoint: str, 
                      data: Any = None, 
                      params: Dict = None, 
                      headers: Dict = None,
                      timeout: float = 30.0) -> Tuple[bool, Any]:
        """
        发送HTTP请求
        
        Args:
            method: 请求方法(GET, POST等)
            endpoint: 接口端点
            data: 请求数据
            params: 查询参数
            headers: 请求头
            timeout: 超时时间(秒)
            
        Returns:
            Tuple[bool, Any]: (是否成功, 响应数据)
        """
        await self._ensure_session()
        
        url = f"{self.base_url}/{endpoint}"
        headers = headers or {}
        
        for retry in range(self.max_retries):
            try:
                async with self.session.request(
                    method=method,
                    url=url,
                    json=data if method.upper() != "GET" else None,
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    # 检查HTTP状态码
                    if response.status >= 400:
                        error_text = await response.text()
                        self.logger.error(f"HTTP错误 {response.status}: {error_text}")
                        if retry < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay * (retry + 1))
                            continue
                        return False, {"error": f"HTTP {response.status}", "message": error_text}
                    
                    # 尝试解析JSON响应
                    try:
                        result = await response.json()
                        return True, result
                    except json.JSONDecodeError:
                        # 不是JSON格式，返回原始文本或二进制数据
                        if response.content_type == 'application/octet-stream':
                            return True, await response.read()
                        else:
                            return True, await response.text()
            
            except asyncio.TimeoutError:
                self.logger.warning(f"请求超时 {url} (重试 {retry+1}/{self.max_retries})")
                if retry < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (retry + 1))
                    
            except aiohttp.ClientError as e:
                self.logger.error(f"请求错误 {url}: {str(e)} (重试 {retry+1}/{self.max_retries})")
                if retry < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (retry + 1))
                    
            except Exception as e:
                self.logger.error(f"未知错误 {url}: {str(e)}")
                return False, {"error": "未知错误", "message": str(e)}
                
        return False, {"error": "最大重试次数已用完", "message": "服务不可用"}
        
    async def is_available(self) -> bool:
        """
        检查服务是否可用
        
        Returns:
            bool: 服务是否可用
        """
        try:
            await self._ensure_session()
            # 子类应该重写此方法实现具体的可用性检查
            return True
        except Exception as e:
            self.logger.error(f"服务可用性检查失败: {str(e)}")
            return False
            
    async def close(self) -> None:
        """关闭连接并释放资源"""
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.info("客户端会话已关闭")


class OllamaClient(ExternalServiceClient):
    """Ollama API客户端"""
    
    def __init__(self, base_url: str = "http://122.204.161.127:11434/api", model: str = "qwen3:32b"):
        """
        初始化Ollama客户端
        
        Args:
            base_url: API基础URL
            model: 使用的模型名称
        """
        super().__init__(base_url)
        self.model = model
        self.logger = logging.getLogger("OllamaClient")
        self.logger.info(f"Ollama客户端初始化: {base_url}, 模型: {model}")
        
    async def is_available(self) -> bool:
        """
        检查Ollama服务是否可用
        
        Returns:
            bool: 服务是否可用
        """
        success, _ = await self._request("GET", "models")
        return success
        
    async def generate_response(self, prompt: str, context: List[int] = None) -> Tuple[str, List[int]]:
        """
        生成文本回复
        
        Args:
            prompt: 提示词
            context: 上下文
            
        Returns:
            Tuple[str, List[int]]: (回复文本, 新上下文)
        """
        # 构建请求数据
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # 不使用流式响应
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        # 添加上下文(如果有)
        if context:
            request_data["context"] = context
            
        self.logger.info(f"发送请求到Ollama: {len(prompt)} 字符")
        
        # 发送请求
        success, response = await self._request("POST", "generate", data=request_data, timeout=60.0)
        
        if not success:
            self.logger.error(f"Ollama请求失败: {response}")
            return "抱歉，我暂时无法回答这个问题。", context or []
            
        # 提取回复和上下文
        response_text = response.get("response", "")
        new_context = response.get("context", context or [])
        
        self.logger.info(f"收到Ollama回复: {len(response_text)} 字符")
        
        return response_text, new_context


class TTSClient(ExternalServiceClient):
    """TTS API客户端"""
    
    def __init__(self, base_url: str = "http://192.168.202.10:9880", voice_id: str = "default"):
        """
        初始化TTS客户端
        
        Args:
            base_url: API基础URL
            voice_id: 声音ID
        """
        super().__init__(base_url)
        self.voice_id = voice_id
        self.logger = logging.getLogger("TTSClient")
        self.logger.info(f"TTS客户端初始化: {base_url}, 声音ID: {voice_id}")
        
    async def is_available(self) -> bool:
        """
        检查TTS服务是否可用
        
        Returns:
            bool: 服务是否可用
        """
        success, _ = await self._request("GET", "tts/health")
        return success
        
    async def text_to_speech(self, text: str, voice_id: str = None) -> bytes:
        """
        将文本转换为语音
        
        Args:
            text: 要转换的文本
            voice_id: 声音ID(如果不指定则使用默认值)
            
        Returns:
            bytes: WAV格式的音频数据
        """
        if not text:
            return b''
            
        voice_id = voice_id or self.voice_id
            
        # 构建请求数据
        request_data = {
            "text": text,
            "voice_id": voice_id,
            "speed": 1.0  # 语速正常
        }
        
        self.logger.info(f"发送TTS请求: {len(text)} 字符, 声音ID: {voice_id}")
        
        # 发送请求
        success, response = await self._request("POST", "tts", data=request_data)
        
        if not success:
            self.logger.error(f"TTS请求失败: {response}")
            return b''
            
        # 如果返回的是字节数据
        if isinstance(response, bytes):
            return response
            
        # 如果返回的是JSON数据，可能需要进一步处理
        self.logger.error(f"TTS返回了非音频数据: {response}")
        return b''


class ASRClient(ExternalServiceClient):
    """ASR API客户端"""
    
    def __init__(self, base_url: str = "http://192.168.202.10:9880"):
        """
        初始化ASR客户端
        
        Args:
            base_url: API基础URL
        """
        super().__init__(base_url)
        self.logger = logging.getLogger("ASRClient")
        self.logger.info(f"ASR客户端初始化: {base_url}")
        
    async def is_available(self) -> bool:
        """
        检查ASR服务是否可用
        
        Returns:
            bool: 服务是否可用
        """
        success, _ = await self._request("GET", "asr/health")
        return success
        
    async def speech_to_text(self, audio_data: bytes) -> str:
        """
        将语音转换为文本
        
        Args:
            audio_data: 音频数据
            
        Returns:
            str: 识别的文本
        """
        if not audio_data:
            return ""
            
        # 构建请求数据(使用FormData)
        headers = {
            "Content-Type": "application/octet-stream"
        }
        
        self.logger.info(f"发送ASR请求: {len(audio_data)} 字节")
        
        # 发送请求
        success, response = await self._request(
            "POST", 
            "asr", 
            data=audio_data,
            headers=headers
        )
        
        if not success:
            self.logger.error(f"ASR请求失败: {response}")
            return ""
            
        # 如果返回的是JSON
        if isinstance(response, dict):
            return response.get("text", "")
            
        # 如果返回的是字符串
        if isinstance(response, str):
            try:
                return json.loads(response).get("text", "")
            except json.JSONDecodeError:
                return response
                
        self.logger.error(f"ASR返回了意外的数据类型: {type(response)}")
        return ""


class ConversationManager:
    """对话管理器，管理对话历史和上下文"""
    
    def __init__(self, max_history: int = 10):
        """
        初始化对话管理器
        
        Args:
            max_history: 最大保存的对话轮数
        """
        self.max_history = max_history
        self.conversation = []
        self.ollama_context = []
        self.logger = logging.getLogger("ConversationManager")
        
    async def add_message(self, role: str, content: str) -> None:
        """
        添加消息到对话历史
        
        Args:
            role: 消息角色("user"或"assistant")
            content: 消息内容
        """
        if role not in ["user", "assistant"]:
            self.logger.warning(f"无效的消息角色: {role}")
            return
            
        # 添加新消息
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        
        self.conversation.append(message)
        
        # 限制历史长度
        if len(self.conversation) > self.max_history * 2:  # 每轮对话有两条消息(用户和助手)
            self.conversation = self.conversation[-self.max_history * 2:]
            
        self.logger.debug(f"添加{role}消息: {len(content)} 字符")
        
    async def get_conversation_context(self) -> List[Dict]:
        """
        获取对话上下文
        
        Returns:
            List[Dict]: 对话历史
        """
        return self.conversation.copy()
        
    def get_last_n_exchanges(self, n: int) -> List[Dict]:
        """
        获取最近n轮对话
        
        Args:
            n: 轮数
            
        Returns:
            List[Dict]: 对话历史
        """
        return self.conversation[-n * 2:] if self.conversation else []
        
    async def clear_context(self) -> None:
        """清除当前对话上下文"""
        self.conversation = []
        self.ollama_context = []
        self.logger.info("对话上下文已清除")
        
    async def set_ollama_context(self, context: List[int]) -> None:
        """
        设置Ollama上下文
        
        Args:
            context: Ollama上下文数组
        """
        self.ollama_context = context
        
    async def get_ollama_context(self) -> List[int]:
        """
        获取Ollama上下文
        
        Returns:
            List[int]: Ollama上下文
        """
        return self.ollama_context
        
    async def save_to_file(self, filename: str) -> bool:
        """
        保存对话历史到文件
        
        Args:
            filename: 文件名
            
        Returns:
            bool: 是否成功
        """
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump({
                    "conversation": self.conversation,
                    "timestamp": time.time()
                }, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"对话历史已保存到 {filename}")
            return True
        except Exception as e:
            self.logger.error(f"保存对话历史失败: {str(e)}")
            return False
            
    async def load_from_file(self, filename: str) -> bool:
        """
        从文件加载对话历史
        
        Args:
            filename: 文件名
            
        Returns:
            bool: 是否成功
        """
        if not os.path.exists(filename):
            return False
            
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            self.conversation = data.get("conversation", [])
            self.logger.info(f"从 {filename} 加载了 {len(self.conversation)} 条对话历史")
            return True
        except Exception as e:
            self.logger.error(f"加载对话历史失败: {str(e)}")
            return False 
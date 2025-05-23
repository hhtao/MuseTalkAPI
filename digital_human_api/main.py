import asyncio
import logging
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import json
import time
import signal
import uuid

# 使用之前修改的相对导入
from core.state_manager import StateManager, DigitalHumanState
from core.audio_monitor import AudioMonitor
from core.interruption_handler import InterruptionHandler
from core.external_services import OllamaClient, TTSClient, ASRClient, ConversationManager
from core.lip_sync_service import LipSyncService
from server.webrtc_server import WebRTCServer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('digital_human_api.log')
    ]
)

class DigitalHumanService:
    """数字人服务主类，集成所有组件"""
    
    def __init__(self, config_path: str = None):
        """
        初始化数字人服务
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.running = False
        
        # 创建状态管理器
        self.state_manager = StateManager()
        
        # 创建外部服务客户端
        self.ollama_client = OllamaClient(
            base_url=self.config.get('ollama_api_url', 'http://122.204.161.127:11434/api'),
            model=self.config.get('ollama_model', 'qwen3:32b')
        )
        
        self.tts_client = TTSClient(
            base_url=self.config.get('tts_api_url', 'http://192.168.202.10:9880'),
            voice_id=self.config.get('tts_voice_id', 'default')
        )
        
        self.asr_client = ASRClient(
            base_url=self.config.get('asr_api_url', 'http://192.168.202.10:9880')
        )
        
        # 创建会话管理器
        self.conversation_manager = ConversationManager(
            max_history=self.config.get('max_conversation_history', 10)
        )
        
        # 创建音频监听器
        self.audio_monitor = AudioMonitor(
            state_manager=self.state_manager,
            asr_client=self.asr_client
        )
        
        # 创建打断处理器
        self.interruption_handler = InterruptionHandler(
            state_manager=self.state_manager
        )
        
        # 创建口型同步服务
        lip_sync_config = {
            'gpu_id': self.config.get('gpu_id', 0),
            'vae_type': self.config.get('vae_type', 'sd-vae'),
            'unet_config': self.config.get('unet_config', './models/musetalk/musetalk.json'),
            'unet_model_path': self.config.get('unet_model_path', './models/musetalk/pytorch_model.bin'),
            'whisper_dir': self.config.get('whisper_dir', './models/whisper'),
            'version': self.config.get('musetalk_version', 'v15'),
            'batch_size': self.config.get('batch_size', 8),
            'fps': self.config.get('fps', 25),
            'avatar_base_dir': self.config.get('avatar_base_dir', './results/v15/avatars')
        }
        self.lip_sync_service = LipSyncService(lip_sync_config)
        
        # 创建WebRTC服务器
        self.webrtc_server = WebRTCServer(
            state_manager=self.state_manager,
            host=self.config.get('host', '0.0.0.0'),
            port=self.config.get('port', 8080)
        )
        
        # 注册事件处理函数
        self._register_event_handlers()
        
        logging.info("DigitalHumanService initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            dict: 配置字典
        """
        default_config = {
            'host': '0.0.0.0',
            'port': 8000,  # 修改为8000端口
            'ollama_api_url': 'http://122.204.161.127:11434/api',
            'ollama_model': 'qwen3:32b',
            'tts_api_url': 'http://192.168.202.10:9880',
            'asr_api_url': 'http://192.168.202.10:9880',
            'tts_voice_id': 'default',
            'max_conversation_history': 10,
            'default_avatar_id': 'monalisa',  # 添加默认数字人模型
            'debug': False
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        config = yaml.safe_load(f)
                    elif config_path.endswith('.json'):
                        config = json.load(f)
                    else:
                        logging.warning(f"Unsupported config file format: {config_path}")
                        config = {}
                    
                    # 更新默认配置
                    default_config.update(config)
                    logging.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logging.error(f"Failed to load config file: {str(e)}")
        
        # 设置日志级别
        if default_config.get('debug', False):
            logging.getLogger().setLevel(logging.DEBUG)
            logging.info("Debug mode enabled")
        
        return default_config
    
    def _register_event_handlers(self) -> None:
        """注册事件处理函数"""
        # 状态变化回调
        asyncio.create_task(self.state_manager.register_state_change_callback(
            DigitalHumanState.THINKING, self._on_thinking_state
        ))
        
        asyncio.create_task(self.state_manager.register_state_change_callback(
            DigitalHumanState.SPEAKING, self._on_speaking_state
        ))
        
        asyncio.create_task(self.state_manager.register_state_change_callback(
            DigitalHumanState.INTERRUPTED, self._on_interrupted_state
        ))
        
        # 打断处理回调
        asyncio.create_task(self.interruption_handler.register_callbacks(
            self._handle_confirmation_interruption,
            self._handle_correction_interruption
        ))
        
        # 音频监听器打断回调
        asyncio.create_task(self.audio_monitor.register_interruption_callback(
            self._on_interruption_detected
        ))
        
        # WebRTC连接建立回调
        asyncio.create_task(self.webrtc_server.peer_manager.register_on_connection_established(
            self._on_webrtc_connection_established
        ))
    
    async def _on_thinking_state(self, previous_state, new_state) -> None:
        """思考状态回调，处理用户输入并生成回复"""
        # 获取用户输入
        context = await self.state_manager.get_state_context()
        user_input = context.get("user_input", "")
        
        if not user_input:
            logging.warning("No user input in thinking state")
            await self.state_manager.set_state(DigitalHumanState.IDLE_LISTENING)
            return
        
        # 添加用户消息到对话上下文
        await self.conversation_manager.add_message("user", user_input)
        
        try:
            # 广播状态更新
            await self.webrtc_server.broadcast_message({
                "type": "state_update",
                "state": "THINKING"
            })
            
            # 获取ollama上下文
            ollama_context = await self.conversation_manager.get_ollama_context()
            
            # 构建提示词
            conversation = await self.conversation_manager.get_conversation_context()
            prompt = self._build_prompt(conversation)
            
            # 调用Ollama生成回复
            response, new_context = await self.ollama_client.generate_response(prompt, ollama_context)
            
            # 更新ollama上下文
            await self.conversation_manager.set_ollama_context(new_context)
            
            # 添加助手回复到对话上下文
            await self.conversation_manager.add_message("assistant", response)
            
            # 准备打断处理器的分段回复
            await self.interruption_handler.prepare_response(response)
            
            # 广播文本消息
            await self.webrtc_server.broadcast_message({
                "type": "text_message",
                "text": response
            })
            
            # 转入说话状态
            await self.state_manager.set_state(DigitalHumanState.SPEAKING)
        except Exception as e:
            logging.error(f"Error in thinking state: {str(e)}")
            await self.state_manager.set_state(DigitalHumanState.IDLE_LISTENING)
    
    async def _on_speaking_state(self, previous_state, new_state) -> None:
        """说话状态回调，播放语音回复"""
        # 广播状态更新
        await self.webrtc_server.broadcast_message({
            "type": "state_update",
            "state": "SPEAKING"
        })
        
        # 获取下一个回复段落
        segment = await self.interruption_handler.get_next_segment()
        
        if not segment:
            # 如果没有更多段落，返回空闲状态
            await self.state_manager.set_state(DigitalHumanState.IDLE_LISTENING)
            return
        
        try:
            # 转换文本为语音
            audio_data = await self.tts_client.text_to_speech(segment)
            
            if not audio_data:
                logging.error("Failed to generate speech")
                await self.state_manager.set_state(DigitalHumanState.IDLE_LISTENING)
                return
            
            # 使用MuseTalk生成口型视频
            avatar_id = self.config.get('default_avatar_id', 'default')
            logging.info(f"生成口型视频，使用数字人：{avatar_id}")
            
            # 初始化口型同步服务（如果尚未初始化）
            if not self.lip_sync_service.initialized:
                success = await self.lip_sync_service.initialize()
                if not success:
                    logging.error("无法初始化口型同步服务")
                    video_data = b''  # 使用空视频数据
                else:
                    logging.info("口型同步服务初始化成功")
            
            # 生成口型视频
            video_data = await self.lip_sync_service.generate_lip_sync_video(
                avatar_id=avatar_id,
                audio_data=audio_data,
                max_frames=300  # 限制最大帧数，避免内存溢出
            )
            
            if not video_data:
                logging.warning("口型生成失败，使用空视频数据")
                video_data = b''
                
            # 发送媒体流
            await self.webrtc_server.send_media_to_all_clients(video_data, audio_data)
            
            # 检查是否有更多段落
            if await self.interruption_handler.is_response_complete():
                # 如果回复完成，返回空闲状态
                await self.state_manager.set_state(DigitalHumanState.IDLE_LISTENING)
            else:
                # 否则，继续说话状态播放下一段
                await self.state_manager.set_state(DigitalHumanState.SPEAKING)
        except Exception as e:
            logging.error(f"Error in speaking state: {str(e)}")
            import traceback
            traceback.print_exc()
            await self.state_manager.set_state(DigitalHumanState.IDLE_LISTENING)
    
    async def _on_interrupted_state(self, previous_state, new_state) -> None:
        """被打断状态回调"""
        # 广播状态更新
        await self.webrtc_server.broadcast_message({
            "type": "state_update",
            "state": "INTERRUPTED"
        })
        
        # 获取打断内容
        context = await self.state_manager.get_state_context()
        interruption_text = context.get("interruption_text", "")
        
        if not interruption_text:
            logging.warning("No interruption text in interrupted state")
            await self.state_manager.set_state(DigitalHumanState.IDLE_LISTENING)
            return
        
        # 处理打断
        await self.interruption_handler.handle_interruption(text=interruption_text)
    
    async def _on_interruption_detected(self, interruption_text: str) -> None:
        """检测到打断时的回调"""
        # 添加用户打断消息到对话上下文
        await self.conversation_manager.add_message("user", f"[打断] {interruption_text}")
        
        # 广播用户消息
        await self.webrtc_server.broadcast_message({
            "type": "text_message",
            "text": interruption_text,
            "role": "user",
            "is_interruption": True
        })
    
    async def _handle_confirmation_interruption(self) -> None:
        """确认性打断处理"""
        try:
            # 生成简短确认回应
            confirmation_response = "好的"
            
            # 添加确认回应到对话上下文
            await self.conversation_manager.add_message("assistant", f"[确认] {confirmation_response}")
            
            # 广播确认回应
            await self.webrtc_server.broadcast_message({
                "type": "text_message",
                "text": confirmation_response,
                "is_confirmation": True
            })
            
            # 转换确认回应为语音
            audio_data = await self.tts_client.text_to_speech(confirmation_response)
            
            if audio_data:
                # 发送确认回应的语音
                await self.webrtc_server.send_media_to_all_clients(b'', audio_data)
        except Exception as e:
            logging.error(f"Error handling confirmation interruption: {str(e)}")
    
    async def _handle_correction_interruption(self, user_text: str) -> None:
        """纠正性打断处理"""
        # 已经在打断处理器中处理了
        pass
    
    async def _on_webrtc_connection_established(self, client_id: str, peer_connection) -> None:
        """WebRTC连接建立回调"""
        logging.info(f"WebRTC connection established with client: {client_id}")
        
        # 发送当前状态
        current_state = await self.state_manager.get_current_state()
        await self.webrtc_server.send_message_to_client(client_id, {
            "type": "state_update",
            "state": current_state.value
        })
        
        # 发送最近的对话历史
        conversation = self.conversation_manager.get_last_n_exchanges(5)
        for message in conversation:
            await self.webrtc_server.send_message_to_client(client_id, {
                "type": "text_message",
                "text": message["content"],
                "role": message["role"]
            })
    
    def _build_prompt(self, conversation: list) -> str:
        """
        构建发送给LLM的提示词
        
        Args:
            conversation: 对话历史
            
        Returns:
            str: 提示词
        """
        # 系统提示词
        system_prompt = """
你是MuseTalk自然交互数字人，一个友好、专业、有帮助的AI助手。
你应该：
1. 给出准确、有用且直接的回答
2. 使用礼貌且自然的语气，像真人一样交流
3. 尽量简洁，但确保回答全面
4. 在不确定时承认局限性
5. 避免使用过于正式或机械的语言
        """
        
        # 构建完整提示词
        prompt = system_prompt + "\n\n"
        
        # 添加对话历史
        for message in conversation:
            role = "User" if message["role"] == "user" else "Assistant"
            prompt += f"{role}: {message['content']}\n\n"
        
        # 添加最后的提示
        prompt += "Assistant: "
        
        return prompt
    
    async def start(self) -> None:
        """启动数字人服务"""
        if self.running:
            logging.warning("DigitalHumanService already running")
            return
            
        self.running = True
        
        # 检查并创建必要的目录
        self._ensure_directories()
        
        # 启动各个组件
        await self.audio_monitor.start()
        await self.webrtc_server.start()
        
        # 加载持久化状态（如果有）
        await self.state_manager.load_persisted_state()
        
        logging.info(f"DigitalHumanService started on {self.config['host']}:{self.config['port']}")
    
    def _ensure_directories(self) -> None:
        """检查并创建必要的目录结构"""
        # 创建结果目录
        results_dir = Path('./results/v15/avatars')
        if not results_dir.exists():
            logging.info(f"创建目录: {results_dir}")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建默认数字人示例文件
            default_avatar = results_dir / "monalisa"
            if not default_avatar.exists():
                logging.info(f"创建默认数字人示例文件: {default_avatar}")
                with open(default_avatar, 'w') as f:
                    f.write("示例数字人配置文件")
        
        # 检查模型目录
        models_dir = Path('./models/musetalk')
        if not models_dir.exists():
            logging.info(f"创建目录: {models_dir}")
            models_dir.mkdir(parents=True, exist_ok=True)
    
    async def stop(self) -> None:
        """停止数字人服务"""
        if not self.running:
            return
            
        self.running = False
        
        # 停止各个组件
        await self.audio_monitor.stop()
        await self.webrtc_server.stop()
        
        # 关闭口型同步服务
        if hasattr(self, 'lip_sync_service'):
            await self.lip_sync_service.close()
        
        # 关闭外部服务连接
        await self.ollama_client.close()
        await self.tts_client.close()
        await self.asr_client.close()
        
        # 保存对话历史
        await self.conversation_manager.save_to_file("conversation_history.json")
        
        logging.info("DigitalHumanService stopped")


async def main():
    """主函数"""
    # 获取配置文件路径
    config_path = os.environ.get('CONFIG_PATH', 'config/api_config.yaml')
    
    # 创建数字人服务
    service = DigitalHumanService(config_path)
    
    # 设置信号处理
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(service)))
    
    # 启动服务
    await service.start()
    
    # 保持运行
    try:
        while service.running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await service.stop()


async def shutdown(service: DigitalHumanService):
    """关闭服务"""
    logging.info("Shutting down...")
    await service.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    except Exception as e:
        logging.error(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1) 
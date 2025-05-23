import asyncio
import logging
import json
import uuid
import os
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
import weakref
from pathlib import Path

import av
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.media import MediaPlayer, MediaRelay, MediaStreamTrack, MediaRecorder
from aiortc.mediastreams import MediaStreamError, VideoStreamTrack, AudioStreamTrack

from aiohttp import web
from aiohttp.web import Request, Response

from digital_human_api.core.state_manager import StateManager, DigitalHumanState

# 自定义媒体轨道类
class CustomVideoStreamTrack(VideoStreamTrack):
    """自定义视频流轨道，支持实时添加帧"""
    
    def __init__(self, track_id: str = None):
        """
        初始化视频轨道
        
        Args:
            track_id: 轨道ID，不提供则自动生成
        """
        super().__init__()
        self._id = track_id or str(uuid.uuid4())
        self._queue = asyncio.Queue()
        self._start = time.time()
        self._timestamp = 0
        logging.info(f"CustomVideoStreamTrack created with id: {self._id}")
    
    async def recv(self) -> av.VideoFrame:
        """
        接收下一个视频帧
        
        Returns:
            av.VideoFrame: 视频帧
        """
        frame = await self._queue.get()
        frame.time_base = av.Fraction(1, 90000)  # 使用标准视频时基
        frame.pts = self._timestamp
        self._timestamp += 3000  # 假设30fps，每帧增加3000个时间单位
        return frame
    
    async def add_frame(self, frame: av.VideoFrame) -> None:
        """
        添加视频帧到队列
        
        Args:
            frame: 视频帧
        """
        await self._queue.put(frame)


class CustomAudioStreamTrack(AudioStreamTrack):
    """自定义音频流轨道，支持实时添加音频样本"""
    
    def __init__(self, track_id: str = None, sample_rate: int = 48000, channels: int = 1):
        """
        初始化音频轨道
        
        Args:
            track_id: 轨道ID，不提供则自动生成
            sample_rate: 采样率
            channels: 声道数
        """
        super().__init__()
        self._id = track_id or str(uuid.uuid4())
        self._queue = asyncio.Queue()
        self._start = time.time()
        self._timestamp = 0
        self._sample_rate = sample_rate
        self._channels = channels
        logging.info(f"CustomAudioStreamTrack created with id: {self._id}, "
                    f"sample_rate: {sample_rate}, channels: {channels}")
    
    async def recv(self) -> av.AudioFrame:
        """
        接收下一个音频帧
        
        Returns:
            av.AudioFrame: 音频帧
        """
        frame = await self._queue.get()
        frame.time_base = av.Fraction(1, self._sample_rate)
        frame.pts = self._timestamp
        self._timestamp += frame.samples
        return frame
    
    async def add_samples(self, frame: av.AudioFrame) -> None:
        """
        添加音频样本到队列
        
        Args:
            frame: 音频帧
        """
        await self._queue.put(frame)


class MediaStreamManager:
    """媒体流管理器，处理音视频流"""
    
    def __init__(self):
        """初始化媒体流管理器"""
        self.video_relay = MediaRelay()
        self.active_tracks = {}  # 类型: Dict[str, Tuple[CustomVideoStreamTrack, CustomAudioStreamTrack]]
        logging.info("MediaStreamManager initialized")
    
    async def create_stream_from_file(self, video_file: str, audio_file: str) -> Tuple[CustomVideoStreamTrack, CustomAudioStreamTrack]:
        """
        从文件创建媒体流轨道
        
        Args:
            video_file: 视频文件路径
            audio_file: 音频文件路径
            
        Returns:
            Tuple[CustomVideoStreamTrack, CustomAudioStreamTrack]: 视频和音频轨道
        """
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"Video file not found: {video_file}")
            
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
        # 创建媒体播放器
        video_player = MediaPlayer(video_file)
        audio_player = MediaPlayer(audio_file)
        
        # 获取轨道
        video_track = self.video_relay.subscribe(video_player.video)
        audio_track = audio_player.audio
        
        # 创建自定义轨道
        custom_video_track = CustomVideoStreamTrack()
        custom_audio_track = CustomAudioStreamTrack()
        
        # 启动转发任务
        asyncio.create_task(self._forward_track(video_track, custom_video_track))
        asyncio.create_task(self._forward_track(audio_track, custom_audio_track))
        
        # 存储轨道
        track_id = str(uuid.uuid4())
        self.active_tracks[track_id] = (custom_video_track, custom_audio_track)
        
        logging.info(f"Created stream from files: {video_file}, {audio_file}")
        return custom_video_track, custom_audio_track
    
    async def create_empty_stream(self) -> Tuple[CustomVideoStreamTrack, CustomAudioStreamTrack]:
        """
        创建空的媒体流轨道
        
        Returns:
            Tuple[CustomVideoStreamTrack, CustomAudioStreamTrack]: 视频和音频轨道
        """
        video_track = CustomVideoStreamTrack()
        audio_track = CustomAudioStreamTrack()
        
        track_id = str(uuid.uuid4())
        self.active_tracks[track_id] = (video_track, audio_track)
        
        logging.info(f"Created empty stream with id: {track_id}")
        return video_track, audio_track, track_id
    
    async def add_video_frame(self, track_id: str, frame_data: bytes) -> None:
        """
        添加视频帧到指定轨道
        
        Args:
            track_id: 轨道ID
            frame_data: 视频帧数据
        """
        if track_id not in self.active_tracks:
            logging.warning(f"Track not found: {track_id}")
            return
            
        video_track, _ = self.active_tracks[track_id]
        
        # 创建视频帧
        packet = av.Packet(frame_data)
        frames = av.VideoFrame.from_ndarray(packet.to_ndarray(), format='bgr24')
        
        # 添加到轨道
        await video_track.add_frame(frames)
    
    async def add_audio_samples(self, track_id: str, audio_data: bytes) -> None:
        """
        添加音频样本到指定轨道
        
        Args:
            track_id: 轨道ID
            audio_data: 音频样本数据
        """
        if track_id not in self.active_tracks:
            logging.warning(f"Track not found: {track_id}")
            return
            
        _, audio_track = self.active_tracks[track_id]
        
        # 创建音频帧
        packet = av.Packet(audio_data)
        frames = av.AudioFrame.from_ndarray(packet.to_ndarray(), format='s16', layout='mono')
        
        # 添加到轨道
        await audio_track.add_samples(frames)
    
    async def remove_stream(self, track_id: str) -> None:
        """
        移除流轨道
        
        Args:
            track_id: 轨道ID
        """
        if track_id in self.active_tracks:
            del self.active_tracks[track_id]
            logging.info(f"Removed stream with id: {track_id}")
    
    async def segment_response(self, audio_data: bytes, segment_duration: float = 2.0) -> List[bytes]:
        """
        将音频分段处理，便于流式传输
        
        Args:
            audio_data: 完整的音频数据
            segment_duration: 每段的持续时间（秒）
            
        Returns:
            List[bytes]: 分段后的音频数据列表
        """
        try:
            # 创建临时文件
            temp_file = f"temp_{uuid.uuid4()}.wav"
            with open(temp_file, "wb") as f:
                f.write(audio_data)
            
            # 使用AV解析音频
            container = av.open(temp_file)
            audio_stream = next(s for s in container.streams if s.type == 'audio')
            
            # 计算每段的帧数
            frames_per_segment = int(segment_duration * audio_stream.rate)
            
            segments = []
            current_segment = bytearray()
            frame_count = 0
            
            # 读取音频帧并分段
            for frame in container.decode(audio_stream):
                frame_bytes = frame.to_ndarray().tobytes()
                current_segment.extend(frame_bytes)
                frame_count += 1
                
                if frame_count >= frames_per_segment:
                    segments.append(bytes(current_segment))
                    current_segment = bytearray()
                    frame_count = 0
            
            # 添加最后一段（如果有）
            if current_segment:
                segments.append(bytes(current_segment))
            
            # 清理临时文件
            os.remove(temp_file)
            
            return segments
        except Exception as e:
            logging.error(f"Error segmenting audio: {str(e)}")
            return [audio_data]
    
    async def _forward_track(self, source_track: MediaStreamTrack, target_track: Union[CustomVideoStreamTrack, CustomAudioStreamTrack]) -> None:
        """
        转发媒体轨道数据
        
        Args:
            source_track: 源轨道
            target_track: 目标轨道
        """
        try:
            while True:
                frame = await source_track.recv()
                if isinstance(target_track, CustomVideoStreamTrack):
                    await target_track.add_frame(frame)
                else:
                    await target_track.add_samples(frame)
        except MediaStreamError as e:
            logging.error(f"Media stream error: {str(e)}")
        finally:
            logging.info("Track forwarding stopped")


class WebRTCPeerManager:
    """WebRTC对等连接管理器"""
    
    def __init__(self):
        """初始化对等连接管理器"""
        self.peer_connections = {}
        self.data_channels = {}
        self.connection_established_callbacks = []
        self.logger = logging.getLogger("WebRTCPeerManager")
    
    async def create_peer_connection(self, client_id: str) -> RTCPeerConnection:
        """
        创建对等连接
        
        Args:
            client_id: 客户端ID
            
        Returns:
            RTCPeerConnection: 创建的对等连接
        """
        # 创建ICE服务器配置
        ice_servers = [
            RTCIceServer(urls="stun:stun.l.google.com:19302"),
            RTCIceServer(urls="stun:stun1.l.google.com:19302")
        ]
        
        config = RTCConfiguration(iceServers=ice_servers)
        
        # 创建对等连接
        pc = RTCPeerConnection(configuration=config)
        
        # 设置连接状态变化回调
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            self.logger.info(f"客户端 {client_id} 连接状态变为 {pc.connectionState}")
            
            if pc.connectionState == "connected":
                # 调用连接建立回调
                for callback in self.connection_established_callbacks:
                    asyncio.create_task(callback(client_id, pc))
            
            elif pc.connectionState == "failed" or pc.connectionState == "closed":
                await self.close_peer_connection(client_id)
        
        # 设置数据通道回调
        @pc.on("datachannel")
        def on_datachannel(channel):
            self.logger.info(f"客户端 {client_id} 创建数据通道: {channel.label}")
            self.data_channels[client_id] = channel
            
            @channel.on("message")
            def on_message(message):
                self.logger.info(f"从客户端 {client_id} 收到消息: {message}")
                # 这里可以添加消息处理逻辑
        
        # 保存对等连接
        self.peer_connections[client_id] = pc
        
        return pc
    
    async def close_peer_connection(self, client_id: str) -> None:
        """
        关闭对等连接
        
        Args:
            client_id: 客户端ID
        """
        if client_id in self.peer_connections:
            pc = self.peer_connections[client_id]
            
            # 关闭数据通道
            if client_id in self.data_channels:
                self.data_channels[client_id].close()
                del self.data_channels[client_id]
            
            # 关闭对等连接
            await pc.close()
            del self.peer_connections[client_id]
            
            self.logger.info(f"客户端 {client_id} 的连接已关闭")
    
    async def close_all_connections(self) -> None:
        """关闭所有连接"""
        client_ids = list(self.peer_connections.keys())
        
        for client_id in client_ids:
            await self.close_peer_connection(client_id)
            
        self.logger.info("所有连接已关闭")
    
    async def send_message(self, client_id: str, message: Any) -> bool:
        """
        发送消息到指定客户端
        
        Args:
            client_id: 客户端ID
            message: 要发送的消息
            
        Returns:
            bool: 是否成功发送
        """
        if client_id not in self.data_channels:
            return False
            
        channel = self.data_channels[client_id]
        
        if channel.readyState != "open":
            return False
            
        try:
            if isinstance(message, dict):
                message = json.dumps(message)
                
            channel.send(message)
            return True
        except Exception as e:
            self.logger.error(f"发送消息到客户端 {client_id} 失败: {str(e)}")
            return False
    
    async def broadcast_message(self, message: Any) -> None:
        """
        向所有客户端广播消息
        
        Args:
            message: 要广播的消息
        """
        for client_id in list(self.data_channels.keys()):
            await self.send_message(client_id, message)
    
    async def register_on_connection_established(self, callback: Callable) -> None:
        """
        注册连接建立回调
        
        Args:
            callback: 回调函数，接收client_id和peer_connection参数
        """
        self.connection_established_callbacks.append(callback)


class WebRTCServer:
    """WebRTC服务器"""
    
    def __init__(self, state_manager, host="0.0.0.0", port=8000):
        """
        初始化WebRTC服务器
        
        Args:
            state_manager: 状态管理器
            host: 主机地址
            port: 端口
        """
        self.state_manager = state_manager
        self.host = host
        self.port = port
        self.app = web.Application()
        self.peer_manager = WebRTCPeerManager()
        self.logger = logging.getLogger("WebRTCServer")
        self.running = False
        self.site = None
        self.runner = None
        
        # 设置路由
        self._setup_routes()
    
    def _setup_routes(self):
        """设置HTTP路由"""
        self.app.router.add_post("/webrtc/offer", self.handle_offer)
        self.app.router.add_post("/webrtc/ice", self.handle_ice)
        self.app.router.add_post("/reset_conversation", self.handle_reset_conversation)
        self.app.router.add_post("/process_audio", self.handle_process_audio)
        
        # 添加CORS中间件
        @web.middleware
        async def cors_middleware(request, handler):
            response = await handler(request)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response
            
        self.app.middlewares.append(cors_middleware)
    
    async def handle_offer(self, request):
        """处理SDP offer请求"""
        try:
            # 生成客户端ID
            client_id = str(uuid.uuid4())
            
            # 解析请求数据
            data = await request.json()
            
            if 'sdp' not in data:
                return web.json_response({"error": "Missing SDP"}, status=400)
                
            # 创建对等连接
            pc = await self.peer_manager.create_peer_connection(client_id)
            
            # 设置远程描述
            offer = RTCSessionDescription(sdp=data['sdp']['sdp'], type=data['sdp']['type'])
            await pc.setRemoteDescription(offer)
            
            # 创建应答
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            # 返回应答
            return web.json_response({
                "client_id": client_id,
                "sdp": {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            })
            
        except Exception as e:
            self.logger.error(f"处理offer请求失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_ice(self, request):
        """处理ICE candidate请求"""
        try:
            # 解析请求数据
            data = await request.json()
            
            if 'client_id' not in data or 'candidate' not in data:
                return web.json_response({"error": "Missing client_id or candidate"}, status=400)
                
            client_id = data['client_id']
            
            # 检查客户端是否存在
            if client_id not in self.peer_manager.peer_connections:
                return web.json_response({"error": "Client not found"}, status=404)
                
            pc = self.peer_manager.peer_connections[client_id]
            
            # 添加ICE候选
            candidate = RTCIceCandidate(
                component=data['candidate'].get('component', None),
                foundation=data['candidate'].get('foundation', None),
                ip=data['candidate'].get('ip', None),
                port=data['candidate'].get('port', None),
                priority=data['candidate'].get('priority', None),
                protocol=data['candidate'].get('protocol', None),
                type=data['candidate'].get('type', None),
                sdpMid=data['candidate'].get('sdpMid', None),
                sdpMLineIndex=data['candidate'].get('sdpMLineIndex', None)
            )
            
            await pc.addIceCandidate(candidate)
            
            return web.json_response({"success": True})
            
        except Exception as e:
            self.logger.error(f"处理ICE candidate请求失败: {str(e)}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_reset_conversation(self, request):
        """处理重置对话请求"""
        try:
            # 这里可以添加重置对话的逻辑
            # 例如重置状态管理器，清除对话历史等
            
            # 广播状态更新消息
            await self.peer_manager.broadcast_message({
                "type": "state_update",
                "state": "idle_listening"
            })
            
            return web.json_response({"success": True})
            
        except Exception as e:
            self.logger.error(f"处理重置对话请求失败: {str(e)}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_process_audio(self, request):
        """处理音频文件，执行ASR并返回文本"""
        try:
            # 解析请求数据（multipart/form-data格式）
            reader = await request.multipart()
            
            # 获取音频文件字段
            field = await reader.next()
            if field.name != 'audio':
                return web.json_response({"error": "Missing audio field"}, status=400)
                
            # 读取音频数据
            audio_data = bytearray()
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                audio_data.extend(chunk)
                
            if not audio_data:
                return web.json_response({"error": "Empty audio data"}, status=400)
                
            # 保存临时文件
            import tempfile
            import os
            
            temp_dir = tempfile.mkdtemp()
            temp_file = os.path.join(temp_dir, "audio.wav")
            
            with open(temp_file, "wb") as f:
                f.write(audio_data)
            
            # 导入ASR客户端
            from core.external_services import ASRClient
            
            # 创建ASR客户端
            asr_client = ASRClient()
            
            # 调用ASR服务
            text = await asr_client.speech_to_text(bytes(audio_data))
            
            # 清理临时文件
            try:
                os.remove(temp_file)
                os.rmdir(temp_dir)
            except:
                pass
                
            # 检查是否有文本结果
            if not text:
                return web.json_response({"error": "Failed to transcribe audio"}, status=500)
                
            # 返回识别结果
            return web.json_response({"text": text})
            
        except Exception as e:
            self.logger.error(f"处理音频文件失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)
    
    async def start(self):
        """启动WebRTC服务器"""
        if self.running:
            return
            
        self.running = True
        
        # 启动HTTP服务器
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        
        self.logger.info(f"WebRTC服务器已启动 {self.host}:{self.port}")
    
    async def stop(self):
        """停止WebRTC服务器"""
        if not self.running:
            return
            
        self.running = False
        
        # 关闭所有WebRTC连接
        await self.peer_manager.close_all_connections()
        
        # 关闭HTTP服务器
        if self.site:
            await self.site.stop()
            
        if self.runner:
            await self.runner.cleanup()
            
        self.logger.info("WebRTC服务器已停止")
    
    async def send_message_to_client(self, client_id: str, message: Any) -> bool:
        """
        发送消息到指定客户端
        
        Args:
            client_id: 客户端ID
            message: 要发送的消息
            
        Returns:
            bool: 是否成功发送
        """
        return await self.peer_manager.send_message(client_id, message)
    
    async def broadcast_message(self, message: Any) -> None:
        """
        向所有客户端广播消息
        
        Args:
            message: 要广播的消息
        """
        await self.peer_manager.broadcast_message(message)
    
    async def send_media_to_all_clients(self, video_data: bytes, audio_data: bytes) -> None:
        """
        向所有客户端发送媒体数据
        
        Args:
            video_data: 视频数据
            audio_data: 音频数据
        """
        # 这里应该实现向客户端发送视频和音频数据的逻辑
        # 在完整实现中，这里应该使用MediaStreamTrack等将数据发送到客户端
        
        # 简化版实现：发送消息通知客户端有新的媒体数据
        await self.broadcast_message({
            "type": "media_ready",
            "video_size": len(video_data),
            "audio_size": len(audio_data)
        }) 
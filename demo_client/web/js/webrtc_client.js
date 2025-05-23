/**
 * WebRTC客户端，用于与数字人服务器建立实时音视频通信
 */
class WebRTCClient {
    /**
     * 创建WebRTC客户端
     * @param {Object} config - 配置选项
     * @param {string} config.serverUrl - 服务器URL（不含协议和端口）
     * @param {number} config.serverPort - 服务器端口
     * @param {boolean} config.secure - 是否使用安全连接（https/wss）
     * @param {HTMLVideoElement} config.videoElement - 视频元素
     * @param {HTMLAudioElement} config.audioElement - 音频元素（可选）
     * @param {Function} config.onStateChange - 状态变化回调
     * @param {Function} config.onMessage - 接收消息回调
     * @param {Function} config.onError - 错误回调
     */
    constructor(config) {
        this.config = Object.assign({
            serverUrl: window.location.hostname,
            serverPort: 8080,
            secure: window.location.protocol === 'https:',
            videoElement: null,
            audioElement: null,
            onStateChange: () => {},
            onMessage: () => {},
            onError: (error) => console.error('WebRTC Error:', error)
        }, config);

        // 检查必要的参数
        if (!this.config.videoElement) {
            throw new Error('videoElement is required');
        }

        // WebRTC状态
        this.state = 'disconnected';
        this.clientId = null;
        this.peerConnection = null;
        this.signalingChannel = null;
        this.mediaStream = null;
        this.dataChannel = null;

        // 绑定方法的this上下文
        this._onSignalingMessage = this._onSignalingMessage.bind(this);
        this._onIceCandidate = this._onIceCandidate.bind(this);
        this._onTrack = this._onTrack.bind(this);
        this._onIceConnectionStateChange = this._onIceConnectionStateChange.bind(this);
        this._onDataChannel = this._onDataChannel.bind(this);
        this._onDataChannelMessage = this._onDataChannelMessage.bind(this);

        console.log('WebRTC client initialized');
    }

    /**
     * 生成服务器URL
     * @param {string} path - 路径
     * @param {boolean} websocket - 是否是WebSocket URL
     * @returns {string} URL
     */
    _buildServerUrl(path, websocket = false) {
        const protocol = websocket 
            ? (this.config.secure ? 'wss://' : 'ws://') 
            : (this.config.secure ? 'https://' : 'http://');
        
        return `${protocol}${this.config.serverUrl}:${this.config.serverPort}${path}`;
    }

    /**
     * 设置状态并触发回调
     * @param {string} newState - 新状态
     */
    _setState(newState) {
        const oldState = this.state;
        this.state = newState;
        console.log(`WebRTC state changed: ${oldState} -> ${newState}`);
        this.config.onStateChange(newState, oldState);
    }

    /**
     * 连接到WebRTC服务器
     * @returns {Promise<void>}
     */
    async connect() {
        if (this.state !== 'disconnected') {
            console.warn(`Cannot connect while in state: ${this.state}`);
            return;
        }

        this._setState('connecting');

        try {
            // 创建WebSocket连接
            await this._createSignalingChannel();
            console.log('Signaling channel established');

            // 创建RTC对等连接
            await this._createPeerConnection();
            console.log('Peer connection created');

            // 创建数据通道
            this._createDataChannel();
            console.log('Data channel created');

            // 创建并发送offer
            await this._createAndSendOffer();
            console.log('Offer sent');

            this._setState('connected');
        } catch (error) {
            this._setState('failed');
            this.config.onError(error);
            throw error;
        }
    }

    /**
     * 断开WebRTC连接
     */
    async disconnect() {
        if (this.state === 'disconnected') {
            return;
        }

        this._setState('disconnecting');

        // 关闭数据通道
        if (this.dataChannel) {
            this.dataChannel.close();
            this.dataChannel = null;
        }

        // 关闭对等连接
        if (this.peerConnection) {
            this.peerConnection.close();
            this.peerConnection = null;
        }

        // 关闭信令通道
        if (this.signalingChannel) {
            if (this.signalingChannel.readyState === WebSocket.OPEN) {
                this.signalingChannel.send(JSON.stringify({
                    type: 'close',
                    client_id: this.clientId
                }));
            }
            this.signalingChannel.close();
            this.signalingChannel = null;
        }

        // 释放媒体流
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        this.clientId = null;
        this._setState('disconnected');
        console.log('WebRTC disconnected');
    }

    /**
     * 发送数据消息到服务器
     * @param {Object} data - 要发送的数据
     * @returns {boolean} 是否成功发送
     */
    sendMessage(data) {
        if (this.state !== 'connected' || !this.dataChannel || this.dataChannel.readyState !== 'open') {
            console.warn('Cannot send message: data channel not ready');
            return false;
        }

        try {
            const message = typeof data === 'string' ? data : JSON.stringify(data);
            this.dataChannel.send(message);
            return true;
        } catch (error) {
            console.error('Error sending message:', error);
            this.config.onError(error);
            return false;
        }
    }

    /**
     * 创建信令通道
     * @returns {Promise<void>}
     */
    async _createSignalingChannel() {
        return new Promise((resolve, reject) => {
            const wsUrl = this._buildServerUrl('/webrtc/signaling', true);
            this.signalingChannel = new WebSocket(wsUrl);

            this.signalingChannel.onopen = () => {
                console.log('Signaling channel opened');
                resolve();
            };

            this.signalingChannel.onmessage = this._onSignalingMessage;

            this.signalingChannel.onerror = (error) => {
                console.error('Signaling channel error:', error);
                reject(error);
            };

            this.signalingChannel.onclose = (event) => {
                console.log('Signaling channel closed:', event.code, event.reason);
                if (this.state === 'connected' || this.state === 'connecting') {
                    this._setState('disconnected');
                }
            };
        });
    }

    /**
     * 处理信令通道消息
     * @param {MessageEvent} event - WebSocket消息事件
     */
    async _onSignalingMessage(event) {
        try {
            const message = JSON.parse(event.data);
            console.log('Received signaling message:', message.type);

            switch (message.type) {
                case 'connect':
                    this.clientId = message.client_id;
                    console.log(`Client ID assigned: ${this.clientId}`);
                    break;
                case 'answer':
                    await this._handleAnswer(message.answer);
                    break;
                case 'ice':
                    await this._handleIceCandidate(message.candidate);
                    break;
                default:
                    this.config.onMessage(message);
                    break;
            }
        } catch (error) {
            console.error('Error handling signaling message:', error, event.data);
        }
    }

    /**
     * 创建对等连接
     * @returns {Promise<void>}
     */
    async _createPeerConnection() {
        const configuration = {
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' }
            ]
        };

        this.peerConnection = new RTCPeerConnection(configuration);
        
        // 设置事件处理器
        this.peerConnection.onicecandidate = this._onIceCandidate;
        this.peerConnection.ontrack = this._onTrack;
        this.peerConnection.ondatachannel = this._onDataChannel;
        this.peerConnection.oniceconnectionstatechange = this._onIceConnectionStateChange;
    }

    /**
     * ICE候选处理
     * @param {RTCIceCandidate} event - ICE候选事件
     */
    _onIceCandidate(event) {
        if (!event.candidate) return;

        console.log('ICE candidate generated:', event.candidate.candidate.substr(0, 50) + '...');

        // 发送ICE候选到服务器
        if (this.signalingChannel && this.signalingChannel.readyState === WebSocket.OPEN) {
            this.signalingChannel.send(JSON.stringify({
                type: 'ice',
                client_id: this.clientId,
                candidate: {
                    sdpMLineIndex: event.candidate.sdpMLineIndex,
                    sdpMid: event.candidate.sdpMid,
                    candidate: event.candidate.candidate
                }
            }));
        }
    }

    /**
     * 媒体轨道处理
     * @param {RTCTrackEvent} event - 轨道事件
     */
    _onTrack(event) {
        console.log('Track received:', event.track.kind);

        // 为视频和音频元素设置媒体流
        if (event.track.kind === 'video') {
            if (!this.mediaStream) {
                this.mediaStream = new MediaStream();
                this.config.videoElement.srcObject = this.mediaStream;
            }
            this.mediaStream.addTrack(event.track);
        } else if (event.track.kind === 'audio') {
            if (this.config.audioElement) {
                if (!this.config.audioElement.srcObject) {
                    this.config.audioElement.srcObject = new MediaStream();
                }
                this.config.audioElement.srcObject.addTrack(event.track);
            } else {
                // 如果没有提供单独的音频元素，将音频轨道添加到视频元素的媒体流中
                if (this.mediaStream) {
                    this.mediaStream.addTrack(event.track);
                }
            }
        }
    }

    /**
     * ICE连接状态变化处理
     */
    _onIceConnectionStateChange() {
        console.log('ICE connection state:', this.peerConnection.iceConnectionState);

        if (this.peerConnection.iceConnectionState === 'failed' ||
            this.peerConnection.iceConnectionState === 'disconnected' ||
            this.peerConnection.iceConnectionState === 'closed') {
            
            if (this.state === 'connected') {
                this._setState('reconnecting');
                // 尝试重新建立连接
                setTimeout(() => {
                    if (this.state === 'reconnecting') {
                        this.disconnect().then(() => this.connect());
                    }
                }, 2000);
            }
        } else if (this.peerConnection.iceConnectionState === 'connected') {
            if (this.state === 'reconnecting') {
                this._setState('connected');
            }
        }
    }

    /**
     * 创建数据通道
     */
    _createDataChannel() {
        this.dataChannel = this.peerConnection.createDataChannel('messages', {
            ordered: true
        });

        this.dataChannel.onopen = () => {
            console.log('Data channel opened');
        };

        this.dataChannel.onclose = () => {
            console.log('Data channel closed');
            this.dataChannel = null;
        };

        this.dataChannel.onmessage = this._onDataChannelMessage;
    }

    /**
     * 数据通道消息处理
     * @param {MessageEvent} event - 消息事件
     */
    _onDataChannelMessage(event) {
        try {
            const message = typeof event.data === 'string' 
                ? JSON.parse(event.data) 
                : event.data;
            this.config.onMessage(message);
        } catch (error) {
            console.error('Error parsing data channel message:', error);
            this.config.onError(error);
        }
    }

    /**
     * 远程数据通道处理
     * @param {RTCDataChannelEvent} event - 数据通道事件
     */
    _onDataChannel(event) {
        console.log('Remote data channel received:', event.channel.label);
        
        if (!this.dataChannel) {
            this.dataChannel = event.channel;
            this.dataChannel.onmessage = this._onDataChannelMessage;
        }
    }

    /**
     * 创建并发送SDP offer
     * @returns {Promise<void>}
     */
    async _createAndSendOffer() {
        try {
            const offer = await this.peerConnection.createOffer({
                offerToReceiveAudio: true,
                offerToReceiveVideo: true
            });
            
            await this.peerConnection.setLocalDescription(offer);

            // 等待ICE收集完成
            if (this.peerConnection.iceGatheringState !== 'complete') {
                await new Promise(resolve => {
                    const checkState = () => {
                        if (this.peerConnection.iceGatheringState === 'complete') {
                            resolve();
                        } else {
                            setTimeout(checkState, 100);
                        }
                    };
                    checkState();
                });
            }

            // 通过WebSocket发送offer
            this.signalingChannel.send(JSON.stringify({
                type: 'offer',
                client_id: this.clientId,
                offer: {
                    type: this.peerConnection.localDescription.type,
                    sdp: this.peerConnection.localDescription.sdp
                }
            }));
        } catch (error) {
            console.error('Error creating offer:', error);
            throw error;
        }
    }

    /**
     * 处理SDP answer
     * @param {Object} answer - SDP answer
     * @returns {Promise<void>}
     */
    async _handleAnswer(answer) {
        try {
            const remoteDesc = new RTCSessionDescription(answer);
            await this.peerConnection.setRemoteDescription(remoteDesc);
            console.log('Remote description set');
        } catch (error) {
            console.error('Error setting remote description:', error);
            this.config.onError(error);
        }
    }

    /**
     * 处理ICE候选
     * @param {Object} candidate - ICE候选
     * @returns {Promise<void>}
     */
    async _handleIceCandidate(candidate) {
        try {
            await this.peerConnection.addIceCandidate(new RTCIceCandidate(candidate));
            console.log('Added ICE candidate');
        } catch (error) {
            console.error('Error adding ICE candidate:', error);
            this.config.onError(error);
        }
    }

    /**
     * 获取WebRTC连接状态
     * @returns {string} 连接状态
     */
    getState() {
        return this.state;
    }

    /**
     * 获取客户端ID
     * @returns {string|null} 客户端ID
     */
    getClientId() {
        return this.clientId;
    }
}

// 如果在浏览器环境中，将类暴露给全局作用域
if (typeof window !== 'undefined') {
    window.WebRTCClient = WebRTCClient;
}

// 如果支持ES模块，导出类
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WebRTCClient;
} 
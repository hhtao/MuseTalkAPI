# MuseTalk Wrapper

MuseTalk视频生成接口封装库，提供多级降级能力和依赖隔离的数字人视频生成解决方案。

## 项目特点

- **多级降级能力**：根据环境自动选择最佳可用模式
  - 高级模式：完整使用MuseTalk的Avatar推理功能
  - 标准模式：使用简化的Avatar生成
  - 基础模式：使用ffmpeg静态帧+音频合成方案

- **依赖管理优化**：
  - 精细的依赖检测机制
  - 清晰的依赖安装指南
  - 高级依赖设为可选，确保基础功能不受影响
  - 隔离的conda环境配置文件

- **简化的API设计**：
  - 统一的核心接口
  - 自动降级处理
  - 友好的错误报告
  - 支持命令行和Web服务模式

## 安装方法

### 基础安装

```bash
# 克隆项目
git clone https://github.com/your-username/musetalk-wrapper.git
cd musetalk-wrapper

# 安装基础依赖
pip install -r musetalk_wrapper/config/requirements_basic.txt
```

### 使用Conda环境（推荐）

```bash
# 生成conda环境文件
python -m musetalk_wrapper.main generate-config

# 创建conda环境（选择适合的级别）
conda env create -f musetalk_wrapper/config/environment_basic.yml
# 或
conda env create -f musetalk_wrapper/config/environment_standard.yml
# 或
conda env create -f musetalk_wrapper/config/environment_advanced.yml

# 激活环境
conda activate musetalk_basic  # 或 musetalk_standard 或 musetalk_advanced
```

### 检查环境

```bash
# 检查当前环境
python -m musetalk_wrapper.main check-env

# 获取安装指南
python -m musetalk_wrapper.main install-guide --level advanced
```

## 使用方法

### 命令行模式

```bash
# 生成视频
python -m musetalk_wrapper.main generate \
  --text "你好，我是数字人" \
  --avatar-id avatar1 \
  --output output.mp4 \
  --subtitle

# 列出可用数字人
python -m musetalk_wrapper.main list-avatars

# 启动Web服务
python -m musetalk_wrapper.main server
```

### Python API

```python
from musetalk_wrapper import AvatarGenerator

# 初始化生成器
generator = AvatarGenerator()

# 生成视频
result = generator.generate_video(
    text="你好，我是数字人",
    avatar_id="avatar1",
    output_path="output.mp4",
    subtitle=True
)

# 检查结果
if result["success"]:
    print(f"视频生成成功：{result['path']}")
    print(f"使用模式：{result['mode']}")
else:
    print(f"视频生成失败：{result['error']}")
```

### Web服务API

启动服务器：

```bash
python -m musetalk_wrapper.main server
```

API端点：

- `POST /api/generate` - 生成视频
- `GET /api/check_status/{task_id}` - 查询任务状态
- `GET /api/avatars` - 获取可用数字人列表
- `GET /api/check_environment` - 检查系统环境
- `GET /api/installation_guide/{level}` - 获取安装指南
- `WebSocket /ws/status` - 实时进度通知

示例请求：

```bash
curl -X POST "http://localhost:8000/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，我是数字人",
    "avatar_id": "avatar1",
    "subtitle": true
  }'
```

## 依赖项

### 基础级别依赖

- Python 3.8+
- FFmpeg
- PyTorch (CPU版本即可)
- 基本的图像和音频处理库

### 标准级别依赖

- Python 3.8+
- FFmpeg
- CUDA支持的PyTorch
- Whisper模型
- 基本的MuseTalk模型文件

### 高级级别依赖

- Python 3.8+
- FFmpeg
- CUDA支持的PyTorch
- MMCV, MMPose等高级库
- 完整的MuseTalk模型文件

## 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 许可证

[MIT License](LICENSE)

## 鸣谢

- [MuseTalk](https://github.com/TMElyralab/MuseTalk) - 原始项目和技术支持
- [FFmpeg](https://ffmpeg.org/) - 视频处理核心组件 
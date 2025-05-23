import os
import sys
import subprocess
import logging
import importlib
import pkg_resources
from typing import Dict, List, Tuple, Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DependencyManager")

class DependencyManager:
    """依赖检测与管理工具，负责检查系统依赖和提供安装指南"""
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        初始化依赖管理器
        
        Args:
            base_dir: 项目基础目录，如果为None则自动检测
        """
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_dir = os.path.join(self.base_dir, "config")
        
        # 定义依赖级别和对应的包
        self.basic_dependencies = [
            ("ffmpeg", None),  # 只需要检测ffmpeg命令行，不检查版本
            ("numpy", "1.20.0"),
            ("torch", "1.10.0")
        ]
        
        self.standard_dependencies = self.basic_dependencies + [
            ("transformers", "4.20.0"),
            ("diffusers", "0.19.0"),
            ("librosa", "0.9.0")
        ]
        
        self.advanced_dependencies = self.standard_dependencies + [
            ("mmcv", "2.0.0"),
            ("mmpose", "1.1.0")
        ]
        
        # conda环境文件路径
        self.conda_env_file = {
            "basic": os.path.join(self.config_dir, "environment_basic.yml"),
            "standard": os.path.join(self.config_dir, "environment_standard.yml"),
            "advanced": os.path.join(self.config_dir, "environment_advanced.yml")
        }
        
        # 模型文件清单
        self.model_files = {
            "basic": [
                os.path.join(self.base_dir, "..", "models", "whisper", "config.json")
            ],
            "standard": [
                os.path.join(self.base_dir, "..", "models", "whisper", "config.json"),
                os.path.join(self.base_dir, "..", "models", "sd-vae", "config.json")
            ],
            "advanced": [
                os.path.join(self.base_dir, "..", "models", "whisper", "config.json"),
                os.path.join(self.base_dir, "..", "models", "sd-vae", "config.json"),
                os.path.join(self.base_dir, "..", "models", "musetalkV15", "unet.pth"),
                os.path.join(self.base_dir, "..", "models", "dwpose", "dw-ll_ucoco_384.pth")
            ]
        }
    
    def check_ffmpeg(self) -> bool:
        """检查ffmpeg是否可用且支持libx264编码器"""
        try:
            # 首先检查系统ffmpeg是否可用
            result = subprocess.run(
                ["/usr/bin/ffmpeg", "-version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=False
            )
            if result.returncode != 0:
                logger.warning("系统FFmpeg检测失败")
                return False
            
            # 然后检查是否支持libx264编码器
            encoders_result = subprocess.run(
                ["/usr/bin/ffmpeg", "-encoders"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if "libx264" not in encoders_result.stdout:
                logger.warning("系统FFmpeg不支持libx264编码器")
                return False
            
            logger.info("系统FFmpeg可用且支持libx264编码器")
            return True
        except Exception as e:
            logger.warning(f"检测ffmpeg时发生错误: {str(e)}")
            return False
    
    def check_cuda(self) -> Tuple[bool, Optional[str]]:
        """
        检查CUDA是否可用
        
        Returns:
            (可用状态, 版本信息)
        """
        try:
            import torch
            if torch.cuda.is_available():
                version = torch.version.cuda
                return True, version
            return False, None
        except Exception as e:
            logger.warning(f"检测CUDA时发生错误: {str(e)}")
            return False, None
    
    def check_package(self, package_name: str, min_version: Optional[str] = None) -> bool:
        """
        检查Python包是否已安装并且版本符合要求
        
        Args:
            package_name: 包名称
            min_version: 最低版本要求，如果为None则只检查是否安装
            
        Returns:
            是否满足要求
        """
        # 对ffmpeg特殊处理
        if package_name == "ffmpeg":
            return self.check_ffmpeg()
            
        try:
            # 尝试导入包
            module = importlib.import_module(package_name)
            
            # 如果不需要检查版本，则返回True
            if min_version is None:
                return True
            
            # 获取包版本
            try:
                package = pkg_resources.get_distribution(package_name)
                installed_version = package.version
                
                # 比较版本
                import pkg_resources.extern.packaging.version as version
                if version.parse(installed_version) >= version.parse(min_version):
                    return True
                else:
                    logger.warning(f"包 {package_name} 版本过低: {installed_version} < {min_version}")
                    return False
            except Exception:
                # 如果无法获取版本，则假设版本满足要求
                logger.warning(f"无法获取包 {package_name} 的版本信息")
                return True
                
        except ImportError:
            logger.warning(f"包 {package_name} 未安装")
            return False
        except Exception as e:
            logger.warning(f"检测包 {package_name} 时发生错误: {str(e)}")
            return False
    
    def check_model_files(self, level: str = "basic") -> bool:
        """
        检查指定级别所需的模型文件是否存在
        
        Args:
            level: 功能级别 (basic/standard/advanced)
            
        Returns:
            是否所有文件都存在
        """
        if level not in self.model_files:
            logger.warning(f"未知的功能级别: {level}")
            return False
        
        all_exist = True
        for file_path in self.model_files[level]:
            if not os.path.exists(file_path):
                logger.warning(f"模型文件不存在: {file_path}")
                all_exist = False
        
        return all_exist
    
    def check_all_dependencies(self, level: str = "basic") -> Dict[str, bool]:
        """
        检查指定级别的所有依赖
        
        Args:
            level: 功能级别 (basic/standard/advanced)
            
        Returns:
            依赖检查结果字典
        """
        results = {
            "ffmpeg": self.check_ffmpeg(),
            "cuda": self.check_cuda()[0],
            "model_files": self.check_model_files(level)
        }
        
        # 检查Python包依赖
        if level == "basic":
            dependencies = self.basic_dependencies
        elif level == "standard":
            dependencies = self.standard_dependencies
        elif level == "advanced":
            dependencies = self.advanced_dependencies
        else:
            logger.warning(f"未知的功能级别: {level}")
            dependencies = []
        
        for package, min_version in dependencies:
            results[package] = self.check_package(package, min_version)
        
        return results
    
    def get_installation_guide(self, level: str = "basic") -> str:
        """
        获取安装指南，包含缺失的依赖安装说明
        
        Args:
            level: 功能级别 (basic/standard/advanced)
            
        Returns:
            安装指南文本
        """
        check_results = self.check_all_dependencies(level)
        
        # 收集缺失的依赖
        missing_deps = []
        for name, status in check_results.items():
            if not status:
                missing_deps.append(name)
        
        if not missing_deps:
            return "所有依赖项已正确安装。"
        
        # 生成安装指南
        guide = f"### {level.capitalize()} 级别功能需要安装以下缺失依赖:\n\n"
        
        # CUDA安装指南
        if "cuda" in missing_deps:
            guide += "#### CUDA 安装:\n"
            guide += "请访问 NVIDIA 官方网站 (https://developer.nvidia.com/cuda-downloads) 下载并安装适合您系统的 CUDA 工具包。\n\n"
        
        # ffmpeg安装指南
        if "ffmpeg" in missing_deps:
            guide += "#### FFmpeg 安装:\n"
            
            # 根据系统提供不同的安装指南
            if sys.platform == "win32":
                guide += "Windows: 下载预编译版本 (https://ffmpeg.org/download.html) 并将bin目录添加到系统PATH环境变量。\n"
            elif sys.platform == "darwin":
                guide += "macOS: 使用Homebrew安装: `brew install ffmpeg`\n"
            else:
                guide += "Linux: 使用包管理器安装，例如 Ubuntu: `sudo apt update && sudo apt install ffmpeg`\n"
            
            guide += "\n"
        
        # 模型文件安装指南
        if "model_files" in missing_deps:
            guide += "#### 模型文件下载:\n"
            guide += "请运行以下命令下载必要的模型文件:\n"
            
            if sys.platform == "win32":
                guide += "```\ndownload_weights.bat\n```\n"
            else:
                guide += "```\n./download_weights.sh\n```\n"
            
            guide += "\n"
        
        # Python包安装指南
        python_deps = [dep for dep in missing_deps if dep not in ["cuda", "ffmpeg", "model_files"]]
        if python_deps:
            guide += "#### Python 包安装:\n"
            guide += "请运行以下命令安装缺失的Python包:\n"
            
            if level == "basic":
                guide += "```\npip install -r requirements_basic.txt\n```\n"
            elif level == "standard":
                guide += "```\npip install -r requirements_standard.txt\n```\n"
            else:
                guide += "```\npip install -r requirements_advanced.txt\n```\n"
            
            guide += "\n或者使用conda创建新环境:\n"
            guide += f"```\nconda env create -f {os.path.basename(self.conda_env_file[level])}\n```\n"
        
        guide += "\n完成上述安装后，请重新运行以验证所有依赖是否已正确安装。"
        
        return guide
    
    def generate_conda_env_file(self, level: str = "basic") -> str:
        """
        生成conda环境配置文件
        
        Args:
            level: 功能级别 (basic/standard/advanced)
            
        Returns:
            生成的配置文件路径
        """
        if level not in self.conda_env_file:
            logger.warning(f"未知的功能级别: {level}")
            return ""
        
        # 根据级别选择依赖列表
        if level == "basic":
            dependencies = self.basic_dependencies
        elif level == "standard":
            dependencies = self.standard_dependencies
        else:
            dependencies = self.advanced_dependencies
        
        # 生成conda环境文件内容
        content = f"""name: musetalk_{level}
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip=22.3
"""
        
        # 添加核心依赖
        if level == "advanced":
            content += "  - pytorch=2.0.1\n"
            content += "  - torchvision=0.15.2\n"
            content += "  - torchaudio=2.0.2\n"
            content += "  - pytorch-cuda=11.8\n"
        elif level == "standard":
            content += "  - pytorch=2.0.1\n"
            content += "  - torchvision=0.15.2\n"
            content += "  - torchaudio=2.0.2\n"
            content += "  - pytorch-cuda=11.8\n"
        else:
            content += "  - pytorch=1.10.0\n"
            content += "  - cpuonly\n"
        
        # 添加pip安装的包
        content += "  - pip:\n"
        for package, version in dependencies:
            # 跳过已作为conda包添加的
            if package in ["torch", "torchvision", "torchaudio"]:
                continue
            content += f"    - {package}>={version}\n"
        
        # 添加特殊安装的包
        if level == "advanced":
            content += "    - openmim\n"
            content += "    - mmengine\n"
            content += "    - mmcv==2.0.1\n"
            content += "    - mmdet==3.1.0\n"
            content += "    - mmpose==1.1.0\n"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.conda_env_file[level]), exist_ok=True)
        
        # 写入文件
        with open(self.conda_env_file[level], 'w', encoding='utf-8') as f:
            f.write(content)
        
        return self.conda_env_file[level]
    
    def generate_requirements_files(self) -> List[str]:
        """
        生成不同功能级别的requirements.txt文件
        
        Returns:
            生成的文件路径列表
        """
        file_paths = []
        
        # 生成基础级别requirements
        basic_path = os.path.join(self.base_dir, "config", "requirements_basic.txt")
        with open(basic_path, 'w', encoding='utf-8') as f:
            f.write("# 基础功能依赖包\n")
            for package, version in self.basic_dependencies:
                f.write(f"{package}>={version}\n")
        file_paths.append(basic_path)
        
        # 生成标准级别requirements
        standard_path = os.path.join(self.base_dir, "config", "requirements_standard.txt")
        with open(standard_path, 'w', encoding='utf-8') as f:
            f.write("# 标准功能依赖包\n")
            for package, version in self.standard_dependencies:
                f.write(f"{package}>={version}\n")
        file_paths.append(standard_path)
        
        # 生成高级级别requirements
        advanced_path = os.path.join(self.base_dir, "config", "requirements_advanced.txt")
        with open(advanced_path, 'w', encoding='utf-8') as f:
            f.write("# 高级功能依赖包\n")
            for package, version in self.advanced_dependencies:
                f.write(f"{package}>={version}\n")
            
            # 添加特殊安装的包
            f.write("\n# 以下包需要特殊安装方式\n")
            f.write("# 请执行以下命令:\n")
            f.write("# pip install --no-cache-dir -U openmim\n")
            f.write("# mim install mmengine\n")
            f.write("# mim install \"mmcv==2.0.1\"\n")
            f.write("# mim install \"mmdet==3.1.0\"\n")
            f.write("# mim install \"mmpose==1.1.0\"\n")
        file_paths.append(advanced_path)
        
        return file_paths 

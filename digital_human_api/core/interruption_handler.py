import asyncio
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Callable
import difflib

from digital_human_api.core.state_manager import StateManager, DigitalHumanState

class InterruptionClassifier:
    """打断意图分类器，用于区分确认性打断和纠正性打断"""
    
    def __init__(self):
        """初始化打断意图分类器"""
        # 确认性打断关键词
        self.confirmation_keywords = {
            "好的", "嗯", "是的", "对", "知道了", "明白", "继续", 
            "没错", "理解", "懂了", "可以", "行", "有道理"
        }
        
        # 纠正性打断关键词
        self.correction_keywords = {
            "等等", "不对", "错了", "不是", "停", "等一下", "换一个", 
            "我想问", "我的意思是", "不是这样", "你误会了", "不是这个"
        }
        
        self.logger = logging.getLogger("InterruptionClassifier")
        
    def add_confirmation_keyword(self, keyword: str) -> None:
        """
        添加确认性打断关键词
        
        Args:
            keyword: 关键词
        """
        self.confirmation_keywords.add(keyword)
        
    def add_correction_keyword(self, keyword: str) -> None:
        """
        添加纠正性打断关键词
        
        Args:
            keyword: 关键词
        """
        self.correction_keywords.add(keyword)
        
    async def classify_interruption(self, text: str, context: str = "") -> str:
        """
        分类打断意图
        
        Args:
            text: 打断文本
            context: 当前回答上下文
            
        Returns:
            str: "CONFIRMATION" 或 "CORRECTION"
        """
        # 转为小写便于匹配
        text_lower = text.lower()
        
        # 关键词匹配
        for keyword in self.confirmation_keywords:
            if keyword.lower() in text_lower:
                self.logger.info(f"检测到确认性打断关键词: {keyword}")
                return "CONFIRMATION"
                
        for keyword in self.correction_keywords:
            if keyword.lower() in text_lower:
                self.logger.info(f"检测到纠正性打断关键词: {keyword}")
                return "CORRECTION"
        
        # 语义相似度分析（简化版）
        similarity = self.calculate_semantic_similarity(text, context)
        
        if similarity > 0.7:
            self.logger.info(f"基于语义相似度({similarity})判定为确认性打断")
            return "CONFIRMATION"
        elif similarity < 0.3:
            self.logger.info(f"基于语义相似度({similarity})判定为纠正性打断")
            return "CORRECTION"
            
        # 默认处理为纠正性打断
        self.logger.info("无法确定打断类型，默认处理为纠正性打断")
        return "CORRECTION"
        
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本语义相似度(简化版)
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            float: 相似度(0.0-1.0)
        """
        # 实际项目中应该使用更高级的语义相似度算法
        # 这里简化为关键词重叠率
        if not text2:
            return 0.3  # 没有上下文时返回中等相似度
            
        # 分词（简化为字符级别分词）
        words1 = set(text1)
        words2 = set(text2)
        
        # 计算重叠率
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))
        
        if total == 0:
            return 0.0
            
        similarity = overlap / total
        return similarity


class ResponseSegmenter:
    """回复分段器，将长文本回复分割为逻辑段落"""
    
    def __init__(self, 
                max_segment_length: int = 100,
                min_segment_length: int = 20):
        """
        初始化回复分段器
        
        Args:
            max_segment_length: 最大段落长度
            min_segment_length: 最小段落长度
        """
        self.max_segment_length = max_segment_length
        self.min_segment_length = min_segment_length
        self.logger = logging.getLogger("ResponseSegmenter")
        
    async def segment_response(self, text: str) -> List[str]:
        """
        将文本分段
        
        Args:
            text: 要分段的文本
            
        Returns:
            List[str]: 分段后的文本列表
        """
        if not text:
            return []
            
        # 如果文本长度小于最小段落长度，直接返回
        if len(text) <= self.min_segment_length:
            return [text]
            
        # 按句子分割
        sentences = self._split_into_sentences(text)
        
        # 合并句子成段落
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            # 如果当前段落加上新句子超过最大长度，开始新段落
            if len(current_segment) + len(sentence) > self.max_segment_length and len(current_segment) >= self.min_segment_length:
                segments.append(current_segment.strip())
                current_segment = sentence
            else:
                current_segment += sentence
                
        # 添加最后一个段落
        if current_segment:
            segments.append(current_segment.strip())
            
        self.logger.info(f"文本被分割为 {len(segments)} 个段落")
        return segments
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        将文本分割为句子
        
        Args:
            text: 要分割的文本
            
        Returns:
            List[str]: 句子列表
        """
        # 使用正则表达式按句子分割
        # 中文句子结束符：。！？；
        # 英文句子结束符：.!?;
        sentence_endings = r'(?<=[。！？；.!?;])'
        sentences = re.split(sentence_endings, text)
        
        # 处理结果，确保句子包含标点
        result = []
        start = 0
        for sentence in sentences:
            if not sentence:
                continue
                
            # 找到原文中的结束位置
            end = text.find(sentence, start) + len(sentence)
            if end < len(text):
                # 确保包含标点
                sentence += text[end]
                
            result.append(sentence)
            start = end + 1
            
        return result


class InterruptionHandler:
    """打断处理器，处理用户在数字人说话过程中的插话"""
    
    def __init__(self, state_manager: StateManager):
        """
        初始化打断处理器
        
        Args:
            state_manager: 状态管理器
        """
        self.state_manager = state_manager
        self.classifier = InterruptionClassifier()
        self.segmenter = ResponseSegmenter()
        
        # 当前回复信息
        self.current_response = ""
        self.response_segments = []
        self.current_segment_index = 0
        
        # 回调函数
        self.confirmation_callback = None
        self.correction_callback = None
        
        self.logger = logging.getLogger("InterruptionHandler")
        
    async def prepare_response(self, response: str) -> None:
        """
        准备回复内容，分段处理
        
        Args:
            response: 完整的回复内容
        """
        self.current_response = response
        self.response_segments = await self.segmenter.segment_response(response)
        self.current_segment_index = 0
        self.logger.info(f"准备回复内容，共 {len(self.response_segments)} 个段落")
        
    async def get_next_segment(self) -> str:
        """
        获取下一个回复段落
        
        Returns:
            str: 下一个段落，如果没有更多段落则返回空字符串
        """
        if not self.response_segments or self.current_segment_index >= len(self.response_segments):
            return ""
            
        segment = self.response_segments[self.current_segment_index]
        self.current_segment_index += 1
        return segment
        
    async def is_response_complete(self) -> bool:
        """
        检查回复是否已完成
        
        Returns:
            bool: 是否已完成
        """
        return not self.response_segments or self.current_segment_index >= len(self.response_segments)
        
    async def handle_interruption(self, text: str) -> None:
        """
        处理打断
        
        Args:
            text: 打断内容
        """
        self.logger.info(f"处理打断: {text}")
        
        # 获取当前上下文
        context = ""
        if self.current_segment_index > 0 and self.response_segments:
            # 使用已经播放的段落作为上下文
            context = " ".join(self.response_segments[:self.current_segment_index])
            
        # 分类打断意图
        interruption_type = await self.classifier.classify_interruption(text, context)
        
        # 根据打断类型处理
        if interruption_type == "CONFIRMATION":
            await self.handle_confirmation_interruption()
        else:  # CORRECTION
            await self.handle_correction_interruption(text)
            
    async def handle_confirmation_interruption(self) -> None:
        """处理确认性打断"""
        self.logger.info("处理确认性打断")
        
        # 调用确认性打断回调
        if self.confirmation_callback:
            await self.confirmation_callback()
            
        # 回到说话状态继续回答
        await self.state_manager.set_state(DigitalHumanState.SPEAKING)
        
    async def handle_correction_interruption(self, user_text: str) -> None:
        """
        处理纠正性打断
        
        Args:
            user_text: 用户纠正内容
        """
        self.logger.info(f"处理纠正性打断: {user_text}")
        
        # 清空当前回复
        self.current_response = ""
        self.response_segments = []
        self.current_segment_index = 0
        
        # 保存用户输入到状态上下文
        await self.state_manager.save_state_context({"user_input": user_text})
        
        # 调用纠正性打断回调
        if self.correction_callback:
            await self.correction_callback(user_text)
            
        # 切换到思考状态处理新问题
        await self.state_manager.set_state(DigitalHumanState.THINKING)
        
    async def register_callbacks(self, 
                               confirmation_callback: Callable, 
                               correction_callback: Callable) -> None:
        """
        注册打断回调函数
        
        Args:
            confirmation_callback: 确认性打断回调
            correction_callback: 纠正性打断回调
        """
        self.confirmation_callback = confirmation_callback
        self.correction_callback = correction_callback
        self.logger.info("打断回调函数已注册")
        
    def get_breakpoint_position(self) -> int:
        """
        获取当前断点位置
        
        Returns:
            int: 当前段落索引
        """
        return self.current_segment_index
        
    async def resume_from_breakpoint(self, position: int = None) -> None:
        """
        从断点恢复
        
        Args:
            position: 恢复位置，默认为当前位置
        """
        if position is not None:
            self.current_segment_index = max(0, min(position, len(self.response_segments) if self.response_segments else 0))
            
        self.logger.info(f"从断点恢复，位置: {self.current_segment_index}")
        
        # 回到说话状态继续回答
        await self.state_manager.set_state(DigitalHumanState.SPEAKING) 
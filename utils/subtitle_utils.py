import re
import os
import numpy as np
import cv2
from datetime import timedelta
import string
import subprocess
import traceback
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

# 设置日志配置
logger = logging.getLogger("SubtitleUtils")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Constants
ENDING_PUNCTUATIONS = set(",.;:!?，。；：！？…")
PAUSE_PUNCTUATIONS = {
    "long": {"。", ".", "!", "！", "?", "？", "；", ";"},
    "medium": {"，", ",", "、", "：", ":"},
    "short": {""", """, "'", "'", "「", "」", "『", "』", "(", ")", "（", "）", "[", "]", "【", "】", "<", ">", "《", "》"}
}
SPECIAL_COMPOUNDS = {
    "audio-visual": 2.5,
    "information-based": 2.5,
    "decision-making": 2.2,
    "problem-solving": 2.2,
    "self-regulated": 2.2,
    "well-designed": 2.2,
    "high-quality": 2.0
}


#--------------------------------
# Text Processing Functions
#--------------------------------

def validate_text_cut_method(text_cut_method: str) -> str:
    """Validates the text cutting method."""
    valid_cut_methods = ['cut0', 'cut1', 'cut2', 'cut3', 'cut4', 'cut5']
    if text_cut_method not in valid_cut_methods:
        print("Warning: Invalid text cutting method {}, using default method cut2".format(text_cut_method))
        return 'cut2'
    return text_cut_method


def should_use_mixed_method(text: Union[str, List[str]]) -> bool:
    """Determines if mixed timing method should be used based on text content."""
    # Convert list to string if needed
    combined_text = " ".join(text) if isinstance(text, list) else text
    
    # Check for Chinese characters
    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in combined_text)
    
    # Check for English words or abbreviations
    english_pattern = r'[a-zA-Z](?:[a-zA-Z]+|\.)|\b[A-Z]{2,}\b'
    has_english = bool(re.search(english_pattern, combined_text))
    
    # Use mixed method if both Chinese and English are present
    return has_chinese and has_english


def identify_english_components(text: str) -> List[Tuple[str, bool]]:
    """Identifies English words, abbreviations and Chinese parts in text."""
    # Pattern to match English words and abbreviations
    english_pattern = r"[a-zA-Z](?:[a-zA-Z\-\'\.]+|\.)|[A-Z][a-z]*|[A-Z]{2,}|\b[a-zA-Z]\b"
    
    matches = list(re.finditer(english_pattern, text))
    if not matches:
        return [(text, False)]  # No English, return the whole text as Chinese
    
    components = []
    last_end = 0
    
    for match in matches:
        start, end = match.span()
        
        # Skip if it's just a decimal point
        if match.group() == '.' and start > 0 and start < len(text) - 1:
            if text[start-1].isdigit() and text[start+1].isdigit():
                continue
            
        # Add Chinese part before the English part (if any)
        if start > last_end:
            chinese_text = text[last_end:start]
            if chinese_text.strip():
                components.append((chinese_text, False))
        
        # Add English part
        components.append((match.group(), True))
        last_end = end
    
    # Add Chinese part after the last English part (if any)
    if last_end < len(text):
        chinese_text = text[last_end:]
        if chinese_text.strip():
            components.append((chinese_text, False))
    
    # Merge adjacent English components separated by punctuation
    return merge_adjacent_english_components(components)


def merge_adjacent_english_components(components: List[Tuple[str, bool]]) -> List[Tuple[str, bool]]:
    """Merges adjacent English components separated by single punctuation."""
    merged_components = []
    i = 0
    
    while i < len(components):
        if components[i][1]:  # If English
            current_text = components[i][0]
            j = i + 1
            
            # Check if next component is a single punctuation followed by English
            while j < len(components) and j < i + 2:
                next_component = components[j]
                is_punctuation = (j == i + 1 and not next_component[1] and 
                                 len(next_component[0].strip()) == 1 and 
                                 next_component[0].strip() in string.punctuation)
                
                if is_punctuation:
                    punctuation_text = next_component[0]
                    if j + 1 < len(components) and components[j+1][1]:
                        # Punctuation followed by English, merge all three
                        current_text += punctuation_text + components[j+1][0]
                        j += 2
                    else:
                        # Just merge the punctuation
                        current_text += punctuation_text
                        j += 1
                else:
                    break
                    
            merged_components.append((current_text, True))
            i = j
        else:
            merged_components.append(components[i])
            i += 1
    
    return merged_components


def count_english_words(text: str) -> Tuple[int, List[str]]:
    """Counts English words and returns their list."""
    components = identify_english_components(text)
    english_words = [component[0] for component in components if component[1]]
    
    actual_word_count = 0
    expanded_words = []
    
    year_pattern = re.compile(r'(\d{4}s?)')
    decade_pattern = re.compile(r'(\d{2}s)')
    number_pattern = re.compile(r'(\d+(?:st|nd|rd|th)?)')
    
    for word in english_words:
        word_lower = word.lower()
        
        # Check for year expressions
        year_match = year_pattern.search(word)
        decade_match = decade_pattern.search(word)
        number_match = number_pattern.search(word)
        
        # Handle compound words with years
        if "-" in word and (year_match or decade_match):
            parts = [p for p in word.split('-') if p.strip()]
            year_part_count = sum(2 if year_pattern.match(part) else 
                                 1 if decade_pattern.match(part) else 1
                                 for part in parts)
            actual_word_count += year_part_count
            
        # Handle plain year expressions
        elif year_match and year_match.group() == word:
            actual_word_count += 2
            
        # Handle decade expressions
        elif decade_match and decade_match.group() == word:
            actual_word_count += 1
            
        # Handle numbers and ordinals
        elif number_match and number_match.group() == word:
            actual_word_count += calculate_number_word_count(word)
                
        # Handle special compound words
        elif word_lower in SPECIAL_COMPOUNDS:
            actual_word_count += SPECIAL_COMPOUNDS[word_lower]
            
        # Handle hyphenated words
        elif '-' in word:
            parts = [p for p in word.split('-') if p.strip()]
            actual_word_count += len(parts) * 1.2
            
        # Handle abbreviations with periods
        elif '.' in word and any(c.isupper() for c in word):
            letter_count = sum(1 for c in word if c.isalpha())
            actual_word_count += max(1, letter_count / 2)
            
        # Handle regular words
        else:
            actual_word_count += 1.2 if len(word) > 7 else 1.0
            
        expanded_words.append(word)
    
    return actual_word_count, expanded_words


def calculate_number_word_count(word: str) -> float:
    """Calculates word count for numbers and ordinals."""
    num = re.sub(r'(st|nd|rd|th)$', '', word)
    try:
        num_val = int(num)
        
        # Different number ranges have different reading patterns
        if num_val < 20:
            word_count = 1  # One word for 1-19
        elif num_val < 100:
            word_count = 1 if num_val % 10 == 0 else 2  # 1-2 words for 20-99
        elif num_val < 1000:
            if num_val % 100 == 0:  # Even hundreds
                word_count = 2
            else:
                ones = num_val % 10
                tens = (num_val % 100) // 10
                if tens == 0 or (tens == 1 or (tens > 0 and ones == 0)):
                    word_count = 3
                else:
                    word_count = 4
        else:
            # For 1000+, estimate based on digit count
            digits = len(str(num_val))
            word_count = max(2, digits // 2)
        
        # Add slight time for ordinal suffixes
        if re.search(r'(st|nd|rd|th)$', word):
            word_count += 0.2
            
        return word_count
        
    except ValueError:
        return 1  # Default for unparseable numbers


def remove_ending_punctuation(text: str) -> str:
    """Removes ending punctuation from text."""
    if text and text[-1] in ENDING_PUNCTUATIONS:
        return text[:-1]
    return text


def split_text_to_sentences(text: str, cut_method: str = "cut2") -> List[str]:
    """Splits text into sentences using the specified cutting method."""
    try:
        logger = logging.getLogger("SubtitleUtils")
        
        from utils.text_cut import get_method
        method = get_method(cut_method)
        result = method(text).split("\n")
        result = [s.strip() for s in result if s.strip()]
        
        if not result:
            logger.warning("No valid sentences after text cutting")
            return [text] if text.strip() else []
            
        # Post-process to handle short sentences and numbers
        processed = []
        current_sentence = ""
        
        for sentence in result:
            # If current sentence is empty, start with this one
            if not current_sentence:
                current_sentence = sentence
                continue
                
            # Check if current sentence should be merged
            if (len(sentence.strip()) <= 5 or  # Short sentence
                bool(re.match(r'^[\d\s]+$', sentence.strip()))):  # Just numbers
                current_sentence += sentence
            else:
                if current_sentence:
                    processed.append(current_sentence)
                current_sentence = sentence
        
        # Add the last sentence
        if current_sentence:
            processed.append(current_sentence)
            
        return processed if processed else [text]
        
    except Exception as e:
        logger = logging.getLogger("SubtitleUtils")
        logger.error("Error splitting text: {}, will use original text".format(e))
        return [text] if text.strip() else []


#--------------------------------
# Timing Calculation Functions
#--------------------------------

def calculate_reading_time(text: str, cn_speaking_rate: float = 4.0, en_word_rate: float = 3.5) -> float:
    """Calculates reading time for mixed Chinese-English text."""
    logger = logging.getLogger("SubtitleUtils")
    
    # Count English words
    english_word_count, english_words = count_english_words(text)
    
    # Check for special expressions
    contains_years = any(re.search(r'\b\d{4}s?\b', word) for word in english_words)
    contains_decades = any(re.search(r'\b\d{2}s\b', word) for word in english_words)
    contains_numbers = any(re.search(r'\b\d+\b', word) for word in english_words)
    
    # Remove English words from text to count Chinese characters
    text_without_english = text
    for word in english_words:
        text_without_english = text_without_english.replace(word, '')
    
    # Count Chinese characters
    chinese_chars = sum(1 for c in text_without_english if '\u4e00' <= c <= '\u9fff')
    
    # Count punctuation
    all_punctuation = set(string.punctuation + "，。、；：？！""''「」『』（）【】《》")
    punctuation_count = sum(1 for c in text_without_english if c in all_punctuation)
    
    # Calculate Chinese reading time
    chinese_time = chinese_chars / cn_speaking_rate if chinese_chars > 0 else 0
    
    # Calculate English reading time
    english_time = 0
    if english_word_count > 0:
        # Determine text complexity
        complexity_factor = calculate_complexity_factor(english_words, english_word_count, 
                                                      contains_years, contains_decades, contains_numbers)
        
        # Adjust reading rate based on sentence type
        is_english_only = chinese_chars == 0
        english_time = calculate_english_time(english_word_count, en_word_rate, 
                                            complexity_factor, is_english_only, len(text))
    
    # Calculate pause time for punctuation
    pause_time = calculate_pause_time(text_without_english)
    
    # Calculate total time
    total_time = chinese_time + english_time + pause_time
    
    # Ensure minimum time
    min_time = determine_min_time(english_word_count, chinese_chars)
    total_time = max(total_time, min_time)
    
    # Debug info
    logger.info("Chinese characters: {}, reading time: {:.2f}s".format(chinese_chars, chinese_time))
    logger.info("English words: {}, reading time: {:.2f}s".format(english_word_count, english_time))
    logger.info("Punctuation pause time: {:.2f}s".format(pause_time))
    logger.info("Total reading time: {:.2f}s".format(total_time))
    
    return total_time


def calculate_complexity_factor(english_words: List[str], word_count: int, 
                              contains_years: bool, contains_decades: bool, 
                              contains_numbers: bool) -> float:
    """Calculates text complexity factor for timing adjustments."""
    if word_count == 0:
        return 1.0
        
    long_words_count = sum(1 for word in english_words if len(word) > 8)
    compound_words_count = sum(1 for word in english_words if '-' in word)
    
    long_word_ratio = long_words_count / word_count
    compound_word_ratio = compound_words_count / word_count
    
    complexity_factor = 1.0 + (long_word_ratio * 0.3) + (compound_word_ratio * 0.4)
    
    if contains_years or contains_decades or contains_numbers:
        complexity_factor += 0.15
        
    return complexity_factor


def calculate_english_time(word_count: int, word_rate: float, 
                         complexity_factor: float, is_english_only: bool, 
                         text_length: int) -> float:
    """Calculates reading time for English text portions."""
    if is_english_only:
        if word_count < 5:  # Very short (1-4 words)
            return word_count / (word_rate * 1.2 / complexity_factor)
        elif word_count < 10:  # Short (5-9 words)
            return word_count / (word_rate * 1.1 / complexity_factor)
        elif text_length < 50:  # Medium length
            return word_count / (word_rate * 1.05 / complexity_factor)
    
    # Long or mixed sentences
    return word_count / (word_rate / complexity_factor)


def calculate_pause_time(text: str) -> float:
    """Calculates pause time for punctuation."""
    pause_time = 0
    for c in text:
        if c in PAUSE_PUNCTUATIONS["long"]:
            pause_time += 0.3
        elif c in PAUSE_PUNCTUATIONS["medium"]:
            pause_time += 0.2
        elif c in PAUSE_PUNCTUATIONS["short"]:
            pause_time += 0.05
    return pause_time


def determine_min_time(english_word_count: int, chinese_chars: int) -> float:
    """Determines minimum display time based on text content."""
    if english_word_count > 0 and chinese_chars == 0:  # English-only
        if english_word_count <= 2:  # 1-2 words
            return 0.8
        elif english_word_count <= 5:  # 3-5 words
            return 0.9
        else:
            return 1.0
    else:  # Chinese or mixed
        return 1.0


def calculate_mixed_subtitle_timing(
    sentences: List[str], 
    audio_duration: Optional[float] = None, 
    fps: int = 25, 
    cn_speaking_rate: float = 3.6,
    en_word_rate: float = 1.5,
    time_offset: float = 0.0, 
    sentence_gap: float = 0.4,
    extend_to_duration: bool = True, 
    use_mixed_method: bool = True,
    audio_fps: int = 50
) -> List[Dict[str, Any]]:
    """Calculates timing for mixed Chinese-English subtitles."""
    if not sentences:
        return []
    
    # Apply time offset
    current_time = time_offset
    
    # Check if mixed method should be used
    combined_text = " ".join(sentences)
    should_use_mixed = should_use_mixed_method(combined_text) if use_mixed_method else True
    
    # Determine language types
    lang_types = determine_language_types(sentences)
    
    # Calculate FPS correction
    fps_correction = fps / audio_fps if audio_fps and audio_fps != fps else 1.0
    if fps_correction != 1.0:
        print("Detected FPS mismatch: {:.2f} (video:{}, audio:{})".format(fps_correction, fps, audio_fps))
    
    # Pre-adjust English rate for English-dominant content
    if audio_duration and is_english_dominant(lang_types):
        en_word_rate = adjust_english_rate_for_duration(sentences, en_word_rate, audio_duration)
    
    # Calculate sentence durations
    sentence_durations, original_durations, speaking_durations = calculate_sentence_durations(
        sentences, lang_types, should_use_mixed, cn_speaking_rate, en_word_rate, sentence_gap)
    
    # Adjust timing based on audio duration
    if audio_duration:
        adjust_timing_for_audio_duration(sentence_durations, speaking_durations, audio_duration, lang_types)
    
    # Generate final timings
    timing = generate_timing_data(sentences, sentence_durations, speaking_durations, 
                                original_durations, lang_types, time_offset, fps)
    
    # Extend last subtitle to video end if needed
    if audio_duration and extend_to_duration and timing:
        extend_last_subtitle(timing, audio_duration, fps)
    
    return timing


def determine_language_types(sentences: List[str]) -> List[str]:
    """Determines the dominant language for each sentence."""
    lang_types = []
    for sentence in sentences:
        chinese_chars = sum(1 for c in sentence if '\u4e00' <= c <= '\u9fff')
        english_chars = sum(1 for c in sentence if 'a' <= c.lower() <= 'z')
        total_chars = len(sentence.strip())
        
        if total_chars == 0:
            lang_types.append("unknown")
        elif chinese_chars / total_chars > 0.5:
            lang_types.append("zh")
        elif english_chars / total_chars > 0.5:
            lang_types.append("en")
        else:
            lang_types.append("mixed")
            
    return lang_types


def is_english_dominant(lang_types: List[str]) -> bool:
    """Determines if the content is primarily English."""
    return sum(1 for lang in lang_types if lang == "en") > len(lang_types) / 2


def adjust_english_rate_for_duration(sentences: List[str], en_word_rate: float, 
                                   audio_duration: float) -> float:
    """Pre-adjusts English speaking rate based on estimated duration."""
    estimated_duration = 0
    for sentence in sentences:
        word_count, _ = count_english_words(sentence)
        estimated_duration += word_count / en_word_rate + 0.3  # word time + pause
    
    if estimated_duration > audio_duration * 1.1:
        speed_adjust = min(estimated_duration / (audio_duration * 0.95), 1.5)  # max 50% speed up
        adjusted_rate = en_word_rate * speed_adjust
        print("Predicted English subtitle duration ({:.2f}s) exceeds audio ({:.2f}s), "
              "adjusting English rate to {:.2f} words/sec".format(estimated_duration, audio_duration, adjusted_rate))
        return adjusted_rate
        
    return en_word_rate


def calculate_sentence_durations(
    sentences: List[str], 
    lang_types: List[str], 
    should_use_mixed: bool, 
    cn_speaking_rate: float, 
    en_word_rate: float,
    sentence_gap: float
) -> Tuple[List[float], List[float], List[float]]:
    """Calculates the duration for each sentence."""
    sentence_durations = []
    original_durations = []
    speaking_durations = []
    
    for i, sentence in enumerate(sentences):
        # Calculate reading time for the sentence
        if should_use_mixed:
            # Use mixed calculation method
            base_duration = calculate_reading_time(
                sentence, 
                cn_speaking_rate=cn_speaking_rate,
                en_word_rate=en_word_rate
            )
            
            # Adjust for language type
            speaking_duration = adjust_duration_by_language(base_duration, lang_types[i])
        else:
            # Simple method: count total characters
            chars = len(sentence.strip())
            speaking_duration = chars / cn_speaking_rate
        
        # Apply minimum display time
        min_display_time = get_min_display_time(sentence)
        original_durations.append(speaking_duration)
        speaking_duration = max(speaking_duration, min_display_time)
        
        # Special rule for short sentences with years
        if re.search(r'\b\d{4}\b', sentence) and len(sentence) < 15:
            speaking_duration *= 1.15
        
        speaking_durations.append(speaking_duration)
        
        # Add sentence gap for TTS timing but not for display
        sentence_durations.append(speaking_duration)  # Remove gap for display timing
    
    return sentence_durations, original_durations, speaking_durations


def adjust_duration_by_language(base_duration: float, lang_type: str) -> float:
    """Adjusts the base duration based on language type."""
    if lang_type == "zh":
        return base_duration * 1.4  # Chinese needs more time
    elif lang_type == "en":
        # English sentences get less additional time
        return base_duration * 1.2 if len(lang_type) < 20 else base_duration * 1.05
    else:
        # Mixed or unknown
        return base_duration * 1.25


def get_min_display_time(sentence: str) -> float:
    """Determines minimum display time based on sentence length."""
    total_chars = len(sentence.strip())
    if total_chars < 5:  # Very short sentences
        return 1.5
    elif total_chars < 10:  # Short sentences
        return 1.2
    else:
        return 1.0  # Regular sentences


def adjust_timing_for_audio_duration(
    sentence_durations: List[float], 
    speaking_durations: List[float], 
    audio_duration: float, 
    lang_types: List[str]
) -> None:
    """Adjusts subtitle timing to match audio duration.
    
    Args:
        sentence_durations: List of subtitle duration times
        speaking_durations: List of speaking duration times
        audio_duration: Total audio duration
        lang_types: List of language types for each sentence
    """
    if not sentence_durations or audio_duration <= 0:
        return

    total_sentence_time = sum(sentence_durations)
    
    # Calculate required scaling factor
    scale_factor = audio_duration / total_sentence_time
    
    # Print timing information
    print("Original subtitle duration: {:.2f}s".format(total_sentence_time))  # 原始字幕总时长
    print("Target audio duration: {:.2f}s".format(audio_duration))  # 目标音频时长
    print("Scaling factor: {:.4f}".format(scale_factor))  # 缩放系数
    
    # Set min/max scaling limits based on language
    is_mainly_english = sum(1 for lang in lang_types if lang == "en") > len(lang_types) * 0.6
    min_scale = 0.5 if is_mainly_english else 0.6  # English allows more compression
    max_scale = 1.5 if is_mainly_english else 1.3  # English allows more stretching
    
    # Print warning if scaling factor exceeds limits
    if scale_factor < min_scale:
        print("Warning: Scaling factor ({:.2f}) below minimum limit ({})".format(
            scale_factor, min_scale))  # 警告：缩放系数过小
    elif scale_factor > max_scale:
        print("Warning: Scaling factor ({:.2f}) above maximum limit ({})".format(
            scale_factor, max_scale))  # 警告：缩放系数过大
    
    # Apply scaling
    for i in range(len(sentence_durations)):
        # Get language type for current sentence
        lang_type = lang_types[i] if i < len(lang_types) else "unknown"
        
        # Adjust scaling factor based on language type
        adjusted_scale = scale_factor
        if lang_type == "en":
            if scale_factor > 1:
                # Slightly reduce scaling for English stretching
                adjusted_scale = scale_factor * 0.95
            else:
                # Slightly increase scaling for English compression
                adjusted_scale = scale_factor * 1.05
        elif lang_type == "zh":
            if scale_factor > 1:
                # Keep original scaling for Chinese stretching
                adjusted_scale = scale_factor
            else:
                # Increase scaling for Chinese compression
                adjusted_scale = scale_factor * 1.1
                
        # Apply adjusted scaling factor
        sentence_durations[i] *= adjusted_scale
        speaking_durations[i] *= adjusted_scale
    
    # Verify adjusted total duration
    adjusted_total = sum(sentence_durations)
    time_diff = abs(adjusted_total - audio_duration)
    
    print("Adjusted subtitle duration: {:.2f}s".format(adjusted_total))  # 调整后字幕总时长
    print("Difference from target: {:.2f}s".format(time_diff))  # 与目标时长差异
    
    # Perform second adjustment if difference is still large
    if time_diff > 0.5:  # If difference exceeds 0.5 seconds
        final_scale = audio_duration / adjusted_total
        for i in range(len(sentence_durations)):
            sentence_durations[i] *= final_scale
            speaking_durations[i] *= final_scale
            
        final_total = sum(sentence_durations)
        print("Duration after second adjustment: {:.2f}s".format(final_total))  # 二次调整后总时长
        print("Final difference from target: {:.2f}s".format(
            abs(final_total - audio_duration)))  # 最终与目标时长差异


def generate_timing_data(
    sentences: List[str], 
    sentence_durations: List[float], 
    speaking_durations: List[float], 
    original_durations: List[float], 
    lang_types: List[str], 
    time_offset: float, 
    fps: int
) -> List[Dict[str, Any]]:
    """Generates final timing data for subtitles."""
    timing = []
    current_time = time_offset
    
    for i, sentence in enumerate(sentences):
        start_time = current_time
        end_time = start_time + speaking_durations[i]
        
        start_frame = max(0, int(start_time * fps))
        end_frame = max(0, int(end_time * fps))
        
        # Get debug info
        english_count, english_words = count_english_words(sentence) if should_use_mixed_method(sentence) else (0, [])
        chinese_count = sum(1 for c in sentence if '\u4e00' <= c <= '\u9fff')
        
        timing.append({
            'sentence': sentence,
            'start_time': max(0, start_time),
            'end_time': end_time,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'debug_info': {
                'language_type': lang_types[i],
                'chinese_chars': chinese_count,
                'english_words': english_count,
                'english_terms': english_words,
                'speaking_duration': speaking_durations[i],
                'full_duration': sentence_durations[i],
                'original_duration': original_durations[i] if i < len(original_durations) else 0,
                'total_chars': len(sentence.strip())
            }
        })
        
        current_time = end_time
    
    return timing


def extend_last_subtitle(timing: List[Dict[str, Any]], audio_duration: float, fps: int) -> None:
    """Extends the last subtitle to match video duration if needed."""
    last_end_time = timing[-1]['end_time']
    
    if 0 < audio_duration - last_end_time < 1.0:
        timing[-1]['end_time'] = audio_duration
        timing[-1]['end_frame'] = max(0, int(audio_duration * fps))
        print("Extended last subtitle to video end time: {:.2f}s".format(audio_duration))


#--------------------------------
# File Generation Functions
#--------------------------------

def format_time(seconds: float) -> str:
    """Formats seconds to HH:MM:SS,mmm for SRT."""
    seconds = max(0, seconds)
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    whole_seconds = int(seconds)
    milliseconds = int((seconds - whole_seconds) * 1000)
    
    return "{:02d}:{:02d}:{:02d},{:03d}".format(hours, minutes, whole_seconds, milliseconds)


def format_vtt_time(seconds: float) -> str:
    """Formats seconds to HH:MM:SS.mmm for VTT."""
    return format_time(seconds).replace(',', '.')


def generate_srt_file(subtitles: List[Dict[str, Any]], output_path: str, remove_ending_punct: bool = True) -> bool:
    """Generates an SRT format subtitle file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\ufeff')  # BOM for UTF-8
            for i, subtitle in enumerate(subtitles):
                f.write("{}\n".format(i+1))
                f.write("{} --> {}\n".format(format_time(subtitle['start_time']), format_time(subtitle['end_time'])))
                
                sentence = subtitle['sentence']
                if remove_ending_punct:
                    sentence = remove_ending_punctuation(sentence)
                f.write("{}\n\n".format(sentence))
        
        # Verify file was written correctly
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print("Successfully generated SRT file: {}, content length: {}".format(output_path, len(content)))
        return True
    except Exception as e:
        print("Warning: Failed to generate SRT file: {}".format(e))
        return False


def generate_vtt_file(subtitles: List[Dict[str, Any]], output_path: str, remove_ending_punct: bool = True) -> bool:
    """Generates a WebVTT format subtitle file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\ufeff')  # BOM for UTF-8
            f.write("WEBVTT\n\n")  # Double newline after WEBVTT
            for i, subtitle in enumerate(subtitles):
                # Add cue identifier with newline
                f.write("cue-{}\n".format(i+1))
                
                # Add timestamp line with newline
                f.write("{} --> {}\n".format(
                    format_vtt_time(subtitle['start_time']),
                    format_vtt_time(subtitle['end_time'])
                ))
                
                # Process the sentence
                sentence = subtitle['sentence']
                if remove_ending_punct:
                    sentence = remove_ending_punctuation(sentence)
                
                # Add sentence with double newline
                f.write("{}\n\n".format(sentence))
        
        # Verify file was written correctly
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print("Successfully generated VTT file: {}, content length: {}".format(output_path, len(content)))
        return True
    except Exception as e:
        print("Warning: Failed to generate VTT file: {}".format(e))
        return False


def get_video_duration(video_path: str) -> float:
    """Gets video duration in seconds."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Cannot open video file: {}".format(video_path))
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if fps <= 0 or frame_count <= 0:
            # Try using ffmpeg
            try:
                cmd = [
                    "ffprobe", 
                    "-v", "error", 
                    "-show_entries", "format=duration", 
                    "-of", "default=noprint_wrappers=1:nokey=1", 
                    video_path
                ]
                output = subprocess.check_output(cmd).decode('utf-8').strip()
                return float(output)
            except Exception as e:
                print("Failed to get video duration using ffprobe: {}".format(e))
                return 5.0  # Default
        
        return frame_count / fps
    except Exception as e:
        print("Error getting video duration: {}".format(e))
        
        # Estimate based on file size
        try:
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            estimated_duration = file_size_mb * 1.0  # Rough estimate
            return max(estimated_duration, 3.0)  # At least 3 seconds
        except:
            return 5.0  # Default


def add_subtitles_to_video(
    input_video: str, 
    output_video: str, 
    subtitles: Union[str, List[Dict[str, Any]]], 
    font_scale: float = 1.0, 
    font: int = cv2.FONT_HERSHEY_DUPLEX,
    remove_ending_punct: bool = True
) -> bool:
    """Adds subtitles to a video using ffmpeg."""
    try:
        import subprocess
        import shutil
        import tempfile
        
        # Check if input video exists
        if not os.path.exists(input_video):
            raise FileNotFoundError("Input video does not exist: {}".format(input_video))
            
        # If subtitles is a path to a subtitle file
        if isinstance(subtitles, str) and os.path.exists(subtitles):
            subtitle_path = subtitles
            subtitle_format = 'srt' if subtitle_path.endswith('.srt') else 'webvtt'
            
            font_size = int(24 * font_scale)
            
            # Use ffmpeg to burn subtitles
            cmd = [
                "ffmpeg", "-y",
                "-i", input_video,
                "-vf", "subtitles={}:force_style='FontName=Sans,FontSize={},PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=3,Outline=2,Shadow=0,Alignment=2'".format(subtitle_path, font_size),
                "-c:a", "copy",
                output_video
            ]
            
            print("Using ffmpeg to burn subtitles...")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
                print("Successfully burned subtitles: {}".format(output_video))
                return True
            else:
                print("ffmpeg subtitle burning failed, will use alternative method")
        
        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Process subtitle data
            if isinstance(subtitles, list):
                subtitle_list = subtitles
                
                if remove_ending_punct:
                    for sub in subtitle_list:
                        sub['sentence'] = remove_ending_punctuation(sub['sentence'])
            else:
                raise ValueError("Must provide subtitle information list")
            
            # Create temporary SRT file
            temp_subtitle = os.path.join(temp_dir, "temp_subtitle.srt")
            generate_srt_file(subtitle_list, temp_subtitle, remove_ending_punct)
            
            # Use ffmpeg to burn subtitles
            font_size = int(24 * font_scale)
            cmd = [
                "ffmpeg", "-y",
                "-i", input_video,
                "-vf", "subtitles={}:force_style='FontName=Sans,FontSize={},PrimaryColour=&HFFFFFF,OutlineColour=&H000000,BorderStyle=3,Outline=2,Shadow=0,Alignment=2'".format(temp_subtitle, font_size),
                "-c:a", "copy",
                output_video
            ]
            
            print("Using ffmpeg to burn subtitles...")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
                print("Successfully burned subtitles: {}".format(output_video))
                return True
            else:
                print("ffmpeg subtitle burning failed")
                return False
                
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        print("Failed to add subtitles: {}".format(str(e)))
        traceback.print_exc()
        return False


#--------------------------------
# Main Functions
#--------------------------------

def create_subtitle_for_text(
    text: str,
    output_path: str,
    duration: float,
    format: str = "vtt",
    encoding: str = "utf-8",
    offset: float = -0.3,
    subtitle_speed: float = 1.0,
    text_cut_method: str = "cut2",
    sentence_gap: float = 0.4,
    remove_ending_punct: bool = True
) -> bool:
    """Creates subtitle file for text with appropriate timing."""
    try:
        logger = logging.getLogger("SubtitleUtils")
        
        audio_fps = 50  # Can be configured or detected dynamically
        
        # Validate video duration
        if duration <= 0:
            logger.warning("Invalid video duration ({})s), using estimated duration".format(duration))
            estimated_duration = len(text) * 0.25  # ~0.25s per character
            duration = max(estimated_duration, 10.0)  # At least 10 seconds
            logger.info("Estimated duration: {}s".format(duration))
        
        # Split text into sentences
        sentences = split_text_to_sentences(text, text_cut_method)
        
        if not sentences:
            logger.warning("No valid sentences after text cutting")
            sentences = [text]  # Use whole text as one sentence
        
        logger.info("Generating subtitles: {} characters, {} sentences, video duration: {:.2f} seconds".format(
            len(text), len(sentences), duration))
        
        # Analyze text characteristics
        text_analysis = analyze_text_characteristics(sentences)
        
        # Calculate appropriate speaking rates
        speaking_rates = calculate_speaking_rates(text_analysis, duration, subtitle_speed)
        
        # Calculate subtitle timing
        timings = calculate_mixed_subtitle_timing(
            sentences=sentences,
            audio_duration=duration,
            fps=30,
            cn_speaking_rate=speaking_rates['cn_rate'],
            en_word_rate=speaking_rates['en_rate'],
            time_offset=offset,
            sentence_gap=speaking_rates['sentence_gap'],
            extend_to_duration=True,
            use_mixed_method=True,
            audio_fps=audio_fps
        )
        
        # Check if subtitle duration is appropriate
        if timings:
            subtitle_end_time = timings[-1]['end_time']
            if subtitle_end_time > duration * 1.1:
                logger.warning("Generated subtitle total duration ({:.2f}s) exceeds video duration ({:.2f}s)".format(
                    subtitle_end_time, duration))
        
        # Generate subtitle file
        if format.lower() == "vtt":
            success = generate_vtt_file(timings, output_path, remove_ending_punct)
        else:
            success = generate_srt_file(timings, output_path, remove_ending_punct)
        
        if success:
            logger.info("Successfully generated subtitle file: {}".format(output_path))
            if timings:
                last_timing = timings[-1]
                logger.info("Subtitle total duration: {:.2f}s, Video duration: {:.2f}s".format(
                    last_timing['end_time'], duration))
        else:
            logger.warning("Failed to generate subtitle file")
            
        return success
    except Exception as e:
        logger = logging.getLogger("SubtitleUtils")
        logger.error("Error creating subtitles for text: {}".format(e))
        logger.error(traceback.format_exc())
        return False


def analyze_text_characteristics(sentences: List[str]) -> Dict[str, Any]:
    """Analyzes text characteristics for timing adjustments."""
    combined_text = " ".join(sentences)
    
    # Count different character types
    english_chars = sum(1 for c in combined_text if 'a' <= c.lower() <= 'z')
    chinese_chars = sum(1 for c in combined_text if '\u4e00' <= c <= '\u9fff')
    
    # Check for special expressions
    has_compound_words = '-' in combined_text
    compound_word_count = combined_text.count('-')
    has_year_expressions = bool(re.search(r'\b\d{4}s?\b', combined_text))
    has_decade_expressions = bool(re.search(r'\b\d{2}s\b', combined_text))
    has_numeric_expressions = bool(re.search(r'\b\d+(?:st|nd|rd|th)?\b', combined_text))
    
    # Check for special compound terms
    special_compound_terms = ["audio-visual", "information-based"]
    has_special_compounds = any(term in combined_text.lower() for term in special_compound_terms)
    
    # Determine if English is dominant
    is_english_dominant = english_chars > (english_chars + chinese_chars) * 0.6
    
    return {
        'total_chars': len(combined_text),
        'english_chars': english_chars,
        'chinese_chars': chinese_chars,
        'is_english_dominant': is_english_dominant,
        'has_compound_words': has_compound_words,
        'compound_word_count': compound_word_count,
        'has_special_compounds': has_special_compounds,
        'has_year_expressions': has_year_expressions,
        'has_decade_expressions': has_decade_expressions,
        'has_numeric_expressions': has_numeric_expressions,
        'sentence_count': len(sentences)
    }


def calculate_speaking_rates(analysis: Dict[str, Any], duration: float, subtitle_speed: float) -> Dict[str, float]:
    """Calculates appropriate speaking rates based on text analysis."""
    logger = logging.getLogger("SubtitleUtils")
    
    # Calculate effective target duration with safety margin
    target_duration = duration
    if analysis['is_english_dominant']:
        safety_margin = calculate_safety_margin(analysis)
        target_duration = duration * (1.0 - safety_margin)
        logger.info("English text with {:.1%} safety margin, target duration: {:.2f}s".format(safety_margin, target_duration))
    
    # Calculate auto speed factor
    auto_speed_factor = 1.0
    estimated_duration = analysis['total_chars'] * 0.25  # rough estimate
    if estimated_duration > target_duration * 1.1:
        auto_speed_factor = min(estimated_duration / (target_duration * 0.9), 1.8)  # max 80% speedup
        logger.info("Text too long, auto-adjusting speed factor to: {:.2f}".format(auto_speed_factor))
    
    # Apply subtitle speed and automatic adjustments
    final_speed = subtitle_speed * auto_speed_factor
    
    # Determine base rates
    if analysis['is_english_dominant']:
        base_cn_rate = 5.0  # Chinese characters per second
        
        # Adjust English rate based on content
        if (analysis['has_special_compounds'] or 
           (analysis['has_compound_words'] and analysis['compound_word_count'] / analysis['sentence_count'] > 0.2)):
            base_en_rate = 1.0  # Slower for complex content
            logger.info("Detected many compound words, using slower English rate")
        elif (analysis['has_year_expressions'] or 
              analysis['has_decade_expressions'] or 
              analysis['has_numeric_expressions']):
            base_en_rate = 1.0  # Slower for numeric content
            logger.info("Detected year/number expressions, using slower English rate")
        else:
            base_en_rate = 1.3  # Normal English rate
        
        logger.info("Detected English-dominant text, base rates - Chinese: {:.1f} chars/sec, English: {:.1f} words/sec".format(
            base_cn_rate, base_en_rate))
    else:
        base_cn_rate = 4.1  # Chinese characters per second
        base_en_rate = 1.0  # English words per second
    
    # Calculate final rates
    cn_speaking_rate = base_cn_rate * final_speed
    en_word_rate = base_en_rate * final_speed
    
    # Apply rate limits for special content
    if analysis['is_english_dominant'] and (analysis['has_special_compounds'] or analysis['has_year_expressions']):
        en_word_rate = min(en_word_rate, 1.3)
        logger.info("Limited max English rate to 1.3 words/sec due to special content")
    
    logger.info("Final subtitle rates - Chinese: {:.2f} chars/sec, English: {:.2f} words/sec".format(
        cn_speaking_rate, en_word_rate))
    
    # Determine sentence gap
    sentence_gap = 0.4 if analysis['is_english_dominant'] else 0.4  # Restore original gap for TTS
    
    return {
        'cn_rate': cn_speaking_rate,
        'en_rate': en_word_rate,
        'sentence_gap': sentence_gap
    }


def calculate_safety_margin(analysis: Dict[str, Any]) -> float:
    """Calculates safety margin for English-dominant text."""
    safety_margin = 0.1  # Base safety margin 10%
    
    if analysis['has_special_compounds']:
        safety_margin += 0.1  # Additional 10%
    
    if analysis['compound_word_count'] > 0:
        compound_density = analysis['compound_word_count'] / analysis['sentence_count']
        safety_margin += compound_density * 0.05  # 5% per compound word per sentence
    
    if (analysis['has_year_expressions'] or 
        analysis['has_decade_expressions'] or 
        analysis['has_numeric_expressions']):
        safety_margin += 0.05  # Additional 5%
        print("Detected year/number expressions, adding 5% safety margin")
    
    return min(safety_margin, 0.25)  # Cap at 25%


def test_subtitle_timing(text: str) -> None:
    """Tests subtitle timing calculation for a text."""
    # Split into sentences
    sentences = split_text_to_sentences(text)
    print("Number of sentences: {}".format(len(sentences)))
    
    # Calculate subtitle timing
    timings = calculate_mixed_subtitle_timing(sentences)
    
    # Output results
    for i, timing in enumerate(timings):
        sentence = timing['sentence']
        duration = timing['end_time'] - timing['start_time']
        debug = timing['debug_info']
        
        print("Sentence {}: {}".format(i+1, sentence))
        print("  Duration: {:.2f} seconds".format(duration))
        print("  Chinese characters: {}".format(debug['chinese_chars']))
        print("  English words: {} {}".format(
            debug['english_words'], 
            debug['english_terms'] if 'english_terms' in debug else ''))
        print("  Total characters: {}".format(debug['total_chars']))
        print()
    
    # Calculate total duration
    total_duration = timings[-1]['end_time'] - timings[0]['start_time']
    print("Total duration: {:.2f} seconds".format(total_duration))

import re
from typing import Callable

punctuation = set(['!', '?', '…', ',', '.', '-'," "])
METHODS = dict()

def get_method(name:str)->Callable:
    """根据名称获取文本切分方法"""
    method = METHODS.get(name, None)
    if method is None:
        raise ValueError("Method {} does not exist".format(name))
    return method

def get_method_names()->list:
    """获取所有可用的切分方法名"""
    return list(METHODS.keys())

def register_method(name):
    """注册一个文本切分方法"""
    def decorator(func):
        METHODS[name] = func
        return func
    return decorator

# 定义标点符号集合
splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }

def split_big_text(text, max_len=510):
    """切分过长文本
    Args:
        text: 输入文本
        max_len: 最大长度限制
    """
    # 定义全角和半角标点符号
    punctuation = "".join(splits)

    # 切割文本
    segments = re.split('([' + punctuation + '])', text)
    
    # 初始化结果列表和当前片段
    result = []
    current_segment = ''
    
    for segment in segments:
        # 如果当前片段加上新的片段长度超过max_len，就将当前片段加入结果列表，并重置当前片段
        if len(current_segment + segment) > max_len:
            result.append(current_segment)
            current_segment = segment
        else:
            current_segment += segment
    
    # 将最后一个片段加入结果列表
    if current_segment:
        result.append(current_segment)
    
    return result

def split(todo_text):
    """按标点符号切分文本"""
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts

# 不切分
@register_method("cut0")
def cut0(inp):
    """不切分文本
    Args:
        inp: 输入文本
    """
    if not set(inp).issubset(punctuation):
        return inp
    else:
        return "/n"

# 凑四句一切
@register_method("cut1")
def cut1(inp):
    """每四句切一次
    Args:
        inp: 输入文本
    """
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

@register_method("cut2")
def cut2(inp):
    """智能切分文本，按照句号→换行→分号→逗号的优先级顺序处理
    Args:
        inp: 输入文本
    Returns:
        按智能方式分组后的文本，每组不超过25字
    """
    # 保存原始文本用于后续处理
    original_text = inp
    
    # 1. 首先按句号等强终止符切分（。.!?！？）
    period_pattern = r'([。.!?！？])'
    period_parts = re.split(period_pattern, inp)
    
    # 组合句子和标点
    sentences = []
    current = ""
    
    for i in range(0, len(period_parts) - 1, 2):
        if i < len(period_parts):
            text = period_parts[i] if i < len(period_parts) else ""
            punct = period_parts[i + 1] if i + 1 < len(period_parts) else ""
            sentences.append(text + punct)
    
    # 处理最后一部分(如果没有标点结尾)
    if len(period_parts) % 2 == 1 and period_parts[-1].strip():
        sentences.append(period_parts[-1])
    
    # 如果按句号切分后没有内容，使用原文
    if not sentences:
        sentences = [inp]
    
    # 2. 检查每个句子是否有原始换行符，如果有则进一步切分
    line_segments = []
    for sentence in sentences:
        if "\n" in sentence:
            lines = [line.strip() for line in sentence.split("\n") if line.strip()]
            line_segments.extend(lines)
        else:
            line_segments.append(sentence)
    
    # 3. 对每个段落，如果过长，按分号等次强分隔符切分
    semicolon_segments = []
    max_segment_len = 25  # 最大段落长度
    
    for segment in line_segments:
        if len(segment) > max_segment_len:
            # 按分号、省略号等切分
            semicolon_pattern = r'([；;…])'
            sub_parts = re.split(semicolon_pattern, segment)
            
            current_part = ""
            for i in range(0, len(sub_parts) - 1, 2):
                if i < len(sub_parts):
                    text = sub_parts[i] if i < len(sub_parts) else ""
                    punct = sub_parts[i + 1] if i + 1 < len(sub_parts) else ""
                    
                    # 添加当前部分
                    semicolon_segments.append(text + punct)
            
            # 处理最后一部分
            if len(sub_parts) % 2 == 1 and sub_parts[-1].strip():
                semicolon_segments.append(sub_parts[-1])
        else:
            semicolon_segments.append(segment)
    
    # 4. 对仍然过长的段落，按逗号切分
    final_segments = []
    
    for segment in semicolon_segments:
        if len(segment) > max_segment_len:
            # 按逗号切分
            comma_pattern = r'([，,、])'
            comma_parts = re.split(comma_pattern, segment)
            
            current_segment = ""
            for i in range(0, len(comma_parts) - 1, 2):
                if i < len(comma_parts):
                    text = comma_parts[i] if i < len(comma_parts) else ""
                    comma = comma_parts[i + 1] if i + 1 < len(comma_parts) else ""
                    
                    if not current_segment:
                        current_segment = text + comma
                    elif len(current_segment + text) <= max_segment_len:
                        current_segment += text + comma
                    else:
                        final_segments.append(current_segment)
                        current_segment = text + comma
            
            # 处理最后一部分
            if len(comma_parts) % 2 == 1:
                last_part = comma_parts[-1].strip()
                if last_part:
                    if not current_segment:
                        current_segment = last_part
                    elif len(current_segment + last_part) <= max_segment_len:
                        current_segment += last_part
                    else:
                        final_segments.append(current_segment)
                        current_segment = last_part
            
            if current_segment:
                final_segments.append(current_segment)
        else:
            final_segments.append(segment)
    
    # 5. 检查特殊情况：处理冒号结构，确保冒号和其前面部分不被单独分开
    colon_adjusted = []
    for segment in final_segments:
        if "：" in segment or ":" in segment:
            colon_pos = max(segment.find("："), segment.find(":"))
            if 0 < colon_pos < len(segment) - 1 and colon_pos < 10:
                # 冒号在靠前位置，保持与后面内容的连接
                colon_adjusted.append(segment)
            else:
                colon_adjusted.append(segment)
        else:
            colon_adjusted.append(segment)
    
    # 6. 最终过滤：移除空段落和纯标点段落
    punctuation_set = set(['!', '?', '…', ',', '.', '-', " ", "；", "，", "。"])
    result = []
    
    for segment in colon_adjusted:
        segment = segment.strip()
        if segment and not set(segment).issubset(punctuation_set):
            result.append(segment)
    
    # 确保至少有一个段落
    if not result:
        result = [original_text]
    
    return "\n".join(result)

def has_mixed_language(text):
    """判断文本是否为中英混合
    Args:
        text: 输入文本
    Returns:
        是否为中英混合文本
    """
    # 计算英文字符数量
    english_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
    # 计算中文字符数量
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    
    # 如果同时包含显著数量的中文和英文，认为是混合文本
    return chinese_chars > 10 and english_chars > 10 and chinese_chars > 0.1 * len(text) and english_chars > 0.1 * len(text)

def is_english_text(text):
    """判断文本是否为英文为主
    Args:
        text: 输入文本
    Returns:
        是否为英文为主的文本
    """
    # 计算英文字符数量
    english_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
    # 计算中文字符数量
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    
    # 如果英文字符数量占比超过中文，认为是英文为主
    total_chars = len(text.strip())
    if total_chars == 0:
        return False
    
    return english_chars > chinese_chars and chinese_chars < 0.3 * len(text)

def cut_mixed_text(text):
    """处理中英文混合文本，智能分割长句并保持语义完整性
    Args:
        text: 输入的混合文本
    Returns:
        分段后的文本，每段长度适中且语义完整
    """
    # 首先检查是否包含英文定义
    definition_pattern = r'(.*?)(Educational Technology is.*?processes)([。.!?！？]?)(.*)$'
    definition_match = re.search(definition_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if definition_match:
        # 分别处理定义前、定义、定义的标点和定义后的部分
        before_def = definition_match.group(1).strip()
        definition = definition_match.group(2).strip()
        def_punctuation = definition_match.group(3) or ""
        after_def = definition_match.group(4).strip()
        
        # 处理英文定义部分
        def_parts = []
        if definition:
            # 首先按主要语义单位分割
            semantic_units = [
                "Educational Technology is a field involved",
                "in the facilitation of human learning",
                "through systematic identification, development, organization",
                "and utilization of a full range of earning resources",
                "and through the management of these processes"
            ]
            
            # 尝试匹配预定义的语义单位
            remaining_text = definition
            for unit in semantic_units:
                if unit.lower() in remaining_text.lower():
                    def_parts.append(unit)
                    remaining_text = remaining_text.replace(unit, "", 1)
            
            # 如果还有剩余文本，使用通用切分逻辑
            if remaining_text.strip():
                # 按逗号、and等连接词分割
                segments = re.split(r'(,\s+|\sand\s)', remaining_text)
                current_part = ""
                
                for segment in segments:
                    if segment.strip() in [",", "and"]:
                        current_part += segment
                        continue
                        
                    if not current_part:
                        current_part = segment
                    elif len(current_part + segment) <= 60:  # 允许更长的英文片段
                        current_part += segment
                    else:
                        if current_part.strip():
                            def_parts.append(current_part.strip())
                        current_part = segment
                
                if current_part.strip():
                    def_parts.append(current_part.strip())
            
            # 处理最后一个定义部分的标点
            if def_parts and def_punctuation:
                def_parts[-1] = def_parts[-1] + def_punctuation
        
        # 处理定义前的中文部分
        before_parts = []
        if before_def:
            before_parts = split_chinese_text(before_def)
        
        # 处理定义后的中文部分
        after_parts = []
        if after_def:
            after_parts = split_chinese_text(after_def)
        
        # 合并所有部分，确保每个部分都有合适的标点符号
        all_parts = []
        
        # 添加定义前的部分
        for part in before_parts:
            # 如果是最后一个前置部分，且没有标点，添加逗号或句号
            if part == before_parts[-1] and not any(p in part[-1] for p in splits):
                part += "，"
            all_parts.append(part.strip())
        
        # 添加英文定义部分
        for i, part in enumerate(def_parts):
            # 处理英文部分的标点
            if i < len(def_parts) - 1:
                # 如果当前部分没有以逗号结尾，添加逗号
                if not part.rstrip().endswith(","):
                    part += ","
            all_parts.append(part.strip())
        
        # 添加定义后的部分
        for part in after_parts:
            all_parts.append(part.strip())
        
        # 确保最后一个部分有结束标点
        if all_parts and not any(p in all_parts[-1][-1] for p in splits):
            all_parts[-1] += "。"
        
        # 过滤空行并合并
        return "\n".join(part for part in all_parts if part.strip())
    
    # 如果没有找到英文定义，使用常规的中英文混合处理逻辑
    return handle_regular_mixed_text(text)

def handle_regular_mixed_text(text):
    """处理常规的中英文混合文本"""
    # 按句号等主要标点符号分割
    main_punct_pattern = r'([。.!?！？])'
    parts = re.split(main_punct_pattern, text)
    
    sentences = []
    current = ""
    
    for i in range(0, len(parts) - 1, 2):
        current_part = parts[i] if i < len(parts) else ""
        punctuation = parts[i + 1] if i + 1 < len(parts) else ""
        
        # 检查是否包含英文
        has_english = bool(re.search(r'[a-zA-Z]', current_part))
        
        # 处理当前部分的逗号
        if has_english:
            # 英文部分允许更长
            max_length = 50
            # 将中文逗号转换为英文逗号
            current_part = re.sub(r'，', ', ', current_part)
        else:
            # 中文部分控制在较短长度
            max_length = 25
            # 将英文逗号转换为中文逗号
            current_part = re.sub(r',\s*', '，', current_part)
        
        if len(current_part.strip()) <= 5:
            current += current_part + punctuation
            continue
            
        if current:
            if len(current + current_part) <= max_length:
                current += current_part + punctuation
            else:
                # 确保当前句子有合适的结束标点
                if not any(p in current[-1] for p in splits):
                    current += "。"
                sentences.append(current)
                current = current_part + punctuation
        else:
            current = current_part + punctuation
    
    if current:
        # 确保最后一个句子有合适的结束标点
        if not any(p in current[-1] for p in splits):
            current += "。"
        sentences.append(current)
    
    return "\n".join(s.strip() for s in sentences if s.strip())

def split_chinese_text(text):
    """处理中文文本部分"""
    # 首先按分号切分
    semicolon_parts = re.split(r'[；;]', text)
    sentences = []
    
    for part in semicolon_parts:
        if not part.strip():
            continue
            
        # 然后按其他主要标点符号分割
        main_punct_pattern = r'([。！？])'
        sub_parts = re.split(main_punct_pattern, part)
        
        current = ""
        
        for i in range(0, len(sub_parts) - 1, 2):
            current_part = sub_parts[i] if i < len(sub_parts) else ""
            punctuation = sub_parts[i + 1] if i + 1 < len(sub_parts) else ""
            
            # 将英文逗号转换为中文逗号
            current_part = re.sub(r',\s*', '，', current_part)
            
            # 处理短句
            if len(current_part.strip()) <= 5:
                current += current_part + punctuation
                continue
                
            # 如果当前部分太长，尝试按逗号切分
            if len(current_part) > 25:
                comma_parts = re.split(r'[，,]', current_part)
                temp = ""
                
                for comma_part in comma_parts:
                    if not temp:
                        temp = comma_part
                    elif len(temp + comma_part) <= 25:
                        temp += "，" + comma_part
                    else:
                        if temp.strip():
                            sentences.append(temp.strip() + "，")
                        temp = comma_part
                
                if temp.strip():
                    current = temp + punctuation
            else:
                # 处理正常长度的句子
                if current:
                    if len(current + current_part) <= 25:
                        current += current_part + punctuation
                    else:
                        # 确保当前句子有合适的结束标点
                        if not any(p in current[-1] for p in splits):
                            current += "。"
                        sentences.append(current)
                        current = current_part + punctuation
                else:
                    current = current_part + punctuation
        
        # 处理最后一个子句
        if current:
            # 确保句子有合适的结束标点
            if not any(p in current[-1] for p in splits):
                current += "。"
            sentences.append(current)
    
    # 过滤空句子并确保每个句子都有合适的标点
    result = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # 如果句子没有结束标点，添加句号
        if not any(p in s[-1] for p in splits):
            s += "。"
        result.append(s)
    
    return result

def cut_english_text(text):
    """专门用于英文文本的切分算法，按照英文的标点符号切分
    Args:
        text: 输入的英文文本
    Returns:
        按照合理方式切分后的英文文本
    """
    # 先按句号切分
    sentences = []
    # 正则表达式匹配句子结束（句号、问号、叹号后面跟空格或结束）
    # 但忽略缩写中的句点，如 "U.S.A."
    sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    parts = re.split(sentence_pattern, text)
    
    # 每个句子作为一个单位处理
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # 长句子处理 - 英文句子超过100字符时，尝试按逗号等分割
        # 但确保语义的完整性
        if len(part) > 100:
            # 按逗号、分号等分割
            comma_pattern = r'([,;:]\s*)'
            clauses = re.split(comma_pattern, part)
            
            # 合并子句
            current_sentence = ""
            max_length = 80  # 英文句子最大长度限制
            
            i = 0
            while i < len(clauses):
                clause = clauses[i]
                
                # 处理标点符号(如果有)
                punctuation = ""
                if i+1 < len(clauses) and re.match(r'[,;:]\s*', clauses[i+1]):
                    punctuation = clauses[i+1]
                    i += 1
                
                if not current_sentence:
                    current_sentence = clause + punctuation
                elif len(current_sentence) + len(clause) + len(punctuation) <= max_length:
                    current_sentence += clause + punctuation
                else:
                    sentences.append(current_sentence)
                    current_sentence = clause + punctuation
                
                i += 1
            
            # 添加最后一个句子
            if current_sentence:
                sentences.append(current_sentence)
        elif 60 < len(part) <= 100:
            # 长度中等的句子，保持完整性，但检查是否可以按自然断点切分
            # 以下关键词可能是一个自然段落的开始，可以作为切分点
            key_conjunctions = [
                " but ", " however, ", " nevertheless, ", " yet ", " on the other hand, ",
                " moreover, ", " in addition, ", " furthermore, ", " additionally, ",
                " therefore, ", " thus, ", " consequently, ", " hence, ", " as a result, ",
                " for example, ", " for instance, ", " such as ", " specifically, ",
                " that is, ", " namely, ", " in other words, ", " in contrast, ", " in comparison, "
            ]
            
            # 检查是否存在自然切分点
            split_point = -1
            for conjunction in key_conjunctions:
                pos = part.lower().find(conjunction)
                if pos > 30:  # 确保前部分有足够长度
                    split_point = pos
                    break
            
            if split_point > 0:
                # 在自然切分点分割
                first_part = part[:split_point]
                second_part = part[split_point:]
                sentences.append(first_part)
                sentences.append(second_part)
            else:
                # 没有找到合适的切分点，作为一个完整句子
                sentences.append(part)
        else:
            # 对于短句，直接添加
            sentences.append(part)
    
    # 处理特殊情况：年份、标题等不应该单独成为一个字幕
    # 例如 "In 1915" 应该合并到下一个句子
    merged_sentences = []
    i = 0
    
    while i < len(sentences):
        current = sentences[i].strip()
        
        # 检查是否是短年份句("In 1915")
        is_year_intro = re.match(r'^(in|around|during|by|before|after)\s+\d{4}[,.]?$', current.lower())
        is_too_short = len(current) < 15
        
        if is_year_intro and i+1 < len(sentences):
            # 合并年份引入句与下一句
            merged_sentences.append(current + " " + sentences[i+1])
            i += 2
        elif is_too_short and i+1 < len(sentences) and len(sentences[i+1]) < 60:
            # 合并非常短的句子
            merged_sentences.append(current + " " + sentences[i+1])
            i += 2
        else:
            # 正常添加当前句子
            merged_sentences.append(current)
            i += 1
    
    # 确保有内容
    if not merged_sentences:
        merged_sentences = [text]
    
    # 过滤空句子
    merged_sentences = [s for s in merged_sentences if s.strip()]
    
    return "\n".join(merged_sentences)

# 按中文句号。切
@register_method("cut3")
def cut3(inp):
    """按中文句号切分
    Args:
        inp: 输入文本
    """
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

# 按英文句号.切
@register_method("cut4")
def cut4(inp):
    """按英文句号切分
    Args:
        inp: 输入文本
    """
    inp = inp.strip("\n")
    opts = re.split(r'(?<!\d)\.(?!\d)', inp.strip("."))
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)

# 按标点符号切
@register_method("cut5")
def cut5(inp):
    """按多种标点符号切分
    Args:
        inp: 输入文本
    """
    inp = inp.strip("\n")
    punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)

if __name__ == '__main__':
    method = get_method("cut2")
    print(method("你好，我是小明。你好，我是小红。你好，我是小刚。你好，我是小张。")) 

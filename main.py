#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt优化脚本 - 使用DeepSeek API将prompt转换为token数较少的英文版本
"""

import json
import sys
import os
from pathlib import Path
from openai import OpenAI
from llmlingua import PromptCompressor

# 强制使用CPU，避免CUDA错误
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# 运行模式配置
RUN_MODE = {
    'deepseek_only': False,   # 只运行DeepSeek优化
    'llmlingua_only': False,  # 只运行LLMLingua压缩
    'both': True              # 两者都运行
}

# 注意：只能选择一种模式，如果都设为True，优先顺序：both > deepseek_only > llmlingua_only



def clean_compressed_text(text):
    """
    清理LLMLingua压缩后留下的多余空格
    
    Args:
        text (str): 压缩后的文本
    
    Returns:
        str: 清理后的文本
    """
    import re
    
    # 将多个连续空格替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    # 清理行首行尾空格
    text = text.strip()
    
    # 清理段落间的多余空行（保留最多两个连续换行）
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # 清理行首行尾的空格
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines]
    text = '\n'.join(cleaned_lines)
    
    # 移除空行
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    # 消除中文字之间的空格
    # 匹配中文字符之间的空格
    text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)
    
    # 消除中文字符与标点符号之间的空格
    text = re.sub(r'([\u4e00-\u9fff])\s+([，。！？；：""''（）【】])', r'\1\2', text)
    text = re.sub(r'([，。！？；：""''（）【】])\s+([\u4e00-\u9fff])', r'\1\2', text)
    
    # 更全面的中文字符空格清理
    # 清理中文字符与数字之间的空格
    text = re.sub(r'([\u4e00-\u9fff])\s+(\d)', r'\1\2', text)
    text = re.sub(r'(\d)\s+([\u4e00-\u9fff])', r'\1\2', text)
    
    # 清理中文字符与英文字母之间的空格
    text = re.sub(r'([\u4e00-\u9fff])\s+([a-zA-Z])', r'\1\2', text)
    text = re.sub(r'([a-zA-Z])\s+([\u4e00-\u9fff])', r'\1\2', text)
    
    # 清理中文字符与特殊符号之间的空格
    text = re.sub(r'([\u4e00-\u9fff])\s+([#@$%^&*+=<>])', r'\1\2', text)
    text = re.sub(r'([#@$%^&*+=<>])\s+([\u4e00-\u9fff])', r'\1\2', text)
    
    # 清理中文字符与括号之间的空格
    text = re.sub(r'([\u4e00-\u9fff])\s+([\(\)\[\]\{\}])', r'\1\2', text)
    text = re.sub(r'([\(\)\[\]\{\}])\s+([\u4e00-\u9fff])', r'\1\2', text)
    
    # 多次应用，确保所有中文字符间的空格都被清理
    for _ in range(3):
        text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)
    
    return text

def smart_split_text(text, max_chars):
    """
    智能分段文本，在单词边界处截断，避免将单词分割
    
    Args:
        text (str): 要分段的文本
        max_chars (int): 每段最大字符数
    
    Returns:
        list: 分段后的文本列表
    """
    if len(text) <= max_chars:
        return [text]
    
    segments = []
    start = 0
    
    while start < len(text):
        # 计算当前段的结束位置
        end = min(start + max_chars, len(text))
        
        # 如果这是最后一段，直接截取
        if end == len(text):
            segments.append(text[start:end])
            break
        
        # 在最大长度范围内寻找最佳截断点
        # 优先寻找句子结束符
        sentence_endings = ['。', '！', '？', '.', '!', '?', '\n\n']
        best_break = start + max_chars
        
        for ending in sentence_endings:
            pos = text.rfind(ending, start, end)
            if pos > start and pos < end:
                best_break = pos + len(ending)
                break
        
        # 如果没找到句子结束符，寻找空格或标点符号
        if best_break == start + max_chars:
            # 寻找空格
            space_pos = text.rfind(' ', start, end)
            if space_pos > start:
                best_break = space_pos + 1
            else:
                # 寻找其他分隔符
                separators = ['，', '；', '：', ',', ';', ':', '、']
                for sep in separators:
                    pos = text.rfind(sep, start, end)
                    if pos > start and pos < end:
                        best_break = pos + len(sep)
                        break
        
        # 如果还是没找到合适的分隔点，就在最大长度处截断
        if best_break == start + max_chars:
            best_break = end
        
        # 添加当前段
        segment = text[start:best_break].strip()
        if segment:  # 确保段不为空
            segments.append(segment)
        
        start = best_break
    
    return segments

# ========= 动态选择 LLMLingua 模型（短/长）与阈值 =========
# 超过该字符数则优先使用 LongLLMLingua（整段压缩，不再分段）
LONG_INPUT_THRESHOLD_CHARS = int(os.getenv('LONG_LINGUA_THRESHOLD_CHARS', '1000'))

# 允许通过环境变量自定义短/长模型名称
# 示例：
#   export LLMLINGUA_MODEL_SHORT="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
#   export LLMLINGUA_MODEL_LONG="<LongLLMLingua 的长上下文检查点>"
DEFAULT_SHORT_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

def create_prompt_compressor(use_long: bool) -> PromptCompressor:
    """
    根据 use_long 选择短/长上下文模型构造 PromptCompressor。
    - 优先读取环境变量：LLMLINGUA_MODEL_LONG / LLMLINGUA_MODEL_SHORT
    - 未配置长模型而触发 use_long 时，回退到短模型并给出提示
    """
    long_model_from_env = os.getenv('LLMLINGUA_MODEL_LONG', '').strip()
    short_model_from_env = os.getenv('LLMLINGUA_MODEL_SHORT', '').strip()

    if use_long and long_model_from_env:
        model_name = long_model_from_env
        print(f"使用LongLLMLingua模式，模型：{model_name}")
    else:
        model_name = short_model_from_env or DEFAULT_SHORT_MODEL
        if use_long:
            print(f"使用LongLLMLingua模式，模型：{model_name}")
        else:
            print(f"使用LLMLingua-2模式，模型：{model_name}")

    return PromptCompressor(
        model_name=model_name,
        use_llmlingua2=True,
        device_map="cpu"
    )

# 读取配置文件
CONFIG_PATH = Path(__file__).parent / 'config.json'
if not CONFIG_PATH.exists():
    print('缺少config.json配置文件，请创建并填写api_key')
    sys.exit(1)
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)
api_key = config.get('api_key')
if not api_key:
    print('config.json中缺少api_key')
    sys.exit(1)

# 你的prompt字符串，直接写在这里
prompt = '''# 角色
你现在是一个视频内容理解专家，可以通过事件在当事人、事件重要程度、受众群体三个维度进行综合评判，给出最后的定级，从而选出点击量最高的视频。你能帮助用户评估视频内容的流行潜力，并提供深入的分析和建议。定级分为A、B、C三个级别。

- 请按照以下格式进行回复，非json：
{
 "level": <等级>,
 "reason": <具体分析理由>,
"type": <事件类型>
}

## 技能
### 技能 1: 事件伤亡评估
- 分析视频内容，判断是否包含伤亡情况。
- 若视频事件涉及伤亡情况，即定级为A。

### 技能 2: 奇闻异事评估
- 评估视频内容是否为非常猎奇性事件。
- 判断事件是否严重超出常规认知、或违背伦理道德、或非常罕见或非常荒诞、或特别奇葩、或事件属性恶劣，即定级为A。
-若该事件对某一地域带来恶劣影响或能在当地取得很高关注度，即定级为A。
-若仅仅为萌宠日常、萌娃趣事等非常小的事件，则定级为C。
- 若视频内容涉及如伦理纠纷、学术造假、虐待动物等，或包含强烈冲击力的画面，如血腥画面、灾难现场，或结果冲击（巨额数字、极端后果、较大数字反差）。即定级为A。

### 技能 3: 高知名度主体舆论事件评估
- 分析视频中是否涉及高知名度主体的负面事件。
- 判断该事件是否因负面性导致公众形象受损，并引发舆论风暴，即定级为A，若仅为高知名度主体的非舆论性事件则定级为B。
- 若视频内容涉及高知名度主体的争议、丑闻或违规问题，定级为A。

### 技能 4: 极端天气或自然灾害评估
- 分析视频是否包含极端天气或自然灾害情况。
- 若视频内容涉及极端天气、或反常天气、或自然灾害类事件，即定级为A。

### 技能 5: 本地人是否比外地人更为关注此事件
- 分析视频是否能得到本地人的广泛关注（如整个县、市、省···），若可以则定为A，反之为B（如不能引起当地居民注意或只引起某小区居民等微小群体的注意）。

分析链路（举例）：
输入：业主用水242吨被要求按800吨缴费
step1:判断是否属于伤亡事件，该事件不包含伤亡信息，因此不属于
step2：判断该事件是否属于奇闻异事，该事件中描述用水242吨却被要求按800吨缴费，超出常规认知，具有强反差性、与结果冲击（242吨与800钝），因此属于奇闻异事，定级为A
step3：判断该事件是否属于高知名度主体舆论事件，该事件不涉及到高知名度主体，因此不属于高知名度主体舆论事件
step4:判断该事件是否属于极端天气/自然灾害，分析视频不包含极端天气或自然灾害情况，因此不属于极端天气/自然灾害。
因此，该事件的定级为A
输出：
{
"level": "A", 
"type": "奇闻异事", 
"reason": "判断理由"
}

## 约束
- 只输出最终定级与分类、原因即可。
-事件类型只能出自我给你的几个类型中（伤亡事件、奇闻异事、高知名度主体舆论事件、极端天气/自然灾害），不得随意添加。
- 当出现自然灾害类的可定级为A。
-当奇闻异事中出现钱金额元素时可定级为A。
-未符合上述标准的定级为B。
- 专注于视频内容的理解与评估。
- 使用以上提供的维度和标准进行评估。
- 只使用准确且可信的数据和信息。
- 不回答与视频内容理解和评估无关的问题。
'''

def optimize_with_deepseek(prompt_text, api_key):
    """
    使用DeepSeek API优化prompt
    
    Args:
        prompt_text (str): 原始prompt文本
        api_key (str): DeepSeek API密钥
    
    Returns:
        str: 优化后的prompt文本
    """
    # DeepSeek system prompt - 将「非关键部分」翻译为更短英文、关键片段保持原样
    system_prompt = '''你是一名专业的 System Prompt 压缩与翻译专家。
目标：在不改变语义与可执行性的前提下，将输入的中文 System Prompt 中「非关键部分」翻译为更短、更直接的英文，以降低token数；同时严格保留「关键部分」原样（语言、标点、结构均不可变）。

必须原样保留（禁止翻译/改写/改动）：
- 任何以「输出格式」「输出格式，非json」「请按如下JSON格式输出」「Format」开头的段落；
- 这些段落内的所有括号、引号、字段名、枚举字面量、大小写与结构；
- 明确的限制/约束/排除机制/注意事项中影响判定边界的条款；
- “非json/JSON”之类的格式声明文本。

允许翻译与压缩（需保持决策边界不变）：
- 角色说明、技能/能力描述、背景解读、非必要冗余；
- 冗长示例中的赘述（保留界定边界的关键信息即可）；
- 合并近义与重复表述、改写为短句、使用更直接的动词名词；
- 英文输出应简洁、指令化、条目化，避免情绪化或口语化。

禁止事项：
- 不得新增/删除字段、不得更名或更改枚举字面量；
- 不得修改保留段落的任何字符（含中英文标点与空格位置）；
- 不得改变“非json/JSON”声明与示例结构。

判断准确率的标准：
- 我会将处理后的System Prompt和原始的System Prompt在相同的User Prompt输入deepseek模型，对比输出结果，如果输出内容中某些字段（例如：level、isPerson、predictLevel、isduplicate）的值不一致，则判定为不准确；
- 所以处理的结果应该尽量保证约束这些字段的条件不被压缩和省略，以此来保证准确率更高。

输出要求：
- 直接输出处理后的 System Prompt 全文；
- 不添加解释、标题或前后缀；
- 确保结果可直接作为模型的 System Prompt 使用；
- 若某句无法安全压缩或翻译，会影响输出格式或判定边界，则保持该句原样中文不变。
'''

    # 使用OpenAI SDK调用DeepSeek API
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/")
    
    print('正在使用DeepSeek优化prompt，请稍候...')
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.3,  # 降低温度以获得更稳定的结果
            stream=False
        )
        
        optimized = response.choices[0].message.content.strip()
        print('\nDeepSeek优化结果：')
        print('='*50)
        print(optimized)
        print('='*50)
        return optimized
    except Exception as e:
        print(f'DeepSeek API调用出错: {e}')
        return None

def compress_with_llmlingua(prompt_text, api_key, target_tokens=500):
    """
    使用DeepSeek判断 + LLMLingua-2压缩prompt
    
    Args:
        prompt_text (str): 要压缩的prompt文本
        api_key (str): DeepSeek API密钥
        target_tokens (int): 目标token数，默认500
    
    Returns:
        dict: 包含压缩结果的字典，包括compressed_prompt, ratio
    """
    print('\n正在使用DeepSeek判断可压缩部分...')
    
    try:
        # 第一步：使用DeepSeek判断哪些部分可以压缩
        system_prompt = '''你是一个prompt分析专家。请分析以下prompt，将其分为两部分：
1. 可压缩部分：可以简化、缩写或删除的部分
2. 不可压缩部分：必须保持原样的部分（如输出格式、关键指令等）
请按以下格式输出：
===可压缩部分===
[这里放可以压缩的内容]

===不可压缩部分===
[这里放不能压缩的内容]

注意：
- 输出格式部分（如"输出格式：{...}"）必须放在不可压缩部分
- 角色描述、技能描述、例子等可以放在可压缩部分
- 确保两部分的内容完整覆盖原prompt'''

        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=1,
                stream=False
            )
            
            analysis_result = response.choices[0].message.content.strip()
            print('\nDeepSeek分析结果：')
            print('='*50)
            print(analysis_result)
            print('='*50)
            
        except Exception as e:
            print(f'DeepSeek分析失败: {e}')
            return None
        
        # 第二步：解析DeepSeek的分析结果
        import re
        compressible_match = re.search(r'===可压缩部分===\s*(.*?)(?=\s*===不可压缩部分===)', analysis_result, re.DOTALL)
        non_compressible_match = re.search(r'===不可压缩部分===\s*(.*?)(?=\s*$)', analysis_result, re.DOTALL)
        
        if not compressible_match or not non_compressible_match:
            print('无法解析DeepSeek的分析结果，回退到普通压缩')
            return compress_with_llmlingua_fallback(prompt_text, target_tokens)
        
        compressible_part = compressible_match.group(1).strip()
        non_compressible_part = non_compressible_match.group(1).strip()
        
        print(f'\n可压缩部分长度: {len(compressible_part)} 字符')
        print(f'不可压缩部分长度: {len(non_compressible_part)} 字符')
        
        # 第三步：使用LLMLingua压缩可压缩部分
        use_long = len(compressible_part) >= LONG_INPUT_THRESHOLD_CHARS
        if use_long:
            print('\n正在使用LongLLMLingua压缩可压缩部分...')
        else:
            print('\n正在使用LLMLingua-2压缩可压缩部分...')

        llm_lingua = create_prompt_compressor(use_long)

        if use_long:
            # 长文本：直接整段压缩（采用 LongLLMLingua 官方参数）
            try:
                result = llm_lingua.compress_prompt(
                    compressible_part,
                    rate=0.8,
                    force_tokens=['\n', '?', '：', '。', '，', ' ', '，', '；', '！', '（', '）'],
                    condition_in_question="after_condition",
                    reorder_context="sort",
                    dynamic_context_compression_ratio=0.3,
                    condition_compare=True,
                    context_budget="+100",
                    rank_method="longllmlingua",
                )
                compressed_part = result.get('compressed_prompt', '') if isinstance(result, dict) else str(result)
                compressed_part = clean_compressed_text(compressed_part)
                print(f'整段压缩完成，从{len(compressible_part)}字符压缩到{len(compressed_part)}字符')
            except Exception as compress_error:
                print(f"整段压缩失败: {compress_error}")
                # 回退到分段
                print('回退：启用分段压缩。')
                max_chars = 512
                segments = smart_split_text(compressible_part, max_chars)
                compressed_segments = []
                for i, segment in enumerate(segments):
                    print(f'正在压缩第{i+1}段（{len(segment)}字符）...')
                    try:
                        result = llm_lingua.compress_prompt(
                            segment,
                            rate=0.8,
                            force_tokens=['\n', '?', '：', '。', '，', ' ', '，', '；', '！', '（', '）'],
                            condition_in_question="after_condition",
                            reorder_context="sort",
                            dynamic_context_compression_ratio=0.3,
                            condition_compare=True,
                            context_budget="+100",
                            rank_method="longllmlingua",
                        )
                        compressed_segment = result.get('compressed_prompt', '') if isinstance(result, dict) else str(result)
                        compressed_segment = clean_compressed_text(compressed_segment)
                        compressed_segments.append(compressed_segment)
                        print(f'第{i+1}段压缩完成，从{len(segment)}字符压缩到{len(compressed_segment)}字符')
                    except Exception as e2:
                        print(f"第{i+1}段压缩失败: {e2}")
                        compressed_segments.append(segment)
                compressed_part = '\n\n'.join(compressed_segments)
        else:
            # 短文本：保留原有的分段逻辑（小段更稳）
            max_chars = 512
            if len(compressible_part) > max_chars:
                print(f'警告：可压缩部分过长（{len(compressible_part)}字符），将分段压缩')
                segments = smart_split_text(compressible_part, max_chars)
                print(f'将分为{len(segments)}段进行压缩')
            else:
                segments = [compressible_part]

            compressed_segments = []
            for i, segment in enumerate(segments):
                print(f'正在压缩第{i+1}段（{len(segment)}字符）...')
                try:
                    result = llm_lingua.compress_prompt(
                        segment,
                        rate=0.8,
                        force_tokens=['\n', '?', '：', '。', '，', ' ', '，', '；', '！', '（', '）']
                    )
                    if isinstance(result, dict):
                        compressed_segment = result.get('compressed_prompt', '')
                    else:
                        compressed_segment = str(result)
                    compressed_segment = clean_compressed_text(compressed_segment)
                    compressed_segments.append(compressed_segment)
                    print(f'第{i+1}段压缩完成，从{len(segment)}字符压缩到{len(compressed_segment)}字符')
                except Exception as compress_error:
                    print(f"第{i+1}段压缩失败: {compress_error}")
                    compressed_segments.append(segment)

            compressed_part = '\n\n'.join(compressed_segments)
        
        # 第四步：拼接压缩后的可压缩部分和不可压缩部分
        final_prompt = compressed_part + '\n\n' + non_compressible_part
        
        ratio = len(final_prompt) / len(prompt_text) if len(prompt_text) > 0 else 0
        
        print('\n最终压缩结果：')
        print('='*50)
        print(final_prompt)
        print('='*50)
        
        return {
            'compressed_prompt': final_prompt,
            'ratio': ratio
        }
        
    except Exception as e:
        print(f'智能压缩失败: {e}')
        print(f'错误详情: {type(e).__name__}: {str(e)}')
        return None

def compress_with_llmlingua_fallback(prompt_text, target_tokens=500):
    """
    回退的普通LLMLingua压缩方法
    """
    print('\n使用回退的普通LLMLingua压缩...')
    
    try:
        use_long = len(prompt_text) >= LONG_INPUT_THRESHOLD_CHARS
        llm_lingua = create_prompt_compressor(use_long)

        if use_long:
            print('长文本：尝试整段压缩...')
            try:
                result = llm_lingua.compress_prompt(
                    prompt_text,
                    rate=0.8,
                    force_tokens=['\n', '?', '：', '。', '，', ' ', '，', '；', '！', '（', '）'],
                    condition_in_question="after_condition",
                    reorder_context="sort",
                    dynamic_context_compression_ratio=0.3,
                    condition_compare=True,
                    context_budget="+100",
                    rank_method="longllmlingua",
                )
                final_prompt = result.get('compressed_prompt', '') if isinstance(result, dict) else str(result)
                final_prompt = clean_compressed_text(final_prompt)
                print(f'整段压缩完成，从{len(prompt_text)}字符压缩到{len(final_prompt)}字符')
            except Exception as e:
                print(f'整段压缩失败：{e}，回退到分段压缩。')
                max_chars = 512
                segments = smart_split_text(prompt_text, max_chars)
                compressed_segments = []
                for i, segment in enumerate(segments):
                    print(f'正在压缩第{i+1}段（{len(segment)}字符）...')
                    try:
                        result = llm_lingua.compress_prompt(
                            segment,
                            rate=0.8,
                            force_tokens=['\n', '?', '：', '。', '，', ' ', '，', '；', '！', '（', '）'],
                            condition_in_question="after_condition",
                            reorder_context="sort",
                            dynamic_context_compression_ratio=0.3,
                            condition_compare=True,
                            context_budget="+100",
                            rank_method="longllmlingua",
                        )
                        compressed_segment = result.get('compressed_prompt', '') if isinstance(result, dict) else str(result)
                        compressed_segment = clean_compressed_text(compressed_segment)
                        compressed_segments.append(compressed_segment)
                        print(f'第{i+1}段压缩完成，从{len(segment)}字符压缩到{len(compressed_segment)}字符')
                    except Exception as e2:
                        print(f"第{i+1}段压缩失败: {e2}")
                        compressed_segments.append(segment)
                final_prompt = '\n\n'.join(compressed_segments)
        else:
            max_chars = 512
            if len(prompt_text) > max_chars:
                print(f'警告：prompt过长（{len(prompt_text)}字符），将分段压缩')
                segments = smart_split_text(prompt_text, max_chars)
                print(f'将分为{len(segments)}段进行压缩')
            else:
                segments = [prompt_text]

            compressed_segments = []
            for i, segment in enumerate(segments):
                print(f'正在压缩第{i+1}段（{len(segment)}字符）...')
                try:
                    result = llm_lingua.compress_prompt(
                        segment,
                        rate=0.8,
                        force_tokens=['\n', '?', '：', '。', '，', ' ', '，', '；', '！', '（', '）']
                    )
                    if isinstance(result, dict):
                        compressed_segment = result.get('compressed_prompt', '')
                    else:
                        compressed_segment = str(result)
                    compressed_segment = clean_compressed_text(compressed_segment)
                    compressed_segments.append(compressed_segment)
                    print(f'第{i+1}段压缩完成，从{len(segment)}字符压缩到{len(compressed_segment)}字符')
                except Exception as compress_error:
                    print(f"第{i+1}段压缩失败: {compress_error}")
                    compressed_segments.append(segment)

            final_prompt = '\n\n'.join(compressed_segments)
        
        ratio = len(final_prompt) / len(prompt_text) if len(prompt_text) > 0 else 0
        
        return {
            'compressed_prompt': final_prompt,
            'ratio': ratio
        }
        
    except Exception as e:
        print(f'回退压缩也失败: {e}')
        return None

def main():
    """主函数，协调整个优化流程"""
    print('='*60)
    print('Prompt优化工具')
    print('='*60)
    
    # 确定运行模式
    if RUN_MODE['both']:
        mode = 'both'
        print('运行模式：DeepSeek优化 + LLMLingua压缩')
    elif RUN_MODE['deepseek_only']:
        mode = 'deepseek_only'
        print('运行模式：仅DeepSeek优化')
    elif RUN_MODE['llmlingua_only']:
        mode = 'llmlingua_only'
        print('运行模式：仅LLMLingua压缩')
    else:
        print('错误：未选择任何运行模式，请在RUN_MODE中设置')
        return
    
    print('='*60)
    
    original_length = len(prompt)
    print(f'原始prompt: {original_length} 字符')
    print('='*60)
    
    if mode in ['deepseek_only', 'both']:
        # 第一步：使用DeepSeek优化
        optimized_prompt = optimize_with_deepseek(prompt, api_key)
        
        if optimized_prompt is None:
            print('DeepSeek优化失败，程序退出')
            return
        
        optimized_length = len(optimized_prompt)
        print(f'\nDeepSeek优化效果：')
        print(f'优化后: {optimized_length} 字符')
        print(f'减少: {original_length - optimized_length} 字符')
        print(f'压缩率: {((original_length - optimized_length) / original_length * 100):.1f}%')
        
        if mode == 'deepseek_only':
            return  # 如果只运行DeepSeek，到这里结束
    
    if mode in ['llmlingua_only', 'both']:
        # 第二步：使用LLMLingua压缩
        input_text = optimized_prompt if mode == 'both' else prompt
        compression_result = compress_with_llmlingua(input_text, api_key)
        
        if compression_result is not None:
            compressed_prompt = compression_result['compressed_prompt']
            ratio = compression_result['ratio']
            
            print(f'\nLLMLingua压缩效果：')
            print(f'压缩后: {len(compressed_prompt)} 字符')
            print(f'压缩比: {ratio:.2f}')
            
            if mode == 'both':
                # 显示完整的压缩效果统计
                print(f'\n总压缩效果：')
                print(f'原始prompt: {original_length} 字符')
                print(f'DeepSeek优化后: {optimized_length} 字符')
                print(f'LLMLingua压缩后: {len(compressed_prompt)} 字符')
                print(f'DeepSeek优化压缩率: {((original_length - optimized_length) / original_length * 100):.1f}%')
                print(f'总压缩率: {((original_length - len(compressed_prompt)) / original_length * 100):.1f}%')
            else:
                # 只运行LLMLingua的情况
                print(f'减少: {original_length - len(compressed_prompt)} 字符')
                print(f'压缩率: {((original_length - len(compressed_prompt)) / original_length * 100):.1f}%')
        else:
            print('\nLLMLingua压缩失败')
            if mode == 'llmlingua_only':
                print('程序退出')
                return

if __name__ == "__main__":
    main()
    


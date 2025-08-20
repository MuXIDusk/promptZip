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

def split_text_by_punctuation(text, max_chars=512):
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
prompt = '''你是一个内容理解与人物识别专家，请根据输入的视频标题与内容判断其是否属于【人物类内容】，并输出清晰的评估结果，帮助我筛选具备“人物打造与运营潜力”的内容；请先判断其归属于网络红人类或素人类，再根据对应维度进行判定。

对素人类和网络红人类内容，需严格满足以下两项，方可进入人物类内容判断细则：1. 个人形象或者事迹有传播、放大可能性；
2. 个人属性较强有特点/有故事性/经历波折有反差，适合延展，可做当事人回应或者内容玩法以及内容扩充。

一、网络红人类
只筛选：网红人物重大突发事件，例如：结婚、领证、离婚、去世等

二、素人类
1. 人物特质突出
    正例：标题突出人物身份、特质、技能、经历
    例：「90岁奶奶跳民族舞惊艳网友」「异瞳男孩沙漠治沙10年」
    反例：人物仅作为事件背景，不具辨识度
    例：「游客与商贩争执」「男子当街打人」

2. 人物成长或成就
    正例：体现个人突破、正向成长、社会影响
    例：「28岁外卖员考上清华」「非遗传人月入三千坚持传承」
    反例：人物处于被动或负面情境
    例：「男子赌博遭殴打」「女子黄体破裂入院」

3. 背景故事完整性
    正例：标注人物身份、经历、情节细节
    例：「3岁失明却自学芭蕾」「#包子姐二登央视舞台。王霞，初中文化，在镇上经营着一家包子铺，为鼓励儿子学习，她一边包包子一边自学英语，靠一口流利英语走红网络。 #包子姐   #用平凡造就非凡 」
    反例：信息空泛，无人物维度延伸
    例：「男子地铁打人」「女子摘枇杷被拘」

4. 结构以具体人物为主导
    正例：人物为主语，动作为核心
    例：「云南男孩徒步千里」「新疆姑娘跳民族舞惊艳众人」
    反例：事件主导，人物弱化或缺席
    例：「高铁上发生激烈冲突」「游客遭遇勒索」「有一种痛叫消防员觉得你痛有一种痛叫消防员觉得你痛！」

5. 人设可延展性（高权重）
    正例：具备持续人物塑造空间（如励志青年、民俗达人、高龄偶像），而不仅仅是事件本身的内容延展。
    例：「从沙漠走出的工程师」「80岁奶奶成健身博主」
    反例：事件单一，缺乏后续运营空间
    例：「跳楼男子救回」「男子为讨薪裸奔」「8旬老人走失17小时获救#今日平江  惊险17小时！平江蓝天救援队成功救援一名8旬走失老人 #蓝天救援队 #救援 #公益」

【内容排除机制】：
为确保人物内容具备较强的传播性与运营价值，以下特定来源或角色类型不纳入“人物类内容”识别范围：
1）涉及政治、军事的所有人物内容；例如：「尹锡悦抵达特检组接受调查」、「李竟被双开」
2）涉及明星相关内容（如明星个人动态、影视综艺参与、公众讨论、八卦新闻、角色出演、二创混剪等）；例如：「大S离世后小S首度公开亮相」
3）涉及运动员及其比赛表现展开（如夺冠、受伤、退赛、赛场情绪等）；例如：「永康队刚满18岁的准大学生厉昱呈」
4）涉及已故历史人物（如古代帝王、历史事件人物、近代已定型人物传记等）；例如：「白求恩临终前仍惦记伤员」
5）涉及明显负面冲突、暴力恐吓、极端维权或过度猎奇元素（如：被骗追款、死亡威胁、畸形奇观等）；例如：「女孩被拐遭养父虐待致终身伤残」
6）涉及残疾人内容，且仅以生理差异、悲情卖惨或猎奇为主导
7）涉及寻亲报道内容

【注意点】：
1.视频标题若包含「走红」「爆火」等关键词，是一个非常明确的人物类热点信号，可直接输出为【人物类内容】和【预测运营包装潜力：高】；
2.素人类及网络红人类判断的五个细分维度，必须全部满足才可判断为【预测运营包装潜力：高】，模型执行时不得“均值考虑”，不能满足 1-2 项就判断为【预测运营包装潜力：高】。

【快速决策流程图】
1. 是否有明确人物作为主角？ → 否 → 排除
2. 是否具备人物特质/成就/情感表达？ → 否 → 排除
3. 是否能引发对“人物本身”的关注？ → 否 → 排除
4. 是否有潜在人设可持续塑造？ → 是 → 判定为人物类内容

限制：
当人物类内容输出为‘否’时，可不做接下来的判断。

【输出格式】
请按如下JSON格式输出：
{
"isPerson": "是", // 是否为人物类内容：「是」或者「否」
"predictLevel": "高", // 预测运营包装潜力：高、中、低
"reason": "这里写判断原因，简洁说明得分维度和关键要素（不少于2句）",
"personType": "名人明星类", // <人物分类：（素人类/网络红人类）>  
"eventType": "吸毒", // <事件类型：不超过5个字（如离婚、去世）>
"recommend": "这里写推荐备注，如具备成长叙事、可延展运营、适合情感包装等"
}
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
    # DeepSeek system prompt - 更明确的指令
    system_prompt = '''你是一个专业的prompt优化专家，擅长将复杂的中文prompt转换为简洁高效的英文版本。
你的任务是将用户提供的prompt转换为token数更少的英文版本，同时保持其核心功能和效果。
**严格遵循以下规则：**
1. **绝对不可翻译的部分**：
   - 所有以"输出格式"、"格式"、"Format"开头的部分及其完整内容
   - 任何JSON格式、代码块或特殊格式要求
2. **优化策略**：
   - 删除冗余的修饰词、重复表达
   - 将长句拆分为简洁的短句
   - 使用更直接的动词和名词
   - 避免不必要的副词、介词、连词
   - 保持逻辑清晰和指令明确
3. **输出要求**：
   - 直接输出优化后的英文prompt
   - 不要添加任何解释或说明
   - 确保输出格式部分完全保持原样（包括中文内容）
   - 保持prompt的完整性和可执行性
**质量检查**：确保优化后的prompt仍然能够准确传达原始意图，并且更容易被AI模型理解和执行。'''


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
- 输出格式部分（如"输出格式，非json：{...}"）必须放在不可压缩部分
- 比较关键的指令通常放在不可压缩部分，不太关键的可以压缩
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
        
        # 第三步：使用LLMLingua-2压缩可压缩部分
        print('\n正在使用LLMLingua-2压缩可压缩部分...')
        
        llm_lingua = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="cpu"
        )
        
        # 处理可压缩部分，如果过长则分段压缩
        max_chars = 512  # 每段最大字符数（确保不超过512个token）
        if len(compressible_part) > max_chars:
            print(f'警告：可压缩部分过长（{len(compressible_part)}字符），将分段压缩')
            # 优先按标点分段
            segments = split_text_by_punctuation(compressible_part, max_chars)
            print(f'将分为{len(segments)}段进行压缩')
        else:
            segments = [compressible_part]
        
        # 分别压缩每一段
        compressed_segments = []
        for i, segment in enumerate(segments):
            print(f'正在压缩第{i+1}段（{len(segment)}字符）...')
            try:
                result = llm_lingua.compress_prompt(
                    segment,
                    rate=0.8,
                    force_tokens=['\n', '?', '：', '。', '，', ' ', '，', '；', '！', '（', '）']
                )
                
                # 处理压缩结果
                if isinstance(result, dict):
                    compressed_segment = result.get('compressed_prompt', '')
                else:
                    compressed_segment = str(result)
                
                # 清理压缩后的多余空格
                compressed_segment = clean_compressed_text(compressed_segment)
                compressed_segments.append(compressed_segment)
                print(f'第{i+1}段压缩完成，从{len(segment)}字符压缩到{len(compressed_segment)}字符')
                
            except Exception as compress_error:
                print(f"第{i+1}段压缩失败: {compress_error}")
                # 如果压缩失败，保留原段
                compressed_segments.append(segment)
        
        # 合并所有压缩后的段落
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
        llm_lingua = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="cpu"
        )
        
        max_chars = 512  # 每段最大字符数（确保不超过512个token）
        if len(prompt_text) > max_chars:
            print(f'警告：prompt过长（{len(prompt_text)}字符），将分段压缩')
            # 优先按标点分段
            segments = split_text_by_punctuation(prompt_text, max_chars)
            print(f'将分为{len(segments)}段进行压缩')
        else:
            segments = [prompt_text]
        
        # 分别压缩每一段
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
                
                # 清理压缩后的多余空格
                compressed_segment = clean_compressed_text(compressed_segment)
                compressed_segments.append(compressed_segment)
                print(f'第{i+1}段压缩完成，从{len(segment)}字符压缩到{len(compressed_segment)}字符')
                
            except Exception as compress_error:
                print(f"第{i+1}段压缩失败: {compress_error}")
                # 如果压缩失败，保留原段
                compressed_segments.append(segment)
        
        # 合并所有压缩后的段落
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
    


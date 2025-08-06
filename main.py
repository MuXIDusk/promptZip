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
    
    return text

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
你是一个视频内容分析专家，擅长理解用户输入的视频内容描述，并能将当前视频内容与线上事件进行匹配。你的能力可以帮助用户判断视频内容是否与在线事件重复。

## 技能
### 技能1：视频内容理解
- 解和分析用户输入的视频内容。
- 理解视频内容的主题、关键词和相关信息。
- 提取视频的核心要素：主体、事件

### 技能2：在线事件理解
- 分析每个在线事件的含义和内容。

### 技能3：重复识别
只有同一件事，才能算作重复。如下列情况：
规则1：文案完全一致
如：人民文娱发文评“配角上桌” 与 人民文娱评配角上桌；
河南位列热门目的地第七 与 河南位列热门目的地第七
规则2：涉及地点+人物+事件完全一致
如：遭同学杀害男孩父亲发声 与 遭同学杀害男孩父亲发朋友圈；
女子买肉被坑偷拍下证据 与 女子多次买排骨被坑后拍下证据15投
规则3：没有任何信息增量
同一件事相同节点、同一件事同一发声主体、同一事件同一进展
规则4：涉及地点、人物、事件、表述完全一致
规则5：同一件事的不同进展不能判定为重复，不同进展指的是事件回应、时间进展、事件通报，其都不能应该与事件本身算作重复。
如：【沈阳通报健康证办理乱象】与【央视曝光多地健康证办理乱象】不是重复，虽然主体事件都是健康证办理乱象，但是节点不同，一个节点是沈阳通报了这件事，一个是央视曝光了这件事，因此不能算为重复。

### 技能4：重复度打分
1.给出重复得分，满分10分，可以精确到小数点后两位
2.打分依据所输入的视频标题与在线事件对比后，其信息增量的程度来进行。

### 例子
1.【下午到夜间大部地区有雷阵雨】与【广西象州水塘干涸旱情持续】不是重复：
step1地点不同：‘未点名地点’与‘广西象州’
step2 事件不同：‘雷阵雨’与‘干涸旱情’
step3主体不同：'未点明'与'广西象州'
step4 地点、事件、主体皆不同，所以不是重复
2.【爱护花草善待动物】与【武汉首办动物保护灯光秀】不是重复
step1地点不同：‘未点名地点’与‘武汉’
step2 事件不同：‘爱护花草动物 ’与‘动物保护灯光秀’
step3主体不同：‘未点名地点’与‘武汉’
step4 地点、事件、主体皆不同，所以不是重复
3. 【外国游客为张家界风景而流泪】与【游客乘坐张家界百龙天梯哇声一片】不是重复：
step1地点相同：‘张家界’与‘张家界’
step2 事件不同：‘为张家界风景而流泪’与‘乘坐张家界百龙天梯哇声一片’
step3主体不同：'外国游客'与'游客'
step4 地点相同、但事件、主体不同，所以不是重复

 输出格式，非json：
{
 "isduplicate": <"是"/"否">,
 "duplicateWord": <在线事件名称>,
 "reason": <判断原因>,
 "score": <重复分数>
}

## 约束
-视频标题中的tag只作参考，不可依赖tag进行重复性判定。
-只有地点、事件、主体、地点完全相同才算重复，任意一元素有不同即算不重复。
-当输入视频为在线事件的回应、通报时，不能算作重复，必须给出9分以下的评分。
- 使用规定的输出格式。
- 仅限于判断视频内容描述和在线事件重复的问题。
- 如果用户未输入事件列表，则无重复在线事件。
- 无重复在线事件时，<在线事件名称>为“无”。
- 确保事件来自于用户输入，避免擅自编造事件名称。
- 如果至少存在一个热词与视频内容重复，则判定为重复。
- 避免回答与判定重复在线事件无关的问题。
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
            temperature=1.0,  # 降低温度以获得更稳定的结果
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
            # 分段压缩
            segments = []
            for i in range(0, len(compressible_part), max_chars):
                segment = compressible_part[i:i + max_chars]
                segments.append(segment)
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
            # 分段压缩
            segments = []
            for i in range(0, len(prompt_text), max_chars):
                segment = prompt_text[i:i + max_chars]
                segments.append(segment)
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
    


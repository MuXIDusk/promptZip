# Prompt压缩工具

一个智能的Prompt压缩优化工具，支持DeepSeek语义优化和LLMLingua文本压缩。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

在项目根目录创建 `config.json` 文件：

```json
{
  "api_key": "YOUR_DEEPSEEK_API_KEY"
}
```

将 `YOUR_DEEPSEEK_API_KEY` 替换为你的真实DeepSeek API密钥。

### 3. 修改运行模式

在 `main.py` 中找到 `RUN_MODE` 配置，选择运行模式：

```python
RUN_MODE = {
    'deepseek_only': False,   # 只运行DeepSeek优化
    'llmlingua_only': False,  # 只运行LLMLingua压缩
    'both': True              # 两者都运行（推荐）
}
```

**运行模式说明：**
- `deepseek_only`: 仅使用DeepSeek进行中文到英文的语义优化
- `llmlingua_only`: 仅使用LLMLingua进行文本压缩
- `both`: 先DeepSeek优化再LLMLingua压缩，获得最佳效果

### 4. 修改Prompt内容

在 `main.py` 中找到 `prompt` 变量，替换为你的prompt内容：

```python
prompt = '''# 角色
你是一个视频内容分析专家，擅长理解用户输入的视频内容描述...

## 技能
### 技能1：视频内容理解
- 解和分析用户输入的视频内容...

# 在这里写入你的prompt内容
'''
```

### 5. 运行脚本

如果使用虚拟环境：
```bash
source venv/bin/activate && python3 main.py
```

如果未使用虚拟环境：
```bash
python3 main.py
```

### 6. 查看结果

脚本会在终端输出详细的压缩过程和结果，包括：
- 原始prompt长度
- 各阶段压缩结果
- 最终压缩率
- 压缩后的prompt内容

## 输出示例

```
============================================================
Prompt优化工具
============================================================
运行模式：DeepSeek优化 + LLMLingua压缩
============================================================
原始prompt: 1488 字符
============================================================

DeepSeek优化效果：
优化后: 1587 字符
减少: -99 字符
压缩率: -6.7%

LLMLingua压缩效果：
压缩后: 1273 字符
压缩比: 0.80

总压缩效果：
原始prompt: 1488 字符
DeepSeek优化后: 1587 字符
LLMLingua压缩后: 1273 字符
总压缩率: 14.4%
```

## 常见问题

**Q: 提示"缺少config.json配置文件"？**
A: 在项目根目录创建config.json文件，并添加你的DeepSeek API密钥。

**Q: 提示"Token indices sequence length is longer than 512"？**
A: 这是正常的，脚本会自动分段处理长文本，无需担心。

**Q: DeepSeek优化后长度反而增加了？**
A: 这是正常现象，因为中文转英文可能增加字符数，但会减少token数。最终的LLMLingua压缩会显著减少长度。

**Q: 如何获得DeepSeek API密钥？**
A: 访问 [DeepSeek官网](https://www.deepseek.com/) 注册账号并获取API密钥。 
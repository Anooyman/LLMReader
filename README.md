# PDFReader

PDFReader 是一个基于大语言模型（LLM）的 PDF 文档分析与智能问答工具，支持多种 LLM provider（Azure OpenAI、OpenAI、Ollama），可自动将 PDF 按页转图片、提取内容、生成摘要、构建向量数据库并支持多轮智能问答。

---

## 功能简介

- 支持将 PDF 按页转为图片并用 LLM 提取内容
- 自动分析并输出基于章节的 summary
- 构建向量数据库，支持高效内容检索
- 支持基于内容的智能问答，推荐带章节名提问以提升检索效果
- 支持多轮对话，自动缓存和复用历史检索内容
- 支持多种 LLM provider（Azure OpenAI、OpenAI、Ollama）
- 所有缓存和中间数据自动存储于 `data/` 目录下，便于管理

---

## 更新日志

- 2025-07-23
  - 新增摘要文件导出功能，支持将简要摘要（brief_summary）和详细摘要（detail_summary）导出至 data/output 文件夹
  - 支持两种文件格式：Markdown（.md）和 PDF（.pdf），满足不同场景需求
  - 新增控制参数 save_data_flag，用于开关文件导出功能，默认值为 True（自动导出）
  - 优化文件生成逻辑：仅当目标文件不存在时才会生成，已存在的文件将自动跳过，减少重复计算和 IO 操作

---

## 环境与依赖

- Python 3.12+
- 推荐使用虚拟环境
- 安装依赖：

```bash
pip install -r requirements.txt
```

---

## 目录结构

```
PDFReader/
├── data/
│   ├── json_data/         # 自动生成的内容 json
│   ├── output/            # 自动生成的文章摘要
│   ├── pdf/               # 你的 PDF 文件请放在这里(需手动创建)
│   ├── pdf_image/         # 自动生成的图片
│   └── vector_db/         # 自动生成的向量数据库
├── src/
│   ├── __init__.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── config.py      # 配置与路径常量
│   │   ├── llm.py         # LLM 封装与多 provider 支持
│   │   ├── mcp_client.py  # mcp clinet 封装
│   │   └── utility.py     # 工具函数
│   ├── reader/
│   │   ├── __init__.py
│   │   ├── pdf_reader.py  # PDF 的处理与问答主流程
│   └── main.py
├── .gitignore
├── README.md
└── requirements.txt
```

- **所有缓存和中间数据均自动存储在 `data/` 目录下**，无需手动管理。
- repo 中 data 目录下仅存放样例的 pdf 和 output

---

## 使用方法

1. **准备 PDF 文件**
   - 请将你要分析的 PDF 文件放入 `data/pdf/` 目录下。
   - 例如：`data/pdf/your_file.pdf`

2. **运行主程序**
   - 在项目根目录下运行：
   ```bash
   python main.py
   ```
   - 启动后会自动处理默认 PDF 文件（可在代码中修改），或根据提示输入 PDF 文件名（如 `your_file.pdf`），无需包含路径。
   - 自动生成简要摘要（brief_summary）和详细摘要（detail_summary）导出至 data/output 文件夹。

3. **交互问答**
   - 首次运行会自动将 PDF 按页转为图片，提取每页内容，并输出章节结构 summary。
   - 程序会自动在 `data/` 目录下生成图片、json、向量数据库等缓存文件，加速后续访问。
   - 你可以在交互模式下继续提问，**建议提问时带上章节名**，如：
     > “请详细解释 Introduction 章节的主要内容”
   - 输入 `退出`、`再见`、`bye`、`exit`、`quit` 可结束对话。

---

## LLM Provider 支持

- 支持 Azure OpenAI、OpenAI、Ollama 三种 LLM provider。
- 可在 `PDFReader` 初始化时通过 `provider` 参数指定（默认 `azure`）。
- 相关 API key、endpoint、模型等配置请在 `src/config.py` 中修改。

---

## 缓存与数据存储说明

- **图片缓存**：`data/pdf_image/`，每个 PDF 会生成对应的图片文件夹。
- **内容缓存**：`data/json_data/`，每个 PDF 会生成对应的内容 JSON 文件。
- **向量数据库**：`data/vector_db/`，每个 PDF 会生成对应的向量数据库文件夹。
- **缓存机制**：已处理过的 PDF 会自动复用缓存，加速后续访问。

---

## 进阶说明

- **多轮对话**：每轮对话会自动复用历史检索内容，提升上下文连贯性和效率。
- **章节检索**：每次 summary 输出后，建议后续提问时直接引用 summary 中的章节标题，检索更精准。
- **自定义参数**：如需调整分块大小、缓存路径等，可在 `config.py` 或 `pdf_reader.py` 构造参数中修改。
- **异常处理**：如遇部分图片内容提取失败，会有详细日志提示，可手动补充 JSON 数据。

---

## 常见问题

- **Q: PDF 文件未被识别？**
  - 请确保 PDF 文件已放入 `data/pdf/` 目录，文件名输入正确。
- **Q: LLM API 报错？**
  - 请检查 `config.py` 中的 API key、endpoint、模型等配置是否正确。
- **Q: 处理速度慢？**
  - 首次处理大文件时会较慢，后续会自动复用缓存。
- **Q: 如何切换 LLM provider？**
  - 在 `PDFReader` 初始化时传入 `provider` 参数，如 `PDFReader(provider="openai")`。

---

## 许可协议

MIT 
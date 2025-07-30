# PDFReader

PDFReader 是一个基于大语言模型（LLM）的文档分析与智能问答工具，支持 PDF 和网页内容解析，支持多种 LLM provider（Azure OpenAI、OpenAI、Ollama），可自动提取内容、生成摘要、构建向量数据库并支持多轮智能问答。

---

## 功能简介

- **多格式支持**：支持 PDF 文档和网页 URL 内容解析
- **内容提取**：支持将 PDF 按页转为图片并用 LLM 提取内容，通过 MCP 服务获取网页信息
- **智能切分**：根据内容长度自动切分文本，确保最佳处理效果
- **自动摘要**：自动分析并输出基于章节的 summary
- **向量数据库**：构建向量数据库，支持高效内容检索
- **智能问答**：支持基于内容的智能问答，推荐带章节名提问以提升检索效果
- **多轮对话**：支持多轮对话，自动缓存和复用历史检索内容
- **多 provider 支持**：支持多种 LLM provider（Azure OpenAI、OpenAI、Ollama）
- **数据导出**：自动生成简要摘要（brief_summary）和详细摘要（detail_summary）并导出至 data/output 文件夹

---

## 更新日志

- 2025-07-30
  - 新增 Web Reader 功能，支持通过 URL 解析网页内容
  - Web Reader 支持 PDF Reader 的所有功能，包括内容提取、摘要生成和智能问答
  - 通过 MCP 服务获取网页信息，并根据内容长度自动切分
  - 已知问题: 当 URL 内容特别长时，信息拉取可能会错乱，可以重跑获取正确信息; 部分网站无法抓取

- 2025-07-23
  - 新增摘要文件导出功能，支持将简要摘要和详细摘要导出至 data/output 文件夹
  - 支持 Markdown 和 PDF 两种导出格式
  - 新增 save_data_flag 控制参数，默认自动导出

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
│   ├── json_data/         # 自动生成的内容JSON文件
│   ├── output/            # 自动导出的摘要文件（按文档分目录）
│   ├── pdf/               # 存放待处理的PDF文件（需手动创建）
│   ├── pdf_image/         # PDF转图片的缓存
│   └── vector_db/         # 自动生成的向量数据库
├── src/
│   ├── config/            # 配置文件目录
│   │   └── settings.py    # 系统配置（含LLM参数）
│   ├── core/              # 核心功能模块
│   │   ├── llm/           # LLM服务相关
│   │   ├── processing/    # 内容处理
│   │   └── vector_db/     # 向量数据库操作
│   ├── readers/           # 解析器模块
│   │   ├── base.py        # 基础解析器
│   │   ├── pdf.py         # PDF解析器
│   │   └── web.py         # 网页解析器
│   ├── services/          # 外部服务接口
│   │   └── mcp_client.py  # MCP服务客户端
│   └── utils/             # 工具函数
├── .gitignore
├── README.md
├── main.py                # 程序入口
└── requirements.txt       # 依赖列表

```

- **所有缓存和中间数据均自动存储在 `data/` 目录下**，无需手动管理。
- repo 中 data 目录下仅存放样例的 pdf 和 output

---

## 使用方法

### 1. **准备 PDF 文件**
  - 将 PDF 文件放入 data/pdf/ 目录（首次使用需手动创建该目录）。 例如：`data/pdf/your_file.pdf`
  - 如需解析网页，准备好目标 URL

### 2. **运行主程序**
  - 在项目根目录下运行：
   ```bash
   python main.py
   ```
  - 程序默认处理预设 PDF 文件（可在代码中修改默认配置）
  - 可根据提示输入 PDF 文件名（无需包含路径）或网页 URL
  - 自动生成简要摘要（brief_summary）和详细摘要（detail_summary）导出至 data/output 文件夹。

### 3. 交互问答
  - 程序自动提取内容并生成章节摘要
  - 建议提问时带上章节名，如："请解释 Introduction 章节内容"
   - 输入 `退出`、`再见`、`bye`、`exit`、`quit` 可结束对话。


---

## LLM Provider 支持

- 支持 Azure OpenAI、OpenAI、Ollama 三种 LLM provider。
- 可在 `PDFReader` 初始化时通过 `provider` 参数指定（默认 `azure`）。
- 相关 API key、endpoint、模型等配置请在 `src/config.py` 中修改。

---

## 缓存与数据存储说明

- **图片缓存**：`data/pdf_image/`，每个 PDF 会生成对应的图片文件夹。
- **内容缓存**：`data/json_data/`，每个 PDF/URL 会生成对应的内容 JSON 文件。
- **向量数据库**：`data/vector_db/`，每个 PDF/URL 会生成对应的向量数据库文件夹。
- **缓存机制**：已处理过的 PDF/URL 会自动复用缓存，加速后续访问。

---

## 进阶说明

- **多轮对话**：每轮对话会自动复用历史检索内容，提升上下文连贯性和效率。
- **章节检索**：每次 summary 输出后，建议后续提问时直接引用 summary 中的章节标题，检索更精准。
- **自定义参数**：如需调整分块大小、缓存路径等，可在 `config.py` 或 `pdf.py` 构造参数中修改。
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
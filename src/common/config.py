class MCPToolName:
  WEB_SEARCH = "web search" 

MCP_CONFIG = {
  MCPToolName.WEB_SEARCH: {
    "playwright": {
      "command": "npx",
      "args": [
        "@playwright/mcp@latest"
      ]
    }
  }
}

class PDFReaderRole:
  IMAGE_EXTRACT = "image_extract"
  PDF_COMMON = "pdf_common"
  PDF_AGENDA = "pdf_agenda"
  PDF_SUMMARY = "pdf_summary"
  PDF_ANSWER = "pdf_answer"
  PDF_CHAT = "pdf_chat"


LLM_CONFIG = {
    "api_key": "ebf7fd9bfd53414f98597208539eae06",
    "api_version": "2025-01-01-preview",
    "azure_endpoint": "https://edward-ke-ai-aiservices-638508787.openai.azure.com/",
    "deployment_name": "crq-gpt-4.1",
    "model_name": "gpt-4.1"
}

LLM_EMBEDDING_CONFIG = {
    "api_key": "c443d242326749d08ff027583c8ea8c5",
    "api_version": "2023-09-15-preview",
    "azure_endpoint": "https://riskinsights-openai-demo.openai.azure.com/",
    "deployment": "riskinsights-knowledge",
    "model": "text-embedding-ada-002"
}


SYSTEM_PROMPT_CONFIG = {

  PDFReaderRole.IMAGE_EXTRACT: """Analyze the content of an image, extract relevant information, and organize it according to human reading habits into markdown format based on the data type (e.g., text, table, image, code, formulas, flowcharts).

# Key Instructions

- **Text Content**: Extract natural language content directly and organize it for human readability. Do not summarize or outline.
- **Table Data**: 
  1. Generate a summary description based on the table title and content.
  2. Extract table information and describe it clearly in natural language.
- **Image Information**:
  1. Parse image content and understand the details.
  2. Generate a descriptive explanation based on the image title and its elements.
- **Code Blocks**: Extract and present code exactly as it appears in the image, preserving its original meaning in proper code formatting.
- **Formulas**: Convert relevant mathematical expressions or data into **LaTeX** format.
- **Flowcharts**:
  1. Analyze and interpret the content of the flowchart.
  2. Provide a clear, human-readable explanation based on the flowchart structure.
- **General Content**: Filter out meaningless information and retain only relevant and concise details. Return the details in markdown format in a way that aligns with human readability.

# Output Format

The extracted content should be formatted as follows:

## Text Content  
[Organized natural language text extracted from the image.]

## Table Summary  
[A brief summary interpreting the purpose and key points of the table.]

## Table Data  
[Key table information rewritten into human-readable language.]

## Image Description  
[A descriptive text of the content depicted in the image, organized for easy understanding.]

## Code  
```  
[Original code extracted from the image, formatted in code blocks.]  
```

## Formulas  
[Mathematical expressions converted to LaTeX format, with explanations if needed.]  

## Flowchart Description  
[A detailed, human-readable interpretation of the flowchart.]

# Steps

- **1. Extract Information**: Extract all forms of content from the image (text, tables, images, code, formulas, flowcharts, etc.).
- **2. Organize by Type**: Classify and process data according to type:
  - Text: Provide as-is but structured naturally.
  - Tables: Summarize first, then describe contents.
  - Images and Flowcharts: Offer detailed interpretations of visual content.
  - Code: Preserve and render in proper code blocks.
  - Formulas: Convert to LaTeX as needed.
- **3. Format in Markdown**: Ensure the final output is organized by type and presented in clear markdown syntax.
- **4. Exclude Irrelevant Data**: Discard meaningless or visually insignificant content.

# Notes

- All outputs should align with human reading habits and intentional sequences.
- For complex tables, images, or flowcharts, aim for clarity and simplicity when interpreting the data.
- Use only markdown syntax for formatting (e.g., bold/italic for emphasis, headers for sections, lists for organization). Do not mix content with unrelated context.

# Example  

### Input:  
[An image containing a table, an illustration, and a formula.]

### Output in Markdown:

## Text Content  
"The image contains information about sales data and its impact across different regions over the last quarter."

## Table Summary  
"The table provides an overview of sales figures categorized by region and type of product."

## Table Data  
- **Region**: North America, Europe, Asia  
- **Products**: Electronics, Furniture, Clothing  
- **Sales Figures**:  
  - North America: $1M Electronics, $500K Furniture, $600K Clothing  
  - Europe: $800K Electronics, $300K Furniture, $400K Clothing  
  - Asia: $1.5M Electronics, $700K Furniture, $800K Clothing  

## Image Description  
"The illustration depicts a globe highlighting trading routes between continents, with arrows marking the direction of trade flows. North America is shown trading heavily with Asia and Europe, represented by bold arrows."

## Formula  
E = mc^2
(Energy equals mass times the square of the speed of light.)

## Flowchart Description  
"The flowchart represents a decision-making process for a customer choosing a product:  
1. Start at the 'Customer Needs' box.  
2. Evaluate if the need is urgent or not.  
3. If urgent: Proceed to 'Express Shipping.'  
4. If not urgent: Proceed to 'Standard Shipping.'"  

""",

  PDFReaderRole.PDF_COMMON: """提取文章的基本信息（如作者、单位、发布日期等）

# 任务说明

- **提取基本信息**：识别文章的元信息，包括文章名字、作者姓名、单位或机构名称、发布日期等（如果数据存在于文章中）。

# 输出格式

- **基本信息**：输出为 JSON 格式，包含 `author`、`institution`、`publish_date` 等字段，若字段不存在可返回 `null`。

# 注意事项

- 若某些基本信息缺失，请在 JSON 中用 `null`标明，例如 `"author": null`。
""",

  PDFReaderRole.PDF_AGENDA: """解析文章内容，提取其目录结构，并返回一个标准的 JSON 风格列表，其中包含章节标题及对应的起始页码。

# 详细任务说明
1. **内容扫描**：
   - 扫描全文，识别章节标题。
   - 仅提取文章中的一级目录标题，忽略次级标题。
   - 不要添加、总结或修改任何章节标题，保持与原文一致。

2. **层级与范围**：
   - 仅收录最上层的章节结构，忽略次级目录。
   - 如果文章没有显示划分子目录但章节标题清晰，应按其显示逻辑提取。

3. **页码准确性**：
   - 提取每个章节标题的起始页码，确保与文章中标注的页码保持一致性。

4. **单层结构兼容**：
   - 如果文章无明显章节分隔，则将文章整体作为一个单条记录，标题设为全文内容的标题，起始页码为文章第一页。

# 输出格式

以 JSON 列表的方式返回，其中每个章节是一个对象，包含以下 key：
- `title`：章节标题。
- `page`：章节的起始页码。

# 输出规则
- 确保章节顺序与文章编排一致，无遗漏或重新组织。
- 无需额外标注文章标题、作者等非目录信息。

# 注意事项
- **一致性**：输出结构和格式必须始终符合上述 JSON 示范样式。
""",

  PDFReaderRole.PDF_SUMMARY: """对输入的文本内容进行总结和信息提取，确保尽可能保留关键的原始信息和结构。

**关键要求**：
1. 确保总结清晰且忠实于原文内容，无删减主要信息。
2. 对数据或者列表中的要点，按照原有顺序进行提取和缩写。
3. 针对长文本，可以按模块分段总结，每段保留对应的核心信息。

# Steps 
1. 阅读和理解输入的文本内容。
2. 提取核心信息，包括数据、论点、或规定的上下文。
3. 对每一个模块进行分点或简明叙述，保留每段的内容关系和逻辑。
4. 输出的总体呈现可以继续维护文本长度的相对压缩度，但对专业术语避免删减。

# Output Format
1. 输出为分段总结，例如：
   - 如果文本逻辑有自然的章节/段落/子标题，按照原顺序分别提取每段的要点。
   - 如果没有明确逻辑，则直接按行或列表方式呈现。
2. 内容可以使用 Markdown 格式呈现，以便提升阅读性，例如：标题 (#)、列表 (-)，强调点子（**）。

# Examples
## 输入:
以下是一段复杂的用户给到的内容，[...]

## 输出:
```
### 标题/子主题 (如果有)
- 核心要点列出保持逻辑：
- [EXAMPLE PLACEHOLDER STRUCTURE ,...]
""",

  PDFReaderRole.PDF_ANSWER: """根据提供的背景信息（Background info），尽可能精准地回答客户问题。如果信息不足，则明确告知客户无法提供更多细节，而不要编造或假设答案。

# 任务步骤

1. **阅读背景信息**: 提取客户提供的 Background info 中的关键内容，理解其核心信息与上下文。
2. **分析客户问题**: 明确客户提出的问题需求，确保对其意图的准确理解。
3. **匹配信息**: 检查 Background info 是否包含与问题相关的内容。
4. **回答规则**:
   - 如果背景信息中包含明确答案或相关内容，直接提供答案。
   - 如果背景信息中不包含足够的信息，提示客户背景信息不足，并避免编造。
5. **组织语言**: 使用清晰、简练和礼貌的语气回答客户问题。

# 注意事项

- **绝不编造答案**: 所有回答必须完全基于背景信息，如果背景信息不足，应明确表明。
- **礼貌表达**: 即使不能回答问题，应保持礼貌和客户至上的态度。
- **简洁清晰**: 回答用词直接，避免含糊其词或冗长叙述。
- **避免偏离背景**: 严格以Background info为界限，避免使用外部推测。
""",

   PDFReaderRole.PDF_CHAT: """分析用户输入及历史对话内容，根据文章章节目录确定需要检索的章节，并返回章节的标题。
结合用户输入、历史对话内容和文章章节目录 `{agenda_dict}`，匹配最相关的章节并输出章节名字。

# Steps

1. **提取核心需求**  
   从用户输入的语句中提取关键需求或关键点。分析输入是否需要内容检索。
   
2. **整合上下文**  
   根据用户输入和历史对话，综合当前上下文，明确用户意图。确保准确捕捉可能隐含的需求。

3. **匹配章节标题**  
   在提供的章节目录中匹配最可能相关的章节标题。匹配逻辑可以根据用户需求。

4. **生成 JSON 结果**  
   - 如果存在匹配的章节，返回格式化的 JSON 对象。以 title 作为 key，value 是需要检索的章节名的 list。
   - 如果无需检索特定章节，直接返回答案/信息。例如在客户提问文章的章节目录时，并不需要检索向量数据库，因为 system prompt 中包含了相关信息。这种情况下直接返回结果就可以了

# Notes
- 匹配的章节标题必须严格使用原始命名，不允许直接推断或生成非标准标题。
- 对于不需要检索的问题，直接返回答案。
""",



}

# 数据根目录
DATA_ROOT = "data"
PDF_IMAGE_PATH = f"{DATA_ROOT}/pdf_image"
JSON_DATA_PATH = f"{DATA_ROOT}/json_data"
PDF_PATH = f"{DATA_ROOT}/pdf"
VECTOR_DB_PATH = f"{DATA_ROOT}/vector_db"
OUTPUT_PATH = f"{DATA_ROOT}/output"


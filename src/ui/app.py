import os
import sys
import asyncio
import pathlib
import streamlit as st
import shutil
import glob

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import makedir, get_pdf_name
from src.readers.pdf import PDFReader
from src.readers.web import WebReader


def ensure_data_dirs():
    makedir("data")
    makedir("data/pdf")
    makedir("data/json_data")
    makedir("data/pdf_image")
    makedir("data/vector_db")
    makedir("data/output")


def init_session_state():
    if "provider" not in st.session_state:
        st.session_state.provider = "azure"
    if "pdf_reader" not in st.session_state:
        st.session_state.pdf_reader = None
    if "web_reader" not in st.session_state:
        st.session_state.web_reader = None
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [(role, content)]
    if "current_doc_name" not in st.session_state:
        st.session_state.current_doc_name = None


def reset_chat():
    st.session_state.messages = []


def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # Fallback if an event loop is already running
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def get_available_documents():
    """获取可用的文档列表"""
    json_files = glob.glob("data/json_data/*.json")
    documents = []
    for json_file in json_files:
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        documents.append(base_name)
    return sorted(documents)





def load_summary_content(doc_name, summary_type):
    """加载总结内容"""
    summary_file = f"data/output/{doc_name}/{summary_type}.md"
    if os.path.exists(summary_file):
        with open(summary_file, 'r', encoding='utf-8') as f:
            return f.read()
    return None


def get_document_info(doc_name):
    """获取指定文档的详细信息"""
    info = {
        "name": doc_name,
        "json_exists": False,
        "vector_db_exists": False,
        "output_exists": False,
        "pdf_image_exists": False,
        "summary_files": [],
        "file_sizes": {}
    }
    
    # 检查JSON文件
    json_file = f"data/json_data/{doc_name}.json"
    if os.path.exists(json_file):
        info["json_exists"] = True
        info["file_sizes"]["json"] = os.path.getsize(json_file)
    
    # 检查向量数据库
    vector_db_dir = f"data/vector_db/{doc_name}_data_index"
    if os.path.exists(vector_db_dir):
        info["vector_db_exists"] = True
        total_size = 0
        for root, dirs, files in os.walk(vector_db_dir):
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))
        info["file_sizes"]["vector_db"] = total_size
    
    # 检查输出文件
    output_dir = f"data/output/{doc_name}"
    if os.path.exists(output_dir):
        info["output_exists"] = True
        total_size = 0
        summary_files = []
        for file in os.listdir(output_dir):
            if file.endswith(('.md', '.pdf')):
                file_path = os.path.join(output_dir, file)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                summary_files.append({
                    "name": file,
                    "size": file_size,
                    "path": file_path
                })
        info["file_sizes"]["output"] = total_size
        info["summary_files"] = summary_files
    
    # 检查PDF图片文件
    pdf_image_dir = f"data/pdf_image/{doc_name}"
    if os.path.exists(pdf_image_dir):
        info["pdf_image_exists"] = True
        total_size = 0
        for root, dirs, files in os.walk(pdf_image_dir):
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))
        info["file_sizes"]["pdf_image"] = total_size
    
    return info


def format_file_size(size_bytes):
    """格式化文件大小显示"""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"





def get_document_files(doc_name):
    """获取指定文档的所有文件列表"""
    files_info = {
        "json_files": [],
        "vector_db_files": [],
        "output_files": [],
        "pdf_image_files": []
    }
    
    # JSON文件
    json_file = f"data/json_data/{doc_name}.json"
    if os.path.exists(json_file):
        files_info["json_files"].append({
            "name": f"{doc_name}.json",
            "path": json_file,
            "size": os.path.getsize(json_file),
            "type": "json"
        })
    
    # 向量数据库文件
    vector_db_dir = f"data/vector_db/{doc_name}_data_index"
    if os.path.exists(vector_db_dir):
        for root, dirs, files in os.walk(vector_db_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, vector_db_dir)
                files_info["vector_db_files"].append({
                    "name": f"向量库/{rel_path}",
                    "path": file_path,
                    "size": os.path.getsize(file_path),
                    "type": "vector_db"
                })
    
    # 输出文件
    output_dir = f"data/output/{doc_name}"
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith(('.md', '.pdf')):
                file_path = os.path.join(output_dir, file)
                files_info["output_files"].append({
                    "name": file,
                    "path": file_path,
                    "size": os.path.getsize(file_path),
                    "type": "output"
                })
    
    # PDF图片文件
    pdf_image_dir = f"data/pdf_image/{doc_name}"
    if os.path.exists(pdf_image_dir):
        for root, dirs, files in os.walk(pdf_image_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, pdf_image_dir)
                files_info["pdf_image_files"].append({
                    "name": f"图片/{rel_path}",
                    "path": file_path,
                    "size": os.path.getsize(file_path),
                    "type": "pdf_image"
                })
    
    return files_info


def delete_specific_files(file_list):
    """删除指定的文件列表"""
    results = []
    
    for file_info in file_list:
        try:
            if os.path.exists(file_info["path"]):
                if os.path.isfile(file_info["path"]):
                    os.remove(file_info["path"])
                    results.append(f"✅ 已删除文件: {file_info['name']}")
                elif os.path.isdir(file_info["path"]):
                    shutil.rmtree(file_info["path"])
                    results.append(f"✅ 已删除目录: {file_info['name']}")
            else:
                results.append(f"⚠️ 文件不存在: {file_info['name']}")
        except Exception as e:
            results.append(f"❌ 删除失败 {file_info['name']}: {str(e)}")
    
    return results


def ui_data_management():
    """数据管理界面"""
    st.subheader("📁 数据管理")
    
    documents = get_available_documents()
    if not documents:
        st.info("暂无已处理的文档")
        return
    
    # 选择要管理的文档
    selected_doc = st.selectbox(
        "选择要管理的文档",
        documents,
        index=documents.index(st.session_state.current_doc_name) if st.session_state.current_doc_name in documents else 0
    )
    
    if selected_doc:
        # 获取文档详细信息
        doc_info = get_document_info(selected_doc)
        files_info = get_document_files(selected_doc)
        
        # 显示文档信息
        st.markdown(f"### 📄 文档: {selected_doc}")
        
        # 创建信息展示区域
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**数据文件状态:**")
            status_items = []
            if doc_info["json_exists"]:
                size = format_file_size(doc_info["file_sizes"].get("json", 0))
                status_items.append(f"📄 JSON数据: {size}")
            if doc_info["vector_db_exists"]:
                size = format_file_size(doc_info["file_sizes"].get("vector_db", 0))
                status_items.append(f"🔍 向量数据库: {size}")
            if doc_info["output_exists"]:
                size = format_file_size(doc_info["file_sizes"].get("output", 0))
                status_items.append(f"📋 输出文件: {size}")
            if doc_info["pdf_image_exists"]:
                size = format_file_size(doc_info["file_sizes"].get("pdf_image", 0))
                status_items.append(f"🖼️ PDF图片: {size}")
            
            if status_items:
                for item in status_items:
                    st.write(item)
            else:
                st.write("❌ 无数据文件")
        
        with col2:
            st.markdown("**总结文件:**")
            if doc_info["summary_files"]:
                for file_info in doc_info["summary_files"]:
                    size = format_file_size(file_info["size"])
                    st.write(f"📝 {file_info['name']}: {size}")
            else:
                st.write("❌ 无总结文件")
        
        # 文件级别删除选项
        st.markdown("### 🗑️ 文件级别删除")
        
        # 收集所有文件
        all_files = []
        all_files.extend(files_info["json_files"])
        all_files.extend(files_info["vector_db_files"])
        all_files.extend(files_info["output_files"])
        all_files.extend(files_info["pdf_image_files"])
        
        if all_files:
            # 按类型分组显示文件
            st.markdown("**选择要删除的具体文件:**")
            
            # JSON文件
            if files_info["json_files"]:
                st.markdown("📄 **JSON数据文件:**")
                json_selected = st.multiselect(
                    "JSON文件",
                    options=files_info["json_files"],
                    format_func=lambda x: f"{x['name']} ({format_file_size(x['size'])})",
                    key="json_files"
                )
            
            # 向量数据库文件
            if files_info["vector_db_files"]:
                st.markdown("🔍 **向量数据库文件:**")
                vector_selected = st.multiselect(
                    "向量数据库文件",
                    options=files_info["vector_db_files"],
                    format_func=lambda x: f"{x['name']} ({format_file_size(x['size'])})",
                    key="vector_files"
                )
            
            # 输出文件
            if files_info["output_files"]:
                st.markdown("📋 **输出文件:**")
                output_selected = st.multiselect(
                    "输出文件",
                    options=files_info["output_files"],
                    format_func=lambda x: f"{x['name']} ({format_file_size(x['size'])})",
                    key="output_files"
                )
            
            # PDF图片文件
            if files_info["pdf_image_files"]:
                st.markdown("🖼️ **PDF图片文件:**")
                image_selected = st.multiselect(
                    "PDF图片文件",
                    options=files_info["pdf_image_files"],
                    format_func=lambda x: f"{x['name']} ({format_file_size(x['size'])})",
                    key="image_files"
                )
            
            # 合并所有选中的文件
            selected_files = []
            if 'json_selected' in locals():
                selected_files.extend(json_selected)
            if 'vector_selected' in locals():
                selected_files.extend(vector_selected)
            if 'output_selected' in locals():
                selected_files.extend(output_selected)
            if 'image_selected' in locals():
                selected_files.extend(image_selected)
            
            if selected_files:
                st.warning(f"⚠️ 即将删除选中的 {len(selected_files)} 个文件")
                
                # 显示选中的文件列表
                st.markdown("**选中的文件:**")
                for file_info in selected_files:
                    st.write(f"  - {file_info['name']} ({format_file_size(file_info['size'])})")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("✅ 确认删除文件", key=f"confirm_delete_files_{selected_doc}"):
                        results = delete_specific_files(selected_files)
                        for result in results:
                            st.write(result)
                        
                        # 检查是否删除了当前文档的关键文件
                        deleted_types = set(file_info["type"] for file_info in selected_files)
                        if "json" in deleted_types or "vector_db" in deleted_types:
                            if selected_doc == st.session_state.current_doc_name:
                                st.session_state.current_doc_name = None
                                st.session_state.pdf_reader = None
                                st.session_state.web_reader = None
                                reset_chat()
                        
                        st.success("文件删除操作完成")
                        st.rerun()
                
                with col2:
                    if st.button("❌ 取消", key=f"cancel_delete_files_{selected_doc}"):
                        st.rerun()
            
            # 快速删除选项
            st.markdown("---")
            st.markdown("**快速操作:**")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("🗑️ 删除所有文件", key=f"delete_all_files_{selected_doc}"):
                    st.warning("⚠️ 确定要删除此文档的所有文件吗？")
                    col_confirm1, col_confirm2 = st.columns([1, 1])
                    with col_confirm1:
                        if st.button("✅ 确认", key=f"confirm_all_files_{selected_doc}"):
                            results = delete_specific_files(all_files)
                            for result in results:
                                st.write(result)
                            
                            if selected_doc == st.session_state.current_doc_name:
                                st.session_state.current_doc_name = None
                                st.session_state.pdf_reader = None
                                st.session_state.web_reader = None
                                reset_chat()
                            
                            st.success("所有文件已删除")
                            st.rerun()
                    with col_confirm2:
                        if st.button("❌ 取消", key=f"cancel_all_files_{selected_doc}"):
                            st.rerun()
            
            with col2:
                if st.button("📋 仅保留总结", key=f"keep_summary_files_{selected_doc}"):
                    to_delete = [f for f in all_files if f["type"] != "output"]
                    if to_delete:
                        results = delete_specific_files(to_delete)
                        for result in results:
                            st.write(result)
                        st.success("已删除除总结外的所有文件")
                        st.rerun()
                    else:
                        st.info("没有可删除的文件")
            
            with col3:
                if st.button("🔍 仅保留向量库", key=f"keep_vector_files_{selected_doc}"):
                    to_delete = [f for f in all_files if f["type"] != "vector_db"]
                    if to_delete:
                        results = delete_specific_files(to_delete)
                        for result in results:
                            st.write(result)
                        st.success("已删除除向量库外的所有文件")
                        st.rerun()
                    else:
                        st.info("没有可删除的文件")
        else:
            st.info("该文档没有可删除的文件")


def ui_sidebar():
    with st.sidebar:
        st.header("LLMReader 设置")
        st.session_state.provider = st.selectbox(
            "LLM Provider",
            options=["azure", "openai", "ollama"],
            index=["azure", "openai", "ollama"].index(st.session_state.provider),
        )
        st.caption("在 `.env` 或环境变量中配置密钥与模型信息。")
        st.markdown("---")
        
        # 数据管理部分
        ui_data_management()
        
        st.markdown("---")
        st.caption("输出保存在 `data/output` 下的对应文档目录中。")


def ui_pdf_tab():
    st.subheader("📄 PDF Reader")
    
    # 文件上传和处理
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_pdf = st.file_uploader("上传 PDF 文件", type=["pdf"], accept_multiple_files=False)
    with col2:
        save_outputs = st.checkbox("生成并保存摘要到本地 (MD/PDF)", value=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        start_pdf = st.button("🚀 开始处理 PDF", type="primary")
    with col2:
        if st.button("🔄 重置对话"):
            reset_chat()

    if start_pdf:
        if uploaded_pdf is None:
            st.warning("请先上传 PDF 文件。")
            return

        ensure_data_dirs()
        pdf_save_path = os.path.join("data", "pdf", uploaded_pdf.name)
        with open(pdf_save_path, "wb") as f:
            f.write(uploaded_pdf.read())

        base_name = get_pdf_name(uploaded_pdf.name)
        st.session_state.current_doc_name = base_name

        with st.spinner("开始处理，请耐心等待。首次处理会比较耗时（PDF 转图片、内容提取与总结、向量库构建）。"):
            reader = PDFReader(provider=st.session_state.provider)
            # 仅处理，不进入 CLI 循环
            reader.process_pdf(base_name, save_data_flag=save_outputs)
            st.session_state.pdf_reader = reader
        st.success("✅ 处理完成！现在可以在下方查看总结或开始聊天。")

    # 显示总结内容
    if st.session_state.current_doc_name:
        st.divider()
        st.markdown("### 📋 自动生成的总结")
        
        # 检查是否有总结文件
        brief_summary = load_summary_content(st.session_state.current_doc_name, "brief_summary")
        detail_summary = load_summary_content(st.session_state.current_doc_name, "detail_summary")
        
        if brief_summary or detail_summary:
            tab1, tab2 = st.tabs(["📝 简要总结", "📖 详细总结"])
            
            with tab1:
                if brief_summary:
                    st.markdown(brief_summary)
                else:
                    st.info("简要总结文件不存在")
            
            with tab2:
                if detail_summary:
                    st.markdown(detail_summary)
                else:
                    st.info("详细总结文件不存在")
        else:
            st.info("暂无总结文件，请确保在处理时勾选了'生成并保存摘要到本地'选项")

    # 聊天区域
    st.divider()
    st.markdown("### 💬 智能问答")
    st.caption("基于当前已处理文档进行问答")
    
    chat_container = st.container()
    for role, content in st.session_state.messages:
        with chat_container.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input("请输入你的问题…")
    if user_input:
        if st.session_state.pdf_reader is None:
            st.warning("请先处理一个 PDF 文件。")
            return
        st.session_state.messages.append(("user", user_input))
        with chat_container.chat_message("assistant"):
            with st.spinner("正在思考..."):
                answer = st.session_state.pdf_reader.chat(user_input)
            st.markdown(answer)
        st.session_state.messages.append(("assistant", answer))


def ui_web_tab():
    st.subheader("🌐 Web Reader")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        url = st.text_input("输入网页 URL")
    with col2:
        save_outputs = st.checkbox("生成并保存摘要到本地 (MD/PDF)", value=True, key="web_save")

    col1, col2 = st.columns([1, 1])
    with col1:
        start_web = st.button("🚀 开始处理 URL", type="primary")
    with col2:
        if st.button("🔄 重置对话", key="web_reset"):
            reset_chat()

    if start_web:
        if not url or not (url.startswith("http://") or url.startswith("https://")):
            st.warning("请输入合法的 URL（必须以 http:// 或 https:// 开头）。")
            return

        ensure_data_dirs()
        # 从URL生成文档名称
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        doc_name = parsed_url.netloc.replace(".", "_") + "_" + str(hash(url))[-8:]
        st.session_state.current_doc_name = doc_name
        
        with st.spinner("开始处理网页内容，可能需要较长时间（抓取、摘要/分块、向量库构建）。"):
            reader = WebReader(provider=st.session_state.provider)
            run_async(reader.process_web(url, save_data_flag=save_outputs))
            st.session_state.web_reader = reader
        st.success("✅ 处理完成！现在可以在下方查看总结或开始聊天。")

    # 显示总结内容
    if st.session_state.current_doc_name:
        st.divider()
        st.markdown("### 📋 自动生成的总结")
        
        # 检查是否有总结文件
        brief_summary = load_summary_content(st.session_state.current_doc_name, "brief_summary")
        detail_summary = load_summary_content(st.session_state.current_doc_name, "detail_summary")
        
        if brief_summary or detail_summary:
            tab1, tab2 = st.tabs(["📝 简要总结", "📖 详细总结"])
            
            with tab1:
                if brief_summary:
                    st.markdown(brief_summary)
                else:
                    st.info("简要总结文件不存在")
            
            with tab2:
                if detail_summary:
                    st.markdown(detail_summary)
                else:
                    st.info("详细总结文件不存在")
        else:
            st.info("暂无总结文件，请确保在处理时勾选了'生成并保存摘要到本地'选项")

    # 聊天区域
    st.divider()
    st.markdown("### 💬 智能问答")
    st.caption("基于当前已处理网页进行问答")
    
    chat_container = st.container()
    for role, content in st.session_state.messages:
        with chat_container.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input("请输入你的问题…", key="web_chat")
    if user_input:
        if st.session_state.web_reader is None:
            st.warning("请先处理一个 URL。")
            return
        st.session_state.messages.append(("user", user_input))
        with chat_container.chat_message("assistant"):
            with st.spinner("正在思考..."):
                answer = st.session_state.web_reader.chat(user_input)
            st.markdown(answer)
        st.session_state.messages.append(("assistant", answer))


def main():
    st.set_page_config(page_title="LLMReader UI", page_icon="📚", layout="wide")
    st.title("📚 LLMReader 智能文档分析系统")
    st.caption("支持 PDF 与网页内容解析、摘要与基于向量库的问答")

    init_session_state()
    ui_sidebar()

    tab_pdf, tab_web = st.tabs(["📄 PDF 文档", "🌐 网页内容"])
    with tab_pdf:
        ui_pdf_tab()
    with tab_web:
        ui_web_tab()


if __name__ == "__main__":
    main()



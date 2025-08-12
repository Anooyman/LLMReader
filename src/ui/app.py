import os
import sys
import asyncio
import pathlib
import streamlit as st

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
        st.caption("输出保存在 `data/output` 下的对应文档目录中。")


def ui_pdf_tab():
    st.subheader("PDF Reader")
    uploaded_pdf = st.file_uploader("上传 PDF 文件", type=["pdf"], accept_multiple_files=False)
    save_outputs = st.checkbox("生成并保存摘要到本地 (MD/PDF)", value=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        start_pdf = st.button("开始处理 PDF", type="primary")
    with col2:
        if st.button("重置对话"):
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

        st.info("开始处理，请耐心等待。首次处理会比较耗时（PDF 转图片、内容提取与总结、向量库构建）。")
        reader = PDFReader(provider=st.session_state.provider)
        # 仅处理，不进入 CLI 循环
        reader.process_pdf(base_name, save_data_flag=save_outputs)
        st.session_state.pdf_reader = reader
        st.success("处理完成！现在可以在下方聊天框提问。")

    # 聊天区域
    st.divider()
    st.markdown("**对话**（基于当前已处理文档）")
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
            answer = st.session_state.pdf_reader.chat(user_input)
            st.markdown(answer)
        st.session_state.messages.append(("assistant", answer))


def ui_web_tab():
    st.subheader("Web Reader")
    url = st.text_input("输入网页 URL")
    save_outputs = st.checkbox("生成并保存摘要到本地 (MD/PDF)", value=True, key="web_save")

    col1, col2 = st.columns([1, 1])
    with col1:
        start_web = st.button("开始处理 URL", type="primary")
    with col2:
        if st.button("重置对话", key="web_reset"):
            reset_chat()

    if start_web:
        if not url or not (url.startswith("http://") or url.startswith("https://")):
            st.warning("请输入合法的 URL（必须以 http:// 或 https:// 开头）。")
            return

        ensure_data_dirs()
        st.info("开始处理网页内容，可能需要较长时间（抓取、摘要/分块、向量库构建）。")
        reader = WebReader(provider=st.session_state.provider)
        run_async(reader.process_web(url, save_data_flag=save_outputs))
        st.session_state.web_reader = reader
        st.success("处理完成！现在可以在下方聊天框提问。")

    # 聊天区域
    st.divider()
    st.markdown("**对话**（基于当前已处理网页/向量库）")
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
            answer = st.session_state.web_reader.chat(user_input)
            st.markdown(answer)
        st.session_state.messages.append(("assistant", answer))


def main():
    st.set_page_config(page_title="LLMReader UI", page_icon="📚", layout="wide")
    st.title("LLMReader 本地 UI")
    st.caption("支持 PDF 与网页内容解析、摘要与基于向量库的问答。")

    init_session_state()
    ui_sidebar()

    tab_pdf, tab_web = st.tabs(["PDF", "Web"])
    with tab_pdf:
        ui_pdf_tab()
    with tab_web:
        ui_web_tab()


if __name__ == "__main__":
    main()



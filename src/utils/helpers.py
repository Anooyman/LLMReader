import fitz  # 导入pymupdf库，它在导入时别名为fitz
import json
import os
import unicodedata
import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

def list_pdf_files(folder_path="data/pdf"):
    """
    读取指定文件夹下的所有文件，返回文件名列表
    """
    if not os.path.exists(folder_path):
        logging.warning(f"文件夹不存在: {folder_path}")
        return []
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


def load_md_file(file_path):
        """
        读取本地Markdown文件的内容
        
        参数:
            file_path (str): Markdown文件的路径
            
        返回:
            str: 文件内容，如果出错则返回None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                return content
        except FileNotFoundError:
            print(f"错误: 找不到文件 '{file_path}'")
        except UnicodeDecodeError:
            print(f"错误: 无法解码文件 '{file_path}'，可能不是UTF-8编码")
        except Exception as e:
            print(f"读取文件时发生错误: {str(e)}")
        return None

def read_images_in_directory(directory_path: str) -> List[str]:
    """
    读取指定目录下所有支持格式的图片文件路径。
    Read all supported image files in a directory.
    Args:
        directory_path (str): 目录路径。
    Returns:
        List[str]: 图片文件路径列表。
    """
    image_files = []
    valid_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in valid_image_extensions:
                image_path = os.path.join(root, file)
                image_files.append(image_path)
    logger.info(f"读取到{len(image_files)}张图片 in {directory_path}")
    return image_files

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Folder {path} created")
    else:
        logger.debug(f"Folder {path} already exists")
 
def get_pdf_name(file_name: str) -> str:
    """
    获取去除扩展名后的文件名。
    Get file name without extension.
    Args:
        file_name (str): 文件名。
    Returns:
        str: 去除扩展名后的文件名。
    """
    dot_index = file_name.rfind('.')
    if dot_index != -1:
        file_name_without_ext = file_name[:dot_index]
        logger.debug(f"PDF文件名去后缀: {file_name_without_ext}")
    else:
        file_name_without_ext = file_name
        logger.debug(f"PDF文件名无后缀: {file_name}")
    return file_name_without_ext

def pdf_to_images(pdf_path: str, output_folder: str) -> None:
    """
    将 PDF 文件每一页转换为图片并保存到指定文件夹。
    Convert each page of a PDF to an image and save to output folder.
    Args:
        pdf_path (str): PDF 文件路径。
        output_folder (str): 图片保存文件夹。
    Returns:
        None
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.info(f"Output folder created: {output_folder}")
    else:
        logger.debug(f"Output folder already exists: {output_folder}")

    logger.info(f"开始将PDF转为图片: {pdf_path}")
    doc = fitz.open(pdf_path)  # 打开 PDF 文档
    for page_num in range(doc.page_count):
        page = doc[page_num]
        pix = page.get_pixmap()
        image_path = f"{output_folder}/page_{page_num + 1}.png"
        pix.save(image_path)
        logger.info(f"Saved image: {image_path}")
    logger.info(f"PDF 转图片完成，共 {doc.page_count} 页。")
    doc.close()

def extract_page_num(path: str) -> Optional[str]:
    """
    从图片路径中提取页码数字。
    Extract page number from image file path.
    Args:
        path (str): 图片文件路径。
    Returns:
        Optional[str]: 提取到的页码数字，未找到则为 None。
    """
    file_name = os.path.basename(path)
    pattern = r'\d+'
    match = re.search(pattern, file_name)

    if match:
        number = match.group(0)
        logger.debug(f"Extracted page number {number} from {file_name}")
        return number
    else:
        logger.warning(f"未找到数字 in {file_name}。")
        return None

def extract_data_from_LLM_res(content) -> dict:

    data = None
    try:
        data = json.loads(content)
    except Exception as e:
        pattern = re.compile(fr'```json\n(.*?)```', re.DOTALL)

        match = pattern.search(content)

        if match:
            json_str = match.group(1)
            try:
                data= json.loads(json_str)
            except Exception as e:
                print(f"@@@@{json_str} json loads error: {e}")
                try:
                    data= eval(json_str)
                except Exception as e:
                    print(f"@@@@{json_str} eval error: {e}")
                    data = None
        else:
            print("No JSON data found.")

    return data

def parse_latest_plugin_call(text: str):  
    i = text.rfind('\nAction:')  
    j = text.rfind('\nAction Input:')  
    k = text.rfind('\nObservation:')  
    final_answer_index = text.rfind('\nFinal Answer:')  
  
    if 0 <= i < j:  # If the text has `Action` and `Action input`,  
        if k < j:  # but does not contain `Observation`,  
            # then it is likely that `Observation` is omitted by the LLM,  
            # because the output text may have discarded the stop word.  
            text = text.rstrip() + '\nObservation:'  # Add it back.  
            k = text.rfind('\nObservation:')  
  
    plugin_name, plugin_args, final_answer = '', '', ''  
  
    if 0 <= i < j < k:  
        plugin_name = text[i + len('\nAction:'):j].strip()  
        plugin_args = text[j + len('\nAction Input:'):k].strip()  
  
    if final_answer_index != -1:  
        final_answer = text[final_answer_index + len('\nFinal Answer:'):].strip()  
  
    return plugin_name, plugin_args, final_answer  


def full_to_half(text):
    """将全角字符转换为半角"""
    normalized = []
    for char in text:
        # 全角转半角
        if unicodedata.east_asian_width(char) == 'F':
            normalized_char = unicodedata.normalize('NFKC', char)
            normalized.append(normalized_char)
        else:
            normalized.append(char)
    return ''.join(normalized)

def normalize_chapter(name):
    # 1. 全角转半角
    name = full_to_half(name)
    # 2. 移除所有标点和空白
    name = re.sub(r'[^\w\u4e00-\u9fa5]', '', name)
    # 3. 转为小写（如果有英文）
    return name.lower()

def deduplicate_by_title(data):
    seen = set()
    result = []
    for item in data:
        title = normalize_chapter(item.get('title'))
        #title = item.get('title')
        if title not in seen:
            seen.add(title)
            result.append(item)
    return result

def group_data_by_sections_with_titles(total_sections, raw_data):
    logger.info(f"开始分组数据，输入章节数: {len(total_sections)}, 原始数据条数: {len(raw_data)}")
    unsort_sections = deduplicate_by_title(total_sections)
    logger.info(f"去重后章节数: {len(unsort_sections)}")

    try:
        sections = sorted(
            unsort_sections,
            key=lambda x: int(x.get("page", [float('inf')])) if x.get("page") else float('inf')
        )
        logger.info(f"章节排序完成，排序后章节如下: {sections}")
    except Exception as e:
        logger.error(f"章节排序出错: {e}")
        sections = unsort_sections

    # 页码转内容
    page_to_data = {}
    for item in raw_data:
        page = int(item['page'])
        if page not in page_to_data:
            page_to_data[page] = {}
        page_to_data[page] = item['data']

    # 找到最大页码
    max_page = max(int(item['page']) for item in raw_data) if raw_data else 0

    # 将章节按连续相同page分组
    groups = []
    if not sections:
        return []
    
    current_group = [sections[0]]
    current_page = int(sections[0]['page'])
    
    for sec in sections[1:]:
        page = int(sec['page'])
        if page == current_page:
            current_group.append(sec)
        else:
            groups.append(current_group)
            current_group = [sec]
            current_page = page
    groups.append(current_group)  # 添加最后一个组

    data_result = []
    agenda_result = []
    # 处理每个组
    for group_idx, group in enumerate(groups):
        # 获取下一组的起始页码
        if group_idx < len(groups) - 1:
            next_group_start = int(groups[group_idx + 1][0]['page'])
        else:
            next_group_start = max_page  # 最后一组的结束页为max_page
        
        # 处理组内每个章节
        for sec_idx, sec in enumerate(group):
            start = int(sec['page'])
            # 组内最后一个章节才延伸到下一组起始页
            if sec_idx == len(group) - 1:
                end = next_group_start if group_idx < len(groups) - 1 else max_page
            else:
                end = start  # 组内非最后章节只包含当前页
            
            # 生成页码范围
            if start == end:
                pages = [start]
            else:
                pages = list(range(start, end + 1))  # 包含end
            
            # 收集数据
            section_data = {}
            for page in pages:
                section_data[page] = page_to_data.get(page, "")
            
            data_result.append({
                'title': sec['title'],
                'pages': pages,
                'data': section_data
            })
            agenda_result.append({
                'title': sec['title'],
                'pages': pages,
            })
    
    return data_result, agenda_result

def add_data_keep_order(total_dict, add_dict):
    """
    合并 add_dict 到 total_dict，若 key 重复则用 add_dict 的 value 覆盖，
    最终返回一个 dict，key 按照数字从小到大排序。
    """
    # 更新 total_dict
    total_dict.update(add_dict)
    # 按 key 升序排序并返回新的 dict
    return dict(sorted(total_dict.items(), key=lambda x: x[0]))

def is_file_exists(file_path: str) -> bool:
    """
    判断指定路径下的文件是否存在

    参数:
        file_path (str): 完整的文件路径，包括文件名和扩展名

    返回值:
        bool: 如果文件存在且是一个文件（不是目录）则返回True，否则返回False

    异常处理:
        捕获并处理路径解析过程中可能出现的异常（如权限问题、无效路径等）
    """
    try:
        # 检查路径是否存在且是一个文件
        if os.path.exists(file_path) and os.path.isfile(file_path):
            logger.info(f"文件存在: {file_path}")
            return True
        else:
            # 路径不存在或不是文件
            if not os.path.exists(file_path):
                logger.warning(f"路径不存在: {file_path}")
            else:
                logger.warning(f"路径指向的是目录而非文件: {file_path}")
            return False
    except Exception as e:
        # 处理其他可能的异常（如权限错误、路径格式错误等）
        logger.error(f"检查文件存在性时发生错误: {str(e)}")
        return False

def extract_name_from_url(url):
    """
    从URL中提取可能作为name的关键信息
    Args:
        url: 待提取信息的URL字符串
    Returns:
        提取到的可能作为name的字符串
    """
    # 去除URL中的协议部分（如http://、https://）
    url_without_protocol = url.split('://')[-1]
    # 去除域名部分（如medium.com、@lucknitelol等）
    path_parts = url_without_protocol.split('/')[-2:]  # 针对该URL结构，取域名后的路径部分
    # 提取URL中以连字符连接的关键内容部分
    if path_parts:
        if path_parts[-1]:
            name_part = path_parts[-1]
        else:
            name_part = path_parts[-2]
        # 将连字符替换为空格
        name = name_part.replace('-', ' ')
        return name
    else:
        return "无法从URL中提取有效信息"

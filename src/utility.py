
import hashlib
import json
import re
import logging
import subprocess


logger = logging.getLogger(__name__)

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

def is_pdf(file_path):
    return file_path.lower().endswith('.pdf')

def deduplicate_by_title(data):
    seen = set()
    result = []
    for item in data:
        title = item.get('title')
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
            page_to_data[page] = []
        page_to_data[page].append(item['data'])

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
            section_data = []
            for page in pages:
                section_data.extend(page_to_data.get(page, []))
            
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

def url_to_id(url: str) -> str:
    """
    根据 url 生成唯一的 hash id（sha256，取前16位更短）。
    """
    hash_obj = hashlib.sha256(url.encode('utf-8'))
    # 取前16位更短，也可用 hash_obj.hexdigest() 得到完整64位
    return hash_obj.hexdigest()[:16]


def download_with_curl(url, dest_path):
    try:
        # -L 跟随重定向，-o 指定输出文件
        result = subprocess.run(
            ["curl", "-L", "-o", dest_path, url],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Downloaded to {dest_path}")
        return dest_path
    except subprocess.CalledProcessError as e:
        print(f"curl failed: {e.stderr}")
        return None

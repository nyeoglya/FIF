import os
import re
import json
import urllib.parse
import typing as tp

from type import JSONDict

def safe_doc_id(name: str, max_len: int = 50) -> str:
    if not name: return "unknown"
    name = re.sub(r'[\/\\\:\*\?\"\<\>\|]', '_', name)
    name = name.strip()
    return name[:max_len]

def get_clean_savepath_from_url_with_custom_extension(image_save_folder_path: str, original_url: str, extension: str) -> str:
    name_without_ext, _ = get_clean_filename_and_extension_from_url(original_url)
    return get_clean_savepath(image_save_folder_path, name_without_ext, extension)

def get_clean_savepath_from_url(image_save_folder_path: str, original_url: str) -> str:
    name_without_ext, extension = get_clean_filename_and_extension_from_url(original_url)
    extension = extension.replace(".","")
    if not extension:
        extension: str = "png"
    return get_clean_savepath(image_save_folder_path, name_without_ext, extension)

def get_clean_filename_from_url(original_url: str) -> str:
    url_path: str = original_url.split('?')[0]
    raw_filename: str = url_path.split('/')[-1]
    decoded_name: str = urllib.parse.unquote(raw_filename)
    name_without_ext, _ = os.path.splitext(decoded_name)
    return get_clean_filename(name_without_ext)

def get_clean_filename_and_extension_from_url(original_url: str) -> tp.Tuple[str, str]:
    url_path: str = original_url.split('?')[0]
    raw_filename: str = url_path.split('/')[-1]
    decoded_name: str = urllib.parse.unquote(raw_filename)
    name_without_ext, extension = os.path.splitext(decoded_name)
    return get_clean_filename(name_without_ext), extension

def get_clean_filename_with_extension_from_filepath(original_path: str) -> str:
    raw_filename: str = original_path.split('/')[-1]
    name_without_ext, extension = os.path.splitext(raw_filename)
    return get_clean_filename(name_without_ext) + extension

def get_clean_savepath(save_folderpath: str, filename: str, extension: str = "") -> str:
    clean_file_name: str = get_clean_filename(filename)
    clean_file_path: str = os.path.join(save_folderpath, f"{clean_file_name}.{extension}")
    return clean_file_path

def get_clean_filename(filename: str) -> str:
    clean_file_name: str = "".join([c for c in filename if c.isalnum() or c in (' ', '_', '-')]).rstrip()
    return clean_file_name

def read_json_or_jsonl(file_path: str) -> tp.List[JSONDict]:
    try:
        if file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line.strip()) for line in f if line.strip()]
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            print(file_path)
            raise ValueError("Unsupported file format. Please provide a .json or .jsonl file.")
        return data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        raise

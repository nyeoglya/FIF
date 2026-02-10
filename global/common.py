import os
import urllib.parse
import typing as tp

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

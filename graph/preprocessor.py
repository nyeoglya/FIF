import os
import re
import json
import base64
import pickle
import asyncio
import typing as tp

from pathlib import Path

import httpx
import numpy as np
from tqdm import tqdm
from numpy.typing import NDArray

from common import get_clean_savepath_from_url_with_custom_extension
from type import JSONDict
from query import DOCUMENT_SUMMARIZATION_QUERY
import compatibility # type: ignore

class LightweightComponent:
    def __init__(self) -> None:
        self.groundtruth_component_label_list: tp.List[str] = []
        self.original_json_data: JSONDict = dict()
        self.component_embedding: NDArray[np.floating[tp.Any]] = np.array([])
        self.subcomponent_embeddings: tp.List[NDArray[np.floating[tp.Any]]] = []
        self.linked_document_title_list: tp.List[str] = []

class FIFDocument:
    def __init__(self) -> None:
        self.document_title: str = ""
        self.document_summary: str = ""
        self.document_embedding: NDArray[np.floating[tp.Any]] = np.array([])
        self.original_json_data: JSONDict = dict()
        self.processed_component_list: tp.List[LightweightComponent] = []
    
    def save_to_filepath(self, save_filepath: str) -> bool:
        with open(save_filepath, 'wb') as f:
            pickle.dump(self, f)
        tqdm.write(f"Successfully saved to {save_filepath}")
        return True

    @staticmethod
    def load_from_ldoc_filepath(ldoc_filepath: str) -> tp.Optional["FIFDocument"]:
        try:
            if not os.path.exists(ldoc_filepath):
                tqdm.write("File not found.")
                return None
            
            with open(ldoc_filepath, 'rb') as f:
                obj = pickle.load(f)
                new_fif_document: FIFDocument = FIFDocument()
                new_fif_document.document_title = obj.doc_title
                new_fif_document.original_json_data = obj.original_json_data
                for processed_component in obj.processed_components:
                    new_processed_component = LightweightComponent()
                    new_processed_component.groundtruth_component_label_list = processed_component.component_uuid
                    new_processed_component.component_embedding = processed_component.component_embedding
                    new_processed_component.linked_document_title_list = processed_component.neighbor_components
                    new_processed_component.original_json_data = processed_component.original_component
                    new_processed_component.subcomponent_embeddings = processed_component.subcomponent_embeddings
                    new_fif_document.processed_component_list.append(new_processed_component)
            return new_fif_document
        except Exception as e:
            tqdm.write(f"Load failed: {e}")
            return None

    @staticmethod
    def load_from_filepath(fif_filepath: str) -> tp.Optional["FIFDocument"]:
        try:
            if not os.path.exists(fif_filepath):
                tqdm.write("File not found.")
                return None
            
            with open(fif_filepath, 'rb') as f:
                obj = pickle.load(f)
            return obj
        except Exception as e:
            tqdm.write(f"Load failed: {e}")
            return None

class DocumentSummarizer:
    def __init__(self) -> None:
        self.image_folderpath: str = ""
        self.image_counter: int = 0
        self.image_link_pattern = r"\[\[([^\]]+)\]\]"
        self.unprocessed_fif_document_list: tp.List[FIFDocument] = []
        self.llm_server_header = {"Content-Type": "application/json"}
        self.llm_server_payload: JSONDict = {
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": []
                }
            ],
            "temperature": 0.2,
            "repetition_penalty": 1.05,
            "top_p": 0.85,
            "max_tokens": 512
        }
    
    def load_data_from_ldoc_folder(self, lilac_doc_folderpath: str, image_folderpath: str) -> bool:
        assert os.path.exists(lilac_doc_folderpath)
        assert os.path.exists(image_folderpath)
        
        self.image_folderpath = image_folderpath
        
        lilac_doc_filename_list = os.listdir(lilac_doc_folderpath)[:5]
        for lilac_doc_filename in tqdm(lilac_doc_filename_list, desc="Loading LILaC Document"):
            lilac_doc_filepath: str = os.path.join(lilac_doc_folderpath, lilac_doc_filename)
            new_fif_document: tp.Optional[FIFDocument] = FIFDocument.load_from_ldoc_filepath(lilac_doc_filepath)
            if new_fif_document:
                self.unprocessed_fif_document_list.append(new_fif_document)
        
        return False
    
    async def _prepare_payload(self, doc: "FIFDocument") -> tp.Dict[str, tp.Any]:
        final_text_prompt, image_filepath_list = self._serialize_document(
            doc.document_title, doc.processed_component_list
        )
        encoded_images = self._serialize_images_to_base64(image_filepath_list[:5]) # TODO: VRAM Limitation
        
        image_content_list: tp.List[JSONDict] = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}
            for data in encoded_images
        ]

        return {
            **self.llm_server_payload,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": final_text_prompt}, *image_content_list]
            }]
        }

    async def _save_to_file(self, filepath: str, data: tp.Any, lock: asyncio.Lock, is_json: bool = True):
        async with lock:
            with open(filepath, "a", encoding="utf-8") as f:
                if is_json:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                else:
                    f.write(str(data) + "\n")

    async def _process_single_document(
        self, 
        client: httpx.AsyncClient, 
        server_url: str, 
        doc: "FIFDocument", 
        semaphore: asyncio.Semaphore,
        write_lock: asyncio.Lock,
        result_path: str,
        failed_path: str
    ):
        async with semaphore:
            try:
                payload = await self._prepare_payload(doc)
                response = await client.post(
                    server_url, 
                    headers=self.llm_server_header, 
                    json=payload, 
                    timeout=180.0
                )
                
                if response.status_code == 200:
                    result_text = response.json()['choices'][0]['message']['content']
                    save_dict: tp.Dict[str, str] = {"doc_title": doc.document_title, "summary": result_text}
                    await self._save_to_file(result_path, save_dict, write_lock)
                else:
                    raise Exception(f"Server {server_url} returned {response.status_code}")

            except Exception as e:
                error_msg = f"Failed to summarize '{doc.document_title}'. Error: {e}"
                await self._save_to_file(failed_path, error_msg, write_lock, is_json=False)

    async def run_parallel_summarize(self, llm_server_list: tp.List[str], llm_result_filepath: str, failed_filepath: str) -> bool:
        assert self.image_folderpath
        assert self.unprocessed_fif_document_list
        assert not os.path.exists(llm_result_filepath)
        assert not os.path.exists(failed_filepath)

        write_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(len(llm_server_list))
        
        async with httpx.AsyncClient() as client:
            tasks: tp.List[tp.Coroutine[tp.Any, tp.Any, None]] = []
            for i, doc in enumerate(self.unprocessed_fif_document_list):
                target_server = llm_server_list[i % len(llm_server_list)]
                tasks.append(
                    self._process_single_document(
                        client, target_server, doc, semaphore, 
                        write_lock, llm_result_filepath, failed_filepath
                    )
                )
            for future in tqdm(
                asyncio.as_completed(tasks), 
                total=len(tasks), 
                unit="doc", 
                desc="Summarizing Documents"
            ):
                await future
        
        return True

    def _serialize_images_to_base64(self, image_filepath_list: tp.List[str]) -> tp.List[str]:
        encoded_data_list: tp.List[str] = []
        for filepath in image_filepath_list:
            path = Path(filepath)
            if not path.is_file(): continue
            try:
                with path.open("rb") as image_file:
                    binary_data = image_file.read()
                    base64_string = base64.b64encode(binary_data).decode('utf-8')
                    encoded_data_list.append(base64_string)
            except (IOError, OSError) as e:
                tqdm.write(f"Error: {e}")
                continue                
        return encoded_data_list
    
    def _serialize_document(self, document_title: str, component_list: tp.List[LightweightComponent]) -> tp.Tuple[str, tp.List[str]]:
        self.image_counter = 0
        serialized_component_list: tp.List[str] = []
        image_filepath_list: tp.List[str] = []
        for component_data in component_list[:10]: # TODO: VRAM Limitation
            component_json_data: JSONDict = component_data.original_json_data
            if component_json_data["type"] == "paragraph":
                serialized_component_list.append(self._serialize_text_component_for_prompt(document_title, component_json_data))
            elif component_json_data["type"] == "table":
                serialized_text, first_image_path = self._serialize_table_component_for_prompt(document_title, component_json_data)
                serialized_component_list.append(serialized_text)
                if first_image_path:
                    image_filepath_list.append(get_clean_savepath_from_url_with_custom_extension(self.image_folderpath, first_image_path, "png"))
            elif component_json_data["type"] == "image":
                serialized_text, image_path = self._serialize_image_component_for_prompt(document_title, component_json_data)
                serialized_component_list.append(serialized_text)
                if image_path:
                    image_filepath_list.append(get_clean_savepath_from_url_with_custom_extension(self.image_folderpath, image_path, "png"))
        
        final_text_prompt: str = (
            DOCUMENT_SUMMARIZATION_QUERY
            + "\n".join(serialized_component_list)
            + "\n[OUTPUT]\n"
        )
        return final_text_prompt, image_filepath_list

    def _serialize_text_component_for_prompt(self, document_title: str, text_component_data: JSONDict) -> str:
        return f"/*\n[Passage]\nTitle: {document_title}\nSection: {', '.join(text_component_data['heading_path'])}\n\n{text_component_data['paragraph']}\n*/\n\n"

    def _serialize_table_cell_for_prompt(self, cell_textdata: str, first_image_taken: bool) -> tp.Tuple[str, str]:
        serialized_text: str = ""
        image_list: tp.List[str] = [item for item in re.findall(self.image_link_pattern, cell_textdata)]
        cleaned_text: str = re.sub(self.image_link_pattern, '', cell_textdata).strip()
        image_path: str = image_list[0] if image_list and not first_image_taken and os.path.exists(image_list[0]) else ""
        if image_path:
            serialized_text += f"<Image {self.image_counter}>"
            self.image_counter += 1
        if serialized_text:
            serialized_text += ", "
        serialized_text += cleaned_text
        
        normalized_serialized_text: str = re.sub(r'\s*,\s*', ' , ', serialized_text)
        return normalized_serialized_text, image_path

    def _serialize_table_component_for_prompt(self, document_title: str, table_component_data: JSONDict) -> tp.Tuple[str, str]:
        serialized_text: str = f"/*\n[Table]\nTitle: {document_title}\nSection: {', '.join(table_component_data['heading_path'])}\n\n"
        first_img_taken: bool = False
        first_image_path: str = ""
        
        serialized_table: tp.List[str] = []
        for table_row in table_component_data["table"]:
            serialized_cell_text_list: tp.List[str] = []
            for table_cell in table_row:
                serialized_cell_text, image_path = self._serialize_table_cell_for_prompt(table_cell, first_img_taken)
                if image_path:
                    first_img_taken = True
                    first_image_path = image_path
                serialized_cell_text_list.append(serialized_cell_text)
            serialized_table.append(" | ".join(serialized_cell_text_list))
        serialized_table.append("*/\n")

        serialized_text += '\n'.join(serialized_table) + "\n"
        return serialized_text, first_image_path

    def _serialize_image_component_for_prompt(self, document_title: str, image_component_data: JSONDict) -> tp.Tuple[str, str]:
        serialized_text: str = f"/*\n[Image]\nTitle: {document_title}\nSection: {', '.join(image_component_data['heading_path'])}\n\n"

        image_path: str = get_clean_savepath_from_url_with_custom_extension(self.image_folderpath, image_component_data["src"], "png")
        if os.path.exists(image_path):
            serialized_text += f"<Image {self.image_counter}>\nCaption: {image_component_data['caption']}\n"
            self.image_counter += 1
        else:
            image_path = ""
        
        serialized_text += "*/\n\n"
        return serialized_text, image_path

from __future__ import annotations

import os
import re
import json
import base64
import pickle
import asyncio
import requests
import typing as tp
from pathlib import Path
import itertools

import cv2
import torch
import pysbd # type: ignore
import httpx
import numpy as np
from tqdm.asyncio import tqdm
from numpy.typing import NDArray
from ultralytics import YOLO # type: ignore

from common import get_clean_savepath_from_url_with_custom_extension
from type import JSONDict
from query import DOCUMENT_SUMMARIZATION_QUERY, IMAGE_OCR_QUERY, IMAGE_EXPLANATION_QUERY
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
    def load_from_ldoc_filepath(ldoc_filepath: str) -> tp.Optional[FIFDocument]:
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
    def load_from_filepath(fif_filepath: str) -> tp.Optional[FIFDocument]:
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

class LLMQuerySerializer:
    def __init__(self, image_folderpath: str) -> None:
        assert os.path.exists(image_folderpath)
        
        self.image_folderpath: str = image_folderpath
        self.image_counter: int = 0
    
    def serialize_images_to_base64(self, image_filepath_list: tp.List[str]) -> tp.List[str]:
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
    
    def serialize_document(self, document_title: str, json_component_list: tp.List[JSONDict]) -> tp.Tuple[str, tp.List[str]]:
        self.image_counter = 0
        serialized_component_list: tp.List[str] = []
        image_filepath_list: tp.List[str] = []
        for json_component_data in json_component_list[:10]: # TODO: VRAM Limitation
            if json_component_data["type"] == "paragraph":
                serialized_component_list.append(self._serialize_text_component_for_prompt(document_title, json_component_data))
            elif json_component_data["type"] == "table":
                serialized_text, first_image_path = self._serialize_table_component_for_prompt(document_title, json_component_data)
                serialized_component_list.append(serialized_text)
                if first_image_path:
                    image_filepath_list.append(get_clean_savepath_from_url_with_custom_extension(self.image_folderpath, first_image_path, "png"))
            elif json_component_data["type"] == "image":
                serialized_text, image_path = self._serialize_image_component_for_prompt(document_title, json_component_data)
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
        return f"/*\n[Passage]\nTitle: {document_title}\nSection: {', '.join(text_component_data['heading_path'])}\n\n{text_component_data['text']}\n*/\n\n"

    def _serialize_table_cell_for_prompt(self, cell_data: JSONDict, first_image_taken: bool) -> tp.Tuple[str, str]:
        serialized_text: str = ""
        image_name: str = cell_data.get("image_name", "")
        image_path: str = image_name[0] if (
            image_name
            and not first_image_taken
            and os.path.exists(os.path.join(self.image_folderpath, image_name + ".png"))
        ) else ""
        if image_path:
            serialized_text += f"<Image {self.image_counter}>"
            self.image_counter += 1
        if serialized_text:
            serialized_text += ", "
        serialized_text += cell_data["text"].strip()
        
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

        image_name = image_component_data.get("image_name")
        if image_name: image_path: str = os.path.join(self.image_folderpath, image_name)
        else: return "", ""
        
        if os.path.exists(image_path):
            serialized_text += f"<Image {self.image_counter}>\nCaption: {image_component_data['caption']}\n"
            self.image_counter += 1
        else:
            image_path = ""
        
        serialized_text += "*/\n\n"
        return serialized_text, image_path

class DocumentSummarizer:
    def __init__(self, document_serializer: LLMQuerySerializer) -> None:
        self.document_serializer: LLMQuerySerializer = document_serializer
        self.fif_json_list: tp.List[JSONDict] = []
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
    
    def load_data_from_json_folderpath(self, json_doc_folderpath: str) -> bool:
        assert os.path.exists(json_doc_folderpath)
        
        json_doc_filename_list = os.listdir(json_doc_folderpath)
        for json_doc_filename in tqdm(json_doc_filename_list, desc="Loading JSON Document"):
            if not json_doc_filename.endswith(".json"): continue
            
            json_doc_filepath: str = os.path.join(json_doc_folderpath, json_doc_filename)
            with open(json_doc_filepath, "r", encoding="utf-8") as json_doc_file:
                new_fif_json: JSONDict = json.load(json_doc_file)
                if new_fif_json:
                    self.fif_json_list.append(new_fif_json)
        
        return True
    
    async def _prepare_payload(self, json_doc: JSONDict) -> tp.Dict[str, tp.Any]:
        json_text_list = list(json_doc["text"].values())
        json_table_list = list(json_doc["table"].values())
        json_image_list = [
            item for item in json_doc["image"].values() 
            if item.get("image_name") is not None
        ]
        
        for item in json_text_list: item["type"] = "text"
        for item in json_table_list: item["type"] = "table"
        for item in json_image_list: item["type"] = "image"
        
        json_component_list = json_text_list + json_table_list + json_image_list
        json_component_list = [json_component for json_component in json_component_list if not json_component["heading_path"]]
        
        final_text_prompt, image_filepath_list = self.document_serializer.serialize_document(json_doc["title"], json_component_list)
        encoded_images = self.document_serializer.serialize_images_to_base64(image_filepath_list[:5]) # TODO: VRAM Limitation
        
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
        doc: JSONDict, 
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
                    save_dict: tp.Dict[str, str] = {"doc_title": doc["title"], "summary": result_text}
                    await self._save_to_file(result_path, save_dict, write_lock)
                else:
                    raise Exception(f"Server {server_url} returned {response.status_code}")
            except Exception as e:
                error_msg = f"Failed to summarize '{doc['title']}'. Error: {e}"
                await self._save_to_file(failed_path, error_msg, write_lock, is_json=False)

    async def run_parallel_summarize(self, llm_server_list: tp.List[str], llm_result_filepath: str, failed_filepath: str) -> bool:
        assert self.fif_json_list
        
        processed_titles: tp.Set[str] = set()
        if os.path.exists(llm_result_filepath):
            try:
                with open(llm_result_filepath, 'r', encoding='utf-8') as llm_result_file:
                    for llm_result_line in llm_result_file:
                        llm_result_line = llm_result_line.strip()
                        if not llm_result_line: continue
                        try:
                            llm_result_data = json.loads(llm_result_line)
                            if "doc_title" in llm_result_data:
                                processed_titles.add(llm_result_data["doc_title"])
                        except json.JSONDecodeError:
                            continue
                tqdm.write(f"Skipping {len(processed_titles)} already processed documents.")
            except Exception as e:
                tqdm.write(f"Error reading existing result file: {e}")

        target_docs = [
            doc for doc in self.fif_json_list 
            if doc.get("title") not in processed_titles
        ]

        if not target_docs:
            tqdm.write("All documents are already processed.")
            return True

        write_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(len(llm_server_list))
        
        async with httpx.AsyncClient() as client:
            tasks: tp.List[tp.Awaitable[tp.Any]] = []
            for i, doc in enumerate(target_docs):
                target_server = llm_server_list[i % len(llm_server_list)]
                tasks.append(
                    self._process_single_document(
                        client, target_server, doc, semaphore, 
                        write_lock, llm_result_filepath, failed_filepath
                    )
                )

            for future in tqdm.as_completed(tasks, total=len(tasks), unit="doc", desc="Summarizing Documents"): # type: ignore
                await future
        
        return True

class BatchImageDescriptor:
    def __init__(self) -> None:
        self.image_filepath_list: tp.List[str] = []
        self.llm_server_header = {"Content-Type": "application/json"}
        self.llm_server_payload: JSONDict = {
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "messages": [],
            "temperature": 0.2,
            "repetition_penalty": 1.05,
            "top_p": 0.85,
            "max_tokens": 512
        }

    def load_image_filelist(self, image_folder_path: str) -> bool:
        assert os.path.exists(image_folder_path)
        self.image_filepath_list = [
            os.path.join(image_folder_path, f) 
            for f in os.listdir(image_folder_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]
        return True

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def _prepare_payload(self, image_path: str, prompt: str) -> JSONDict:
        base64_image = self._encode_image(image_path)
        
        return {
            **self.llm_server_payload,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        }

    async def _save_to_file(self, filepath: str, data: tp.Any, lock: asyncio.Lock, is_json: bool = True):
        async with lock:
            with open(filepath, "a", encoding="utf-8") as f:
                if is_json:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                else:
                    f.write(str(data) + "\n")

    async def _request_llm(self, client: httpx.AsyncClient, server_url: str, payload: JSONDict) -> str:
        response = await client.post(
            server_url,
            headers=self.llm_server_header,
            json=payload,
            timeout=120.0
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    async def _process_single_image(
        self,
        client: httpx.AsyncClient,
        server_url: str,
        image_path: str,
        semaphore: asyncio.Semaphore,
        write_lock: asyncio.Lock,
        result_path: str,
        failed_path: str
    ):
        async with semaphore:
            try:
                explanation_payload = await self._prepare_payload(image_path, IMAGE_EXPLANATION_QUERY)
                explanation_text = await self._request_llm(client, server_url, explanation_payload)

                ocr_payload = await self._prepare_payload(image_path, IMAGE_OCR_QUERY)
                ocr_text = await self._request_llm(client, server_url, ocr_payload)

                result = {
                    "image": os.path.basename(image_path),
                    "explanation": explanation_text,
                    "ocr": ocr_text,
                }
                await self._save_to_file(result_path, result, write_lock)

            except Exception as e:
                error_log = f"Error processing {image_path}: {str(e)}"
                await self._save_to_file(failed_path, error_log, write_lock, is_json=False)

    async def run_parallel_description(
        self, 
        llm_server_list: tp.List[str], 
        description_file_path: str, 
        failed_file_path: str
    ) -> bool:
        processed_pathset: tp.Set[str] = set()
        if os.path.exists(description_file_path):
            with open(description_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        processed_pathset.add(data["image"])
                    except: continue

        target_list = [
            p for p in self.image_filepath_list 
            if os.path.basename(p) not in processed_pathset
        ]

        if not target_list:
            tqdm.write("No images to process.")
            return True

        write_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(len(llm_server_list))
        server_cycle = itertools.cycle(llm_server_list)

        async with httpx.AsyncClient() as client:
            tasks: tp.List[tp.Coroutine[tp.Any, tp.Any, None]] = []
            for image_path in target_list:
                tasks.append(
                    self._process_single_image(
                        client, next(server_cycle), image_path, 
                        semaphore, write_lock, description_file_path, failed_file_path
                    )
                )

            for future in tqdm.as_completed(tasks, total=len(tasks), desc="Processing Images"): # type: ignore
                await future

        return True

class BatchObjectDetector:
    def __init__(
        self,
        device: str = "cuda",
        max_objects: int = 3,
    ):
        MODEL_NAME: str = "yolov8x.pt"

        self.device = device
        self.max_objects = max_objects

        self.model = YOLO(MODEL_NAME)
        self.model.to(device)
        self.model.fuse()

        self.image_filepath_list: tp.List[str] = []
        self.progress_bar = tqdm(total=0, desc="Object Detecting")

    def load_image_filelist(self, image_folder_path: str) -> None:
        assert os.path.exists(image_folder_path)
        self.image_filepath_list = [os.path.join(image_folder_path, filename) for filename in os.listdir(image_folder_path)]
        self.image_filepath_list.sort()

    def run_detection(
        self,
        output_json_path: str,
        failed_file_path: str,
        batch_size: int = 32,
    ):
        self.progress_bar.total = len(self.image_filepath_list)
        
        with open(output_json_path, "w", encoding="utf-8") as output_file:
            failed_filename_list: tp.List[str] = []
            
            for i in range(0, len(self.image_filepath_list), batch_size):
                batch_filepath_list: tp.List[str] = self.image_filepath_list[i : i + batch_size]

                valid_image_list: tp.List[cv2.typing.MatLike] = []
                valid_filepath_list: tp.List[str] = []
                
                for p in batch_filepath_list:
                    try:
                        img = cv2.imread(p)
                        valid_image_list.append(img)
                        valid_filepath_list.append(p)
                    except:
                        failed_filename_list.append(p)
                        self.progress_bar.update(1)
                if not valid_image_list: continue

                batch_outputs = self.batch_detect(valid_image_list)
                self.progress_bar.update(len(valid_filepath_list))

                for path, dets in zip(valid_filepath_list, batch_outputs):
                    line_data: JSONDict = {"image": os.path.basename(path), "bounding_box": dets}
                    output_file.write(json.dumps(line_data, ensure_ascii=False) + "\n")

        if failed_filename_list:
            with open(failed_file_path, "w", encoding="utf-8") as failed_file:
                for path in failed_filename_list:
                    failed_file.write(path + "\n")

    @torch.inference_mode()
    def batch_detect(
        self,
        images: tp.List[cv2.typing.MatLike],
    ) -> tp.List[tp.List[tp.List[int]]]:
        preds = self.model.predict( # type: ignore
            source=images,
            device=self.device,
            conf=0.15,
            iou=0.6,
            agnostic_nms=True,
            half=True,
            verbose=False,
        )

        outputs: tp.List[tp.List[tp.List[int]]] = []

        for r in preds:
            if r.boxes is None:
                outputs.append([])
                continue

            boxes = r.boxes.xyxy.cpu().numpy() # type: ignore
            scores = r.boxes.conf.cpu().numpy() # type: ignore

            order = scores.argsort()[::-1][: self.max_objects] # type: ignore
            selected = boxes[order].astype(int).tolist() # type: ignore

            outputs.append(selected) # type: ignore

        return outputs

SerializedDataForEmbeding = tp.Tuple[str, str, tp.Optional[tp.Tuple[int, int, int, int]]]

class EmbeddingSerializer:
    def __init__(
            self,
            processed_image_folder: str,
        ) -> None:
        self.image_data_folderpath: str = processed_image_folder
        self.text_segmenter = pysbd.Segmenter(language="en", clean=False)
        
        self.doc_title: str = ""
        self.image_description_dict: tp.Dict[str, JSONDict] = dict()
    
    def run(
        self,
        json_data: JSONDict,
        image_description_dict: tp.Dict[str, JSONDict],
    ) -> tp.Tuple[tp.Dict[str, SerializedDataForEmbeding], tp.List[SerializedDataForEmbeding], tp.Dict[str, tp.Tuple[int, int]]]:
        self.doc_title: str = json_data["title"]
        self.image_description_dict = image_description_dict
        
        result_serialized_component_dict: tp.Dict[str, SerializedDataForEmbeding] = dict()
        component_metadata_dict: tp.Dict[str, tp.Tuple[int, int]] = dict()
        result_serialized_subcomponent_list: tp.List[SerializedDataForEmbeding] = []

        for json_key in json_data["text"]:
            component = json_data["text"][json_key]
            prev_length = len(result_serialized_component_dict)
            try:
                serialized_component, serialized_subcomponent_list = self.process_text_component(component)
                result_serialized_component_dict[json_key] = serialized_component
                result_serialized_subcomponent_list.extend(serialized_subcomponent_list)
                component_metadata_dict[json_key] = (prev_length, prev_length + len(serialized_subcomponent_list))
            except:
                component_metadata_dict[json_key] = (prev_length, prev_length)
        for json_key in json_data["table"]:
            component = json_data["table"][json_key]
            prev_length = len(result_serialized_subcomponent_list)
            try:
                serialized_component, serialized_subcomponent_list = self.process_table_component(component)
                result_serialized_component_dict[json_key] = serialized_component
                result_serialized_subcomponent_list.extend(serialized_subcomponent_list)
                component_metadata_dict[json_key] = (prev_length, prev_length + len(serialized_subcomponent_list))
            except:
                component_metadata_dict[json_key] = (prev_length, prev_length)
        for json_key in json_data["image"]:
            component = json_data["image"][json_key]
            prev_length = len(result_serialized_subcomponent_list)
            try:
                serialized_component, serialized_subcomponent_list = self.process_image_component(component)
                result_serialized_component_dict[json_key] = serialized_component
                result_serialized_subcomponent_list.extend(serialized_subcomponent_list)
                component_metadata_dict[json_key] = (prev_length, prev_length + len(serialized_subcomponent_list))
            except:
                component_metadata_dict[json_key] = (prev_length, prev_length)
        
        return result_serialized_component_dict, result_serialized_subcomponent_list, component_metadata_dict
    
    def process_text_component(self, component: JSONDict) -> tp.Tuple[SerializedDataForEmbeding, tp.List[SerializedDataForEmbeding]]:
        serialized_text_prefix = f"{self.doc_title} [SEP] {' , '.join(component['heading_path'])} [SEP] "
        sentence_list: tp.List[str] = self.text_segmenter.segment(component['text']) # type: ignore
        if not sentence_list: sentence_list = [component.get("text", "")]
    
        component_serialized_text = serialized_text_prefix + component['text']
        component_serialized_text = component_serialized_text.replace("\n","")
        subcomponent_serialized_text_list: tp.List[SerializedDataForEmbeding] = [(serialized_text_prefix + sentence.replace("\n",""), "", None) for sentence in sentence_list]
        
        return (component_serialized_text, "", None), subcomponent_serialized_text_list
    
    def process_table_component(self, component: JSONDict) -> tp.Tuple[SerializedDataForEmbeding, tp.List[SerializedDataForEmbeding]]:
        serialized_text_prefix = f"{self.doc_title} [SEP] {' , '.join(component['heading_path'])} [SEP] "
        original_table: tp.List[tp.List[JSONDict]] = component["table"]
        subcomponent_serialized_data_list: tp.List[SerializedDataForEmbeding] = []
        
        assert original_table
        
        table_first_row: tp.List[JSONDict] = original_table[0]
        
        if len(original_table) <= 2:
            text, image_filepath_list = self._flatten_table(original_table)
            serialized_text = serialized_text_prefix + text
            subcomponent_serialized_data_list = [(serialized_text.replace("\n", ""), image_filepath_list[0] if image_filepath_list else "", None)]
        else:
            for table_line in original_table[1:]:
                text, image_filepath_list = self._flatten_table([table_first_row, table_line])
                serialized_text = serialized_text_prefix + text
                subcomponent_serialized_data_list.append((serialized_text.replace("\n", ""), image_filepath_list[0] if image_filepath_list else "", None))
        
        full_text, full_img_filepath_list = self._flatten_table(original_table)
        component_serialized_data = (full_text.replace("\n", ""), full_img_filepath_list[0] if full_img_filepath_list else "", None)
        
        return component_serialized_data, subcomponent_serialized_data_list
    
    def process_image_component(self, component: JSONDict) -> tp.Tuple[SerializedDataForEmbeding, tp.List[SerializedDataForEmbeding]]:
        image_filename = component["image_name"]
        image_filepath = os.path.join(self.image_data_folderpath, image_filename)
        subcomponent_serialized_data_list: tp.List[SerializedDataForEmbeding] = []
        
        assert os.path.exists(image_filepath)
        assert image_filename in self.image_description_dict
        
        image_metadata = self.image_description_dict[image_filename]
        serialized_text = f"{self.doc_title} [SEP] {' , '.join(component['heading_path'])} [SEP] {component['caption']}"
        serialized_text += f" [SEP] {image_metadata['explanation']}" if image_metadata["explanation"] else ""
        serialized_text += f" [SEP] {image_metadata['ocr']}" if image_metadata["ocr"] else ""

        component_serialized_data = (serialized_text, image_filepath, None)
        if not image_metadata["bounding_box"]: subcomponent_serialized_data_list = [component_serialized_data]
        else:
            for bounding_box in image_metadata["bounding_box"]:
                subcomponent_serialized_data_list.append((serialized_text, image_filepath, bounding_box))
        
        return component_serialized_data, subcomponent_serialized_data_list

    def _flatten_table(self, table_data: tp.List[tp.List[JSONDict]]) -> tp.Tuple[str, tp.List[str]]:
        temp_image_name_list: tp.List[str] = []
        result_text_list: tp.List[str] = []
        result_image_path_list: tp.List[str] = []

        for table_row in table_data:
            temp_list: tp.List[str] = []
            for table_elem in table_row:
                text = table_elem.get("text", "").strip()
                image_name = table_elem.get("image_name", "")
                if image_name: temp_image_name_list.append(image_name)
                if text: temp_list.append(text)
            result_text_list.append(" ".join(temp_list))
        result_text = " [SEP] ".join(result_text_list)
        
        image_index = 1
        for image_name in temp_image_name_list:
            image_name_with_ext: str = image_name + ".png"
            image_full_filepath: str = os.path.join(self.image_data_folderpath, image_name_with_ext)
            if os.path.exists(image_full_filepath) and image_name_with_ext in self.image_description_dict:
                result_image_path_list.append(image_full_filepath)
                result_text += f" [SEP] <Image {image_index}> {image_full_filepath} {self.image_description_dict[image_name_with_ext]['explanation']}"
                image_index += 1
        return result_text, result_image_path_list

class DocumentEmbedder:
    def __init__(self, embedding_serializer: EmbeddingSerializer) -> None:
        self.embedding_serializer: EmbeddingSerializer = embedding_serializer
        self.json_filepath_list: tp.List[str] = []
        self.document_summarization_dict: tp.Dict[str, str] = {}
        self.image_description_dict: tp.Dict[str, JSONDict] = {}
        self.object_detection_dict: tp.Dict[str, JSONDict] = {}
    
    def load_files(
        self,
        json_folderpath: str,
        document_summarization_filepath: str,
        object_detection_filepath: str,
        image_description_filepath: str
    ):
        assert os.path.exists(json_folderpath)
        self.json_filepath_list = [os.path.join(json_folderpath, filename) for filename in os.listdir(json_folderpath) if filename.endswith(".json")]
        self.document_summarization_dict = self._load_jsonl_to_dict(document_summarization_filepath, "doc_title", "summary")
        self.image_description_dict = self._load_jsonl_to_dict(image_description_filepath, "image", "explanation")
        self.object_detection_dict = self._load_jsonl_to_dict(object_detection_filepath, "image", "bounding_box")
    
    def _load_jsonl_to_dict(self, file_path: str, title_key: str, data_key: str) -> tp.Dict[str, tp.Any]:
        assert os.path.exists(file_path), "Error: File not found at {file_path}"
        result_dict: tp.Dict[str, JSONDict] = {}
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading {os.path.basename(file_path)}"):
                line = line.strip()
                if not line: continue
                item = json.loads(line)
                result_dict[item[title_key]] = item[data_key]
        return result_dict
    
    def run_embedding(
        self,
        embedding_server_list: tp.List[str],
        embedding_result_folderpath: str,
        failed_filepath: str
    ) -> bool:
        with open(failed_filepath, "a", encoding="utf-8") as failed_file:
            for json_filepath in tqdm(self.json_filepath_list, desc="Embedding Document Summary"): # TODO
                with open(json_filepath, "r", encoding="utf-8") as json_file:
                    json_file_data: JSONDict = json.load(json_file)
                doc_title = json_file_data["title"]
                result_filename = os.path.join(embedding_result_folderpath, os.path.basename(json_filepath) + ".npz")
                
                if os.path.exists(result_filename): continue
                
                # TODO: serialize
                document_summary = self.document_summarization_dict[doc_title]
                result_serialized_component_dict, result_serialized_subcomponent_list, component_metadata_dict = self.embedding_serializer.run(
                    json_file_data,
                    self.image_description_dict
                )
                
                # TODO: request
                result_embedding_dict: tp.Dict[str, NDArray[np.floating[tp.Any]]] = dict()
                result_embedding_dict["doc_emb"] = self._request_embedding(embedding_server_list[0], (document_summary, "", None))
                
                result_embedding_dict: tp.Dict[str, NDArray[np.floating[tp.Any]]] = {}
                for component_key in result_serialized_component_dict:
                    result_serialized_data: SerializedDataForEmbeding = result_serialized_component_dict[component_key]
                    try:
                        result_embedding_dict[component_key] = self._request_embedding(embedding_server_list[0], result_serialized_data)
                    except Exception as e:
                        failed_file.write(f"{doc_title}\t{e}\n")
                
                temp_subcomponents: tp.List[NDArray[np.floating[tp.Any]]] = []
                for subcomponent_data in result_serialized_subcomponent_list:
                    try:
                        temp_subcomponents.append(self._request_embedding(embedding_server_list[0], subcomponent_data))
                    except Exception as e:
                        failed_file.write(f"{doc_title}\t{e}\n")
                
                result_embedding_dict["subembs"] = np.array(temp_subcomponents)
                result_embedding_dict["metadata"] = np.array(component_metadata_dict, dtype=object)
                
                # TODO: save
                np.savez(result_filename, **result_embedding_dict)
                failed_file.flush()
        
        return True

    def _request_embedding(self, embedding_server_url: str, serialized_data: SerializedDataForEmbeding) -> NDArray[np.floating[tp.Any]]:
        payload: JSONDict = {"text": serialized_data[0], "img_path": serialized_data[1], "bounding_box": serialized_data[2]}
        res = requests.post(f"{embedding_server_url}/embed", json=payload, timeout=120)
        res.raise_for_status()
        return np.array(res.json()["embedding"], dtype=np.float32)

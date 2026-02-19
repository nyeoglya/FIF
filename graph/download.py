import re
import json
import typing as tp
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
from datasets import load_dataset, Dataset # type: ignore
from type import JSONDict
from common import safe_doc_id

MergedDocType = tp.Dict[str, tp.Dict[str, tp.Dict[str, tp.Any]]]

class FiFDatasetLoader:
    DEFAULT_CONFIGS = ['text_component', 'table_component', 'image_component', 'image_dump']
    
    def __init__(self, dataset_name: str, save_path: str):
        self.dataset_name = dataset_name
        self.base_out = Path(save_path)
        self.image_out = self.base_out / "images"
        
        print(f"Loading dataset: {self.dataset_name}...")
        self.data: tp.Dict[str, Dataset] = dict()
        
        self.data['text'] = load_dataset(self.dataset_name, 'text_component', split='text')
        self.data['table'] = load_dataset(self.dataset_name, 'table_component', split='table')
        self.data['image_meta'] = load_dataset(self.dataset_name, 'image_component', split='image_meta')
        self.data['image_dump'] = load_dataset(self.dataset_name, 'image_dump', split='dump')
        self.data['dev'] = load_dataset(self.dataset_name, 'dev', split='test')

        print("Mapping image bytes...")
        self.image_bytes_dict = {
            row["image_name"]: row["byte_data"] 
            for row in tqdm(self.data['image_dump'], desc="Caching Images")
        }

        self.merged_doc_dict: tp.DefaultDict[str, MergedDocType] = defaultdict(lambda: {"text": {}, "table": {}, "image": {}})
        self._prepare_merged_data()

    def _safe_filename(self, name: str, max_len: int = 50) -> str:
        if not name: return "unknown"
        name = re.sub(r'[\/\\\:\*\?\"\<\>\|]', '_', name)
        name = name.strip()
        return name[:max_len]

    def _prepare_merged_data(self):
        for row in tqdm(self.data['text'], desc="Processing Text"):
            doc_title: str = row['doc_title']
            self.merged_doc_dict[doc_title]["text"][row["component_id"]] = {
                "text": row["component"],
                "heading_path": json.loads(row["heading_path"]),
                "hyperlinks": json.loads(row["hyperlinks"]),
                "label_id": row["label_id"]
            }

        for row in tqdm(self.data['table'], desc="Processing Tables"):
            doc_title: str = row['doc_title']
            self.merged_doc_dict[doc_title]["table"][row["component_id"]] = {
                "table": json.loads(row["component"]),
                "heading_path": json.loads(row["heading_path"]),
                "hyperlinks": json.loads(row["hyperlinks"]),
                "label_id": row["label_id"]
            }

        for row in tqdm(self.data['image_meta'], desc="Processing Images"):
            doc_title: str = row['doc_title']
            comp_json: JSONDict = json.loads(row["component"])
            self.merged_doc_dict[doc_title]["image"][row["component_id"]] = {
                "image_name": comp_json.get("image_name"),
                "heading_path": json.loads(row["heading_path"]),
                "caption": comp_json.get("caption", ""),
                "hyperlinks": json.loads(row["hyperlinks"]),
                "label_id": row["label_id"]
            }

    def restore(self, overwrite: bool = False):
        if self.base_out.exists() and not overwrite:
            raise FileExistsError(f"Save folder '{self.base_out}' already exists!")
        
        self.image_out.mkdir(parents=True, exist_ok=True)

        for doc_title, content in tqdm(self.merged_doc_dict.items(), desc="Restoring Documents"):
            safe_doc_title = safe_doc_id(doc_title)
            
            for _, img_info in content["image"].items():
                image_name = img_info["image_name"]
                img_bytes: bytes = self.image_bytes_dict.get(image_name)
                
                if image_name and isinstance(img_bytes, bytes):
                    safe_img_name = self._safe_filename(image_name)
                    filename = f"{safe_img_name}.png"
                    with open(self.image_out / filename, "wb") as f:
                        f.write(img_bytes)
                    img_info["image_name"] = filename
                else:
                    img_info["image_name"] = None

            for section in ("text", "table", "image"):
                for comp_data in content[section].values():
                    if "hyperlinks" in comp_data:
                        comp_data["hyperlinks"] = [safe_doc_id(h) for h in comp_data["hyperlinks"]]

            doc_json: JSONDict = {
                "title": safe_doc_title,
                "text": content["text"],
                "table": content["table"],
                "image": content["image"]
            }
            json_path = self.base_out / f"{safe_doc_title}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(doc_json, f, ensure_ascii=False, indent=2)

        print(f"Successfully restored to: {self.base_out.resolve()}")
    
    def restore_dev_jsonl(self, filename: str):
        dev_path = self.base_out / filename
        print(f"Restoring dev queries to {dev_path}...")

        with open(dev_path, "w", encoding="utf-8") as f:
            for row_data in tqdm(self.data['dev'], desc="Processing Dev Queries"):
                qid = row_data.get("qid")
                question = row_data.get("question")
                primary_answer = row_data.get("answer", "")
                evidences_field = row_data.get("evidence", [])
                safe_evidences = [
                    [safe_doc_id(evi[0])] + evi[1:] for evi in evidences_field
                ]
                record = {
                    "qid": qid,
                    "question": question,
                    "answer": str(primary_answer),
                    "evidence": safe_evidences
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Successfully saved {len(self.data['dev'])} queries to {dev_path}")

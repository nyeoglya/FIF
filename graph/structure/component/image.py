import typing as tp
from pathlib import Path

from ._component import Component
from .constants import EmbeddingMode
from type import JSONDict

class Image(Component):
    def __init__(self, 
        docname: str,
        cid: str, 
        document_title: str, 
        hierarchy_dict: JSONDict, 
        raw_component: dict,
        images_dir: str = "",
        image_summaries_dir: str = ""
    ):
        super().__init__(
            document_path = docname, 
            component_id = cid, 
            document_title = document_title, 
            hierarchy_dict = hierarchy_dict, 
            raw_component = raw_component,
        )
        
        self._images_dir = images_dir
        self._image_summaries_dir = image_summaries_dir
    
    def serialize(self, mode: str) -> JSONDict:
        serialized_image = self._document_title + " [SEP] " + self.get_serialized_hierarchy_path()
        representations = {
            "caption": " ",
            EmbeddingMode.IMAGE.value: None,
            EmbeddingMode.SUMMARY.value: " ",
            EmbeddingMode.OCR.value: " ",
        }
    
        if "caption" in self._raw_component:
            if "text" in self._raw_component["caption"]:
                representations["caption"] = " [SEP] " + self._raw_component["caption"]["text"]
            
            if "ocr" in self._raw_component["caption"]:
                representations["ocr"] = " [SEP] " + self._raw_component["caption"]["ocr"]
        
        if "filename" in self._raw_component and self._raw_component["filename"] != None:
            raw = self._raw_component["filename"]
            image_filename = self._normalize_image_filename(raw)
            if image_filename:
                image_filepath = self._get_image_abs_path(image_filename)
                representations[EmbeddingMode.IMAGE.value] = image_filepath

                image_summary = self._get_image_summary(image_filename)
                representations["summary"] = image_summary
            
        to_embed_filepaths = []
        serialized_image += " [SEP] " + representations["caption"]
        
        if EmbeddingMode.IMAGE.value in mode:
            if representations[EmbeddingMode.IMAGE.value] != None:
                to_embed_filepaths.append(representations[EmbeddingMode.IMAGE.value])
        if EmbeddingMode.SUMMARY.value in mode:
            if representations[EmbeddingMode.SUMMARY.value] != None:
                serialized_image += " [SEP] " + representations[EmbeddingMode.SUMMARY.value]
        if EmbeddingMode.OCR.value in mode:
            if representations[EmbeddingMode.OCR.value] != None:  
                serialized_image += " [SEP] " + representations[EmbeddingMode.OCR.value]
        
        serialized_image = serialized_image.replace("\n", " ")
        
        embedding_obj: JSONDict = {
            "id": [self._document_path, self._component_id],
            "target": {
                "text": serialized_image,
                "images": to_embed_filepaths
            }
        }
        
        return embedding_obj
    
    def _get_image_summary(self, image_filename: str) -> str:
        p = Path(image_filename)
        if p.parent == Path("."): stem = p.stem
        else: stem = p.parent.name

        target_txt = f"{stem}.txt"
        summary_path = Path(self._image_summaries_dir) / target_txt
    
        assert summary_path.exists()

        with summary_path.open("r", encoding="utf-8") as fh:
            return fh.read()
    
    def serialize_into_prompt(self, next_image_idx: int):
        title = self.get_doctitle()
        parent_section = self.get_serialized_hierarchy_path()
        image_component = self._raw_component
    
        serialized_text = ""
        image_paths = []
        
        serialized_text = "/*\n"
        serialized_text += "[Image]\n"
        serialized_text += "Title: " + title + "\n"
        serialized_text += "Section: " + parent_section + "\n\n"
        
        if "filename" in image_component and image_component["filename"] != None:
            raw = image_component["filename"]
            image_filename = self._normalize_image_filename(raw)
            if image_filename:
                image_path = self._get_image_abs_path(image_filename)
                if image_path:
                    serialized_text += "<Image " + str(next_image_idx) + ">" + "\n"
                    image_paths = [image_path]
                    next_image_idx += 1
                
        if "caption" in image_component and "text" in image_component["caption"]:
            serialized_text += "Caption: " + image_component["caption"]["text"] + "\n"
            
        serialized_text += "*/\n\n"
        
        return serialized_text, image_paths, next_image_idx
    
    def get_hyperlinked_filenames(self) -> tp.List[str]:
        if "caption" not in self._raw_component\
            or "edges" not in self._raw_component["caption"]:
            return []
        
        edges: tp.List[str] = []
        for edge_obj in self._raw_component["caption"]["edges"]:
            edges.append(edge_obj["edge"])
        
        return edges
    
    def _normalize_image_filename(self, x):
        # Accept str | list | tuple and return a str path or None
        if isinstance(x, (list, tuple)):
            for v in x:
                if isinstance(v, str):
                    return v
            return None
        return x if isinstance(x, str) else None

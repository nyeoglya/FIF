
import os
import re
import json
import typing as tp

from type import JSONDict
from ._component import Component
from .constants import EmbeddingMode

REFS = "refs"

class Table(Component):
    def __init__(
        self, 
        document_path: str,
        component_id: str, 
        document_title: str, 
        hierarchy_dict: JSONDict, 
        raw_component: JSONDict,
        images_dir: str = "",
        image_summaries_dir: str = ""
    ):
        super().__init__(
            document_path = document_path,
            component_id = component_id,
            document_title = document_title,
            hierarchy_dict = hierarchy_dict,
            raw_component = raw_component
        )
        
        self._images_dir = images_dir
        self._image_summaries_dir = image_summaries_dir

    def serialize(self, mode: tp.List[str]) -> JSONDict:
        in_table_image_filepaths: tp.List[str] = []
        serialized_text = (
            f"{self._document_title} [SEP] "
            f"{self.get_serialized_hierarchy_path()} [SEP] "
        )
        
        if self._raw_component["table"]:
            hdr_txt, hdr_imgs = self._serialize_row(
                self._raw_component["table"][0],
                self._raw_component,
                in_table_image_filepaths
            )
            serialized_text += hdr_txt + " [SEP] "
        
        for row in self._raw_component["table"][1:]:
            row_txt, _row_imgs = self._serialize_row(
                row,
                self._raw_component,
                in_table_image_filepaths
            )
            serialized_text += row_txt + " [SEP] "

        image_reprs: tp.Dict[str, tp.List[str]] = {}
        if EmbeddingMode.SUMMARY.value in mode:
            image_reprs[EmbeddingMode.SUMMARY.value] = [
                self._get_image_summary(fn)
                for fn in in_table_image_filepaths
            ]

        image_summaries: tp.List[str] = []
        for idx, img_path in enumerate(in_table_image_filepaths):
            parts = [f"<Image {idx+1}>"]
            if EmbeddingMode.IMAGE.value in mode:
                parts.append(os.path.basename(img_path))
            if EmbeddingMode.SUMMARY.value in mode:
                summ = image_reprs[EmbeddingMode.SUMMARY.value][idx]
                if summ:
                    parts.append(summ)
            image_summaries.append(" ".join(parts) + " [SEP] ")

        for img_line in image_summaries:
            serialized_text += img_line

        serialized_text = serialized_text.replace("\n", " ")

        if EmbeddingMode.IMAGE.value not in mode:
            in_table_image_filepaths = []

        # 8) return single object
        something: JSONDict = {
            "id": [self._document_path, self._component_id],
            "target": {
                "text": serialized_text,
                "images": in_table_image_filepaths
            }
        }
        
        return something 

    def _serialize_row(self, row, table_component, in_table_image_filepaths):
        new_images = []
        serialized_row = ""
        for cell in row:
            if "ref" in cell:
                serialized_cell, new_image = self._serialize_cell(table_component[REFS][str(cell["ref"])], in_table_image_filepaths)
            else:
                serialized_cell, new_image = self._serialize_cell(cell, in_table_image_filepaths)
            serialized_cell = serialized_cell.strip()
            serialized_row += serialized_cell + ", "
            if new_image: new_images.append(new_image - 1)
        return serialized_row[:-2], new_images

    
    def _serialize_cell(self, cell, in_table_image_filepaths) -> tp.Tuple[str, str]:
        serialized_cell: str = ""
        new_image = None

        if "text" in cell:
            serialized_cell += cell["text"]

        if "image" in cell and "filename" in cell["image"] and cell["image"]["filename"] != None:
            # image_filename = cell["image"]["filename"]
            # image_filepath = self._get_image_abs_path(image_filename)    
            raw = cell["image"]["filename"]
            image_filename = self._normalize_image_filename(raw)
            image_filepath = self._get_image_abs_path(image_filename) if image_filename else None
            
            if image_filepath is None:
                pass
            elif image_filepath not in in_table_image_filepaths:
                in_table_image_filepaths.append(image_filepath)
                image_idx = in_table_image_filepaths.index(image_filepath) + 1
                new_image = image_idx
            else:
                image_idx = in_table_image_filepaths.index(image_filepath) + 1
                
            if image_filepath is not None:
                serialized_cell += f" <Image {image_idx}> "

        serialized_cell = re.sub(r'\s*,\s*', ' , ', serialized_cell)

        return serialized_cell, new_image
    
    def _serialize_cell_for_prompt(self, cell, first_img_taken, first_img_path, cur_image_idx) -> tuple[str, bool, str, int]:
        ser = ""
            
        if (
            not first_img_taken
            and "image" in cell
            and "filename" in cell["image"]
            and cell["image"]["filename"] is not None
        ):
            raw = cell["image"]["filename"]
            img_filename = self._normalize_image_filename(raw)
            img_path = self._get_image_abs_path(img_filename) if img_filename else None
            if img_path:
                ser += f"<Image {cur_image_idx}>"
                first_img_taken = True
                first_img_path = img_path
                cur_image_idx += 1

        # 2) text (if any)
        if "text" in cell and cell["text"]:
            if ser: # need comma separator
                ser += ", "
            ser += cell["text"]

        # normalise spaces around commas
        ser = re.sub(r'\s*,\s*', ' , ', ser)
        return ser, first_img_taken, first_img_path, cur_image_idx
    
    def serialize_into_prompt(self, next_image_idx: int):
        serialized_table: tp.List[str] = []
        image_paths: tp.List[str] = [] # will stay empty or 1-length
        first_img_taken: bool = False
        first_img_path: tp.Optional[str] = None
        cur_idx: int = next_image_idx

        serialized_table.append("/*")
        serialized_table.append("[Table]")
        serialized_table.append(f"Title: {self._document_title}")
        serialized_table.append(f"Section: {self.get_serialized_hierarchy_path()}\n")

        # walk rows
        for row in self._raw_component["table"]:
            row_ser_parts = []
            for cell in row:
                if "ref" in cell:
                    cell = self._raw_component["refs"][str(cell["ref"])]
                cell_ser, first_img_taken, first_img_path, cur_idx = self._serialize_cell_for_prompt(
                    cell, first_img_taken, first_img_path, cur_idx
                )
                row_ser_parts.append(cell_ser)
            serialized_table.append(" | ".join(row_ser_parts))
        serialized_table.append("*/\n")

        # record the single image (if any)
        if first_img_path:
            image_paths.append(first_img_path)

        # build final string
        final_str = "\n".join(serialized_table) + "\n"

        return final_str, image_paths, cur_idx

    def get_hyperlinked_filenames(self) -> tp.List[str]:
        refs = self._raw_component.get(REFS, {})
        table = self._raw_component.get("table", [])
        
        edges: tp.List[str] = []
        for row in table:
            for cell in row:
                if "ref" in cell: cell = refs[str(cell["ref"])]
                if "edges" not in cell: continue
                for edge_obj in cell["edges"]:
                    edges.append(edge_obj["edge"])
        
        return list(set(edges))
    
    def _normalize_image_filename(self, x):
        if isinstance(x, (list, tuple)):
            for v in x:
                if isinstance(v, str):
                    return v
            return None
        return x if isinstance(x, str) else None

if __name__ == "__main__":
    
    TARGET_PATH = "/mnt/sdc/jhyun/omdr_mountspace/datasets/_01_multimodalqa/multimodal_documents/dev/2017_Major_League_Baseball_season.json"
    
    with open(TARGET_PATH, "r") as f:
        json_data = json.load(f)
        
    target_table_component = json_data["table"]["t_1"]
    
    table = Table(
        document_path            = TARGET_PATH,     
        document_title      = json_data["title"],
        hierarchy_dict      = json_data["hierarchy"],
        component_id        = "t_1",
        raw_component    = target_table_component,
        images_dir          = "/mnt/sdc/jhyun/omdr_mountspace/datasets/_01_multimodalqa/image_components/dev",
        image_summaries_dir = "/mnt/sdc/jhyun/omdr_mountspace/datasets/_01_multimodalqa/image_summaries/dev"
    )
    
    outputs = table.serialize(mode = ["image", "summary"])
    
    print(json.dumps(outputs, indent = 4))
    
    with open("check.json", "w") as f:
        json.dump(outputs, f, indent = 4)

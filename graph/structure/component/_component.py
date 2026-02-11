import os
import typing as tp
from abc import ABC, abstractmethod

from type import JSONDict, Type_NID

class Component(ABC):
    def __init__(
        self,
        document_path: str,
        component_id: str,
        document_title: str,
        hierarchy_dict: JSONDict,
        raw_component: JSONDict
    ):
        self._document_path: str = document_path
        self._document_title: str = document_title
        self._component_id: str = component_id
        self._hierarchy_path: tp.List[str] = []
        self._raw_component: JSONDict = raw_component
        self._set_hierarchy_path(hierarchy_dict)
    
    def get_docpath(self):
        return self._document_path
    
    def get_doctitle(self):
        return self._document_title
    
    def get_id(self):
        return self._component_id
    
    def get_nid(self) -> Type_NID:
        base_filename = os.path.basename(self._document_path)
        return (base_filename, self._component_id)
    
    def get_highest_component_id(self):
        return self._component_id.split("_")[0] + "_" + self._component_id.split("_")[1]
    
    def get_hierarchy_path(self) -> tp.List[str]:
        return self._hierarchy_path
    
    def get_serialized_hierarchy_path(self) -> str:
        return ', '.join(self._hierarchy_path)
    
    def get_component_object(self):
        return self._raw_component
    
    @abstractmethod
    def serialize(self, mode: tp.Optional[str] = None) -> JSONDict:
        # mode: ["image", "summary", "ocr"]
        raise NotImplementedError("The method serialize_component() is not implemented in the Component class.")
    
    @abstractmethod
    def serialize_into_prompt(self, next_image_idx: int) -> tp.Tuple[str, tp.List[str], int]:
        raise NotImplementedError("The method serialize_into_prompt() is not implemented in the Component class.")
    
    @abstractmethod
    def get_hyperlinked_filenames(self) -> tp.List[str]:
        raise NotImplementedError("The method get_edges() is not implemented in the Component class.")

    def _get_image_abs_path(self, image_filename: str) -> str:
        image_filepath = os.path.join(self._images_dir, image_filename)
        assert os.path.exists(image_filepath)
        return image_filepath
    
    def _get_image_summary(self, image_filename: str) -> str:
        # Get basename, remove extension, and add .txt
        image_filename = os.path.basename(image_filename)
        image_filename = os.path.splitext(image_filename)[0] + ".txt"
        image_summary_path = os.path.join(self._image_summaries_dir, image_filename)
        if not os.path.exists(image_summary_path):
            return None
        with open(image_summary_path, 'r') as file:
            image_summary = file.read()
        return image_summary

def get_highest_component_id(component_id):
    return component_id.split("_")[0] + "_" + component_id.split("_")[1]

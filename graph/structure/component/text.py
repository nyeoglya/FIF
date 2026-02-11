import typing as tp

from type import JSONDict
from ._component import Component

class Text(Component):
    def __init__(
        self,
        document_path: str,
        component_id: str,
        document_title: str, 
        hierarchy_dict: JSONDict,
        raw_component: JSONDict,
    ):
        super().__init__(
            document_path = document_path, 
            component_id = component_id, 
            document_title = document_title, 
            hierarchy_dict = hierarchy_dict, 
            raw_component = raw_component
        )
        self.text = self._raw_component["text"]

    def serialize(self, mode: tp.Optional[str] = None) -> JSONDict:
        serialized_metadata = f"{self._document_title} [SEP] {self.get_serialized_hierarchy_path()} [SEP] "

        serialization = serialized_metadata + self.text
        serialization = serialization.replace("\n", " ")
        serialization = serialization.replace("\t", " ")
        
        embedding_obj: JSONDict = {
            "id": [self._document_path, self._component_id],
            "target": {
                "text": serialization,
                "images": []
            }
        }

        return embedding_obj

    def serialize_into_prompt(self, next_image_idx: int) -> tp.Tuple[str, tp.List[str], int]:
        prompt = "/*\n[Passage]\n"
        prompt += f"Title: {self._document_title}\n"
        prompt += f"Section: {', '.join(self._hierarchy_path)}\n\n"
        prompt += self.text
        prompt += "\n*/\n\n"
        return prompt, [], next_image_idx
    
    def get_text_object_for_split(self) -> JSONDict:
        return {
            "title": self._document_title,
            "section": self._hierarchy_path,
            "text": self.text
        }
    
    def get_hyperlinked_filenames(self) -> tp.List[str]:
        if "edges" not in self._raw_component: return []
        
        edges: tp.List[str] = []
        for edge_obj in self._raw_component["edges"]:
            edges.append(edge_obj["edge"])
        
        return edges

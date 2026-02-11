import os
import typing as tp

from .component.constants import Modality, ParsedWebKeywords
from .component._component import Component, get_highest_component_id
from .component.component_factory import ComponentFactory
from .component.text import Text
from .component.table import Table
from .component.image import Image

from .node import Node
from type import Type_NID, JSONDict
from common import read_json_or_jsonl

# Register component classes with the factory
ComponentFactory.register(Modality.TEXT.value, Text)
ComponentFactory.register(Modality.TABLE.value, Table)
ComponentFactory.register(Modality.IMAGE.value, Image)
ComponentFactory.register(Modality.SENTENCE.value, Text)
ComponentFactory.register(Modality.TABLE_SEGMENT.value, Table)
ComponentFactory.register(Modality.SUBIMAGE.value, Image)

class DocumentTree:
    def __init__(self, file_path: str, images_dir: str = "", subimages_dir: str = "", image_summaries_dir: str = ""):
        self._file_name: str = os.path.basename(file_path)
        self._title: str = ""
        self._root_node: Node = None
        
        self._component_id_to_node: tp.Dict[str, Node] = {}
        self._modality_to_node: tp.Dict[str, tp.List[Node]] = {}
        
        self._generate_from_file(
            file_path,
            images_dir,
            subimages_dir,
            image_summaries_dir
        )
        self._generate_tree_structure()
        
    def get_title(self) -> str: 
        return self._title
    
    def get_root_node(self) -> Node:
        return self._root_node
    
    @property
    def root_node(self) -> Node:
        return self.get_root_node()
    
    def get_component_list(self) -> tp.List[Node]:
        return list(self._component_id_to_node.values())
    
    def get_global_component_ids_list(self) -> tp.List[Type_NID]:
        return [(self._file_name, component_id) for component_id in self._component_id_to_node.keys()]
    
    def get_component_nids(self) -> tp.List[Type_NID]:
        # If the number of "_" in component_id is 1
        return [
            (self._file_name, component_id) for component_id in self._component_id_to_node.keys()
            if component_id.count("_") == 1
        ]
    
    def get_component_by_cid(self, component_id: str) -> Node:
        return self._component_id_to_node.get(component_id)

    def get_components_by_modality(self, modality: Modality) -> tp.List[Node]:
        return self._modality_to_node.get(modality.value, [])
    
    def get_cid_to_component(self) -> tp.Dict[str, Node]:
        return self._component_id_to_node
    
    def get_component_statistics(self) -> tp.Dict[str, int]:
        statistics: tp.Dict[str, int] = {}
        for modality, components in self._modality_to_node.items():
            statistics[modality] = len(components)
        return statistics

    def set_node_resolver(self, resolver) -> None:
        """Propagate a resolver to all nodes (alias for backward compatibility)."""
        return self.set_node_resolver_for_all_nodes(resolver)

    def set_node_resolver_for_all_nodes(self, resolver) -> None:
        """Propagate a resolver to all nodes so they can resolve NIDs to Node objects."""
        if self._root_node is not None:
            self._root_node.set_resolver(resolver)
        for node in self._component_id_to_node.values():
            node.set_resolver(resolver)
        return

    def _generate_from_file(
        self,
        doc_filepath: str,
        images_dir: str,
        subimages_dir: str,
        image_summaries_dir
    ) -> None:
        modalities = [
            Modality.TEXT, Modality.TABLE, Modality.IMAGE,
            Modality.SENTENCE, Modality.TABLE_SEGMENT, Modality.SUBIMAGE
        ]

        raw_json_data = read_json_or_jsonl(doc_filepath)
        self._title = raw_json_data[ParsedWebKeywords.TITLE.value]
        
        factory = ComponentFactory()
        
        root_node: Node = Node(node_id = (self._file_name, "root"))
        self._root_node = root_node
        self._component_id_to_node["root"] = root_node

        for modality in modalities:
            self._modality_to_node[modality.value] = []
            component_id_to_component_obj: tp.Dict[str, JSONDict] = raw_json_data[modality.value]
            
            if modality == Modality.SUBIMAGE: image_dir = subimages_dir
            else: image_dir = images_dir

            for component_id, raw_component in component_id_to_component_obj.items():
                
                nid = (self._file_name, component_id)
                
                node: Node = Node(node_id = nid)

                component: Component = factory.create(
                    modality = modality.value,
                    doc_filepath = doc_filepath,
                    component_id = component_id,
                    document_title = self._title,
                    hierarchy_dict      = raw_json_data[ParsedWebKeywords.HIERARCHY.value],
                    raw_component       = raw_component,
                    images_dir          = image_dir,
                    image_summaries_dir = image_summaries_dir
                )
                
                node.set_component(component)

                self._modality_to_node[modality.value].append(node)
                self._component_id_to_node[component_id] = node
    
    def _generate_tree_structure(self):
        layer1_modality_list: tp.List[Modality] = [ Modality.TEXT, Modality.TABLE, Modality.IMAGE ]
        layer2_modality_list: tp.List[Modality] = [ Modality.SENTENCE, Modality.TABLE_SEGMENT, Modality.SUBIMAGE ]
        
        for layer1_modality in layer1_modality_list:
            nodes = self._modality_to_node[layer1_modality.value]
            for node in nodes:
                parent_node = self._root_node
                node.set_parent_node(parent_node)
                parent_node.add_child_node(node)
        
        for layer2_modality in layer2_modality_list:
            nodes = self._modality_to_node[layer2_modality.value]
            for node in nodes:
                parent_cid = get_highest_component_id(node.cid)
                parent_node = self._component_id_to_node[parent_cid]
                node.set_parent_node(parent_node)
                parent_node.add_child_node(node)

from __future__ import annotations
from typing import Dict, List, Optional, Callable, Union

from type import Type_NID
from .component._component import Component

class Node:
    def __init__(
        self,
        node_id: Type_NID
    ) -> None:
        self._node_id: Type_NID = node_id

        # Relationship state stored as NIDs for lightweight serialization
        self._parent_nid: Optional[Type_NID] = None
        self._children_nids: Dict[Type_NID, Type_NID] = {}
        self._intra_neighbor_nids: Dict[Type_NID, Type_NID] = {}
        self._neighbor_nids: Dict[Type_NID, Type_NID] = {}
        self._incoming_neighbor_nids: Dict[Type_NID, Type_NID] = {}
        self._outgoing_neighbor_nids: Dict[Type_NID, Type_NID] = {}

        # Resolver to map NID -> Node. Must be set by the owning graph.
        self._resolver: Optional[Callable[[Type_NID], "Node"]] = None

        self._component: Component = None

    def get_nid(self) -> Type_NID:
        return self._node_id

    @property
    def _nid(self) -> Type_NID:
        return self.get_nid()
    
    def get_children_nodes(self) -> List[Node]:
        return [self._resolve(nid) for nid in self._children_nids.values()]

    def get_children_nids(self) -> List[Type_NID]:
        return list(self._children_nids.keys())
    
    def get_cid(self) -> str:
        return self._nid[1]
    
    @property
    def cid(self) -> str:
        return self.get_cid()
    
    def set_parent_node(self, parent_node: Union[Node, Type_NID]) -> None:
        nid = self._to_nid(parent_node)
        self._parent_nid = nid
        return
    
    def get_parent_node(self) -> Node:
        return self._resolve(self._parent_nid) if self._parent_nid else None

    def get_parent_nid(self) -> Optional[Type_NID]:
        return self._parent_nid
    
    @property
    def parent_node(self) -> Node:
        return self.get_parent_node()
    
    @property
    def children_nodes(self) -> List[Node]:
        return self.get_children_nodes()
    
    def set_component(self, component: Component) -> None:
        self._component = component
        return
    
    def get_component(self) -> Component:
        return self._component
    
    @property
    def component(self) -> Component:
        return self.get_component()
    
    def get_neighbors(self) -> List[Node]:
        return [self._resolve(nid) for nid in self._neighbor_nids.values()]

    def get_neighbor_nids(self) -> List[Type_NID]:
        return list(self._neighbor_nids.keys())
    
    def get_neighbor_by_nid(self, nid: Type_NID) -> Node:
        if nid not in self._neighbor_nids:
            return None
        return self._resolve(self._neighbor_nids[nid])
    
    
    
    def add_intra_neighbor(self, neighbor_node: Union[Node, Type_NID]) -> None:
        nid = self._to_nid(neighbor_node)
        self._intra_neighbor_nids[nid] = nid
        return

    def add_child_node(
        self, 
        child_node: Union[Node, Type_NID]
    ) -> None:
        nid = self._to_nid(child_node)
        self._children_nids[nid] = nid
        if isinstance(child_node, Node):
            child_node._parent_nid = self._nid
        return
    
    def add_incoming_neighbor(
        self, 
        neighbor_node: Union[Node, Type_NID]
    ) -> None:
        nid = self._to_nid(neighbor_node)
        self._incoming_neighbor_nids[nid] = nid
        return
    
    def add_outgoing_neighbor(
        self, 
        neighbor_node: Union[Node, Type_NID]
    ) -> None:
        nid = self._to_nid(neighbor_node)
        self._outgoing_neighbor_nids[nid] = nid
        return
    
    def collapse_neighbors(self) -> None:
        for nid in self._incoming_neighbor_nids.values():
            self._neighbor_nids[nid] = nid
        for nid in self._outgoing_neighbor_nids.values():
            self._neighbor_nids[nid] = nid
        return

    def set_resolver(self, resolver: Callable[[Type_NID], "Node"]) -> None:
        """Attach a resolver that maps NID -> Node, typically provided by Graph."""
        self._resolver = resolver

    def _resolve(self, nid: Type_NID) -> "Node":
        if nid is None:
            return None
        if self._resolver is None:
            raise RuntimeError(f"Node resolver not set when resolving {nid}")
        return self._resolver(nid)

    @staticmethod
    def _to_nid(node_or_nid: Union["Node", Type_NID]) -> Type_NID:
        return node_or_nid._nid if isinstance(node_or_nid, Node) else node_or_nid

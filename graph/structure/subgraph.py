from typing import List, Tuple, Set, Union

from type import Type_NID
from .document import DocumentTree
from .graph import LayeredComponentGraph


class LayeredComponentSubgraph:
    """
    Subgraph extracted from the main Graph based on a set of top-level components.

    Creates retrieval units (edges and isolated nodes) from a subset of components,
    supporting both 0-hop and 1-hop extraction modes.
    """

    def __init__(
        self,
        graph: LayeredComponentGraph,
        component_nid_list: List[Type_NID] = []
    ) -> None:
        
        """
        Initialize a subgraph from a set of top-level components.

        Args:
            graph: The main Graph object
            global_top_level_component_id_list: List of (filename, component_id) tuples
                representing the top-level components to include in the subgraph
        """
        # List of retrieval units: can be 2-tuples (edges) or 1-tuples (isolated nodes)
        # Each element is Tuple[Type_NID, ...] where inner tuples are (filename, component_id)
        
        self.retrieval_units_list: List[Union[
            Tuple[Type_NID, Type_NID], # 2-tuple for edges
            Tuple[Type_NID] # 1-tuple for isolated nodes
        ]] = []

        self._graph: LayeredComponentGraph = graph

        self.total_nid_list: List[Type_NID] = []
        self.input_nid_list: List[Type_NID] = component_nid_list

        return
    
    
    
    def get_top_level_nids_list(self) -> List[Type_NID]:
        distinct_nid_set: Set[Type_NID] = set()
        for retrieval_unit in self.retrieval_units_list:
            for nid in retrieval_unit:
                distinct_nid_set.add(nid)
        return list(distinct_nid_set)
    
    

    def get_retrieval_units_list(self) -> List[Union[
        Tuple[Type_NID, Type_NID],
        Tuple[Type_NID]
    ]]:
        return self.retrieval_units_list
    
    
    
    def extract_edges(self, mode: str) -> None:

        if mode == "0_hop":
            self._extract_edges_v0()
        elif mode == "1_hop":
            self._extract_edges_v1()
        else:
            raise ValueError(f"Invalid mode: {mode}. Use '0_hop' or '1_hop'.")    
    
    
    def _extract_edges_v1(self) -> None:
        
        """
        Extract edges in 1-hop mode.

        For each component NID in `self.input_nid_list`,
        gathers all edges from the global Graph:

        (A) Intra-document edges:
            (nid, comp_nid) for every other top-level in the same document

        (B) Inter-document edges:
            If `nid` has an inter-document link to document 'B',
            then for every top-level component 'b_i' in document 'B',
            adds an edge (nid, b_i)

        After collecting edges, adds any isolated nodes as 1-tuples.
        """
        
        edges: Set[Tuple[Type_NID, Type_NID]] = set()

        # 1) Loop over each component NID we care about
        for comp_nid in self.input_nid_list:
            docname, cid = comp_nid

            # Can: Can be replaced with edge
            # A) Intra-document edges
            #    → link to every other *top-level* component in the same doc
            document: DocumentTree = self._graph.get_document_by_docname(docname)
            component_nids: List[Type_NID] = document.get_component_nids()
            # Possibly it's named `get_global_top_level_component_ids_list()`,
            # or `get_all_top_level_ids()`. Adapt to your real method name!

            for inter_comp_nid in component_nids:
                other_comp_cid = inter_comp_nid[1]
                if other_comp_cid == cid:
                    continue
                # Build an unordered edge
                edge_tuple = tuple(sorted([
                    (docname, cid),
                    (docname, other_comp_cid),
                ]))
                edges.add(edge_tuple)

            # B) Inter-document edges
            #    → from this component NID to each doc that it references.
            #       Then link to EVERY component NID in that doc.
            try:
                # Get the hyperlinked filenames directly from the node's component
                node                = self._graph.get_node_by_nid(docname, cid)
                linked_docnames     = node.get_component().get_hyperlinked_filenames()
            except (ValueError, AttributeError):
                # Node not found or component doesn't have edges
                linked_docnames = []

            for linked_docname in linked_docnames:
                # Get that doc, gather its top-level IDs
                linked_doc: DocumentTree = self._graph.get_document_by_docname(linked_docname)
                inter_comp_nids = linked_doc.get_component_nids()

                # For each top-level in that doc, form an edge
                for inter_comp_nid in inter_comp_nids:
                    inter_comp_cid = inter_comp_nid[1]
                    edge_tuple = tuple(sorted([
                        (docname, cid),
                        (linked_docname, inter_comp_cid),
                    ]))
                    edges.add(edge_tuple)

        # 2) Convert edges (set of 2-tuples) into a list
        edge_list = list(edges)
        # print(edge_list)

        # 3) Add isolated nodes
        #    Check if any NID from our input never appears in the edges
        connected_nodes = set()
        for e in edge_list:
            # e is a 2-tuple like ((docname_a, cid_a), (docname_b, cid_b))
            connected_nodes.add(e[0])
            connected_nodes.add(e[1])

        # Convert from list[list[str]] → set[Type_NID]
        all_input_nodes = {tuple(g) for g in self.input_nid_list}
        isolated = all_input_nodes - connected_nodes

        # 4) Build final retrieval_units_list: 
        #    We combine the 2-node edges + 1-node tuples for isolates
        self.retrieval_units_list = edge_list
        self.retrieval_units_list.extend([(node,) for node in isolated])

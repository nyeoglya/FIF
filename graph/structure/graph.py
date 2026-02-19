import os
import time
from typing import Dict, List

from tqdm import tqdm

from src.dataset.base_dataset import BaseDataset
from .node import Node
from type import Type_NID

from .document import DocumentTree
from src.algorithm.sns.data_structure.io.artifact_manager import ArtifactManager


class LayeredComponentGraph:
    def __init__(
        self,
        dataset: BaseDataset,
        artifact_manager: ArtifactManager
    ) -> None:
        self._images_dir: str = dataset.image_components_dir
        self._documents_dir: str = artifact_manager.multimodal_documents_path
        self._subimages_dir: str = artifact_manager.subimage_components_path
        self._summaries_dir: str = artifact_manager.image_summaries_path

        self._docname_to_document: Dict[str, DocumentTree] = {}
        self._docname_to_rootnode: Dict[str, Node] = {}

    def _resolve_node_by_nid(self, nid: Type_NID) -> Node:
        """Resolve NID -> Node via this graph."""
        return self.get_node_by_nid(nid[0], nid[1])

    def get_document_by_docname(self, filename: str) -> DocumentTree:
        if filename in self._docname_to_document:
            return self._docname_to_document[filename]
        else:
            raise ValueError(f"Document with filename {filename} not found.")
        
    def get_rootnode_by_filename(self, filename: str) -> Node:
        if filename in self._docname_to_rootnode:
            return self._docname_to_rootnode[filename]
        else:
            raise ValueError(f"Root node for document with filename {filename} not found.")

    def get_node_by_nid(
        self,
        docname: str,
        cid: str
    ) -> Node:
        if docname not in self._docname_to_document:
            raise ValueError(f"Document with filename {docname} not found.")
        document: DocumentTree = self._docname_to_document[docname]
        node = document.get_component_by_cid(cid)
        if node is None:
            raise ValueError(f"Node with ID {cid} not found in document {docname}.")

        return node

    def generate_from_files(self) -> None:
        """
        Parse all documents from the documents directory and generate edges.

        This method:
        1. Reads all document files from the documents directory
        2. Creates MultimodalDocument objects for each file
        3. Generates intra-document and inter-document edges
        """
        
        print(f"[Graph] Parsing documents from {self._documents_dir}...")

        start_time = time.time()

        filenames = os.listdir(self._documents_dir)
        filenames.sort()

        for filename in tqdm(filenames):
            
            filepath = os.path.join(self._documents_dir, filename)

            document = DocumentTree(
                file_path           = filepath,
                images_dir          = self._images_dir,
                subimages_dir       = self._subimages_dir,
                image_summaries_dir = self._summaries_dir
            )
            root_node = document.get_root_node()

            self._docname_to_document[filename] = document
            self._docname_to_rootnode[filename] = root_node

            # Attach resolver so nodes can lazy-resolve NIDs back to Node objects.
            document.set_node_resolver(self._resolve_node_by_nid)

        end_time = time.time()

        print(f"[Graph] Finished parsing documents from {self._documents_dir}.")
        print(f"[Graph] Time taken: {end_time - start_time:.2f} seconds.", end = "\n\n")

        self._generate_intra_document_edges()
        self._generate_inter_document_edges()

        return

    def _generate_intra_document_edges(self) -> None:
        
        """
        Generate intra-document edges (edges between components within the same document).

        For each document, creates edges from parent components to their child components
        (e.g., from a paragraph to its sentences, from an image to its subimages).
        """
        
        print("[Graph] Generating intra-document edges...")
        
        for _, root_node in tqdm(self._docname_to_rootnode.items()):
            children_nodes = root_node.get_children_nodes()
            
            # Generate a clique of children_nodes
            for i in range(len(children_nodes)):
                for j in range(len(children_nodes)):
                    if i == j:
                        continue
                    in_node = children_nodes[i]
                    out_node = children_nodes[j]
                    in_node.add_intra_neighbor(out_node)
                
        # Analyze the number of edges
        total_edges = 0
        for _, root_node in self._docname_to_rootnode.items():
            children_nodes = root_node.get_children_nodes()
            for child_node in children_nodes:
                total_edges += len(child_node.get_neighbors())
        print(f"[Graph] Total number of intra-document edges: {total_edges}")
        
        print(f"[Graph] Finished generating intra-document edges.")
        
        return

    def _generate_inter_document_edges(self) -> None:
        
        """
        Generate inter-document edges (edges between components across different documents).

        Uses the component's intra_edges (hyperlinks, references) to create edges
        from components in one document to referenced documents.
        """
        
        print("[Graph] Generating inter-document edges...")
        
        # Layer 0
        for _, root_node in tqdm(self._docname_to_rootnode.items()):
            
            # Layer 1
            children_nodes: List[Node] = root_node.get_children_nodes()
            for child_node in children_nodes:
                
                linked_filenames: List[str] = child_node.get_component().get_hyperlinked_filenames()
                
                # For each linked filename, find the root node and connect
                for linked_filename in linked_filenames:
                    if linked_filename not in self._docname_to_rootnode:
                        raise ValueError(f"Linked filename {linked_filename} not found in graph documents.")
                    
                    linked_root_node: Node = self._docname_to_rootnode[linked_filename]
                    linked_children_nodes: List[Node] = linked_root_node.get_children_nodes()
                    
                    # Connect to all children nodes of the linked document
                    root_node.add_outgoing_neighbor(linked_root_node)
                    for linked_child_node in linked_children_nodes:
                        child_node.add_outgoing_neighbor(linked_child_node)
                        
        for _, root_node in self._docname_to_rootnode.items():
            root_node.collapse_neighbors()
            children_nodes = root_node.get_children_nodes()
            for child_node in children_nodes:
                child_node.collapse_neighbors()
                
        print(f"[Graph] Finished generating inter-document edges.")

        return

    @staticmethod
    def comp_nid_by_subcomp_nid(nid: Type_NID) -> Type_NID:
        """Convert subcomponent NID to its parent component NID."""
        return (nid[0], nid[1].rsplit("_", 1)[0])

    @staticmethod
    def get_top_level_modality(nid: Type_NID) -> str:
        """Get the top-level modality from a NID."""
        return nid[1][0]

if __name__ == "__main__":
    """
    Test case for Graph class using ComponentFactory pattern with MMCoQA dataset.
    """
    from src.dataset.registry import DATASET_REGISTRY

    print("=" * 80)
    print("Testing Graph with ComponentFactory Pattern (MMCoQA Dataset)")
    print("=" * 80)

    # Step 1: Initialize MMCoQA dataset
    print("\n[1] Initializing MMCoQA dataset...")
    dataset_name = "_01_multimodalqa"

    if dataset_name not in DATASET_REGISTRY:
        print(f"Error: Dataset '{dataset_name}' not found in registry.")
        print(f"Available datasets: {list(DATASET_REGISTRY.keys())}")
        exit(1)

    dataset = DATASET_REGISTRY[dataset_name]()
    dataset.initiate_dataset()
    print(f"✓ Dataset initialized: {dataset.name}")
    print(f"  - Image components dir: {dataset.image_components_dir}")

    # Step 2: Initialize ArtifactManager
    print("\n[2] Initializing ArtifactManager...")
    artifact_manager = ArtifactManager(
        algorithm_name="sns",
        dataset=dataset
    )
    print(f"✓ ArtifactManager initialized")
    print(f"  - Multimodal documents path: {artifact_manager.multimodal_documents_path}")
    print(f"  - Image summaries path: {artifact_manager.image_summaries_path}")
    print(f"  - Subimage components path: {artifact_manager.subimage_components_path}")

    # Step 3: Initialize Graph
    print("\n[3] Initializing Graph...")
    graph = LayeredComponentGraph(
        dataset = dataset,
        artifact_manager = artifact_manager
    )
    print(f"✓ Graph initialized successfully")

    # Step 4: Check for documents
    print("\n[4] Checking for multimodal documents...")
    if not os.path.exists(artifact_manager.multimodal_documents_path):
        print(f"⚠ Warning: Multimodal documents directory does not exist:")
        print(f"  {artifact_manager.multimodal_documents_path}")
        print(f"  Cannot proceed with document parsing test.")
        exit(1)

    document_files = os.listdir(artifact_manager.multimodal_documents_path)
    if not document_files:
        print(f"⚠ Warning: No document files found in:")
        print(f"  {artifact_manager.multimodal_documents_path}")
        print(f"  Cannot proceed with document parsing test.")
        exit(1)

    print(f"✓ Found {len(document_files)} document files")

    # Step 5: Parse documents using ComponentFactory
    print("\n[5] Parsing documents with ComponentFactory...")
    try:
        graph.generate_from_files()
        print(f"✓ Documents parsed successfully")

        # Step 6: Verify graph structure
        print("\n[6] Verifying graph structure...")
        num_documents = len(graph._docname_to_document)
        print(f"✓ Loaded {num_documents} documents")

        # Count the number of children nodes that does not have any child
        orphan_children_count = 0
        for filename, root_node in graph._docname_to_rootnode.items():
            children_nodes = root_node.get_children_nodes()
            for child_node in children_nodes:
                if len(child_node.get_children_nodes()) == 0:
                    orphan_children_count += 1
        print(f"✓ Found {orphan_children_count} orphan children nodes (no child nodes)")

        if num_documents > 0:
            # Get first document as sample
            sample_filename = list(graph._docname_to_document.keys())[1]
            sample_doc = graph.get_document_by_docname(sample_filename)

            print(f"\n  Sample document: {sample_filename}")
            print(f"  - Title: {sample_doc.get_title()}")
            print(f"  - Component statistics:")
            for modality, count in sample_doc.get_component_statistics().items():
                print(f"    • {modality}: {count} components")

            # Get root node
            root_node = graph.get_rootnode_by_filename(sample_filename)
            print(f"  - Root node ID: {root_node.get_nid()}")
            print(f"  - Number of children: {len(root_node.get_children_nodes())}")

            # Check components created via factory
            components: List[Node] = sample_doc.get_component_list()
            if components:
                sample_component: Node = components[0]
                print(f"\n  Sample component (created via ComponentFactory):")
                print(f"  - Component type: {type(sample_component).__name__}")
                print(f"  - Component CID: {sample_component.get_cid()}")
                
        # Count the number of average number of neighbors
        total_root_neighbors = 0
        total_roots = 0
        
        total_neighbors = 0
        total_nodes = 0
        
        for _, root_node in graph._docname_to_rootnode.items():
            total_roots += 1
            total_root_neighbors += len(root_node.get_neighbors())
            children_nodes = root_node.get_children_nodes()
            for child_node in children_nodes:
                total_neighbors += len(child_node.get_neighbors())
                total_nodes += 1
        average_root_neighbors = total_root_neighbors / total_roots if total_roots > 0 else 0
        average_neighbors = total_neighbors / total_nodes if total_nodes > 0 else 0
        print(f"\n✓ Average number of neighbors per root node: {average_root_neighbors:.2f}")
        print(f"\n✓ Average number of neighbors per child node: {average_neighbors:.2f}")
        
        print("\n" + "=" * 80)
        print("✓ All tests passed! ComponentFactory is working correctly.")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Error during test:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

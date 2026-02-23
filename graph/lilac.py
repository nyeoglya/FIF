import typing as tp

import numpy as np
from numpy.typing import NDArray

from memory import FiFMemoryUnit, FiFTraversalContext
from graph import FiFGraph, FiFDocument, FiFComponent
from type import UniqueID

class LILaCTraverser:
    def __init__(self, fif_graph: FiFGraph) -> None:
        self.fif_graph: FiFGraph = fif_graph

    def find_entry(self, ctx: FiFTraversalContext) -> bool:
        ctx.fif_memory.reset_memory()

        component_scores: NDArray[np.float32] = self.fif_graph.component_embeddings @ ctx.query_embedding.T
        top_2048_indices = np.argsort(-component_scores, axis=0)[:2048]

        component_candidate_list: tp.List[tp.Tuple[float, int]] = []
        for component_loc in top_2048_indices:
            component_uid: UniqueID = self.fif_graph.component_loc_to_id_map[int(component_loc)]
            component: FiFComponent = self.fif_graph.get_component_from_unique_id(component_uid)
            subcomponent_embeddings: NDArray[np.float32] = component.get_subcomponent_embeddings()
            if subcomponent_embeddings.size == 0: continue
            subcomponent_scores = subcomponent_embeddings @ ctx.current_subquery_embeddings.T
            component_candidate_list.append((float(subcomponent_scores.max()), component_loc))

        component_candidate_list.sort(key=lambda x: x[0], reverse=True)
        beam_component_ids: tp.List[UniqueID] = [
            self.fif_graph.component_loc_to_id_map[item[1]]
            for item in component_candidate_list[:ctx.beam_size]
        ]

        memory_unit: FiFMemoryUnit = FiFMemoryUnit(component_id_list=beam_component_ids)
        ctx.fif_memory.add_new_history(memory_unit)
        return len(memory_unit.component_id_list) > 0

    def one_hop(self, ctx: FiFTraversalContext) -> bool:
        candidate_edge_dict: tp.Dict[tp.FrozenSet[UniqueID], float] = dict()
        component_unique_id_list: tp.List[UniqueID] = self.get_component_unique_id_list(ctx, ctx.beam_size)
        for component_unique_id in component_unique_id_list:
            component_data: FiFComponent = self.fif_graph.get_component_from_unique_id(component_unique_id)
            neighbor_id_list: tp.List[UniqueID] = []

            # intra-document edges
            same_doc: FiFDocument = self.fif_graph.get_document_from_unique_id(component_unique_id)
            for comp_key in same_doc.component_dict:
                neighbor_id_list.append((component_unique_id[0], comp_key, None))

            # inter-document edges
            for linked_doc_uid in component_data.linked_document_uid_list:
                if linked_doc_uid[0] not in self.fif_graph.doc_title_to_loc_map:
                    continue
                linked_doc: FiFDocument = self.fif_graph.get_document_from_unique_id(linked_doc_uid)
                for component_key in linked_doc.component_dict:
                    neighbor_id_list.append((linked_doc_uid[0], component_key, None))

            if not neighbor_id_list:
                candidate_edge_dict[frozenset([component_unique_id])] = self._calculate_maxsim_score(ctx, component_unique_id)
                continue

            for neighbor_id in neighbor_id_list:
                candidate_edge = frozenset([component_unique_id, neighbor_id])
                if candidate_edge in candidate_edge_dict: continue

                candidate_edge_score = self._calculate_maxsim_score(ctx, component_unique_id, neighbor_id)
                component1_single_score = self._calculate_maxsim_score(ctx, component_unique_id)
                component2_single_score = self._calculate_maxsim_score(ctx, neighbor_id)

                if candidate_edge_score <= max(component1_single_score, component2_single_score) + 1e-5: # epsilon = 1e-5 for numerical stability
                    winner_id = component_unique_id if component1_single_score >= component2_single_score else neighbor_id
                    candidate_edge_dict[frozenset([winner_id])] = max(component1_single_score, component2_single_score)
                else:
                    candidate_edge_dict[candidate_edge] = candidate_edge_score

        sorted_candidate_edge_list = sorted(candidate_edge_dict.items(), key=lambda x: x[1], reverse=True)

        new_component_id_list: tp.List[UniqueID] = []
        seen: tp.Set[UniqueID] = set()
        for component_id_set, _ in sorted_candidate_edge_list:
            nodes_in_edge = list(component_id_set)
            if len(nodes_in_edge) > 1:
                nodes_in_edge.sort(
                    key=lambda cid: self._calculate_maxsim_score(ctx, cid),
                    reverse=True
                )
            for cid in nodes_in_edge:
                if cid not in seen:
                    seen.add(cid)
                    new_component_id_list.append(cid)
            if len(new_component_id_list) >= ctx.beam_size:
                break

        old_component_id_list = ctx.fif_memory.get_last_unit().component_id_list
        ctx.fif_memory.add_new_history(
            FiFMemoryUnit(component_id_list = new_component_id_list[:ctx.beam_size])
        )
        return old_component_id_list != new_component_id_list[:ctx.beam_size]

    def multi_hop(self, ctx: FiFTraversalContext, max_hop: int) -> None:
        ctx.hop_count = 0
        while ctx.hop_count < max_hop:
            changed = self.one_hop(ctx)
            if not changed: break
            ctx.hop_count += 1
    
    def _calculate_maxsim_score(
        self,
        ctx: FiFTraversalContext,
        component1_id: UniqueID,
        component2_id: tp.Optional[UniqueID] = None
    ) -> float:
        component1: FiFComponent = self.fif_graph.get_component_from_unique_id(component1_id)
        embedding1: NDArray[np.float32] = component1.get_subcomponent_embeddings()

        if component2_id is not None and component1_id != component2_id:
            component2: FiFComponent = self.fif_graph.get_component_from_unique_id(component2_id)
            embedding2: NDArray[np.float32] = component2.get_subcomponent_embeddings()
            combined_subembeddings = np.concatenate((embedding1, embedding2), axis=0)
        else:
            combined_subembeddings = embedding1

        if combined_subembeddings.size == 0: return -1.0

        sim_matrix = ctx.current_subquery_embeddings @ combined_subembeddings.T
        return float(np.sum(np.max(sim_matrix, axis=1)))

    def get_component_unique_id_list(self, ctx: FiFTraversalContext, top_k: int) -> tp.List[UniqueID]:
        assert ctx.fif_memory.history
        return ctx.fif_memory.history[-1].component_id_list[:top_k]

import asyncio

from download import FiFDatasetLoader
from memory import FiFMemory
from graph import FiFGraph
from lilac import LILaCTraverser
from preprocessor import (
    LLMQuerySerializer,
    DocumentSummarizer,
    BatchImageDescriptor,
    BatchObjectDetector,
    EmbeddingSerializer,
    DocumentEmbedder
)
from sanity import verify_embedding
from evaluation import embedding_evaluation, lilac_retrieval_evaluation
from config import (
    MMCOQA_RESTORE_FOLDERPATH, MMCOQA_RESTORE_DEV_FILEPATH, MMCOQA_RESTORE_IMAGE_FOLDERPATH,
    MMCOQA_DOCUMENT_SUMMARIZATION_FILEPATH, MMCOQA_DOCUMENT_SUMMARIZATION_FAILED_FILEPATH,
    MMCOQA_IMAGE_DESCRIPTION_FILEPATH, MMCOQA_IMAGE_DESCRIPTION_FAILED_FILEPATH,
    MMCOQA_OBJECT_DETECTION_FILEPATH, MMCOQA_OBJECT_DETECTION_FAILED_FILEPATH,
    MMCOQA_DOCUMENT_EMBEDDING_FOLDERPATH, MMCOQA_DOCUMENT_EMBEDDING_FAILED_FILEPATH,
    MMCOQA_DEV_EMBEDDING_CACHE_FILEPATH,
    QWEN_SERVER_URL_LIST, MMEMBED_SERVER_URL_LIST,
    TOP_K, BEAM_SIZE, MAX_HOP
)
from agent import FiFOrchestrator

def preprocess_main():
    print("Start Pre-processing...")
    
    '''Dataset Download + Load (From Huggingface)'''
    # mmcoqa_doc_dataset_loader = FiFDatasetLoader(
    #     dataset_name="JoohyungYun/mmcoqa_doc",
    #     save_path=MMCOQA_RESTORE_FOLDERPATH
    # )
    # mmcoqa_doc_dataset_loader.restore(overwrite=True)
    # mmcoqa_doc_dataset_loader.restore_dev_jsonl(MMCOQA_RESTORE_DEV_FILEPATH)

    '''Document Summarization'''
    # llm_query_serializer = LLMQuerySerializer(MMCOQA_RESTORE_IMAGE_FOLDERPATH)
    # document_summarizer: DocumentSummarizer = DocumentSummarizer(llm_query_serializer)
    # document_summarizer.load_data_from_json_folderpath(MMCOQA_RESTORE_FOLDERPATH)
    # asyncio.run(document_summarizer.run_parallel_summarize(
    #     QWEN_SERVER_URL_LIST,
    #     MMCOQA_DOCUMENT_SUMMARIZATION_FILEPATH,
    #     MMCOQA_DOCUMENT_SUMMARIZATION_FAILED_FILEPATH
    # ))

    '''Image Description'''
    # batch_image_descriptor = BatchImageDescriptor()
    # batch_image_descriptor.load_image_filelist(MMCOQA_RESTORE_IMAGE_FOLDERPATH)
    # asyncio.run(batch_image_descriptor.run_parallel_description(
    #     QWEN_SERVER_URL_LIST,
    #     MMCOQA_IMAGE_DESCRIPTION_FILEPATH,
    #     MMCOQA_IMAGE_DESCRIPTION_FAILED_FILEPATH
    # ))

    '''Object Detection'''
    # batch_object_detector = BatchObjectDetector()
    # batch_object_detector.load_image_filelist(MMCOQA_RESTORE_IMAGE_FOLDERPATH)
    # batch_object_detector.run_detection(
    #     MMCOQA_OBJECT_DETECTION_FILEPATH,
    #     MMCOQA_OBJECT_DETECTION_FAILED_FILEPATH
    # )
    
    '''Document Full Embedding'''
    # embedding_serializer = EmbeddingSerializer(MMCOQA_RESTORE_IMAGE_FOLDERPATH)
    # document_embedder: DocumentEmbedder = DocumentEmbedder(embedding_serializer)
    # document_embedder.load_files(
    #     MMCOQA_RESTORE_FOLDERPATH,
    #     MMCOQA_DOCUMENT_SUMMARIZATION_FILEPATH,
    #     MMCOQA_OBJECT_DETECTION_FILEPATH,
    #     MMCOQA_IMAGE_DESCRIPTION_FILEPATH
    # )
    # asyncio.run(document_embedder.run_embedding(
    #     MMEMBED_SERVER_URL_LIST,
    #     MMCOQA_DOCUMENT_EMBEDDING_FOLDERPATH,
    #     MMCOQA_DOCUMENT_EMBEDDING_FAILED_FILEPATH
    # ))
    
    '''MM-Embed Evaluation'''
    # verify_embedding(MMCOQA_DOCUMENT_EMBEDDING_FOLDERPATH)
    # asyncio.run(
    #     embedding_evaluation(
    #         MMEMBED_SERVER_URL_LIST[0],
    #         MMCOQA_RESTORE_DEV_FILEPATH,
    #         MMCOQA_DOCUMENT_EMBEDDING_FOLDERPATH,
    #         TOP_K
    #     )
    # )
    
    '''FiF Graph Construction'''
    # fif_graph: FiFGraph = FiFGraph.construct_graph(
    #     MMCOQA_RESTORE_FOLDERPATH,
    #     MMCOQA_DOCUMENT_EMBEDDING_FOLDERPATH,
    #     MMCOQA_DOCUMENT_SUMMARIZATION_FILEPATH
    # )
    
    # FiFGraph.save(fif_graph, "mmcoqa")

def process_main():
    print("Start Processing...")
    
    fif_graph: FiFGraph = FiFGraph.load("mmcoqa")
    
    '''LILaC Retrieval Evaluation'''
    # asyncio.run(
    #     lilac_retrieval_evaluation(
    #         fif_graph,
    #         QWEN_SERVER_URL_LIST[0],
    #         MMEMBED_SERVER_URL_LIST[0],
    #         MMCOQA_RESTORE_DEV_FILEPATH,
    #         BEAM_SIZE, TOP_K, MAX_HOP,
    #         MMCOQA_DEV_EMBEDDING_CACHE_FILEPATH
    #     )
    # )
    
    '''FiF Graph Traverse Orchestrator''' # TODO
    from memory import FiFTraversalContext
    from client import request_subqueries, request_subqueries_embedding
    
    fif_orchestrator: FiFOrchestrator = FiFOrchestrator(
        fif_graph,
        QWEN_SERVER_URL_LIST[0],
        MMEMBED_SERVER_URL_LIST[0]
    )
    query = "What sports is the Ben Piazza 1976 movie title?"
    subqueries = asyncio.run(request_subqueries(QWEN_SERVER_URL_LIST[0], query))
    subquery_embeddings = asyncio.run(request_subqueries_embedding(QWEN_SERVER_URL_LIST[0], MMEMBED_SERVER_URL_LIST[0], query, subqueries))
    ctx = FiFTraversalContext()
    ctx.fif_memory.reset_memory(query, subqueries, subquery_embeddings)
    fif_orchestrator.one_hop(ctx)
    
    '''FiF Retrieval Evaluation''' # TODO
    # asyncio.run(
    #     fif_retrieval_evaluation(
    #         fif_graph,
    #         QWEN_SERVER_URL_LIST[0],
    #         MMEMBED_SERVER_URL_LIST[0],
    #         MMCOQA_RESTORE_DEV_FILEPATH,
    #         BEAM_SIZE, TOP_K, MAX_HOP,
    #         MMCOQA_DEV_EMBEDDING_CACHE_FILEPATH
    #     )
    # )
    
    '''LLM Answer Generation''' # TODO
    
    '''LLM Answer Evaluation''' # TODO


if __name__ == "__main__":
    # preprocess_main()
    process_main()

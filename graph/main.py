import asyncio

from download import FIFDatasetLoader
from graph import FIFGraph
from preprocessor import (
    LLMQuerySerializer, DocumentSummarizer, BatchImageDescriptor, BatchObjectDetector, EmbeddingSerializer, DocumentEmbedder
)
from config import (
    MMCOQA_RESTORE_FOLDERPATH, MMCOQA_RESTORE_IMAGE_FOLDERPATH,
    MMCOQA_DOCUMENT_SUMMARIZATION_FILEPATH, MMCOQA_DOCUMENT_SUMMARIZATION_FAILED_FILEPATH,
    MMCOQA_IMAGE_DESCRIPTION_FILEPATH, MMCOQA_IMAGE_DESCRIPTION_FAILED_FILEPATH,
    MMCOQA_OBJECT_DETECTION_FILEPATH, MMCOQA_OBJECT_DETECTION_FAILED_FILEPATH,
    MMCOQA_DOCUMENT_EMBEDDING_FOLDERPATH, MMCOQA_DOCUMENT_EMBEDDING_FAILED_FILEPATH,
    QWEN_SERVER_URL_LIST, MMEMBED_SERVER_URL_LIST,
)

def preprocess_main():
    '''Dataset Download + Load (From Huggingface)'''
    # mmcoqa_doc_dataset_loader = FIFDatasetLoader(
    #     dataset_name="JoohyungYun/mmcoqa_doc",
    #     save_path=MMCOQA_RESTORE_FOLDERPATH
    # )
    # mmcoqa_doc_dataset_loader.restore(overwrite=True)

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
    
    '''Document Full Embedding''' # TODO
    embedding_serializer = EmbeddingSerializer(MMCOQA_RESTORE_IMAGE_FOLDERPATH)
    document_embedder: DocumentEmbedder = DocumentEmbedder(embedding_serializer)
    document_embedder.load_files(
        MMCOQA_RESTORE_FOLDERPATH,
        MMCOQA_DOCUMENT_SUMMARIZATION_FILEPATH,
        MMCOQA_OBJECT_DETECTION_FILEPATH,
        MMCOQA_IMAGE_DESCRIPTION_FILEPATH
    )
    document_embedder.run_embedding(
        MMEMBED_SERVER_URL_LIST,
        MMCOQA_DOCUMENT_EMBEDDING_FOLDERPATH,
        MMCOQA_DOCUMENT_EMBEDDING_FAILED_FILEPATH
    )
    
    '''MM-Embed Evaluation'''
    
    '''FIF Graph Construction''' # TODO
    # FIFGraph.construct_graph()


def process_main():
    pass
    '''Orchestrator''' # TODO
    '''LLM Answer Generation''' # TODO
    '''Evaluation''' # TODO

if __name__ == "__main__":
    preprocess_main()
    process_main()

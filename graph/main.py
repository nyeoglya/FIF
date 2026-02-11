import asyncio

from download import FIFDatasetContainer
from preprocessor import DocumentSerializer, DocumentSummarizer, DocumentEmbedder
from config import (
    MMQA_LILAC_DOC_FOLDERPATH, MMQA_PROCESSED_IMAGE_FOLDERPATH,
    MMQA_DOCUMENT_SUMMARIZATION_FILEPATH, MMQA_DOCUMENT_SUMMARIZATION_FAILED_FILEPATH,
    MMQA_DOCUMENT_EMBEDDING_FILEPATH, MMQA_DOCUMENT_EMBEDDING_FAILED_FILEPATH,
    QWEN_SERVER_URL_LIST, MMEMBED_SERVER_URL_LIST
)

def preprocess_main():
    '''Dataset Download (From Huggingface)'''
    mmqa_doc_dataset_manager = FIFDatasetContainer("JoohyungYun/multimodalqa_doc")
    # mmcoqa_doc_dataset_manager = FIFDatasetContainer("JoohyungYun/mmcoqa_doc")
    # webqa_doc_dataset_manager = FIFDatasetContainer("JoohyungYun/webqa_doc")
    
    print(mmqa_doc_dataset_manager.get_docdata_from_doctitle("Cloris_Leachman"))
    
    '''Object Detection''' # TODO
    
    
    '''Image Summarization''' # TODO
    
    
    '''Document Serialization''' # TODO
    document_serializer = DocumentSerializer()
    
    '''Document Summarization''' # TODO
    document_summarizer: DocumentSummarizer = DocumentSummarizer()
    document_summarizer.load_data_from_ldoc_folder(MMQA_LILAC_DOC_FOLDERPATH)
    asyncio.run(document_summarizer.run_parallel_summarize(
        QWEN_SERVER_URL_LIST,
        MMQA_DOCUMENT_SUMMARIZATION_FILEPATH,
        MMQA_DOCUMENT_SUMMARIZATION_FAILED_FILEPATH
    ))
    
    '''Document Embedding''' # TODO
    document_embedder: DocumentEmbedder = DocumentEmbedder(MMEMBED_SERVER_URL_LIST[0])
    document_embedder.run_embedding(
        MMQA_DOCUMENT_SUMMARIZATION_FILEPATH,
        MMQA_DOCUMENT_EMBEDDING_FILEPATH,
        MMQA_DOCUMENT_EMBEDDING_FAILED_FILEPATH
    )
    
    '''FIF Graph Construction''' # TODO
    

def process_main():
    pass
    '''Orchestrator''' # TODO
    '''LLM Answer Generation''' # TODO
    '''Evaluation''' # TODO

if __name__ == "__main__":
    preprocess_main()
    # process_main()

import asyncio

from preprocessor import DocumentSummarizer
from config import (
    MMQA_LILAC_DOC_FOLDERPATH, MMQA_PROCESSED_IMAGE_FOLDERPATH,
    MMQA_DOCUMENT_SUMMARIZATION_FILEPATH, MMQA_DOCUMENT_SUMMARIZATION_FAILED_FILEPATH,
    QWEN_SERVER_URL_LIST
)

def preprocess_main():
    '''Document Summarization'''
    document_summarizer = DocumentSummarizer()
    document_summarizer.load_data_from_ldoc_folder(MMQA_LILAC_DOC_FOLDERPATH, MMQA_PROCESSED_IMAGE_FOLDERPATH)
    asyncio.run(document_summarizer.run_parallel_summarize(
        QWEN_SERVER_URL_LIST,
        MMQA_DOCUMENT_SUMMARIZATION_FILEPATH,
        MMQA_DOCUMENT_SUMMARIZATION_FAILED_FILEPATH
    ))
    
    # TODO: FIF Graph Construction

def process_main():
    pass
    # TODO: Orchestrator
    # TODO: LLM Answer Generation
    # TODO: Evaluation


if __name__ == "__main__":
    preprocess_main()
    # process_main()

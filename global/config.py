import typing as tp

MMQA_LILAC_DOC_FOLDERPATH: str       = "/dataset/mmqa_ldoc"
MMQA_PROCESSED_IMAGE_FOLDERPATH: str = "/dataset/processed_mmqa_images/"

MMQA_DOCUMENT_SUMMARIZATION_FILEPATH: str        = "/dataset/mmqa_doc_summarization.jsonl"
MMQA_DOCUMENT_SUMMARIZATION_FAILED_FILEPATH: str = "/dataset/mmqa_doc_summarization_failed.txt"
MMQA_DOCUMENT_EMBEDDING_FILEPATH: str            = "/dataset/mmqa_doc_embedding_database"
MMQA_DOCUMENT_EMBEDDING_FAILED_FILEPATH: str     = "/dataset/mmqa_doc_embedding_failed.txt"

QWEN_SERVER_URL_LIST: tp.List[str] = [
    "http://fif-qwen-worker-0:8000/v1/chat/completions",
    # "http://fif-qwen-worker-1:8000/v1/chat/completions",
    # "http://fif-qwen-worker-2:8000/v1/chat/completions"
]
MMEMBED_SERVER_URL_LIST: tp.List[str] = [
    "http://fif-mmembed:8000"
]

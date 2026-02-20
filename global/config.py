import typing as tp

MMCOQA_RESTORE_FOLDERPATH: str       = "/dataset/original/mmcoqa_doc/"
MMCOQA_RESTORE_IMAGE_FOLDERPATH: str = "/dataset/original/mmcoqa_doc/images"
MMCOQA_RESTORE_DEV_FILEPATH: str     = "/dataset/original/mmcoqa_doc_dev.jsonl"

MMCOQA_DOCUMENT_SUMMARIZATION_FILEPATH: str        = "/dataset/artifact/mmcoqa_doc_summarization.jsonl"
MMCOQA_DOCUMENT_SUMMARIZATION_FAILED_FILEPATH: str = "/dataset/artifact/mmcoqa_doc_summarization_failed.txt"
MMCOQA_IMAGE_DESCRIPTION_FILEPATH: str             = "/dataset/artifact/mmcoqa_image_description.jsonl"
MMCOQA_IMAGE_DESCRIPTION_FAILED_FILEPATH: str      = "/dataset/artifact/mmcoqa_image_description_failed.txt"
MMCOQA_OBJECT_DETECTION_FILEPATH: str              = "/dataset/artifact/mmcoqa_object_detection.jsonl"
MMCOQA_OBJECT_DETECTION_FAILED_FILEPATH: str       = "/dataset/artifact/mmcoqa_object_detection_failed.txt"
MMCOQA_IMAGE_EMBEDDING_FILEPATH: str               = "/dataset/artifact/mmcoqa_image_embedding_database"
MMCOQA_IMAGE_EMBEDDING_FAILED_FILEPATH: str        = "/dataset/artifact/mmcoqa_image_embedding_failed.txt"
MMCOQA_DOCUMENT_EMBEDDING_FOLDERPATH: str          = "/dataset/artifact/mmcoqa_doc_embedding/"
MMCOQA_DOCUMENT_EMBEDDING_FAILED_FILEPATH: str     = "/dataset/artifact/mmcoqa_doc_embedding_failed.txt"
MMCOQA_DEV_EMBEDDING_CACHE_FILEPATH: str           = "/dataset/artifact/mmcoqa_dev_embedding_cache.npz"


QWEN_SERVER_URL_LIST: tp.List[str] = [
    "http://fif-qwen-worker-0:8000/v1/chat/completions",
    "http://fif-qwen-worker-1:8000/v1/chat/completions",
    "http://fif-qwen-worker-2:8000/v1/chat/completions"
]
MMEMBED_SERVER_URL_LIST: tp.List[str] = [
    "http://fif-mmembed:8000",
    "http://fif-mmembed:8001"
]

TOP_K: int = 9
BEAM_SIZE: int = 30
MAX_HOP: int = 100

import os
import typing as tp

from tqdm import tqdm
from structure.document import DocumentTree
from structure.component.constants import Modality

base_path = "/mnt/sdc/jhyun/omdr_mountspace/algorithm_artifacts/sns/_02_mmcoqa/multimodal_documents/dev"
images_dir = "/mnt/sdc/jhyun/omdr_mountspace/datasets/_02_mmcoqa/image_components/dev"
subimages_dir = "/mnt/sdc/jhyun/omdr_mountspace/datasets/_02_mmcoqa/subimage_components/dev"
image_summaries_dir = "/mnt/sdc/jhyun/omdr_mountspace/datasets/_02_mmcoqa/image_summaries/dev"

if __name__ == "__main__":
    parsed_documents: tp.List[DocumentTree] = []
    
    modality_to_count: tp.Dict[str, int] = {
        Modality.TEXT.value: 0,
        Modality.TABLE.value: 0,
        Modality.IMAGE.value: 0,
        Modality.SENTENCE.value: 0,
        Modality.TABLE_SEGMENT.value: 0,
        Modality.SUBIMAGE.value: 0
    }

    for file_name in tqdm(os.listdir(base_path)):
        file_path = os.path.join(base_path, file_name)
        document = DocumentTree(file_path)
        parsed_documents.append(document)
         
        statistics = document.get_component_statistics()
        
        for modality, count in statistics.items():
            modality_to_count[modality] += count

    print("Modality statistics:")
    for modality, count in modality_to_count.items():
        print(f"{modality}: {count}")
    
    print("Total components:", sum(modality_to_count.values()))
    print("Total documents:", len(parsed_documents))
    print("Average components per document:", sum(modality_to_count.values()) / len(parsed_documents))

import json
import typing as tp
from collections import defaultdict

from tqdm import tqdm
from datasets import load_dataset, DatasetDict, Dataset # type: ignore

from type import JSONDict

class FIFDatasetContainer:
    DEFAULT_CONFIGS = ['text_component', 'table_component', 'image_component', 'image_dump']
    
    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        self.data: tp.Dict[str, DatasetDict] = dict()

        for config in self.DEFAULT_CONFIGS:
            self.data[config] = load_dataset(self.dataset_name, config)
        
        self.merged_doc_dict: tp.DefaultDict[str, tp.List[JSONDict]] = defaultdict(list)

        def append_data(dataset: Dataset, desc: str = "", component_deserialize: bool = True):
            rows: tp.Iterable[JSONDict] = tp.cast(tp.Iterable[JSONDict], dataset)
            for row in tqdm(rows, desc=desc):
                doc_title = row['doc_title']
                item = {k: v for k, v in row.items() if k != 'doc_title'}
                item['heading_path'] = json.loads(item['heading_path'])
                item['hyperlinks'] = json.loads(item['hyperlinks'])
                if component_deserialize:
                    item['component'] = json.loads(item['component'])
                self.merged_doc_dict.setdefault(doc_title, []).append(item)
        
        append_data(dataset=self.get_text_data(), desc="Loading Text", component_deserialize=False)
        append_data(dataset=self.get_table_data(), desc="Loading Table")
        append_data(dataset=self.get_image_data(), desc="Loading Image")
        
        self.doc_title_set: tp.Set[str] = set(tp.cast(tp.List[str], self.get_text_data()['doc_title']))
        self.doc_title_set.update(set(tp.cast(tp.Set[str], self.get_table_data()['doc_title'])))
        self.doc_title_set.update(set(tp.cast(tp.Set[str], self.get_image_data()['doc_title'])))
        
    def get_document_title_set(self) -> tp.Set[str]:
        return self.doc_title_set
    
    def get_docdata_from_doctitle(self, doc_title: str) -> tp.List[JSONDict]:
        return self.merged_doc_dict[doc_title]

    def get_text_data(self) -> Dataset:
        return self.data['text_component']["text"]

    def get_table_data(self) -> Dataset:
        return self.data['table_component']["table"]
    
    def get_image_data(self) -> Dataset:
        return self.data['image_component']["image_meta"]
    
    def get_image_dump_data(self) -> Dataset:
        return self.data['image_dump']["dump"]

    def __getitem__(self, config_name: str) -> DatasetDict:
        return self.data[config_name]

    def __repr__(self) -> str:
        return f"DatasetManager(name={self.dataset_name}, loaded_configs={list(self.data.keys())})"

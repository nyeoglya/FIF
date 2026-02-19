import os
import numpy as np
import typing as tp
from tqdm import tqdm

from type import JSONDict

def verify_embedding(folder_path: str):
    embedding_file_list: tp.List[str] = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".npz")]
    
    results: JSONDict = {
        "total": len(embedding_file_list),
        "valid": 0,
        "invalid": [],
        "errors": []
    }

    essential_keys = {"doc_emb", "subembs", "metadata"}

    for filepath in tqdm(embedding_file_list, desc="Verifying"):
        filename: str = os.path.basename(filepath)
        try:
            with np.load(filepath, allow_pickle=True) as data:
                actual_keys: tp.Set[str] = set(data.files)
                missing_essential: tp.Set[str] = essential_keys - actual_keys
                if missing_essential:
                    results["invalid"].append(f"{filename} | Essential Key Missing: {missing_essential}")
                    continue
                
                # metadata Load and Synchronize
                metadata: tp.Dict[str, tp.Tuple[int, int]] = data['metadata'].item()
                metadata_keys: tp.Set[str] = set(metadata.keys())

                missing_content: tp.Set[str] = metadata_keys - actual_keys
                if missing_content:
                    results["invalid"].append(f"{filename} | Key Missing: {missing_content}")
                    continue

                # Metadata Tuple Sanity
                tuple_values: tp.List[tp.Tuple[int, int]] = list(metadata.values())
                unique_values: tp.Set[tp.Tuple[int, int]] = set(tuple_values)
                if len(tuple_values) != len(unique_values):
                    seen: tp.Set[tp.Tuple[int, int]] = set()
                    dupes = [x for x in tuple_values if x in seen or seen.add(x)]
                    results["invalid"].append(f"{filename} | Redundant Metadata: {dupes}")
                    continue
                
                results["valid"] += 1
        except Exception as e:
            results["errors"].append(f"{filename} | File Read Error: {str(e)}")

    print("\n[Summary]")
    print(f"=> Normal: {results['valid']} / {results['total']}")
    print(f"=> Error: {len(results['invalid'])} / {results['total']}")
    print(f"=> Corrupted: {len(results['errors'])} / {results['total']}")

    if results["invalid"] or results["errors"]:
        print("\n[Details]")
        error_results = results["invalid"] + results["errors"]
        for index, issue in enumerate(error_results):
            print(f"({index:03d}) {issue}")
    
    return results

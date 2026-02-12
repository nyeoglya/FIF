import numpy as np

def inspect_npz_structure(file_path: str):
    print(f"--- Inspecting File: {file_path} ---")
    
    try:
        data = np.load(file_path, allow_pickle=True)
        
        keys = data.files
        print(f"[*] Found Keys: {keys}")
        print("-" * 40)
        
        for key in keys:
            content = data[key]
            
            if key == "metadata":
                meta_dict = content.item()
                print(f"[Key: {key}] (Dictionary)")
                print(f"    - Type: {type(meta_dict)}")
                print(f"    - Keys in metadata: {list(meta_dict.keys())}")
                for i, (k, v) in enumerate(meta_dict.items()):
                    if i > 1: break
                    print(f"    - Sample -> {k}: {v}")
            
            else:
                print(f"[Key: {key}] (NumPy Array)")
                print(f"    - Shape: {content.shape}")
                print(f"    - Dtype: {content.dtype}")
                if content.size > 0:
                    sample = content[0] if content.ndim > 1 else content
                    print(f"    - Sample Data: {sample[:3]}...") 

    except Exception as e:
        print(f"[!] Error loading npz: {e}")

inspect_npz_structure("/dataset/artifact/mmcoqa_doc_embedding/Chinmayi.json.npz")

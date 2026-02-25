import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding.CacheDataset import generate_cache_pic_new, preprocess_traces_dir


if __name__ == "__main__":
    RAW_DIR = "./cache_dataset/cache_dataset_llc_tvm/"
    preprocess_traces_dir(RAW_DIR, skip=0)

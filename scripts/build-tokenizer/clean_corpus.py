"""
01 — Download, Filter, and Clean the Corpus
============================================

Step 1 — download every pretrain shard from HuggingFace (~5 GB, cached).
Step 2 — measure whether Type-A cleaning still earns its place.
Step 3 — demo the Type-B normalizer (unchanged, still the golden rule).
Step 4 — write a SAMPLED corpus for the tokenizer to train on.

"""

import os

# Must be set before importing huggingface_hub: the slow/stalling Xet backend
# is bypassed in favor of the classic HTTP downloader.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from cleaning import prepare_document, clean_dataframe, normalize_text
from huggingface_hub import hf_hub_download
from pathlib import Path
import pandas as pd

REPO = 'bakrianoo/jabarti-llm-dataset'

HF_FILES = {
    'phase1_train': 'pretrain/phase1_train-00000-of-00001.parquet',
    'phase1_eval': 'pretrain/phase1_eval-00000-of-00001.parquet',

    'phase2_train': 'pretrain/phase2_train-00000-of-00001.parquet',
    'phase2_eval': 'pretrain/phase2_eval-00000-of-00001.parquet',

    'ft_train': 'finetune/train-00000-of-00001.parquet',
    'ft_eval': 'finetune/eval-00000-of-00001.parquet',
}

TRAIN_SPLITS = {"phase1_train", "phase2_train"}
EVAL_SPLITS = {"phase1_eval", "phase2_eval"}


OUT_DIR = Path(__file__).parent / 'output'

MAX_CHUNKS = 20

def download_and_filter():

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, file_name in HF_FILES.items():
        local = hf_hub_download(
            repo_id=REPO, 
            repo_type='dataset', 
            filename=file_name)
        df = pd.read_parquet(local)
        print(f'Loaded {name}: {len(df)} rows')
        if name in TRAIN_SPLITS:
            df = (
                df.sort_values(['article_id', 'chunk_index'])
                .groupby('article_id', sort=False)
                .head(MAX_CHUNKS)
                .reset_index(drop=True)
            )
            print(f'Filtered {name}: {len(df)} rows')
        output_path = OUT_DIR / f'{name}_filtered.parquet'
        df.to_parquet(output_path, index=False)

def clean_phase(phase, input_path, output_path):
    print(f'Starting cleaning phase for {phase}')

    df = pd.read_parquet(input_path)
    print(f'Cleaning {phase}: {len(df)} rows before cleaning, input path: {input_path}')
    cleaned_df = clean_dataframe(df)
    print(f'Cleaning {phase}: {len(cleaned_df)} rows after cleaning')

    cleaned_df.to_parquet(output_path, index=False)

def main():
    download_and_filter()
    for name in TRAIN_SPLITS | EVAL_SPLITS:
        input_path = OUT_DIR / f'{name}_filtered.parquet'
        output_path = OUT_DIR / f'{name}_clean.parquet'
        clean_phase(phase=name, input_path=input_path, output_path=output_path)

if __name__ == "__main__":
    main()
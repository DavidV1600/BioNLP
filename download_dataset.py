import os
from datasets import load_dataset
from aidose.dataset.constants import END_POINT_HF_DATASET_PATH, HF_HUB_REPO_ID

print("Downloading dataset from Hugging Face:", HF_HUB_REPO_ID)
ds = load_dataset(HF_HUB_REPO_ID)
print("Dataset loaded:", ds)
print("Saving dataset to disk at:", END_POINT_HF_DATASET_PATH)
ds.save_to_disk(END_POINT_HF_DATASET_PATH)
print("Dataset saved successfully!")

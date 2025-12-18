from transformers import AutoProcessor, AutoModelForCTC
model = AutoModelForCTC.from_pretrained("/home/zyz/data/openslr")
print(model)

from datasets import load_dataset

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset

dataset = load_dataset("/home/zyz/data/Librispeech", "default", split={
    "train": "train",
    "validation": "validation",
    "test": "test"
})
print(dataset)
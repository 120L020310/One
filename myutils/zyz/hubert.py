from transformers import AutoProcessor, HubertModel
from datasets import load_dataset
import soundfile as sf
import torch

processor = AutoProcessor.from_pretrained("/home/zyz/workdir/MVCL-ADD-main/models/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("/home/zyz/workdir/MVCL-ADD-main/models/hubert-large-ls960-ft")


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch

test_audio = torch.rand(64,48000)

# input_values = processor(test_audio, return_tensors="pt").input_values  # Batch size 1
hidden_states = model(test_audio).last_hidden_state
print(hidden_states.shape)
import json
import pickle
import os
import numpy as np
import torch
import torch.nn.functional as F
from train_char import ModelConfig, TrainConfig
from model import Transformer
from dataclasses import dataclass

seed = 12345
meta = json.load(open("data/github/meta.json", "r"))
folder = f"2L4H256C-{seed}"
with open(os.path.join(folder, "model_config.pkl"), "rb") as f:
    model_conf = pickle.load(f)

model = Transformer(model_conf)
model.load_state_dict(torch.load("2L4H256C-12345/ckpt.pt"))

sequence = np.random.randint(0, 80, 5)
tokens = torch.tensor(np.concatenate([sequence for _ in range(3)])).unsqueeze(0)
print(tokens.shape)

logits = model(tokens)
print(logits.shape)
print(tokens)
print(logits[0].argmax(1))


data = np.memmap("data/github/char.bin", dtype=np.uint16, mode="r")
print(len(data) // (256 * 64))

# chars = list(data)
# d = {}
# for c in chars:
#     if c in d:
#         d[c] += 1
#     else:
#         d[c] = 1
# print(sorted([(k, count) for k, count in d.items()], key=lambda x: x[1], reverse=True))

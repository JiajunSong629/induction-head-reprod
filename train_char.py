import json
import pickle
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from model import Transformer
from dataclasses import dataclass, asdict

seed = 1245
data = np.memmap("data/github/char.bin", dtype=np.uint16, mode="r")
meta = json.load(open("data/github/meta.json", "r"))


# train
@dataclass
class TrainConfig:
    batch_size = 64
    lr = 5e-4
    weight_decay = 0.1
    num_epochs = 20000


# model
@dataclass
class ModelConfig:
    n_layer: int = 2
    n_head: int = 4
    d_model: int = 256
    block_size: int = 128
    n_vocab: int = meta["vocab_size"]
    bias: bool = False


torch.manual_seed(seed)


def get_batch(batch_size, block_size, device="cuda"):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    x, y = x.to(device), y.to(device)
    return x, y


def main():
    model_conf = ModelConfig()
    train_conf = TrainConfig()
    model = Transformer(model_conf)
    model.to("cuda")
    print(model)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_conf.lr,
        weight_decay=train_conf.weight_decay,
        betas=(0.9, 0.98),
    )
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 10, 1))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        train_conf.num_epochs,
        eta_min=train_conf.lr / 10,
    )

    # io
    folder = f"2L4H{model_conf.d_model}C-history-{seed}"
    os.makedirs(folder, exist_ok=True)

    losses = {}
    for epoch in range(train_conf.num_epochs):
        x, y = get_batch(train_conf.batch_size, model_conf.block_size)
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1
        )
        loss.backward()
        losses[epoch] = loss.item()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if epoch % 100 == 0:
            print(epoch, loss.item(), "lr", scheduler.get_lr())

        if epoch in range(1000, 40000, 2000):
            torch.save(model.state_dict(), f"{folder}/ckpt-{epoch}.pt")

    torch.save(model.state_dict(), f"{folder}/ckpt-{train_conf.num_epochs}.pt")

    with open(os.path.join(folder, "loss.json"), "w") as f:
        json.dump(losses, f)
    with open(os.path.join(folder, "model_config.json"), "w") as f:
        json.dump(asdict(model_conf), f)
    with open(os.path.join(folder, "train_config.json"), "w") as f:
        json.dump(asdict(train_conf), f)


if __name__ == "__main__":
    main()

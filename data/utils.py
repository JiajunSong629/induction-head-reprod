import requests
from datasets import load_dataset


def download_dataset(dataset, streaming_samples, input_file_path):
    if dataset == "github":
        ds = load_dataset(
            "codeparrot/github-code",
            streaming=True,
            split="train",
            languages=["Python"],
        )
        key = "code"
    elif dataset == "openwebtext":
        ds = load_dataset(
            "Skylion007/openwebtext",
            streaming=True,
            split="train",
        )
        key = "text"
    elif dataset == "wikitext":
        ds = load_dataset(
            "wikitext",
            "wikitext-103-v1",
            streaming=True,
            split="train",
        )
        key = "text"
    elif dataset == "shakespeare":
        print("Downloading", dataset)
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)
    else:
        raise ValueError(
            f"{dataset} not supported! Must be in 'github', 'openwebtext', 'wikitext'."
        )

    print("Downloading", dataset)
    samples = []
    for sample in ds.take(streaming_samples):
        if check(sample[key]):
            samples.append(sample[key])

    with open(input_file_path, "w") as f:
        f.write("\n".join(samples))


def check(sample):
    # sample only includes
    chars = sorted(list(set(sample)))
    return ord(chars[-1]) <= ord("~") and ord(chars[0]) >= ord("\t")


def char_handler(data):
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    ids = [stoi[c] for c in data]

    # save the meta information as well, to help us encode/decode later
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    return ids, meta

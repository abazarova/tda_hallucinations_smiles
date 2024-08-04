import argparse
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import login
from tqdm import trange
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)
DATA_PATH = Path("assets/data/rag_truth/")
os.chdir('/app')


def load_model_and_tokenizer(model_id: str, device: str = "cuda") -> tuple:
    HUGGING_FACE_API_KEY = os.environ["HF_API_KEY"]
    login(token=HUGGING_FACE_API_KEY)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    if device == "cuda":
        model = model.cuda().half()

    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def grab_attention_weights(
    model, tokenizer, sentences: list[tuple], max_len: int = 128, device: str = "cuda"
) -> list[tuple]:
    ids, sentences = zip(*sentences)
    inputs = tokenizer(
        [text_preprocessing(s) for s in sentences],
        return_tensors="pt",
        add_special_tokens=True,
        max_length=max_len,  # max length to truncate/pad
        pad_to_max_length=True,
        truncation=True,
    ).to(device)

    model.to(device)

    with torch.no_grad():
        attn_matrices = model(
            **inputs, output_attentions=True
        ).attentions  # layer x sample x head x n_token x n_token

    ntokens = inputs["attention_mask"].sum(dim=-1).tolist()
    f = lambda x: x.cpu().half().numpy()
    attn_matrices = attn_matrices[15], attn_matrices[-1]
    attn_matrices = np.asarray([f(mx) for mx in attn_matrices])

    return list(zip(ids, ntokens, attn_matrices.swapaxes(0, 1)))


def save_attn_weights(
    model,
    tokenizer,
    sentences: list[tuple],
    max_len: int = 128,
    device: str = "cuda",
    batch_size: int = 64,
    save_path: Path = Path("."),
) -> None:
    data_size = len(sentences)
    total_weights = []
    for i in trange(0, data_size, batch_size):
        batch = sentences[i : i + batch_size]
        attn_weights = grab_attention_weights(model, tokenizer, batch, max_len, device)
        total_weights += attn_weights

    assert (
        len(total_weights) == data_size
    ), f"Data sizes mismatch: {len(total_weights)} != {data_size}. Sanity check failed."

    ids, ntokens, attn_matrices = zip(*total_weights)
    ntokens_dict = dict(zip(ids, ntokens))
    attn_matrices_dict = dict(zip(ids, attn_matrices))

    save_path.mkdir(exist_ok=True)

    with open(f"{save_path}/tokens_count.json", "w") as f:
        json.dump(ntokens_dict, f)

    np.savez_compressed(f"{save_path}/attn_matrices", **attn_matrices_dict)


def text_preprocessing(text: str) -> str:
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r"(@.*?)[\s]", " ", text)
    # Replace '&amp;' with '&'
    text = re.sub(r"&amp;", "&", text)
    # Remove trailing whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, choices=["qa", "summ"], default="qa"
    )
    parser.add_argument(
        "--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.1"
    )
    parser.add_argument(
        "--save_path", type=str, default="assets/attention_maps"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_dump", type=int, default=5)

    args = parser.parse_args()
    save_path = Path(args.save_path)

    data = pd.read_csv(DATA_PATH / f"{args.task}_samples.csv")
    
    data = data[data.apply(lambda x: x['model'].lower() in args.model_id.lower(), axis=1)]
    data["id"] = data["id"].apply(str)
    data.fillna("", inplace=True)

    model, tokenizer = load_model_and_tokenizer(args.model_id, args.device)

    sentences = list(zip(data["id"], data["prompt"] + data["response"]))

    data_size = len(sentences)
    n_dump = args.n_dump
    dump_size = data_size // args.n_dump
    
    for i in range(0, data_size, dump_size):
        save_attn_weights(
            model,
            tokenizer,
            sentences[i : i + dump_size],
            max_len=1024,
            batch_size=4,
            save_path=save_path / f"{args.task}/pt_{i // dump_size}",
        )

        logger.info(f"Dumped part {i // dump_size}/{n_dump}")

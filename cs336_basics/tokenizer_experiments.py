import time
import regex as re
import numpy as np
from cs336_basics.tokenizer import BPETokenizer

def sample_documents(text: str, nsamples: int = 10):
    docs = re.split(re.escape('<|endoftext|>'), text)
    ndocs = len(docs)
    sample_indices = np.random.choice(ndocs, nsamples, replace=False)
    return [docs[idx] for idx in sample_indices]


def tokenizer_stats(text: str, tokenizer: BPETokenizer):
    nbytes = len(text.encode("utf-8"))
    start_time = time.time()
    tokens = tokenizer.encode(text)
    end_time = time.time()
    ntokens = len(tokens)
    return float(nbytes) / ntokens, (nbytes / (end_time - start_time))


def tokenizer_comparison(filepath: str):
    print(f"Comparison on {filepath}")
    with open(filepath, "r") as f:
        text = f.read()
        sample_docs = sample_documents(text, 10)
        avg_ts_throughput = 0
        avg_owt_throughput = 0
        avg_ts_compression_ratio = 0
        avg_owt_compression_ratio = 0
        for doc in sample_docs:
            ts_compression_ratio, ts_throughput = (
                tokenizer_stats(doc, ts_tokenizer)
            )
            avg_ts_compression_ratio += ts_compression_ratio
            avg_ts_throughput += ts_throughput
            print("TinyStory compression ratio (bytes/tokens): ",
                ts_compression_ratio
            )
            owt_compression_ratio, owt_throughput = (
                tokenizer_stats(doc, owt_tokenizer)
            )
            avg_owt_compression_ratio += owt_compression_ratio
            avg_owt_throughput += owt_throughput
            print("Open Web Text compression ratio (bytes/tokens)",
                owt_compression_ratio
            )
        avg_owt_compression_ratio /= 10
        avg_ts_compression_ratio /= 10
        avg_owt_throughput /= 10
        avg_ts_throughput /= 10

        print(f"Average Open Web Text tokenizer compression ratio {avg_owt_compression_ratio},"
              f" throughput {avg_owt_throughput}")

        print(f"Average Tiny Stories tokenizer compression ratio {avg_ts_compression_ratio},"
              f" throughput {avg_ts_throughput}")


def tokenize_file(filepath: str, tokenizer: BPETokenizer, savepath: str):
    with open(filepath, 'r') as f:
        ids = []
        for _id in tokenizer.encode_iterable(f):
            ids.append(_id)
        ids = np.array(ids, dtype=np.uint16)
        np.save(savepath, ids)


if __name__ == "__main__":
    ts_tokenizer = BPETokenizer.from_files(
        "data/TinyStoriesV2-GPT4-train-vocab.pkl",
        "data/TinyStoriesV2-GPT4-train-merges.pkl",
        ["<|endoftext|>"],
    )
    owt_tokenizer = BPETokenizer.from_files(
        "data/owt_train-vocab.pkl",
        "data/owt_train-merges.pkl",
        ["<|endoftext|>"],
    )
    
    # tokenizer_comparison("data/owt_train.txt")
    # tokenizer_comparison("data/TinyStoriesV2-GPT4-train.txt")

    # tokenize_file("tests/fixtures/tinystories_sample.txt", 
    #     ts_tokenizer, "tests/fixtures/tinystories_sample-tokens.npy")
    tokenize_file("data/TinyStoriesV2-GPT4-valid.txt",
        ts_tokenizer, "data/TinyStoriesV2-GPT4-valid-tokens.npy")
    tokenize_file("data/TinyStoriesV2-GPT4-train.txt",
        ts_tokenizer, "data/TinyStoriesV2-GPT4-train-tokens.npy")

    tokenize_file("data/owt_valid.txt",
        ts_tokenizer, "data/owt_valid-tokens.npy")
    tokenize_file("data/owt_train.txt",
        ts_tokenizer, "data/owt_train-tokens.npy")
    



    

    

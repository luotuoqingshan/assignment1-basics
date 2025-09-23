import regex as re 
import time
import json
import pickle
from typing import Any, BinaryIO
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Process, Queue, Pool

def pre_tokenization(
    input_data: str,
    word_count: dict[tuple[bytes], int],
):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for match in re.finditer(PAT, input_data):
        # count the appearances of words
        word = match.group().encode("utf-8")
        # we represent each word as a tuple of bytes
        word_count[word] = word_count.get(word, 0) + 1


def split_special_token(
    input_data: str,
    special_tokens: list[str],
) -> list[str]:
    pattern = "|".join(re.escape(token) for token in special_tokens)
    chunks = re.split(pattern, input_data)
    return chunks


def init_vocab(
    special_tokens: list[str],
) -> dict[int, bytes]:
    # initialize the vocabulary
    vocab = dict[int, bytes]()
    for i in range(256):
        vocab[i] = i.to_bytes()
    
    # the vocabulary size right now
    cur_vocab_size = 256
    for special_token in special_tokens:
        vocab[cur_vocab_size] = special_token.encode("utf-8")
        cur_vocab_size += 1
    return vocab


def pre_tokenization_worker(
    input_path: str,
    start: int, 
    end: int,
    special_tokens: list[str],
) -> dict[tuple[bytes], int]:
    word_count = dict[tuple[bytes], int]()
    with open(input_path, "rb") as f:
        f.seek(start)
        input_data = f.read(end - start).decode("utf-8", errors="ignore")
        chunks = split_special_token(input_data, special_tokens)
        for chunk in chunks:
            pre_tokenization(chunk, word_count)
    return word_count


def merge_word_counts(
    local_word_counts: list[dict[tuple[bytes], int]],
):
    global_word_count = dict[tuple[bytes], int]()
    for word_count in local_word_counts:
        # always merge smaller one to larger one
        if len(word_count) > len(global_word_count):
            global_word_count, word_count = (
                word_count, global_word_count
            )
        for (k, v) in word_count.items():
            global_word_count[k] = (
                global_word_count.get(k, 0) + v
            )
    return global_word_count


def post_process_word_count(
    word_count: dict[tuple[bytes], int]
):
    new_word_count = dict[tuple[bytes], int]()
    for (word, count) in word_count.items():
        byte_level_word = tuple([word[i:i+1] for i in range(len(word))])
        new_word_count[byte_level_word] = count
    return new_word_count


def init_token_pair_count_and_loc(
    word_count: dict[tuple[bytes], int],
):
    words = list[tuple[bytes]]()
    token_pair_count = dict[tuple[bytes, bytes], int]()
    token_pair_loc = dict[tuple[bytes, bytes], set[int]]()
    for idx, (word, count) in enumerate(word_count.items()):
        words.append(word)
        for i in range(len(word) - 1):
            token_pair = tuple([word[i], word[i+1]])
            token_pair_count[token_pair] = (
                token_pair_count.get(token_pair, 0) + count
            )

            if token_pair not in token_pair_loc:
                token_pair_loc[token_pair] = set[int]()
            token_pair_loc[token_pair].add(idx)
    return words, token_pair_count, token_pair_loc


def find_most_common_token_pair(
    token_pair_count: dict[tuple[bytes, bytes], int],
) -> tuple[bytes, bytes]:

    most_common_token_pair = (
        max(
            token_pair_count,
            # If two token pairs have the same frequency,
            # break the tie by merging the lexicographically greater pair
            key=lambda token_pair: (
                token_pair_count[token_pair], 
                token_pair[0],
                token_pair[1],
            ),
        )
    )
    return most_common_token_pair


def add_token_pair_count(
    token_pair_count: dict[tuple[bytes, bytes], int],
    token_pair_loc: dict[tuple[bytes, bytes], set[int]],
    token_pair: tuple[bytes, bytes],
    count: int,
    word_idx: int,
):
    token_pair_count[token_pair] = (
        token_pair_count.get(token_pair, 0) + count
    )
    if token_pair not in token_pair_loc:
        token_pair_loc[token_pair] = set[int]()
    token_pair_loc[token_pair].add(word_idx)


def sub_token_pair_count(
    token_pair_count: dict[tuple[bytes, bytes], int],
    token_pair: tuple[bytes, bytes],
    count: int,
):
    token_pair_count[token_pair] -= count


def merge_most_common_token_pair(
    most_common_token_pair: tuple[bytes, bytes],
    words: list[tuple[bytes]],
    word_count: dict[tuple[bytes], int],
    token_pair_count: dict[tuple[bytes, bytes], int],
    token_pair_loc: dict[tuple[bytes, bytes], set[int]],
):
    token1, token2 = most_common_token_pair
    # print("Most common token pair: ", most_common_token_pair)
    merged_token = b"".join([token1, token2])

    # iterate through all words containing this token pair
    for word_idx in token_pair_loc[most_common_token_pair]:
        word = words[word_idx]
        word_freq = word_count[word]
        new_word = list[bytes]()
        token_idx = 0
        while token_idx < len(word):
            # find the match
            if (
                token_idx + 1 < len(word) 
                and word[token_idx] == token1 
                and word[token_idx + 1] == token2
            ):
                new_word.append(merged_token)

                # adjust the adjacent token pairs
                if token_idx > 0:
                    prev_token = word[token_idx - 1]
                    old_prev_token_pair = tuple([prev_token, token1])
                    sub_token_pair_count(
                        token_pair_count, 
                        old_prev_token_pair, 
                        word_freq
                    )
                if token_idx + 2 < len(word):
                    next_token = word[token_idx + 2]
                    old_next_token_pair = tuple([token2, next_token])
                    sub_token_pair_count(
                        token_pair_count, 
                        old_next_token_pair,
                        word_freq
                    )
                token_idx += 2
            else:
                new_word.append(word[token_idx])
                token_idx += 1
        new_word = tuple(new_word)
        words[word_idx] = new_word
        word_count.pop(word)
        word_count[new_word] = word_freq
        for token_idx in range(len(new_word)):
            token = new_word[token_idx]
            if token == merged_token:
                if token_idx > 0:
                    prev_token = new_word[token_idx - 1]
                    add_token_pair_count(
                        token_pair_count,
                        token_pair_loc,
                        tuple([prev_token, merged_token]),
                        word_freq,
                        word_idx,
                    )
                if token_idx + 1 < len(new_word):
                    next_token = new_word[token_idx + 1]
                    add_token_pair_count(
                        token_pair_count,
                        token_pair_loc,
                        tuple([merged_token, next_token]),
                        word_freq,
                        word_idx,
                    )
    
    token_pair_count.pop(most_common_token_pair)
    token_pair_loc.pop(most_common_token_pair)

    # print(token_pair_count)

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    assert len(special_tokens) + 256 <= vocab_size, "vocba_size too small,"  \
        "cannot initialize the vocabulary"
    
    vocab = init_vocab(special_tokens)
    cur_vocab_size = len(vocab)
    merges = list[tuple[bytes, bytes]]()

    # word_count = dict[tuple[bytes], int]()
# 
    # chunks = split_special_token(input_data, special_tokens)
    # for chunk in chunks:
    #     pre_tokenization(
    #         chunk, 
    #         word_count,
    #     )
    num_processes = 16
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    p = Pool(num_processes)

    local_word_counts = p.starmap(
        pre_tokenization_worker, 
        [(input_path, boundaries[i], boundaries[i+1], special_tokens) 
        for i in range(len(boundaries)-1)]
    )

    word_count = merge_word_counts(local_word_counts)
    word_count = post_process_word_count(word_count)
    
    words, token_pair_count, token_pair_loc = (
        init_token_pair_count_and_loc(word_count)
    )

    # print("Words: ", words)
    # print("Word Count: ", word_count)
    # print(token_pair_count)

    while cur_vocab_size < vocab_size:
        most_common_token_pair = find_most_common_token_pair(token_pair_count)
        merges.append(most_common_token_pair)

        
        merge_most_common_token_pair(
            most_common_token_pair,
            words,
            word_count,
            token_pair_count,
            token_pair_loc,
        )
        
        vocab[cur_vocab_size] = b"".join(most_common_token_pair)
        cur_vocab_size += 1
    
    return vocab, merges


def train_bpe_from_file(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    #with open(input_path, "r") as f:
    #    data = f.read()
    return train_bpe(input_path, vocab_size, special_tokens)


if __name__ == "__main__":
    # local test cases
    # vocab, merges = train_bpe("low low low low low lower lower widest widest widest newest newest newest newest newest newest", 263, ["<|endoftext|>"])
    # print(vocab)
    # print(merges)
    start_time = time.time()
## 
    dataset = "TinyStoriesV2-GPT4-train"
    vocab, merges = train_bpe_from_file(
        f"data/{dataset}.txt",
        10000,
        ["<|endoftext|>"]
    )
    end_time = time.time()
    print("Total time:", end_time - start_time)

    with open(f"data/{dataset}-vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(f"data/{dataset}-merges.pkl", "wb") as f:
        pickle.dump(merges, f)
## 
    
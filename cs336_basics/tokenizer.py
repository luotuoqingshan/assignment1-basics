import regex as re 
import time
from typing import Any, BinaryIO
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Process, Queue

def pre_tokenization(
    input: str,
    word_count: dict[tuple[bytes], int],
):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for match in re.finditer(PAT, input):
        # count the appearances of words
        word = match.group().encode("utf-8")
        # we represent each word as a tuple of bytes
        word = tuple([word[i:i+1] for i in range(len(word))])
        word_count[word] = word_count.get(word, 0) + 1


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
    input_data: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    assert len(special_tokens) + 256 <= vocab_size, "vocba_size too small,"  \
        "cannot initialize the vocabulary"
    
    vocabulary = dict[int, bytes]()
    merges = list[tuple[bytes, bytes]]()

    # initialize the vocabulary
    for i in range(256):
        vocabulary[i] = i.to_bytes()
    
    # the vocabulary size right now
    cur_vocab_size = 256
    for special_token in special_tokens:
        vocabulary[cur_vocab_size] = special_token.encode("utf-8")
        cur_vocab_size += 1

    chunks = re.split(
        "|".join(re.escape(token) for token in special_tokens), 
        input_data
    )

    word_count = dict[tuple[bytes], int]()
    for chunk in chunks:
        pre_tokenization(
            chunk, 
            word_count,
        )
    
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
        
        vocabulary[cur_vocab_size] = b"".join(most_common_token_pair)
        cur_vocab_size += 1
    
    return vocabulary, merges


def train_bpe_from_file(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "r") as f:
        data = f.read()
    return train_bpe(data, vocab_size, special_tokens)


if __name__ == "__main__":
    # local test cases
    # vocab, merges = train_bpe("low low low low low lower lower widest widest widest newest newest newest newest newest newest", 263, ["<|endoftext|>"])
    # print(vocab)
    # print(merges)
    start_time = time.time()
#
    train_bpe_from_file(
        "data/TinyStoriesV2-GPT4-valid.txt",
        1000,
        ["<|endoftext|>"]
    )
#
    end_time = time.time()
    print("Total time:", end_time - start_time)
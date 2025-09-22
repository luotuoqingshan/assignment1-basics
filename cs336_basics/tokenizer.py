import regex as re 
from typing import Any


def pre_tokenization(
    input: str,
    words: list[tuple[bytes]],
    token_pair_count: dict[tuple[bytes, bytes], int],    
    token_pair_appear: dict[tuple[bytes, bytes], set[int]],
):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for match in re.finditer(PAT, input):
        word = match.group().encode("utf-8")
        for i in range(len(word) - 1):
            token_pair = tuple([word[i:i+1], word[i+1:i+2]])
            token_pair_count[token_pair] = token_pair_count.get(token_pair, 0) + 1
            
            if token_pair not in token_pair_appear:
                token_pair_appear[token_pair] = set[int]()
            token_pair_appear[token_pair].add(len(words))
        words.append(tuple([word[i:i+1] for i in range(len(word))]))            
    

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


def adjust(
    token_pair_count: dict[tuple[bytes, bytes], int],
    token_pair_appear: dict[tuple[bytes, bytes], set[int]],
    old_token_pair: tuple[bytes, bytes],
    new_token_pair: tuple[bytes, bytes],
    word_id: int
):
    # one token pair can appear multiple times in one word, 
    # it's not worthwhile to support deletion in token pair appearance dict
    token_pair_count[old_token_pair] -= 1
    token_pair_count[new_token_pair] = (
        token_pair_count.get(new_token_pair, 0) + 1
    )
    if new_token_pair not in token_pair_appear:
        token_pair_appear[new_token_pair] = set[int]()
    token_pair_appear[new_token_pair].add(word_id)


def add_token_pair_count(
    token_pair_count: dict[tuple[bytes, bytes], int],
    token_pair_appear: dict[tuple[bytes, bytes], set[int]],
    token_pair: tuple[bytes, bytes],
    word_idx: int,
):
    token_pair_count[token_pair] = (
        token_pair_count.get(token_pair, 0) + 1
    )
    if token_pair not in token_pair_appear:
        token_pair_appear[token_pair] = set[int]()
    token_pair_appear[token_pair].add(word_idx)


def sub_token_pair_count(
    token_pair_count: dict[tuple[bytes, bytes], int],
    token_pair: tuple[bytes, bytes],
):
    token_pair_count[token_pair] -= 1


def merge_most_common_token_pair(
    most_common_token_pair: tuple[bytes, bytes],
    words,
    token_pair_count: dict[tuple[bytes, bytes], int],
    token_pair_appear: dict[tuple[bytes, bytes], set[int]],
):
    token1, token2 = most_common_token_pair
    merged_token = b"".join([token1, token2])

    # iterate through all words containing this token pair
    for word_idx in token_pair_appear[most_common_token_pair]:
        word = words[word_idx]
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
                    sub_count(token_pair_count, old_prev_token_pair)
                if token_idx + 2 < len(word):
                    next_token = word[token_idx + 2]
                    old_next_token_pair = tuple([token2, next_token])
                    sub_count(token_pair_count, old_next_token_pair)
                token_idx += 2
            else:
                new_word.append(word[token_idx])
                token_idx += 1
        words[word_idx] = tuple(new_word)
        for token_idx in range(len(new_word)):
            token = new_word[token_idx]
            if token == merged_token:
                if token_idx > 0:
                    prev_token = new_word[token_idx - 1]
                    add_count(
                        token_pair_count,
                        token_pair_appear,
                        tuple([prev_token, merged_token]),
                        word_idx,
                    )
                if token_idx + 1 < len(new_word):
                    next_token = new_word[token_idx + 1]
                    add_count(
                        token_pair_count,
                        token_pair_appear,
                        tuple([merged_token, next_token]),
                        word_idx,
                    )
    
    token_pair_count.pop(most_common_token_pair)
    token_pair_appear.pop(most_common_token_pair)

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

    token_pair_count = dict[tuple[bytes, bytes], int]()
    token_pair_appear = dict[tuple[bytes, bytes], set[int]]()
    words = list[tuple[bytes]]()
    for chunk in chunks:
        pre_tokenization(
            chunk, 
            words,
            token_pair_count, 
            token_pair_appear,
        )

    while cur_vocab_size < vocab_size:
        most_common_token_pair = find_most_common_token_pair(token_pair_count)
        merges.append(most_common_token_pair)

        
        merge_most_common_token_pair(
            most_common_token_pair,
            words,
            token_pair_count,
            token_pair_appear,
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
    vocab, merges = train_bpe("low low low low low lower lower widest widest widest newest newest newest newest newest newest", 263, ["<|endoftext|>"])
    print(vocab)
    print(merges)
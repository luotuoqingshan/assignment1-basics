import regex as re 


def pre_tokenization(
    word_count: dict[tuple[bytes], int],
    input: str,
):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for match in re.finditer(PAT, input):
        string = match.group().encode("utf-8")
        token_bytes = tuple([string[i:i+1] for i in range(len(string))])
        word_count[token_bytes] = word_count.get(token_bytes, 0) + 1
    return word_count


def find_most_common_adj_token_pair(
    word_count: dict[tuple[bytes], int], # each word is represented as tuple of bytes
) -> tuple[bytes, bytes]:
    adj_token_pair_count = dict[tuple[bytes, bytes], int]()
    for word_tokens, count in word_count.items():
        for i in range(len(word_tokens) - 1):
            adj_token_pair = tuple([word_tokens[i], word_tokens[i+1]])
            adj_token_pair_count[adj_token_pair] = (
                adj_token_pair_count.get(adj_token_pair, 0) + count
            )

    most_common_adj_token_pair = (
        max(
            adj_token_pair_count, 
            # If two token pairs have the same frequency,
            # break the tie by merging the lexicographically greater pair
            key=lambda adj_token_pair: (
                adj_token_pair_count[adj_token_pair], 
                adj_token_pair[0],
                adj_token_pair[1],
            ),
        )
    )
    return most_common_adj_token_pair


def merge_most_common_adj_token_pair(
    word_count: dict[tuple[bytes], int],
    most_common_adj_token: tuple[bytes, bytes],
) -> dict[tuple[bytes], int]:
    new_word_count = dict[tuple[bytes], int]()
    token1, token2 = most_common_adj_token
    merged_token = b"".join([token1, token2])
    for word_tokens, count in word_count.items():
        idx = 0
        new_word_tokens = list[bytes]()
        while idx < len(word_tokens):
            if (
                idx + 1 < len(word_tokens)
                and word_tokens[idx] == token1 
                and word_tokens[idx+1] == token2
            ):
                # merge
                new_word_tokens.append(merged_token)
                idx += 2
            else:
                new_word_tokens.append(word_tokens[idx])
                idx += 1
        new_word_tokens = tuple(new_word_tokens)
        new_word_count[new_word_tokens] = new_word_count.get(new_word_tokens, 0) + count
    return new_word_count

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
        word_count = pre_tokenization(word_count, chunk)

    while cur_vocab_size < vocab_size:
        most_common_adj_token_pair = find_most_common_adj_token_pair(word_count)
        merges.append(most_common_adj_token_pair)
        word_count = merge_most_common_adj_token_pair(word_count, most_common_adj_token_pair)
        vocabulary[cur_vocab_size] = b"".join(most_common_adj_token_pair)
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
import pickle

def longest_token(
    vocab: dict[tuple[bytes], int],
):
    longest_token_id = max(
            vocab, 
            key = lambda id: (
                len(vocab[id]),
            )
        )
    return vocab[longest_token_id]

if __name__ == "__main__":
    dataset = "owt_train"
    with open(f"data/{dataset}-vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
        print(longest_token(vocab).decode("utf-8"))

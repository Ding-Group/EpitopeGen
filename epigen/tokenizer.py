# Standard library imports
import os
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def make_seq_for_tok(train_csv, n_tcr=2000000, n_epi=2000000):
    df = pd.read_csv(train_csv)
    tcrs = df['tcr'].sample(n_tcr).tolist()
    epitopes = df['epitope'].sample(n_epi).tolist()
    data = tcrs + epitopes
    outdir = str(Path(train_csv).parent)
    with open(f"{outdir}/seq_for_tok.txt", "w") as f:
        f.write("\n".join(data))
    print(f"{outdir}/seq_for_tok.txt")


def average_token_length(tokenizer, sequences):
    total_tokens = 0
    total_length = len(sequences)
    for seq in sequences:
        encoded = tokenizer.encode(seq)
        total_tokens += len(encoded.tokens)
    return total_tokens / total_length


def vocabulary_coverage(tokenizer, sequences):
    unique_tokens = set()
    for seq in sequences:
        encoded = tokenizer.encode(seq)
        unique_tokens.update(encoded.tokens)
    return len(unique_tokens) / tokenizer.get_vocab_size()


def normalize_decoded_sequence(decoded_sequence):
    # Remove leading spaces introduced by byte-level encoding
    normalized_sequence = decoded_sequence.replace('Ġ', '').replace('Ċ', '').replace(' ', '')
    return normalized_sequence


def reconstruction_accuracy(tokenizer, sequences):
    correct_reconstructions = 0
    total_sequences = len(sequences)
    for i, seq in enumerate(sequences):
        encoded = tokenizer.encode(seq)
        decoded = tokenizer.decode(encoded.ids)
        normalized_decoded = normalize_decoded_sequence(decoded)
        if seq == normalized_decoded:
            correct_reconstructions += 1
    return correct_reconstructions / total_sequences


def train_bpe_tokenizer(vocab_size, seq_for_tok, outdir):
    # Initialize a tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel()

    # Initialize a trainer
    trainer = BpeTrainer(
        special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]", "[EOS]"],
        vocab_size=vocab_size
    )

    # List of files to train on
    files = [seq_for_tok]

    # Train the tokenizer
    tokenizer.train(files, trainer)

    # Save the tokenizer
    tokenizer.save(f"{outdir}/tokenizer_vocab_{vocab_size}.json")
    print(f"{outdir}/tokenizer_vocab_{vocab_size}.json")


def evaluate_tokenizers(vocab_sizes, outdir, sequences):
    results = []
    tokenizers = []

    for vocab_size in vocab_sizes:
        tokenizer = Tokenizer.from_file(f"{outdir}/tokenizer_vocab_{vocab_size}.json")
        tokenizers.append(tokenizer)

        avg_length = average_token_length(tokenizer, sequences)
        vocab_coverage = vocabulary_coverage(tokenizer, sequences)
        accuracy = reconstruction_accuracy(tokenizer, sequences)

        results.append({
            "Vocab Size": vocab_size,
            "Avg Token Length": avg_length,
            "Vocab Coverage": vocab_coverage * 100,
            "Reconstruction Accuracy": accuracy * 100
        })

        print(f"Vocab Size: {vocab_size}")
        print(f"Average token length: {avg_length:.2f}")
        print(f"Vocabulary coverage: {vocab_coverage * 100:.2f}% unique tokens")
        print(f"Reconstruction accuracy: {accuracy * 100:.2f}%")
        print()

    results_df = pd.DataFrame(results)

    # Create plots
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    sns.lineplot(data=results_df, x="Vocab Size", y="Avg Token Length")
    plt.title("Average Token Length vs Vocab Size")
    plt.xscale("log")

    plt.subplot(2, 2, 2)
    sns.lineplot(data=results_df, x="Vocab Size", y="Vocab Coverage")
    plt.title("Vocabulary Coverage vs Vocab Size")
    plt.xscale("log")

    plt.subplot(2, 2, 3)
    sns.lineplot(data=results_df, x="Vocab Size", y="Reconstruction Accuracy")
    plt.title("Reconstruction Accuracy vs Vocab Size")
    plt.xscale("log")

    plt.subplot(2, 2, 4)
    sns.heatmap(results_df.set_index("Vocab Size"), annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Heatmap of Tokenizer Metrics")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "tokenizer_evaluation_results.png"))
    plt.close()

    return results_df

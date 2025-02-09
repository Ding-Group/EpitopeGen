# package/epigen/inference.py

import torch
from pathlib import Path
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle

class EpiGenPredictor:
    def __init__(
        self,
        checkpoint_path: str,
        model_path: str = "gpt2-small",
        tokenizer_path: str = "research/regaler/EpiGen",
        device: str = None,
        special_token_id: int = 400,
        batch_size: int = 32
    ):
        """Initialize EpiGen predictor.

        Args:
            checkpoint_path: Path to model checkpoint directory
            model_path: Base model path (default: gpt2-small)
            tokenizer_path: Path to tokenizer (default: regaler/EpiGen)
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            special_token_id: Special token ID used as separator (default: 400)
            batch_size: Batch size for inference (default: 32)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.special_token_id = special_token_id
        self.batch_size = batch_size

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Load config and model
        config_path = Path(tokenizer_path) / "GPT2Config_small.pkl"
        with open(config_path, "rb") as f:
            config = pickle.load(f)

        config.vocab_size = config.vocab_size + 1  # 400 -> 401
        self.model = AutoModelForCausalLM.from_config(config)

        # Load checkpoint
        weights = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(weights)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, tcr_sequences: list, num_predictions: int = 50,
                temperature: float = 0.7, top_k: int = 50, top_p: float = 0.95) -> pd.DataFrame:
        """Generate epitope predictions for TCR sequences.

        Args:
            tcr_sequences: List of TCR sequences
            num_predictions: Number of epitope predictions per TCR (default: 50)
            temperature: Sampling temperature (default: 0.7)
            top_k: Top-k sampling parameter (default: 50)
            top_p: Top-p sampling parameter (default: 0.95)

        Returns:
            DataFrame with TCR sequences and predicted epitopes
        """
        # Prepare input data
        input_data = pd.DataFrame({
            'text': tcr_sequences,
            'label': ['ZZZZZ'] * len(tcr_sequences)  # Placeholder label
        })

        return self.predict_from_df(input_data, num_predictions, temperature, top_k, top_p)

    def predict_from_df(self, df: pd.DataFrame, num_predictions: int = 50,
                       temperature: float = 0.7, top_k: int = 50, top_p: float = 0.95) -> pd.DataFrame:
        """Generate epitope predictions from DataFrame.

        Args:
            df: DataFrame with 'text' column containing TCR sequences
            num_predictions: Number of epitope predictions per TCR
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter

        Returns:
            DataFrame with predictions
        """
        predictions = []

        # Process in batches
        for i in range(0, len(df), self.batch_size):
            batch_df = df.iloc[i:i + self.batch_size]

            # Use the same format as in training (put special token at the end)
            batch_df['text_formatted'] = batch_df['text'] + "<|endoftext|>"

            # Tokenize
            encoded = self.tokenizer(
                batch_df['text_formatted'].tolist(),
                padding='max_length',
                max_length=12,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)['input_ids']

            # Generate predictions
            with torch.no_grad():
                generated_sequences = self.model.generate(
                    encoded,
                    num_return_sequences=num_predictions,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )

            # Process generated sequences
            for batch_idx in range(len(generated_sequences) // num_predictions):
                tcr = self._decode_tcr(
                    encoded[batch_idx].tolist()
                )[:-1]  # trim <|endoftext|>
                preds_for_tcr = [tcr]

                for seq_idx in range(num_predictions):
                    gen_seq = generated_sequences[batch_idx * num_predictions + seq_idx].tolist()
                    special_index = gen_seq.index(self.special_token_id)
                    epi = self.tokenizer.decode(
                        gen_seq[special_index:],
                        skip_special_tokens=True
                    ).replace(" ", "")

                    if not epi:  # avoid empty predictions
                        epi = "GILGFVFTLV"
                    preds_for_tcr.append(epi)

                predictions.append(preds_for_tcr)

        # Create results DataFrame
        results_df = pd.DataFrame(
            predictions,
            columns=['tcr'] + [f'pred_{i}' for i in range(num_predictions)]
        )

        return results_df

    def _trim_sequences(self, input_ids):
        """Trim sequences to include only TCR part."""
        trimmed_ids = []
        labels = []

        for sequence in input_ids:
            special_index = (sequence == self.special_token_id).nonzero(as_tuple=True)[0]
            if len(special_index) > 0:
                end_index = special_index[0] + 1
                seq = sequence[:end_index]
                if len(seq) <= 13:
                    trimmed_ids.append(seq)

                    label = sequence[end_index:]
                    zero_index = (label == 0).nonzero(as_tuple=True)[0]
                    if len(zero_index) > 0:
                        label = label[:zero_index[0]]
                    labels.append(label)

        return torch.stack(trimmed_ids), labels

    def _decode_tcr(self, input_id_tr):
        """Decode TCR sequence from input IDs."""
        try:
            ind_of_0 = input_id_tr.index(0)
        except ValueError:
            ind_of_0 = len(input_id_tr) - 1

        tcr = self.tokenizer.decode(input_id_tr[:ind_of_0])
        return tcr.replace(" ", "")

if __name__ == "__main__":
    # Initialize predictor
    predictor = EpiGenPredictor(
        checkpoint_path="path/to/checkpoint",  # pytorch_model.bin
        tokenizer_path="research/regaler/EpiGen"
    )

    # Predict from TCR sequences
    tcrs = ["CASIPEGGRETQYF", "CAVRATGTASKLTF"]
    results = predictor.predict(tcrs)
    print(results)

    # Or predict from DataFrame
    df = pd.DataFrame({"text": tcrs})
    results = predictor.predict_from_df(df)

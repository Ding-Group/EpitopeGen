# package/epitopegen/inference.py

import torch
import warnings
from pathlib import Path
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import pickle
import requests
import zipfile
import os
from tqdm import tqdm
from typing import List, Dict

from .config import (
    TOKENIZER_PATH,
    MODEL_CHECKPOINTS,
    ZENODO_URL,
    DEFAULT_CHECKPOINT,
    DEFAULT_CACHE_DIR
)

class EpitopeGenPredictor:
    ZENODO_URL = ZENODO_URL
    DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT
    AVAILABLE_CHECKPOINTS = MODEL_CHECKPOINTS

    def __init__(
        self,
        checkpoint_path: str = None,
        model_path: str = "gpt2-small",
        tokenizer_path: str = None,  # Changed default
        device: str = None,
        special_token_id: int = 2,
        batch_size: int = 32,
        cache_dir: str = None
    ):
        """Initialize epitopegen predictor.

        Args:
            checkpoint_path: Path to model checkpoint directory or checkpoint name (e.g., 'ckpt1')
            model_path: Base model path (default: gpt2-small)
            tokenizer_path: Path to tokenizer (default: package's built-in tokenizer)
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            special_token_id: Special token ID used as separator (default: 2)
            batch_size: Batch size for inference (default: 32)
            cache_dir: Directory to store downloaded checkpoints (default: ~/.cache/epitopegen)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.special_token_id = special_token_id
        self.batch_size = batch_size
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR

        # Use package's tokenizer by default
        tokenizer_path = tokenizer_path or TOKENIZER_PATH

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Handle checkpoint path
        if checkpoint_path is None or checkpoint_path in self.AVAILABLE_CHECKPOINTS:
            ckpt_name = checkpoint_path or "ckpt3"  # default to ckpt3
            checkpoint_path = self._ensure_checkpoint(ckpt_name)

        # Rest of initialization remains the same
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        config_path = Path(tokenizer_path) / "GPT2Config_small.pkl"
        with open(config_path, "rb") as f:
            config = pickle.load(f)

        config.vocab_size = config.vocab_size + 1
        self.model = AutoModelForCausalLM.from_config(config)

        weights = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(weights)
        self.model.to(self.device)
        self.model.eval()

    def _download_file(self, url: str, dest_path: str):
        """Download a file with progress bar."""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as f, tqdm(
            desc=f"Downloading checkpoints",
            total=total_size,
            unit='iB',
            unit_scale=True
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)

    def _ensure_checkpoint(self, checkpoint_name: str) -> str:
        """Ensure checkpoint is available, downloading if necessary."""
        checkpoint_path = os.path.join(self.cache_dir, self.AVAILABLE_CHECKPOINTS[checkpoint_name])

        if not os.path.exists(checkpoint_path):
            zip_path = os.path.join(self.cache_dir, "checkpoints.zip")

            # Download if not already present
            if not os.path.exists(zip_path):
                print(f"Downloading checkpoints from Zenodo...")
                self._download_file(self.ZENODO_URL, zip_path)

            # Extract
            print("Extracting checkpoints...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.cache_dir)

            # Verify checkpoint exists after extraction
            if not os.path.exists(checkpoint_path):
                raise RuntimeError(f"Checkpoint {checkpoint_path} not found after extraction")

        return checkpoint_path

    def _calculate_statistics(self, results_df: pd.DataFrame) -> dict:
        """Calculate useful statistics from prediction results."""
        stats = {
            "num_tcrs": len(results_df),
            "num_predictions_per_tcr": len(results_df.columns) - 1,  # -1 for tcr column
            "avg_tcr_length": results_df['tcr'].str.len().mean(),
            "avg_epitope_length": results_df.iloc[:, 1:].apply(lambda x: x.str.len().mean()).mean(),
            "unique_epitopes": len(pd.unique(results_df.iloc[:, 1:].values.ravel())),
            "most_common_epitopes": pd.Series(results_df.iloc[:, 1:].values.ravel()).value_counts().head(5).to_dict()
        }
        return stats

    def predict_all(
        self,
        tcr_sequences: list,
        output_dir: str,
        models: List[str] = None,  # List of checkpoint names to use
        top_k: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.95,
        use_attention_mask = False
    ) -> Dict[str, pd.DataFrame]:
        """Run predictions using multiple model checkpoints.

        Args:
            tcr_sequences: List of TCR sequences
            output_dir: Directory to save prediction results
            models: List of checkpoint names to use (default: all available)
            [other args same as predict()]

        Returns:
            Dictionary mapping checkpoint names to prediction DataFrames
        """
        models = models or list(self.AVAILABLE_CHECKPOINTS.keys())
        results = {}

        print(f"\n=== Running Multi-Model Predictions ===")
        print(f"• Processing {len(tcr_sequences)} TCRs")
        print(f"• Using {len(models)} model checkpoints")

        for ckpt_name in models:
            print(f"\nProcessing checkpoint: {ckpt_name}")

            # Load new checkpoint
            checkpoint_path = self._ensure_checkpoint(ckpt_name)
            weights = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(weights)

            # Run prediction
            output_path = Path(output_dir) / f"predictions_{ckpt_name}.csv"
            results[ckpt_name] = self.predict(
                tcr_sequences,
                output_path=output_path,
                top_k=top_k,
                temperature=temperature,
                top_p=top_p,
                use_attention_mask=use_attention_mask
            )

        return results

    def predict(self, tcr_sequences: list, output_path: str = None, top_k: int = 50,
                temperature: float = 0.7, top_p: float = 0.95, use_attention_mask=False) -> pd.DataFrame:
        """Generate epitope predictions for TCR sequences.

        Args:
            tcr_sequences: List of TCR sequences
            temperature: Sampling temperature (default: 0.7)
            top_k: Number of predictions per TCR (this is NOT top_k in top-k top-p sampling)
            top_p: Top-p sampling parameter (default: 0.95)

        Returns:
            DataFrame with TCR sequences and predicted epitopes
        """
        # Prepare input data
        input_data = pd.DataFrame({
            'text': tcr_sequences,
            'label': ['AAAAA'] * len(tcr_sequences)  # Placeholder label
        })

        return self.predict_from_df(input_data, output_path, top_k, temperature, top_p, use_attention_mask)

    def predict_from_df(self, df: pd.DataFrame, output_path: str = None, top_k: int = 50,
                       temperature: float = 0.7, top_p: float = 0.95, use_attention_mask=False) -> pd.DataFrame:
        """Generate epitope predictions from DataFrame.

        Args:
            df: DataFrame with 'text' column containing TCR sequences
            temperature: Sampling temperature
            top_k: Number of predictions per TCR
            top_p: Top-p sampling parameter
            use_attention_mask: whether or not to use the attention mask. Default to False to align with the training time condition.

        Returns:
            DataFrame with predictions
        """
        predictions = []

        # Process in batches
        for i in tqdm(range(0, len(df), self.batch_size)):
            batch_df = df.iloc[i:i + self.batch_size]

            # Tokenize
            tokenized = self.tokenizer(
                batch_df['text'].tolist(),
                padding='max_length',
                max_length=12,
                truncation=True,
                return_tensors='pt',
            ).to(self.device)

            encoded = tokenized['input_ids'].to(self.device)

            # Create a tensor of the special token with matching batch size
            special_token_tensor = torch.full((encoded.size(0), 1), self.special_token_id, device=encoded.device)

            # Concatenate both the encoded input and attention mask (optional) along dimension 1
            encoded = torch.cat([encoded, special_token_tensor], dim=1)

            # Create attention mask for special token (all ones since we want to attend to it)
            if use_attention_mask:
                attention_mask = tokenized['attention_mask'].to(self.device)
                special_token_mask = torch.ones((encoded.size(0), 1), device=encoded.device)
                attention_mask = torch.cat([attention_mask, special_token_mask], dim=1)
            else:
                attention_mask = None

            # Generate predictions
            logging.set_verbosity_info()
            with torch.no_grad():
                generated_sequences = self.model.generate(
                    encoded,
                    attention_mask=attention_mask,  # should NOT be set
                    pad_token_id=self.special_token_id,  # should NOT be set to 0
                    eos_token_id=self.special_token_id,
                    max_length=20,
                    num_return_sequences=top_k,
                    do_sample=True,
                    temperature=temperature,
                    top_k=50,
                    top_p=top_p,
                )


            # Process generated sequences
            for batch_idx in range(len(generated_sequences) // top_k):
                tcr = self._decode_tcr(
                    encoded[batch_idx].tolist()
                )
                preds_for_tcr = [tcr]

                for seq_idx in range(top_k):
                    gen_seq = generated_sequences[batch_idx * top_k + seq_idx].tolist()
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
            columns=['tcr'] + [f'pred_{i}' for i in range(top_k)]
        )

        # Calculate statistics
        stats = self._calculate_statistics(results_df)

        # Save predictions if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)
            stats['output_path'] = str(output_path)

        # Print informative summary
        print("\n=== epitopegen Prediction Summary ===")
        print(f"• Processed {stats['num_tcrs']} TCR sequences")
        print(f"• Generated {stats['num_predictions_per_tcr']} predictions per TCR")
        print(f"• Average TCR length: {stats['avg_tcr_length']:.1f}")
        print(f"• Average epitope length: {stats['avg_epitope_length']:.1f}")
        print(f"• Generated {stats['unique_epitopes']} unique epitopes")
        print("\n• Most common predicted epitopes:")
        for epi, count in stats['most_common_epitopes'].items():
            print(f"  - {epi}: {count} times")
        if output_path:
            print(f"\n• Results saved to: {output_path}")
        print("=============================")

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

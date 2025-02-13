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
    """A predictor class for generating epitopes from TCR sequences using a GPT-2 based model.

    This class handles model initialization, checkpoint management, and prediction generation
    for TCR-epitope pairs. It supports multiple checkpoints and automatic downloading of
    model weights from Zenodo.

    Attributes:
        ZENODO_URL: URL for downloading model checkpoints.
        DEFAULT_CHECKPOINT: Name of the default checkpoint to use.
        AVAILABLE_CHECKPOINTS: Dictionary mapping checkpoint names to their file paths.
    """
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
        """Initializes the epitope generator predictor with specified parameters.

        Args:
            checkpoint_path: Path to model checkpoint directory or checkpoint name (e.g., 'ckpt1').
                If None, uses the default checkpoint 'ckpt3'.
            model_path: Base model architecture path to use (default: "gpt2-small").
            tokenizer_path: Path to custom tokenizer. If None, uses the package's built-in tokenizer.
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detection).
            special_token_id: Special token ID used as separator between input and output sequences
                (default: 2).
            batch_size: Number of sequences to process simultaneously during inference
                (default: 32).
            cache_dir: Directory to store downloaded checkpoints. If None, uses
                ~/.cache/epitopegen.

        Note:
            The model will automatically download checkpoints from Zenodo if they're not
            found in the cache directory.
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
        """Downloads a file from a URL with a progress bar.

        Downloads a file using streaming to support large files while displaying
        a progress bar using tqdm.

        Args:
            url: The URL of the file to download.
            dest_path: The local path where the downloaded file will be saved.

        Note:
            Uses 1024 byte chunks for streaming and displays progress in
            binary units (iB).
        """
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
        """Ensures the specified model checkpoint is available locally.

        Checks if the checkpoint exists in the cache directory. If not, downloads
        the checkpoint archive from Zenodo and extracts it.

        Args:
            checkpoint_name: Name of the checkpoint to ensure (e.g., 'ckpt1').
                Must be a key in AVAILABLE_CHECKPOINTS.

        Returns:
            str: The full path to the local checkpoint file.

        Raises:
            RuntimeError: If the checkpoint file is not found after extraction.
            KeyError: If checkpoint_name is not in AVAILABLE_CHECKPOINTS.

        Note:
            Downloads are stored in a zip file named "checkpoints.zip" in the
            cache directory before extraction.
        """
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
        """Calculates useful statistics from prediction results.

        Args:
            results_df: A pandas DataFrame containing TCR sequences and their predicted epitopes.
                Expected to have a 'tcr' column and multiple prediction columns.

        Returns:
            dict: A dictionary containing the following statistics:
                - num_tcrs: Total number of TCR sequences analyzed
                - num_predictions_per_tcr: Number of predictions made per TCR
                - avg_tcr_length: Average length of TCR sequences
                - avg_epitope_length: Average length of predicted epitopes
                - unique_epitopes: Number of unique epitopes predicted
                - most_common_epitopes: Dictionary of the 5 most frequently predicted epitopes and their counts
        """
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
        """Runs predictions using multiple model checkpoints.

        Args:
            tcr_sequences: List of TCR amino acid sequences to generate predictions for.
            output_dir: Directory path where prediction results will be saved.
            models: List of checkpoint names to use. If None, uses all available checkpoints.
            top_k: Number of most likely tokens to consider for sampling (default: 50).
            temperature: Sampling temperature, higher values increase diversity (default: 0.7).
            top_p: Nucleus sampling probability threshold (default: 0.95).
            use_attention_mask: Whether to use attention masking during generation (default: False).

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping checkpoint names to prediction DataFrames.
                Each DataFrame contains the input TCR sequences and their predicted epitopes.
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
        """Generates epitope predictions for a list of TCR sequences.

        A convenience wrapper around predict_from_df that accepts a list of TCR sequences
        instead of a DataFrame.

        Args:
            tcr_sequences: List of TCR amino acid sequences to generate predictions for.
            output_path: Path to save the prediction results CSV. If None, results are
                only returned as DataFrame.
            top_k: Number of epitope predictions to generate per TCR sequence (default: 50).
                Note: This is not the top-k parameter used in top-k top-p sampling.
            temperature: Sampling temperature for generation. Higher values increase diversity
                (default: 0.7).
            top_p: Nucleus sampling probability threshold (default: 0.95).
            use_attention_mask: Whether to use attention masking during generation
                (default: False).

        Returns:
            pd.DataFrame: DataFrame containing TCR sequences and their predicted epitopes.
                Columns are ['tcr', 'pred_0', 'pred_1', ..., 'pred_{top_k-1}'].
        """
        # Prepare input data
        input_data = pd.DataFrame({
            'text': tcr_sequences,
            'label': ['AAAAA'] * len(tcr_sequences)  # Placeholder label
        })

        return self.predict_from_df(input_data, output_path, top_k, temperature, top_p, use_attention_mask)

    def predict_from_df(self, df: pd.DataFrame, output_path: str = None, top_k: int = 50,
                       temperature: float = 0.7, top_p: float = 0.95, use_attention_mask=False) -> pd.DataFrame:
        """Generates epitope predictions from a DataFrame containing TCR sequences.

        Main prediction method that processes TCR sequences in batches and generates
        multiple epitope predictions for each sequence using the loaded model.

        Args:
            df: DataFrame containing TCR sequences in a 'text' column.
            output_path: Path to save the prediction results CSV. If None, results are
                only returned as DataFrame.
            top_k: Number of epitope predictions to generate per TCR sequence (default: 50).
            temperature: Sampling temperature for text generation. Higher values increase
                diversity (default: 0.7).
            top_p: Nucleus sampling probability threshold (default: 0.95).
            use_attention_mask: Whether to use attention masking during generation. Defaults
                to False to match training conditions.

        Returns:
            pd.DataFrame: DataFrame containing TCR sequences and their predicted epitopes.
                Columns are ['tcr', 'pred_0', 'pred_1', ..., 'pred_{top_k-1}'].

        Note:
            The method processes sequences in batches defined by self.batch_size and
            prints a detailed summary of the predictions including statistics about
            TCR lengths, epitope lengths, and most common predictions.
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
        """Trims input sequences to contain only the TCR part and extracts labels.

        Processes tokenized sequences by finding the special token that separates TCR
        from epitope sequences. Trims sequences to include only the TCR part (up to
        and including the special token) and extracts the epitope labels.

        Args:
            input_ids: Tensor of tokenized sequences containing both TCR and epitope parts,
                separated by special_token_id.

        Returns:
            tuple:
                - torch.Tensor: Stack of trimmed sequences containing only TCR parts
                  (including special token).
                - list[torch.Tensor]: List of extracted epitope label sequences (everything
                  after special token, before padding).

        Note:
            Only sequences with length <= 13 tokens are included in the output.
            Padding tokens (0) are removed from the extracted labels.
        """
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
        """Decodes a tokenized TCR sequence back to amino acid sequence.

        Converts a list of token IDs back to a TCR amino acid sequence, handling
        padding tokens and removing spaces from the decoded sequence.

        Args:
            input_id_tr: List of token IDs representing a TCR sequence.

        Returns:
            str: Decoded TCR amino acid sequence with spaces removed.

        Note:
            Decoding stops at the first padding token (0) if present.
            All spaces are removed from the final sequence.
        """
        try:
            ind_of_0 = input_id_tr.index(0)
        except ValueError:
            ind_of_0 = len(input_id_tr) - 1

        tcr = self.tokenizer.decode(input_id_tr[:ind_of_0])
        return tcr.replace(" ", "")

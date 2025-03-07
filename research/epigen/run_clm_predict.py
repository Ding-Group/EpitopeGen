#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import pickle
os.environ['HF_HOME'] = "__pycache__"
import random
import numpy as np
from itertools import chain
from pathlib import Path
import pandas as pd

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.39.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv, txt or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv, txt or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--tokenized_dataset",
        type=str,
        default=None,
        help=(
            "Cache path to tokenized dataset"
        ),
    )
    parser.add_argument(
        "--inference_mode",
        action="store_true",
        help=(
            "Run inference: generate predictions on the val/test set and save them as a csv file."
        ),
    )
    parser.add_argument(
        "--gpt2_small",
        action="store_true",
        help=(
            "Use the small model. Need to pre-download the ckpt using the internet. "
        ),
    )
    parser.add_argument(
        "--inf_out_dir",
        type=str,
        help=(
            "The dir where the pred file will be saved. "
        ),
    )
    parser.add_argument(
        "--use_mhc",
        action="store_true",
        help=(
            "Whether or not to use the MHC information. False by default. "
        ),
    )
    parser.add_argument(
        "--special_token_id",
        type=int,
        default=400,
        help=(
            "400 after optimizing the vocab size. Before it was 30000. "
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    return args


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            repo_id = create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id
            # Clone repo locally
            repo = Repository(args.output_dir, clone_from=repo_id, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        # config = AutoConfig.from_pretrained(
        #     args.model_name_or_path,
        #     trust_remote_code=args.trust_remote_code,
        # )
        if args.gpt2_small:
            conf_pkl = f"{args.tokenizer_name}/GPT2Config_small.pkl"
        else:
            conf_pkl = f"{args.tokenizer_name}/GPT2Config.pkl"
        with open(conf_pkl, "rb") as f:
            config = pickle.load(f)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
        # tokenizer.pad_token = tokenizer.eos_token  # Add for padding
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.gpt2_small:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)
    else:
        if args.model_name_or_path:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                trust_remote_code=args.trust_remote_code,
            )
        else:
            logger.info("Training new model from scratch")
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        # Tokenize the text
        tokenized_inputs = tokenizer(examples['text'], padding='max_length', max_length=12, truncation=True)

        # Tokenize the labels. Since GPT-2 uses the same tokenizer for inputs and outputs,
        # we can use the tokenizer directly on the labels as well.
        # Note: If your labels are not textual or need different preprocessing, adjust accordingly.
        tokenized_labels = tokenizer(examples['label'], padding='max_length', max_length=12, truncation=True)

        # Return a dictionary containing both inputs and labels
        return {"input_ids": tokenized_inputs['input_ids'], "labels": tokenized_labels['input_ids']}

    if args.tokenized_dataset:
        print(f"Loading cached tokenized dataset from: {args.tokenized_dataset}")
        with open(args.tokenized_dataset, 'rb') as f:
            lm_datasets = pickle.load(f)
    else:
        # Tokenize the dataset
        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                # num_proc=args.preprocessing_num_workers,
                num_proc=1,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        if args.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > config.max_position_embeddings:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
                )
                block_size = min(1024, config.max_position_embeddings)
        else:
            if args.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(args.block_size, tokenizer.model_max_length)


        def proc_texts(examples):
            # This function processes the examples to format them correctly for conditional generation.
            # It concatenates input_ids and labels with a separator token.

            # Define a separator token. GPT-2 uses <|endoftext|> as the default EOS token.
            sep_token = tokenizer.eos_token_id

            processed_examples = {'input_ids': [], 'labels': []}
            for input_ids, labels in zip(examples['input_ids'], examples['labels']):
                # Concatenate input_ids and labels with the separator token in between
                concatenated_example = input_ids + [sep_token] + labels

                # For the labels, you might want to shift the tokens to ignore the input part during training
                # This is because you only want to calculate loss on the generated part, not the input part
                labels_shifted = [-100] * (len(input_ids) + 1) + labels  # +1 for the sep_token

                processed_examples['input_ids'].append(concatenated_example[:19])  # TODO: Remove hard-coding
                processed_examples['labels'].append(labels_shifted[:19])

            return processed_examples


        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with accelerator.main_process_first():
            lm_datasets = tokenized_datasets.map(
                # group_texts,
                proc_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                # desc=f"Grouping texts in chunks of {block_size}",
                desc="Structuring the datasets for conditional generation"
            )
        tok_desc = str(Path(args.validation_file).stem)
        with open(f"{args.output_dir}/lm_datasets_{tok_desc}.pkl", "wb") as f:
            pickle.dump(lm_datasets, f)
        print(f"{args.output_dir}/lm_datasets_{tok_desc}.pkl was saved. ")

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size, shuffle=True
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.eval()
        losses = []
        cor_pred_stack = 0
        total_pred_stack = 0
        if args.inference_mode:
            predictions = []  # seq info of tcr|mhc,epi
            num_of_gen = 50
        for step, batch in tqdm(enumerate(eval_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
            input_ids = batch['input_ids']

            if args.inference_mode:
                # Stack prediction info
                input_ids_trimmed, labels = trim_sequences(input_ids, special_token_id=args.special_token_id)  # <-> 400
                generated_sequences = model.generate(
                    input_ids_trimmed,
                    num_return_sequences=num_of_gen,
                    do_sample=True,  # Enable sampling to generate multiple sequences
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95
                )
                for batch_idx in range(len(generated_sequences) // num_of_gen):
                    input_id_tr = input_ids_trimmed[batch_idx]
                    label = labels[batch_idx]
                    if args.use_mhc:
                        tcr, mhc = decode_tcr_mhc(tokenizer, input_id_tr.tolist())
                        preds_for_tcr = [tcr, mhc]
                    else:
                        tcr = decode_tcr(tokenizer, input_id_tr.tolist())
                        preds_for_tcr = [tcr]
                    label = tokenizer.decode(label, skip_special_tokens=True).replace(" ", "")
                    preds_for_tcr.append(label)

                    for seq_idx in range(num_of_gen):
                        gen_seq = generated_sequences[batch_idx * num_of_gen + seq_idx].tolist()
                        special_index = gen_seq.index(args.special_token_id)  # Change to 30000 depending on the vocab size
                        epi = tokenizer.decode(gen_seq[special_index:], skip_special_tokens=True).replace(" ", "")
                        if epi == "":  # avoid being empty
                            epi = "GILGFVFTLV"
                        preds_for_tcr.append(epi)

                    predictions.append(preds_for_tcr)
            else:
                # Calculate the next token prediction accuracy
                pred = torch.argmax(outputs['logits'], dim=-1)
                correct_predictions, total_predictions = calculate_next_token_accuracy(
                    batch['input_ids'], torch.argmax(outputs['logits'], dim=-1), tokenizer, special_token_id=args.special_token_id)
                cor_pred_stack += correct_predictions
                total_pred_stack += total_predictions
                loss = outputs.loss
                losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

        if args.inference_mode:
            predictions = pad_predictions(predictions)
            if args.use_mhc:
                df = pd.DataFrame(predictions, columns=['tcr', 'mhc', 'epitope', 'pred'])
            else:
                try:
                    df = pd.DataFrame(predictions, columns=['tcr', 'epitope'] + [f'pred_{i}' for i in range(num_of_gen)])
                except:
                    breakpoint()
            Path('predictions').mkdir(parents=True, exist_ok=True)
            Path(f'predictions/{args.inf_out_dir}').mkdir(parents=True, exist_ok=True)
            outfile = os.path.basename(args.validation_file)
            df.to_csv(f'predictions/{args.inf_out_dir}/{outfile}', index=False)
        else:
            try:
                acc = float(cor_pred_stack) / total_pred_stack
            except:
                exit()
            loss_avg = sum(map(lambda x: torch.mean(x), losses[:-1])) / len(losses[:-1])
            losses = torch.cat(losses)
            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")
            print(f"[ckpt: {args.resume_from_checkpoint}] eval acc={acc}, loss={loss_avg}, perplexity={perplexity}")
            name = str(Path(args.validation_file).stem)
            real_epoch = str(Path(args.resume_from_checkpoint).stem).split("_")[1]
            Path('predictions').mkdir(parents=True, exist_ok=True)
            with open(f"predictions/GPT2_acc_{name}_{args.inf_out_dir}.txt", 'a') as f:
                f.write(f"{epoch},{acc},{loss_avg},{perplexity}\n")
        exit()

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)


def calculate_next_token_accuracy(input_ids, pred, tokenizer=None, special_token_id=30000):
    correct_predictions = 0
    total_predictions = 0

    for input_id_seq, pred_seq in zip(input_ids, pred):
        # Find the index of the special token
        special_token_index = (input_id_seq == special_token_id).nonzero(as_tuple=True)[0]

        if len(special_token_index) == 0:
            # If the special token is not found, skip this sequence
            continue

        start_index = special_token_index[0]  # Start comparing after the special token
        relevant_input_ids = input_id_seq[start_index + 1:]  # Shift by 1 to get next-token labels
        relevant_preds = pred_seq[start_index:]

        # Ensure equal length before comparison
        min_length = min(len(relevant_input_ids), len(relevant_preds))
        relevant_input_ids = relevant_input_ids[:min_length]
        relevant_preds = relevant_preds[:min_length]

        # Calculate accuracy (ignoring padding tokens)
        mask = relevant_input_ids != 0
        correct_predictions += (relevant_input_ids == relevant_preds)[mask].sum().item()
        total_predictions += mask.sum().item()

        inputs = tokenizer.decode(relevant_input_ids.tolist())

        preds = tokenizer.decode(relevant_preds.tolist())
        # print(inputs)
        # print(preds)

    # accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    # return accuracy
    return correct_predictions, total_predictions

# Example usage:
# input_ids and pred are tensors from your batch and model output
# accuracy = calculate_next_token_accuracy(batch['input_ids'], torch.argmax(outputs['logits'], dim=-1))
# print(f"Next-token prediction accuracy: {accuracy:.4f}")


def trim_sequences(input_ids, special_token_id=30000):
    # Separate TCR and epitope (GT) information
    trimmed_ids = []
    labels = []
    for sequence in input_ids:
        # Find the index of the special token and trim the sequence
        special_index = (sequence == special_token_id).nonzero(as_tuple=True)[0]
        if len(special_index) > 0:
            end_index = special_index[0] + 1  # Keep the special token in the sequence
            seq = sequence[:end_index]
            label = sequence[end_index:]
            if len(seq) <= 13:
                trimmed_ids.append(seq)
                # stack label info
                zero_index = (label == 0).nonzero(as_tuple=True)[0]
                if len(zero_index) > 0:
                    label = label[:zero_index[0]]
                labels.append(label)
    return torch.stack(trimmed_ids), labels

def decode_tcr_mhc(tokenizer, input_id_tr):
    """
    [6, 36, 3648, 90, 2253, 26, 435, 0, 0, 0, 0, 0, 30000] -> "CASSTSGSGITDTQYF", "YYATYRNIFTNTYENIAYGWTYDNYYTWAELAYLWHA"
    input_id_tr: list[int]
    """
    ind_of_26 = input_id_tr.index(26)
    tcr = tokenizer.decode(input_id_tr[:ind_of_26])
    if 0 in input_id_tr:
        ind_of_0 = input_id_tr.index(0)
        mhc = tokenizer.decode(input_id_tr[ind_of_26+1:ind_of_0])
    else:
        mhc = tokenizer.decode(input_id_tr[ind_of_26+1:-1])
    return tcr.replace(" ", ""), mhc.replace(" ", "")


def decode_tcr(tokenizer, input_id_tr):
    try:
        ind_of_0 = input_id_tr.index(0)
    except:
        ind_of_0 = len(input_id_tr) - 1
    tcr = tokenizer.decode(input_id_tr[:ind_of_0])
    return tcr.replace(" ", "")


def pad_predictions(predictions, target_length=52, fill_value=None):
    """
    Pad each list in predictions to the target length with the specified fill_value.

    Parameters
    ----------
    predictions : list of lists
        The list containing the predictions to be padded.
    target_length : int, optional
        The target length for each list (default is 52).
    fill_value : any, optional
        The value used to pad the lists (default is None).

    Returns
    -------
    list of lists
        The padded predictions.
    """
    padded_predictions = []
    for pred in predictions:
        if len(pred) < target_length:
            pred.extend([fill_value] * (target_length - len(pred)))
        padded_predictions.append(pred)
    return padded_predictions


if __name__ == "__main__":
    main()

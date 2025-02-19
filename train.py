
"""
Fine-tuning the library models for sequence to sequence.
"""

import logging
import os
import sys
import time
import datasets
import evaluate
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset
from filelock import FileLock
import torch


import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    GenerationConfig,
    BitsAndBytesConfig
)
# from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_offline_mode
from cfg import ModelArguments, DataTrainingArguments, AMTrainingArguments
from utils import get_preprocess_function_fn, get_compute_metrics_fn, postprocess_text
import random

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, AMTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.top_k:
        gen_conf = GenerationConfig.from_pretrained(model_args.model_name_or_path)
        gen_conf.do_sample = True
        gen_conf.top_k = 50
        training_args.generation_config = gen_conf

    # Setup logging
    fh = logging.FileHandler(os.path.join(training_args.output_dir, "output.log"), mode='w')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[ch,fh],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Disable default handler and enable propagation, to achieve logging both to file and stdout.
    # Not tested in distributed training.
    transformers.utils.logging.disable_default_handler()
    transformers.utils.logging.enable_propagation()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"output_dir: {training_args.output_dir}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    logger.info(f"Loading dataset:")
    logger.info(f"Training file: {data_args.train_file}")
    logger.info(f"Validation file: {data_args.validation_file}")
    logger.info(f"Test file: {data_args.test_file}")
    # Block the log of loading dataset, comment this line to see the log of dataset loading.
    ch.setLevel(logging.WARNING)
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )
    ch.setLevel(logging.INFO)
    logger.info(f"Dataset loading")

    # Load pretrained model and tokenizer
    #
    logger.info(f"Loading model {model_args.model_name_or_path} ...")
    # Block the log of model loading, comment this line to see the log of model loading.
    ch.setLevel(logging.WARNING)
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )
    # use T5Tokenizer instead of AutoTokenizer to avoid missing whitespace before added extra tokens problem
    # tokenizer = T5Tokenizer.from_pretrained(
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    if training_args.use_lora:
        if training_args.use_quant:
            compute_dtype = getattr(torch, "bfloat16")
            logger.info(f"Using compute dtype {compute_dtype}")
            bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                    # bnb_4bit_quant_storage=compute_dtype
            )
        else:
            bnb_config = None
        peft_config = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.05,
            r=16,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
            # target_modules= ['q', 'v', 'k', 'o', 'wo', 'wi_0', 'wi_1'],
            target_modules=training_args.lora_target_modules,
            # modules_to_save=["lm_head"],
        )
        # if training_args.deepspeed is None:
        #     device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
        # else:
        #     device_map = None
        device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            quantization_config=bnb_config,
            # device_map={"": 0}
            device_map=device_map
            # torch_dtype=compute_dtype
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
        )
    ch.setLevel(logging.INFO)
    logger.info(f"Model loaded.")

    num_added_toks = tokenizer.add_tokens(model_args.extra_tokens)
    logger.info(f"Added {num_added_toks} tokens:")
    logger.info(model_args.extra_tokens)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    
    if training_args.use_lora:
        # model.add_adapter(peft_config, adapter_name="adapter_4")
        # model.set_adapter("adapter_4")
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
            model = get_peft_model(model, peft_config)
        else:
            model = get_peft_model(model, peft_config)
        training_args.optim=training_args.lora_optim

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    if data_args.source_seq_column is None:
        source_seq_column = column_names[0]
        logger.info("No source sequence column specified, will use %s as source sequence column", source_seq_column)
    else:
        source_seq_column = data_args.source_seq_column
        if source_seq_column not in column_names:
            raise ValueError(
                f"--source_seq_column' value '{data_args.source_seq_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.target_seq_column is None:
        target_seq_column = column_names[1]
        logger.info("No target sequence column specified, will use %s as target sequence column", target_seq_column)
    else:
        target_seq_column = data_args.target_seq_column
        if target_seq_column not in column_names:
            raise ValueError(
                f"--target_seq_column' value '{data_args.target_seq_column}' needs to be one of: {', '.join(column_names)}"
            )

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    preprocess_function = get_preprocess_function_fn(data_args, tokenizer, source_seq_column, target_seq_column)

    selected_sample_ids = None
    if training_args.training_data_amount is not None:
        train_dataset = raw_datasets["train"]
        if 'sv' in data_args.train_file:
            data_args.max_train_samples = int(training_args.training_data_amount * len(train_dataset))
            selected_sample_ids = random.sample(range(len(train_dataset)), data_args.max_train_samples)
        elif 'mv' in data_args.train_file:
            assert len(raw_datasets["train"]) % 3 == 0
            orig_samples_num = len(raw_datasets["train"]) // 3
            max_orig_samples_num = int(training_args.training_data_amount * orig_samples_num)
            # data_args.max_train_samples = int(training_args.training_data_amount * orig_samples_num) * 3
            selected_sample_ids = random.sample(range(orig_samples_num), max_orig_samples_num)
            tmp_selected_sample_ids = [[i * 3, i * 3 + 1, i * 3 + 2] for i in selected_sample_ids]
            selected_sample_ids = [item for sublist in tmp_selected_sample_ids for item in sublist]

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if selected_sample_ids is not None:
            train_dataset = train_dataset.select(selected_sample_ids)
        elif data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    eval_dataset = None
    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
    
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    am_f1_metric = evaluate.load("evaluations/metrics/am_f1.py", cache_dir=model_args.cache_dir)
    compute_metrics = get_compute_metrics_fn(data_args, tokenizer, am_f1_metric, raw_datasets)

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )

    # evaluate on both val and test
    # make sure there are val and test datasets
    if training_args.do_train and data_args.validation_file is not None and data_args.test_file is not None:
        if training_args.eval_both_on_val_and_test:
            val_and_test_datasets = {
                "val": eval_dataset,
                "test": predict_dataset
            }
            eval_ds = val_and_test_datasets
        else:
            eval_ds = eval_dataset
    else:
        eval_ds = eval_dataset

    if training_args.using_adafactor_optim_manually:
        from transformers.optimization import Adafactor, AdafactorSchedule
        if training_args.adafactor_with_none_lr:
            optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
            lr_scheduler = AdafactorSchedule(optimizer)
        else:
            optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=training_args.learning_rate, \
                                eps=(1e-30, 1e-3), clip_threshold=1.0, decay_rate=-0.8, beta1=None, weight_decay=0.0,) # lr=1e-3
            lr_scheduler = None
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_ds if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            optimizers=(optimizer, lr_scheduler)
        )
    else:
        # Initialize our Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_ds if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # trainer.model.merge_and_unload()
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = predict_results.predictions
                # predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                # predictions = tokenizer.batch_decode(
                #     predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                # )
                predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                predictions = tokenizer.batch_decode(predictions, clean_up_tokenization_spaces=False)
                predictions = postprocess_text(predictions)

                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))
            
            for file_name, args in [('training_args.txt', training_args), 
                                    ('data_args.txt', data_args), 
                                    ('model_args.txt', model_args)]:
                with open(os.path.join(training_args.output_dir, file_name), 'w') as f:
                    print(args, file=f)

    return results


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")
from typing import Optional
from dataclasses import dataclass, field
from transformers import Seq2SeqTrainingArguments
import json


@dataclass
class AMTrainingArguments(Seq2SeqTrainingArguments):
    """
    Arguments for training AM models.
    """

    using_adafactor_optim_manually: bool = field(
        default=False, metadata={"help": "Whether to use adafactor optimizer manually."}
    )
    adafactor_with_none_lr: bool = field(
        default=False, metadata={"help": "Whether to use adafactor optimizer with None learning rate."}
    )
    eval_both_on_val_and_test: bool = field(
        default=False, metadata={"help": "Whether to evaluate on both val and test sets."}
    )
    top_k: bool = field(
        default=False, metadata={"help": "Whether to use top-k sampling."}
    )
    hop_config_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the hop config file."}
    )
    training_data_amount: Optional[float] = field(
        default=None, metadata={"help": "The amount of training data to use."}
    )
    use_lora: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use qlora."}
    )
    lora_target_modules: Optional[list[str]] = field(
        default_factory=lambda: [], metadata={"help": "The target modules for qlora."}
    )
    lora_optim: Optional[str] = field(
        default="paged_adamw_8bit", metadata={"help": "The optimizer for qlora."}
    )
    use_quant: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use quantization."}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    extra_tokens: Optional[str] = field(
        default="[]",
        metadata={
            "help": (
                "The list of extra tokens to add to the tokenizer."
            )
        },
    )
    use_flash_attention: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to use flash attention."
            )
        },
    )
    def __post_init__(self):
        self.extra_tokens = json.loads(self.extra_tokens)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    source_seq_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the source sequence."},
    )
    target_seq_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the target sequence."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=4096,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    block_size: Optional[int] = field(
        default=6000,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    view_type: Optional[str] = field(
        default="sv",
        metadata={
            "help": (
                "The type of task to train the model on. "
                )
        },
    )
    task_type: Optional[str] = field(
        default="AM",
        metadata={
            "help": (
                "The type of task to train the model on. "
            )
        },
    )
    voting_threshold: Optional[float] = field(
        default=1,
        metadata={
            "help": (
                "The threshold for voting. "
            )
        },
    )
    use_oracle_span: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to use oracle span. "
            )
        },
    )
    use_fuzzy_search: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to use fuzzy search. "
            )
        },
    )



    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training, validation, or test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length
        if self.view_type == "sv":
            self.voting_threshold = 1

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DefaultDataCollator

NUM_LABELS = 28
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128

EMOTION_NAMES = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]


class MultiLabelDataCollator(DefaultDataCollator):
    """Data collator that casts labels to float32 for BCEWithLogitsLoss."""

    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors=return_tensors)
        if "labels" in batch:
            batch["labels"] = batch["labels"].float()
        return batch


def load_goemotions(tokenizer_name=MODEL_NAME, max_length=MAX_LENGTH):
    """Load GoEmotions dataset, tokenize, and convert labels to multi-hot format.

    Uses the simplified config which has train/validation/test splits.
    Labels come as a list of label IDs and are converted to multi-hot vectors.

    Returns train, validation, test splits, emotion names, and a data collator.
    """
    dataset = load_dataset("go_emotions", "simplified")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def preprocess(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        labels = []
        for label_ids in examples["labels"]:
            multi_hot = [0.0] * NUM_LABELS
            for lid in label_ids:
                multi_hot[lid] = 1.0
            labels.append(multi_hot)
        tokenized["labels"] = labels
        return tokenized

    columns_to_remove = dataset["train"].column_names
    encoded = dataset.map(preprocess, batched=True, remove_columns=columns_to_remove)
    encoded.set_format("torch")

    collator = MultiLabelDataCollator()

    return encoded["train"], encoded["validation"], encoded["test"], EMOTION_NAMES, collator

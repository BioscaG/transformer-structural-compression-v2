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


def load_goemotions(tokenizer_name=MODEL_NAME, max_length=MAX_LENGTH, exclude_emotions=None):
    """Load GoEmotions dataset, tokenize, and convert labels to multi-hot format.

    Uses the simplified config which has train/validation/test splits.
    Labels come as a list of label IDs and are converted to multi-hot vectors.

    Args:
        exclude_emotions: list of emotion name strings to exclude (e.g. ['neutral', 'grief']).
            Examples whose only labels are excluded emotions are dropped entirely.

    Returns train, validation, test splits, emotion names, and a data collator.
    """
    dataset = load_dataset("go_emotions", "simplified")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    exclude_emotions = set(exclude_emotions or [])
    excluded_ids = {i for i, name in enumerate(EMOTION_NAMES) if name in exclude_emotions}
    # Build mapping: old label id -> new label id (None if excluded)
    old_to_new = {}
    new_id = 0
    for old_id, name in enumerate(EMOTION_NAMES):
        if old_id in excluded_ids:
            old_to_new[old_id] = None
        else:
            old_to_new[old_id] = new_id
            new_id += 1
    active_emotions = [name for name in EMOTION_NAMES if name not in exclude_emotions]
    num_labels = len(active_emotions)

    def preprocess(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        labels = []
        keep = []
        for label_ids in examples["labels"]:
            mapped = [old_to_new[lid] for lid in label_ids if old_to_new[lid] is not None]
            if not mapped:
                keep.append(False)
                labels.append([0.0] * num_labels)
            else:
                keep.append(True)
                multi_hot = [0.0] * num_labels
                for nid in mapped:
                    multi_hot[nid] = 1.0
                labels.append(multi_hot)
        tokenized["labels"] = labels
        tokenized["_keep"] = keep
        return tokenized

    columns_to_remove = dataset["train"].column_names
    encoded = dataset.map(preprocess, batched=True, remove_columns=columns_to_remove)
    # Filter out examples with no remaining labels
    encoded = encoded.filter(lambda x: x["_keep"])
    encoded = encoded.map(lambda x: x, remove_columns=["_keep"])
    encoded.set_format("torch")

    collator = MultiLabelDataCollator()

    return encoded["train"], encoded["validation"], encoded["test"], active_emotions, collator

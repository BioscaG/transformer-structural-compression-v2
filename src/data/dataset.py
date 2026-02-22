import torch
from datasets import load_dataset
from transformers import AutoTokenizer

NUM_LABELS = 28
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128


def load_goemotions(tokenizer_name=MODEL_NAME, max_length=MAX_LENGTH):
    """Load GoEmotions dataset, tokenize, and convert labels to multi-hot format.

    Returns train, validation, and test splits as HuggingFace Datasets
    compatible with the Trainer API.
    """
    dataset = load_dataset("go_emotions", "raw")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    emotion_columns = [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness",
        "optimism", "pride", "realization", "relief", "remorse",
        "sadness", "surprise", "neutral",
    ]

    def preprocess(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        labels = []
        for i in range(len(examples["text"])):
            multi_hot = [float(examples[col][i]) for col in emotion_columns]
            labels.append(multi_hot)
        tokenized["labels"] = labels
        return tokenized

    columns_to_remove = dataset["train"].column_names
    encoded = dataset.map(preprocess, batched=True, remove_columns=columns_to_remove)
    encoded.set_format("torch")

    return encoded["train"], encoded["validation"], encoded["test"], emotion_columns

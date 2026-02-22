from transformers import AutoModelForSequenceClassification

from src.data.dataset import MODEL_NAME, NUM_LABELS


def load_bert_classifier(model_name=MODEL_NAME, num_labels=NUM_LABELS):
    """Load BERT base with a multi-label classification head (768 -> num_labels).

    Uses BCEWithLogitsLoss internally via problem_type="multi_label_classification".
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )
    return model

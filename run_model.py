import json
import numpy as np
import torch
import wandb

from argparse import ArgumentParser
from datasets import load_metric
from lra_datasets import ImdbDataset
from lra_config import get_text_classification_config
from ml_collections import ConfigDict
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


# TASKS - IMDb als Standard
TASKS = {
    'imdb': ConfigDict(dict(dataset_fn=ImdbDataset, config_getter=get_text_classification_config)),
    # Weitere Datensätze hier
}

# GPU-Speichernutzung
def track_gpu_usage():
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        return max_memory
    return 0  # Kein GPU-Speicher verwendet

# Predictive Entropy berechnen
def compute_entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-9), axis=1).mean()

# Metriken definieren
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    probs = softmax(predictions, axis=1)
    preds = np.argmax(predictions, axis=1)

    # Standardmetriken
    accuracy = load_metric("accuracy").compute(predictions=predictions, references=labels)["accuracy"]
    precision = load_metric("precision").compute(predictions=predictions, references=labels)["precision"]
    recall = load_metric("recall").compute(predictions=predictions, references=labels)["recall"]
    f1 = load_metric("f1").compute(predictions=predictions, references=labels)["f1"]
    # auc = load_metric("roc_auc").compute(predictions=predictions, references=labels)["roc_auc"]
    auc = roc_auc_score(labels, probs[:, 1])  # Hier mit scikit-learn anstatt Hugging Face datasets (version-dependant)

    # Zusätzliche Metriken
    entropy = compute_entropy(probs)
    gpu_memory = track_gpu_usage()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "predictive_entropy": entropy,
        "gpu_memory_usage_mb": gpu_memory,
    }

# Ergebnisse speichern
def save_results(config, metrics, filename="results.json"):
    results = {
        "config": config.to_dict(),
        "metrics": metrics,
    }
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

# Main-Funktion
if __name__ == "__main__":
    # Argumente parsen
    parser = ArgumentParser()
    parser.add_argument("--task", default="imdb", choices=TASKS.keys(), help="Datensatz auswählen")
    args = parser.parse_args()

    # Passende Konfiguration laden
    task_name = args.task
    task = TASKS[task_name]
    config, _ = task.config_getter()
    train_dataset = task.dataset_fn(config, split="train")
    eval_dataset = task.dataset_fn(config, split="eval")

    # Modell und Tokenizer
    model_name = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Initialize W&B
    wandb.init(project="context-extension")

    # Testen, ob GPU verwendet wird -> cuda?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}. \n Starting training now...")

    # Trainingsargumente
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        logging_steps=100,
        fp16=True, # Optional --> Mixed-Precision beschleunigt Training
        load_best_model_at_end=True,
        report_to="wandb",  # Aktiviert Weights & Biases
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Training starten
    trainer.train()

    # Evaluation und Speicherung der Ergebnisse
    eval_metrics = trainer.evaluate()
    save_results(config, eval_metrics)

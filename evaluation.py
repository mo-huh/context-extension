import evaluate
import json
import numpy as np
import os
import torch
import wandb

from scipy.special import softmax


'''
Weitere Metriken:
- Trainingszeit: Longformer/BigBird brachen mehr Speicher,
    dafür Sliding Window mehr Berechnungszeit
- Speicherzeit-Komplexität (Memory vs. Time Tradeoff):
    Dies könnte in einem Trade-off-Diagramm (Memory vs. Time) visualisiert werden,
    um die Effizienz der Modelle zu vergleichen.
- Durchsatz (Throughput): Anzahl der verarbeiteten Token/Sekunde.
    Dies könnte bei der Skalierung auf größere Textlängen von Bedeutung sein.
'''


def track_gpu_usage():
    """Trackt die GPU-Speichernutzung."""
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        return max_memory
    return 0  # Kein GPU-Speicher verwendet


def compute_entropy(probs):
    """Berechnet die Predictive Entropy."""
    return -np.sum(probs * np.log(probs + 1e-9), axis=1).mean()


def compute_metrics(eval_pred):
    """Berechnet die Metriken für Hugging Face Trainer."""
    predictions, labels = eval_pred

    # Fehlerfindung
    if predictions is None:
        print(f"❌ Fehler: Predictions sind None.")
        predictions = np.zeros((len(labels), 2))  # Dummy-Werte (2 Klassen)

    # Fehlerfindung: Stelle sicher, dass die Logits die Form (batch_size, num_labels) haben
    if len(predictions.shape) > 2:
        print(f"⚠️ Fehler: Predictions haben die falsche Form: {predictions.shape}.")
        predictions = predictions.squeeze(1)  # Entferne unnötige Dimensionen (z. B. (batch_size, 1, num_labels))

    probs = softmax(predictions, axis=1)  # Wahrscheinlichkeiten aus den Logits: (batch_size, 2)
    preds = np.argmax(predictions, axis=1)  # Klassen 0/1 für jede Eingabe

    # Metriken laden
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    roc_auc_metric = evaluate.load("roc_auc")

    # Fehlerfindung: Sicherstellen, dass predictions und labels die gleiche Länge haben
    if len(preds) != len(labels):
        print(f"❌ Fehler: Länge von Predictions ({len(preds)}) passt nicht zu Labels ({len(labels)})")

    # Standardmetriken
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    precision = precision_metric.compute(predictions=preds, references=labels)["precision"]
    recall = recall_metric.compute(predictions=preds, references=labels)["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels)["f1"]
    auc = roc_auc_metric.compute(prediction_scores=probs[:, 1], references=labels)["roc_auc"]

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


def create_run_name(config):
    '''
    Unter diesem Namen werden die Modellergebnisse gespeichert.
    '''
    if config.model_name == "bert-base-uncased":
        model_name = "bert"
    else:
        model_name = config.model_name

    run_name = f"{model_name}-{config.dataset_name}-{config.max_length}"

    if config.context_technique:
        run_name += f"-{config.context_technique}"
    if config.stride and (config.context_technique == "sliding_window"):
        run_name += f"-{config.stride}"
    run_name += f"-{config.batch_size}/{config.grad_acc_steps}"

    return run_name, model_name


def save_results(config, metrics, filename="results.json"):
    """
    Speichert die Ergebnisse in einer JSON-Datei als Dictionary.
    Eindeutige Schlüssel basierend auf Modell, Kontextgröße und Datensatz:
    "model-dataset-max_length-context_technique-stride-batch_size/grad_acc_steps"
    Bsp.: "bert-imdb-128-sliding_window-64-8/2"
    """
    # Eindeutiger Schlüssel für den run
    run_name, model_name = create_run_name(config)
    
    # Config in ein serialisierbares Format umwandeln
    serializable_config = {key: value for key, value in config.to_dict().items() if isinstance(value, (int, float, list, dict))}

    # Ergebnis vorbereiten
    result_entry = {
        "model_name": model_name,
        "dataset_name": config.dataset_name,
        "config": serializable_config,
        "metrics": metrics,
    }

    # Prüfen, ob "results.json" existiert
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                results = json.load(f)  # Vorhandene Daten laden
                if not isinstance(results, dict):  # Datei hat ein unerwartetes Format
                    raise ValueError("JSON-Datei hat ein unerwartetes Format. Erwartet wird ein Dictionary.")
            except json.JSONDecodeError:
                results = {}  # Datei ist leer oder ungültig
    else:
        results = {}  # Datei existiert nicht

    # Ergebnis hinzufügen oder aktualisieren
    results[run_name] = result_entry

    # Ergebnisse speichern
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")


def init_wandb(config):
    """
    Initialisiert W&B mit Run-Details.
    """
    run_name, model_name = create_run_name(config)

    wandb.init(
        project="context-extension",
        name=run_name,
        config=config.to_dict(),
        tags=[model_name, config.dataset_name, str(config.max_length), config.context_technique or "no-technique", str(config.stride) or "no-stride", str(config.batch_size), str(config.grad_acc_steps)],
    )

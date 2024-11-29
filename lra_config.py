import torch
import ml_collections

from transformers import AutoTokenizer


def get_text_classification_config(num_labels=2):
    """
    Konfiguriert die Parameter für die Textklassifikation mit dem IMDb-Datensatz.
    """
    config = ml_collections.ConfigDict()

    # Trainingsparameter
    config.batch_size = 16  # (16 / 8) Batchgröße für das Training
    config.num_epochs = 3  # (3 / 2 / 1) Anzahl der Epochen für das Feintuning

    config.learning_rate = 2e-5  # Lernrate für Feintuning
    config.weight_decay = 1e-2  # Gewichtungsabnahme
    config.warmup_steps = 500  # Anzahl an Warmup-Schritten
    
    # Maximale Sequenzlänge
    config.max_length = 512  # Maximale Sequenzlänge für BERT

    # Pretrained Modell
    config.model_name = "bert-base-uncased"  # Pretrained Modell von Hugging Face
    config.tokenizer = AutoTokenizer.from_pretrained(config.model_name)  # Hugging Face Tokenizer

    return config, None

    # config.eval_frequency = 100  # Häufigkeit der Evaluation während des Trainings
    # config.total_train_samples = 500  # (-1 / 5000 / 1000 / 500) Maximale Anzahl an Trainingssamples
    # config.total_eval_samples = 200  # (-1 / 200) Nutze alle Samples im Evaluierungsdatensatz
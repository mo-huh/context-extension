import ml_collections

from transformers import AutoTokenizer

from model_wrappers import CustomBertForSequenceClassification


def get_text_classification_config(max_length=512, num_labels=2):
    """
    Konfiguriert die Parameter für die Textklassifikation mit dem IMDb-Datensatz.
    """
    config = ml_collections.ConfigDict()

    # Pretrained Modell
    config.model_name = "bert-base-uncased"  # Pretrained Modell von Hugging Face
    config.tokenizer = AutoTokenizer.from_pretrained(config.model_name)  # Hugging Face Tokenizer
    config.model = CustomBertForSequenceClassification.from_pretrained(config.model_name, num_labels=num_labels)

    # Maximale Sequenzlänge
    config.max_length = max_length  # Kontextgröße

    # Trainingsparameter
    config.num_epochs = 3  # Schnell: 1, Quali: 3 - (3 / 2 / 1)
    config.batch_size = 8  # Schnell: 2, Quali: 16/8 - (16 / 8 / 4 / 2)
    config.grad_acc_steps = 2  # Schnell: 1, Quali: 2 - (1 / 2 / 4) Verarbeitet Batch-Größe über mehrere Schritte, bevor Gradientenabstieg erfolgt

    # Train-/Eval- Data
    config.total_train_samples = -1  # (-1 / 5000 / 1000 / 500 / 100)
    config.total_eval_samples = -1  # (-1 / 200 / 10)

    # Hyperparameter
    config.learning_rate = 3e-5  # (2e-5 / 3e-5)
    config.weight_decay = 1e-2
    config.warmup_steps = 500  # Bei weniger Epochen auch 200 möglich

    return config, None

    # config.eval_frequency = 100  # Häufigkeit der Evaluation während des Trainings
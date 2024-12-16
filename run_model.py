import torch

from argparse import ArgumentParser
from ml_collections import ConfigDict
from transformers import TrainingArguments

from evaluation import compute_metrics, create_run_name, save_results, init_wandb
from lra_config import get_text_classification_config
from lra_datasets import ImdbDataset
from model_wrappers import CustomTrainer, CustomDataCollator

# Unterdrücke die Warnung, dass die Eingabesequenz bei Sliding Window länger als 512 Tokens ist
import transformers
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


# TASKS - IMDb als Standard
TASKS = {
    'imdb': ConfigDict(dict(dataset_fn=ImdbDataset, config_getter=get_text_classification_config)),
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", default="imdb", choices=TASKS.keys(), help="Datensatz auswählen")
    parser.add_argument("--max_length", type=int, default=512, help="Maximale Token-Länge für das Kontextfenster")
    parser.add_argument("--context_technique", default=None, choices=["sliding_window", None], help="Technik zur Kontextverlängerung")
    parser.add_argument("--stride", type=int, default=None, help="Stride für Sliding Window (Standard: max_length / 2)")
    args = parser.parse_args()

    # Passende Konfiguration laden
    task_name = args.task
    task = TASKS[task_name]
    config, _ = task.config_getter(args.max_length)
    
    # Config Parameter
    config.dataset_name = task_name
    config.context_technique = args.context_technique
    config.stride = args.stride if args.stride is not None else config.max_length // 2

    # Datensätze laden
    train_dataset = task.dataset_fn(config, split="train")
    eval_dataset = task.dataset_fn(config, split="eval")

    # Data Collator
    data_collator = CustomDataCollator(tokenizer=config.tokenizer)

    # Initialisiere W&B
    init_wandb(config)

    # Testen, ob GPU verwendet wird -> cuda?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}.\nTraining parameters: {create_run_name(config)}\nStarting training now...")

    # Trainingsargumente
    training_args = TrainingArguments(
        output_dir="./final_model",
        eval_strategy="epoch",  # ("epoch" / "steps")
        eval_steps=500,  # (1000 / 500 / 100) - (kann raus, wenn eval_strategy="epoch")
        save_strategy="no",  # ("epoch"/"no") Speichert Checkpoint pro Epoche
        save_total_limit=1,  # Begrenzt die Anzahl der gespeicherten Checkpoints
        logging_dir="./logs",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_accumulation_steps=config.grad_acc_steps,
        logging_steps=100,  # (100 / 500)
        fp16=True,  # Optional -> Mixed-Precision beschleunigt Training -> 16-Bit statt 32-Bit-Gleitkommazahlen
        load_best_model_at_end=False,  # (True / False)
        report_to="wandb",  # ("wandb" / "none")
    )

    # Trainer
    trainer = CustomTrainer(
        model=config.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        data_collator=data_collator,  # CustomDataCollator
        compute_metrics=compute_metrics,
    )

    # Training starten
    trainer.train()
    torch.cuda.empty_cache()  # Gibt durch Fragmentierung belegten Speicher frei

    # Evaluation und Speicherung der Ergebnisse
    print("Starting evaluation now...")
    eval_metrics = trainer.evaluate()
    save_results(config, eval_metrics)

import json
import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from lra_config import get_text_classification_config
from lra_datasets import ImdbDataset
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from ml_collections import ConfigDict

# Helper-Funktionen
def dict_to_device(inputs, device):
    return {key: inputs[key].to(device) for key in inputs}

def transformers_collator(sample_list):
    input_list, target_list = zip(*sample_list)
    keys = input_list[0].keys()
    # Concatenate inputs (e.g., input_ids, attention_mask)
    inputs = {k: torch.cat([inp[k].unsqueeze(0) if inp[k].dim() == 1 else inp[k] for inp in input_list], dim=0) for k in keys}
    # Concatenate targets --> Stelle korrekte Dimensionen sicher
    target = torch.cat([t.unsqueeze(0) if t.dim() == 0 else t for t in target_list], dim=0)
    return inputs, target

# Metriken
def accuracy_score(outp, target): # Loss, Accuracy, Prediction, Recall, F1, AUC (Misst Trennfähigkeit des Modells)
    assert len(outp.shape) == 2, "accuracy score must receive 2d output tensor"
    assert len(target.shape) == 1, "accuracy score must receive 1d target tensor"
    return (torch.argmax(outp, dim=-1) == target).sum().item() / len(target)

def predictive_entropy(probs): # Bewertet Robustheit des Modells --> Misst Unsicherheit in den Vorhersagen
    """Berechnet die prädiktive Entropie eines Modells."""
    return -np.sum(probs * np.log(probs + 1e-12), axis=1).mean()

def gpu_memory_usage(): # Misst max GPU-Speichernutzung
    """Erfasst den maximal genutzten GPU-Speicher."""
    torch.cuda.reset_peak_memory_stats()
    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # in MB
    return max_memory

# TASKS - IMDb als Standard
TASKS = {
    'imdb': ConfigDict(dict(dataset_fn=ImdbDataset, config_getter=get_text_classification_config)),
}

# Hauptfunktionen
def get_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    return model, tokenizer

def train(model, tokenizer, config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    train_dataset = ImdbDataset(config, split='train')
    eval_dataset = ImdbDataset(config, split='eval')

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=transformers_collator)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, collate_fn=transformers_collator)

    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    model.train()
    for epoch in range(1, config.num_epochs + 1):
        running_loss, running_acc = 0.0, 0.0
        for inputs, target in tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}"):
            inputs = dict_to_device(inputs, device)
            target = target.to(device)
            
            def validate_inputs(inputs):
                for key, value in inputs.items():
                    if value.dim() != 2:
                        raise ValueError(f"Input {key} has incorrect shape: {value.shape}, expected 2D (batch_size, seq_length)")
            validate_inputs(inputs)

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = F.cross_entropy(outputs.logits, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += accuracy_score(outputs.logits, target)

        print(f"Epoch {epoch}: Loss={running_loss/len(train_loader):.4f}, Accuracy={running_acc/len(train_loader):.4f}")
        evaluate(model, eval_loader, device)

def evaluate(model, eval_loader, device):
    model.eval()
    eval_loss, eval_acc = 0.0, 0.0
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats()  # Speicherstatistiken zurücksetzen
        for inputs, target in tqdm(eval_loader, desc="Evaluating"):
            inputs = dict_to_device(inputs, device)
            target = target.to(device)
            
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(target.cpu().numpy())

            loss = F.cross_entropy(outputs.logits, target)
            eval_loss += loss.item()
            eval_acc += accuracy_score(outputs.logits, target)

    # Metriken berechnen
    eval_loss /= len(eval_loader)
    eval_acc /= len(eval_loader)
    precision = precision_score(all_targets, all_preds, average='binary')
    recall = recall_score(all_targets, all_preds, average='binary')
    f1 = f1_score(all_targets, all_preds, average='binary')
    auc = roc_auc_score(all_targets, [p[1] for p in all_probs])
    entropy = predictive_entropy(np.array(all_probs))
    max_memory = gpu_memory_usage()

    print(f"Eval Loss={eval_loss:.4f}, Eval Accuracy={eval_acc:.4f}")
    print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}, AUC={auc:.4f}")
    print(f"Predictive Entropy={entropy:.4f}, Max GPU Memory Usage={max_memory:.2f} MB")
    model.train()

# Main
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", default="imdb", choices=TASKS.keys(),
                        help="Name des Datensatzes (Standard: imdb)")
    args = parser.parse_args()

    task_name = args.task
    task = TASKS[task_name]
    config, _ = task.config_getter()
    model, tokenizer = get_model(config)
    train(model, tokenizer, config)

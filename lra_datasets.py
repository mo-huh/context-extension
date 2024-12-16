import numpy as np
import torch

from glob import glob
from itertools import cycle

from context_techniques import SlidingWindowProcessor


class ImdbDataset:
    """
    IMDb-Dataset-Klasse stellt Trainings- & Evaluationsdaten bereit.
    Tokenisiert die Texte und bereitet sie für das Modell vor.
    """
    def __init__(self, config, split='train'):
        data_paths = {'train': "datasets/aclImdb/train", 'eval': "datasets/aclImdb/test"}
        split_path = data_paths[split]
        neg_path = split_path + "/neg"
        pos_path = split_path + "/pos"

        # Laden der negativen und positiven Beispiele
        neg_inputs = zip(glob(neg_path + "/*.txt"), cycle([0]))  # 0 für negative Labels
        pos_inputs = zip(glob(pos_path + "/*.txt"), cycle([1]))  # 1 für positive Labels
        self.data = np.random.permutation(list(neg_inputs) + list(pos_inputs))
        self.data = self.data[:config.total_train_samples] if config.total_train_samples > 0 else self.data  # Für Testing! (weniger samples)

        self.tokenizer = config.tokenizer
        self.max_length = config.max_length
        self.context_technique = config.context_technique
        
        # Sliding Window wird nur initialisiert, wenn es genutzt werden soll
        self.sliding_window = None
        if self.context_technique == "sliding_window":
            self.sliding_window = SlidingWindowProcessor(
                window_size=self.max_length,
                stride=config.stride,
                tokenizer=self.tokenizer
            )
            

    def __getitem__(self, i):
        """
        Holt ein tokenisiertes Beispiel mit Label aus dem Dataset und gibt ein Dictionary zurück.
        """
        data = self.data[i]
        with open(data[0], 'r', encoding='utf-8') as fo:
            source = fo.read()

        label = int(data[1])

        # Sliding Window anwenden, falls aktiviert
        if self.context_technique == "sliding_window" and self.sliding_window:
            encoded = self.tokenizer(
                source,
                truncation=False,  # Nicht abschneiden -> Sliding Window verarbeitet vollständige Sequenz
                padding=False,  # Kein Padding, Sliding Window macht das
                return_attention_mask=True,
                add_special_tokens=False  # Keine CLS- und SEP-Tokens hinzufügen -> Wird dann für jedes Segment gemacht
            )

            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            
            segments = self.sliding_window.process(input_ids, attention_mask, label)
            return segments

        # Ohne Sliding Window: Standard-Tokenisierung mit Padding
        inputs = self.tokenizer(
            source,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        # Label hinzufügen
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}  # Entfernt Batch-Dimension
        inputs["labels"] = torch.tensor(label, dtype=torch.long)  # Label hinzufügen
        return inputs


    def __len__(self):
        """
        Gibt die Länge des Datasets zurück.
        """
        return len(self.data)
import numpy as np
import torch
from glob import glob
from itertools import cycle

class ImdbDataset:
    """
    IMDb-Dataset-Klasse zur Verarbeitung von Trainings- und Evaluationsdaten.
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
        
        self.tokenizer = config.tokenizer
        self.max_length = config.max_length
        
    
    def __getitem__(self, i):
        """
        Holt ein tokenisiertes Beispiel mit Label aus dem Dataset und gibt ein Dictionary zurück.
        """
        data = self.data[i]
        with open(data[0], 'r', encoding='utf-8') as fo:
            source = fo.read()
        inputs = self.tokenizer(
            source,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        # Label hinzufügen
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}  # Entfernt Batch-Dimension
        inputs["labels"] = torch.tensor(int(data[1]), dtype=torch.long)  # Label hinzufügen
        return inputs


    def __len__(self):
        """
        Gibt die Länge des Datasets zurück.
        """
        return len(self.data)

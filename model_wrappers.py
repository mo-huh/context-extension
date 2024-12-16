import torch

from transformers import BertForSequenceClassification, DataCollatorWithPadding, PreTrainedTokenizerBase, Trainer
from typing import List, Dict


class CustomDataCollator:
    """
    Custom Data Collator stellt sicher, dass die Segmente von jeder Eingabesequenz separat verarbeitet werden.
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        if not isinstance(features[0], list):
            # Wenn keine Sliding-Window-Segmente vorhanden sind, verwende den Standard-Collator
            return self.data_collator(features)

        batch_input_ids, batch_attention_masks, batch_labels = [], [], []
        max_num_segments = max(len(feature) if isinstance(feature, list) else 1 for feature in features)

        for feature in features:
            if isinstance(feature, list):
                input_ids = torch.stack([seg['input_ids'] for seg in feature])
                attention_mask = torch.stack([seg['attention_mask'] for seg in feature])
                labels = feature[0]['labels']

                # Jedes Fenster sollte gleich viele Segmente haben (wenn nicht, dann mit Dummy-Segmenten auffüllen)
                if input_ids.size(0) < max_num_segments:
                    pad_size = max_num_segments - input_ids.size(0)
                    input_ids_pad = torch.zeros((pad_size, input_ids.shape[1]), dtype=torch.long)
                    attention_mask_pad = torch.zeros((pad_size, attention_mask.shape[1]), dtype=torch.long)
                    input_ids = torch.cat([input_ids, input_ids_pad], dim=0)
                    attention_mask = torch.cat([attention_mask, attention_mask_pad], dim=0)

                batch_input_ids.append(input_ids)
                batch_attention_masks.append(attention_mask)
                batch_labels.append(labels)
            else:
                batch_input_ids.append(feature['input_ids'].unsqueeze(0))
                batch_attention_masks.append(feature['attention_mask'].unsqueeze(0))
                batch_labels.append(feature['labels'])

        # Fehlerfindung
        if len(batch_input_ids) == 0:
            print(f"❌ Fehler: Batch ist leer! Prüfe die Segmente.")
        # print(f"📊 Input Shape: {torch.stack(batch_input_ids).shape}")

        batch = {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_masks),
            "labels": torch.stack(batch_labels)
        }
        return batch


class CustomBertForSequenceClassification(BertForSequenceClassification):
    """
    Custom BERT: Verarbeitet Sliding-Window-Segmente und mittelt deren Logits für jede Eingabesequenz.
    """
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if input_ids.dim() == 3:  # Sliding Window aktiv (batch_size, segments_size, segment_length)
            batch_size, num_segments, seq_len = input_ids.size()
            logits_list = []

            for i in range(batch_size):  # Für jede Eingabesequenz im Batch
                logits_per_segment = []
                for j in range(num_segments):  # Für jedes Segment einer Eingabesequenz
                    seg_ids = input_ids[i][j].unsqueeze(0)  # (1, seq_len)
                    seg_mask = attention_mask[i][j].unsqueeze(0)  # (1, seq_len)
                    
                    # Forward-Durchlauf für ein Segment
                    outputs = super().forward(input_ids=seg_ids, attention_mask=seg_mask, labels=None)
                    logits_per_segment.append(outputs.logits.squeeze(0))  # (num_labels,)
                
                # Mittelwert der Logits über alle Segmente
                logits = torch.stack(logits_per_segment).mean(dim=0)  # Mitteln der Logits
                logits_list.append(logits)
            
            logits = torch.stack(logits_list)  # (batch_size, num_labels)
        else:
            outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # Fehlerfindung
        if logits is None:
            print(f"❌ Fehler: Logits sind None. input_ids.size() = {input_ids.size()}")
        if logits.dim() != 2 or logits.size(1) != self.num_labels:
            print(f"❌ Fehler: Logits haben falsche Dimension. logits.size() = {logits.size()}")

        return {'loss': loss, 'logits': logits, 'hidden_states': outputs.hidden_states, 'attentions': outputs.attentions}


class CustomTrainer(Trainer):
    """
    Überschreibt den Standard-Prediction-Step, um die Logits zu loggen.
    """
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss, logits, labels = super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys
            )
            
            # Sicherstellen, dass Logits existieren und Tensor sind
            if isinstance(logits, tuple):
                logits = logits[0]  # Extrahiere das Tensor-Element aus dem Tuple

            # Fehlerfindung: Sicherstellen, dass logits ein Tensor ist und die Form korrekt ist
            if not isinstance(logits, torch.Tensor):
                print(f"❌ Fehler: Logits sind kein Tensor. Typ von Logits: {type(logits)}")
            if logits.dim() < 2:
                print(f"❌ Fehler: Logits haben die falsche Dimension: {logits.shape}")
            
            # Loggen der Logits zur Laufzeit
            if logits is not None and isinstance(logits, torch.Tensor):
                print(f"🔍 Logits Shape: {logits.shape}, Logits Sample: {logits[:2]}")
            else:
                print("⚠️ Warnung: Logits sind None oder kein Tensor")
            
            return loss, logits, labels

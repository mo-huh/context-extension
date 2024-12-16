import torch

from transformers import PreTrainedTokenizerBase
from typing import List, Dict


class SlidingWindowProcessor:
    """
    Führt Sliding Window auf einer Eingabesequenz durch.
    Jede Sequenz wird in (überlappende) Segmente unterteilt.
    """
    def __init__(self, window_size: int, stride: int, tokenizer: PreTrainedTokenizerBase):  # Wichtigste Sliding Window Parameter: window_size & stride
        self.window_size = min(window_size, 512) - 2  # -2, da Platz für Special Tokens sein muss: [CLS], [SEP]
        self.stride = stride
        self.tokenizer = tokenizer

    def process(self, input_ids: List[int], attention_mask: List[int], label: int) -> List[Dict[str, torch.Tensor]]:
        segments = []
        start = 0
        while start < len(input_ids):
            end = start + self.window_size
            segment_input_ids = input_ids[start:end]

            # Fehlerfindung
            if len(segment_input_ids) > self.window_size:
                print(f"❌ Fehler: Segment ist zu lang! len(segment_input_ids)={len(segment_input_ids)}, window_size={self.window_size}")

            encoded = self.tokenizer.build_inputs_with_special_tokens(segment_input_ids)
            segment_attention_mask = [1] * len(encoded)

            if len(encoded) < (self.window_size + 2):
                padding_length = (self.window_size + 2) - len(encoded)
                encoded += [0] * padding_length
                segment_attention_mask += [0] * padding_length

            # Fehlerfindung
            if len(encoded) != (self.window_size + 2):
                print(f"❌ Fehler: Segment hat falsche Länge! len(encoded) = {len(encoded)}, erwartet = {self.window_size + 2}")
                encoded = encoded[:self.window_size + 2]  # Truncate die Token

            segments.append({
                "input_ids": torch.tensor(encoded, dtype=torch.long),
                "attention_mask": torch.tensor(segment_attention_mask, dtype=torch.long),
                "labels": torch.tensor(label, dtype=torch.long),
            })
            start += self.stride
        return segments
import torch
print(torch.cuda.is_available())  # Gibt True aus, wenn GPU verfügbar
print(torch.cuda.device_count())  # Gibt die Anzahl der GPUs zurück

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

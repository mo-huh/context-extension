1. Entfernung irrelevanter Datensätze und Funktionen
   - CIFAR-10 und ListOps entfernt (in `lra_datasets.py`, `lra_config.py`, und `fetch_data.py`), sowie `ds_config.json`
   - Tokenizer-Funktionen und spezifische Konfigurationen für andere Datensätze gelöscht.
2. Anpassung der IMDb-Dataset-Verarbeitung
   - `ImdbDataset` optimiert:
      - Sicherstellung der korrekten Tokenisierung mit Hugging Face-Tokenizer.
      - Labels (0, 1) korrekt in numerische Tensoren (torch.tensor) umgewandelt.
3. Flexibilisierung des Datensatz-Downloads
   - `fetch_data.py`: Nur IMDb wird unterstützt und entpackt korrekt in `datasets/`.
4. Vereinfachung der Konfiguration
   - `lra_config.py`: Bereinigung auf IMDb-spezifische Parameter (batch_size, learning_rate, max_length).
   - Integration des Hugging Face-Tokenizers (`AutoTokenizer`).
5. Anpassung der Trainingslogik
   - Dynamische Eingabeverarbeitung: Sicherstellung der korrekten Batch-Dimensionen.
   - `get_model` unterstützt vortrainierte Modelle wie BERT (`bert-base-uncased`).
6. Erweiterung der Evaluationsmetriken
   - Hinzugefügt: Precision, Recall, F1-Score, AUC, prädiktive Entropie und GPU-Speicherverbrauch.
7. Vorbereitung für zukünftige Erweiterungen
   - Einfache Integration von Longformer, BigBird, ...
   - Möglichkeit Erweiterung für mehr Datensätze.
#!/bin/bash

# Erstelle das Verzeichnis für Datensätze
mkdir -p datasets
cd datasets

# Lade und entpacke den IMDb-Datensatz
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xvf aclImdb_v1.tar.gz

echo "IMDb-Datensatz erfolgreich heruntergeladen und entpackt."

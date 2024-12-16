import requests
import tarfile

from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm


def download_url(url, save_path, chunk_size=1024):
    """
    Lädt Datei von der URL herunter und speichert sie lokal.
    """
    r = requests.get(url, stream=True)
    total = int(r.headers.get('content-length', 0))
    with open(save_path, 'wb') as fd, tqdm(unit_scale=True, unit_divisor=chunk_size, total=total, desc="Downloading") as pbar:
        for chunk in r.iter_content(chunk_size=chunk_size):
            size = fd.write(chunk)
            pbar.update(size)
    return save_path

def extract_tar(archive, target_dir, subdir=None, mode="r:gz"):
    """
    Entpackt tar.gz-Archiv in "datasets".
    """
    with tarfile.open(archive, mode) as tar:
        if subdir is None:
            tar.extractall(path=target_dir)
        else:
            members = [tarinfo for tarinfo in tar.getmembers() if tarinfo.name.startswith(subdir)]
            tar.extractall(path=target_dir, members=members)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", choices=["imdb"], default="imdb", help="Name des Datensatzes zum Herunterladen")
    parser.add_argument("--dir", type=Path, default=Path("datasets"), help="Pfad zum Speichern der Datensätze")
    args = parser.parse_args()

    datasets = {
        "imdb": {"url": "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"},
    }

    task = args.task
    path_dir = args.dir
    path_dir.mkdir(parents=True, exist_ok=True)

    if task in datasets:
        archive_path = download_url(datasets[task]["url"], path_dir / f"{task}.tar.gz")
        extract_tar(archive_path, target_dir=path_dir)
        print(f"{task} Datensatz wurde erfolgreich heruntergeladen und entpackt.")
    else:
        raise ValueError(f"Unbekannter Datensatz: {task}")

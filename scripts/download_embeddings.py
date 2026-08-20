"""Download a pre-trained word embedding model for semantic similarity.

The model is stored in data/embeddings/ and loaded lazily by the app.

Usage:
    python scripts/download_embeddings.py [glove-wiki-gigaword-50|glove-twitter-25]

The default is glove-wiki-gigaword-50 (~66 MB), a compact GloVe model with
50-dimensional vectors that is good enough for a coursework project.

The archive is downloaded to the gensim-data cache directory and converted to
gensim's native KeyedVectors format in data/embeddings/.
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import urllib.request

import gensim.downloader as api

MODELS = {
    "glove-wiki-gigaword-50": "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz",
    "glove-twitter-25": "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-twitter-25/glove-twitter-25.gz",
}


def _cache_dir() -> str:
    return os.path.join(os.path.expanduser("~"), "gensim-data")


def download(url: str, dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"Already present: {dest}")
        return
    print(f"Downloading {url.split('/')[-1]} ...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request) as resp, open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
        os.replace(tmp, dest)
    except Exception as exc:  # network issues -> tell the user to retry
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            f"Download failed ({exc}). Please retry, or download the file "
            f"manually from {url} and place it at {dest}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        default="glove-wiki-gigaword-50",
        choices=sorted(MODELS),
        help="Pre-trained embedding model to download.",
    )
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "..", "data", "embeddings")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.model}.kv")

    if os.path.exists(out_path):
        print(f"Already present: {out_path}")
        return

    from gensim.models import KeyedVectors

    cache_file = os.path.join(_cache_dir(), args.model, f"{args.model}.gz")
    if not os.path.exists(cache_file):
        download(MODELS[args.model], cache_file)

    print("Converting to gensim format...")
    kv = KeyedVectors.load_word2vec_format(cache_file, binary=False)
    kv.save(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
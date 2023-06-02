# Semantic Hashing for clustering

This repository repurposes experiments from https://github.com/proger/vae to https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/

To prepare the data, run:

```
make
mkdir -p data
python ./remap.py --output-docword data/docword.three.txt --output-vocab data/vocab.three.txt docword.enron.txt.gz docword.kos.txt.gz docword.nips.txt.gz
gzip data/docword.three.txt
```

# Semantic Hashing for Clustering

[Semantic hashing VAEs](https://github.com/proger/vae) adapted to https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/

To prepare the data, run this. These commands join three datasets together and remaps vocabularies.

```bash
make
mkdir -p data
python ./remap.py --output-docword data/docword.three.txt \
  --output-vocab data/vocab.three.txt \
  docword.enron.txt.gz docword.kos.txt.gz docword.nips.txt.gz
gzip data/docword.three.txt
```

To run experiments do:

```bash
python ./scripts/train_semhash.py --latent_features 3 --vocab-size 1000  \
  --epochs 1000 --finetune-epochs 1500 --seed 20 \
  three20_1k.pt
python ./scripts/ari.py three20_1k.pt # adjusted rand index for clustering given by known labels
```

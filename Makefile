ROOT=https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/

all: docword.enron.txt.gz docword.kos.txt.gz docword.nips.txt.gz docword.nytimes.txt.gz readme.txt vocab.enron.txt vocab.kos.txt vocab.nips.txt vocab.nytimes.txt

docword.enron.txt.gz docword.kos.txt.gz docword.nips.txt.gz docword.nytimes.txt.gz readme.txt vocab.enron.txt vocab.kos.txt vocab.nips.txt vocab.nytimes.txt:
	wget $(ROOT)/$@


"""
The format of the docword.*.txt file is 3 header lines, followed by
NNZ triples:
---
D
W
NNZ
docID wordID count
docID wordID count
docID wordID count
docID wordID count


We are going to read all datasets and remap documents
"""

import argparse
import gzip



def read_vocab(file) -> dict:
    return {word.strip(): wordID for wordID, word in enumerate(file)}


parser = argparse.ArgumentParser()
parser.add_argument('--output-docword', type=argparse.FileType('w'))
parser.add_argument('--output-vocab', type=argparse.FileType('w'))
parser.add_argument('docword_x_txt', nargs='+', type=str)
args = parser.parse_args()

global_id_to_word = {}
global_word_to_id = {}

global_document_id = 0
global_doc_ids = {}
for dataset_id, docword_filename in enumerate(args.docword_x_txt, start=1):
    with open(docword_filename.replace('docword.', 'vocab.').replace('.gz', '')) as vocab_file:
        local_word_to_id = read_vocab(vocab_file)
        for word in local_word_to_id:
            if not word in global_word_to_id:
                global_id = len(global_id_to_word)
                global_id_to_word[global_id] = word
                global_word_to_id[word] = global_id
                print(word, file=args.output_vocab)

    keys = list(local_word_to_id.keys())
    docword_file = iter(gzip.open(docword_filename, 'r'))
    D = next(docword_file)
    W = next(docword_file)
    NNZ = next(docword_file)
    print(docword_filename, 'has', D.decode('utf-8').strip(), 'documents')
    for line in docword_file:
        docID, wordID, count = line.decode('utf-8').split()
        word = keys[int(wordID)-1]
        docID_ = global_doc_ids.get((dataset_id, docID))
        if docID_ is None:
            global_doc_ids[(dataset_id, docID)] = global_document_id
            docID = global_document_id
            global_document_id += 1
        else:
            docID = docID_
        print(docID, local_word_to_id[word], count, file=args.output_docword)

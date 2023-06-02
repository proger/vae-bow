import torch
import sklearn.metrics
import sys

logits = torch.load(sys.argv[1])['test_code_logits'].squeeze(1)

total = 39861+3430+1500
documents = torch.ones(total)
enron = documents[:39861] *0
kos = documents[39861:39861+3430]*1
nips = documents[39861+3430:39861+3430+1500]*2

print(logits)
print('true', torch.cat([enron,kos,nips]).numpy())
print('obsM', logits.argmin(-1).numpy())
print('randM', sklearn.metrics.adjusted_rand_score(torch.cat([enron,kos,nips]).numpy(), logits.argmin(-1).numpy()))
print('probs', logits.softmax(-1))
print('obsP', logits.softmax(-1).argmax(-1).numpy())
print('randP', sklearn.metrics.adjusted_rand_score(torch.cat([enron,kos,nips]).numpy(), logits.softmax(-1).argmax(-1).numpy()))
print('randL', sklearn.metrics.adjusted_rand_score(torch.cat([enron,kos,nips]).numpy(), logits.argmax(-1).numpy()))

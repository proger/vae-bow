import argparse

import torch
import torch.nn as nn

from va.semhash import SematicHasher
from va.newsgroups import make_datasets

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'mps')
parser.add_argument('--latent_features', type=int, default=64, help='number of latent Bernoulli variables (bits)')
parser.add_argument('--seed', type=int, default=10, help='random seed')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('--finetune-epochs', type=int, default=1500, help='number of finetuning epochs')
parser.add_argument('--vocab-size', type=int, default=25000, help='vocabulary size')
parser.add_argument('--encoder_hidden_features', type=int, nargs='*', default=(), help='list of hidden layer dimensions for the encoder, one for each layer')
parser.add_argument('--decoder_hidden_features', type=int, nargs='*', default=(), help='list of hidden layer dimensions for the decoder, one for each layer')
parser.add_argument('output_checkpoint', type=str, help='save the checkpoint')
args = parser.parse_args()

print(vars(args))

torch.manual_seed(args.seed)

model = SematicHasher(
    vocab_size=args.vocab_size,
    latent_features=args.latent_features,
    encoder_hidden_features=args.encoder_hidden_features,
    decoder_hidden_features=args.decoder_hidden_features
)
model.to(args.device)

print(model)

optimizer = model.make_optimizer(lr=1e-3)
train = make_datasets(vocab_size=args.vocab_size)

print('train documents', len(train))

train_loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=False, num_workers=0)


def train(model, train_loader, optimizer, reinforce_steps=1):
    device = next(model.parameters()).device

    train_loss = 0.
    for word_counts, in train_loader:
        word_counts = word_counts.to(device)

        code_logits = model(word_counts)

        for _ in range(reinforce_steps):
            loss = model.disarm_elbo(code_logits, word_counts).mean()
            (loss / reinforce_steps).backward(retain_graph=True)

        with torch.no_grad():
            train_loss += model.sample_elbo(code_logits, word_counts).mean().item()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    train_loss /= len(train_loader)    
    return train_loss

train_losses = []

for epoch in range(args.epochs):
    train_loss = train(model, train_loader, optimizer)
    print(f'Epoch {epoch}: train_loss={train_loss:.2f}', flush=True)
    train_losses.append(train_loss)

# 500 more epochs with 10 REINFORCE steps
for epoch in range(args.epochs, args.finetune_epochs):
    train_loss = train(model, train_loader, optimizer, reinforce_steps=10)
    print(f'Epoch {epoch}: train_loss={train_loss:.2f}', flush=True)
    train_losses.append(train_loss)

@torch.inference_mode()
def test(test_loader):
    device = next(model.parameters()).device

    all_code_logits = []
    for i, (word_counts, ) in enumerate(test_loader):
        word_counts = word_counts.to(device)

        code_logits = model(word_counts)
        all_code_logits.append(code_logits.cpu())
    return torch.stack(all_code_logits)
    

print('Best train loss', min(train_losses))

torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'train_losses': train_losses,
    'test_code_logits': test(test_loader),
    'args': vars(args),
}, args.output_checkpoint)

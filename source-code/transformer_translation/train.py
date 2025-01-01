import math
import time
import torch
from torch import nn, optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
from transformer import Transformer
from bleu import get_bleu
from config import *


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class Dataset:
    def __init__(self, huggingface_dataset_name='bentrevett/multi30k', batch_size=32):

        dataset = load_dataset(huggingface_dataset_name, data_dir="")

        if huggingface_dataset_name == 'bentrevett/multi30k':
            self.tokenizer_src = BertTokenizer.from_pretrained("bert-base-cased")
            self.tokenizer_trg = BertTokenizer.from_pretrained("bert-base-german-cased")

            train_dataset = dataset['train'].map(lambda x: self.tokenizer_src(x['en'], padding='max_length', max_length=max_len, truncation=True), batched=True).select_columns('input_ids').rename_columns({'input_ids': 'src'})
            train_dataset_trg = dataset['train'].map(lambda x: self.tokenizer_trg(x['de'],  padding='max_length', max_length=max_len, truncation=True), batched=True).select_columns('input_ids').rename_columns({'input_ids': 'trg'})
            train_dataset = train_dataset.add_column('trg', train_dataset_trg['trg']) 

            val_dataset = dataset['validation'].map(lambda x: self.tokenizer_src(x['en'], padding='max_length', max_length=max_len, truncation=True), batched=True).select_columns('input_ids').rename_columns({'input_ids': 'src'})
            val_dataset_trg = dataset['validation'].map(lambda x: self.tokenizer_trg(x['de'],  padding='max_length', max_length=max_len, truncation=True), batched=True).select_columns('input_ids').rename_columns({'input_ids': 'trg'})
            val_dataset = val_dataset.add_column('trg', val_dataset_trg['trg'])

            test_dataset = dataset['test'].map(lambda x: self.tokenizer_src(x['en'], padding='max_length', max_length=max_len, truncation=True), batched=True).select_columns('input_ids').rename_columns({'input_ids': 'src'})
            test_dataset_trg = dataset['test'].map(lambda x: self.tokenizer_trg(x['de'], padding='max_length', max_length=max_len, truncation=True), batched=True).select_columns('input_ids').rename_columns({'input_ids': 'trg'})
            test_dataset = test_dataset.add_column('trg', test_dataset_trg['trg'])

            train_dataset.set_format("torch")
            val_dataset.set_format("torch")
            test_dataset.set_format("torch")

            self.train_loader = DataLoader(train_dataset, batch_size=batch_size)
            self.val_loader   = DataLoader(val_dataset, batch_size=batch_size)
            self.test_loader  = DataLoader(test_dataset, batch_size=batch_size)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



dataset = Dataset(huggingface_dataset_name=huggingface_dataset_name, batch_size=batch_size)
src_pad_idx = dataset.tokenizer_src.convert_tokens_to_ids(dataset.tokenizer_src.pad_token)
trg_pad_idx = dataset.tokenizer_trg.convert_tokens_to_ids(dataset.tokenizer_trg.pad_token)
trg_sos_idx = dataset.tokenizer_trg.convert_tokens_to_ids(dataset.tokenizer_trg.sep_token)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=len(dataset.tokenizer_src),
                    dec_voc_size=len(dataset.tokenizer_trg),
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch['src'].to(device)
        trg = batch['trg'].to(device)

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch['src'].to(device)
            trg = batch['trg'].to(device)

            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = dataset.tokenizer_trg.convert_ids_to_tokens(batch['trg'][j], skip_special_tokens=True)
                    output_words = output[j].max(dim=1)[1]
                    output_words = dataset.tokenizer_trg.convert_ids_to_tokens(output_words, skip_special_tokens=True)
                    # print((output_words, trg_words))
                    bleu = get_bleu(hypotheses=output_words, reference=trg_words)
                    total_bleu.append(bleu)
                except Exception as e: 
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu


def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, dataset.train_loader, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, dataset.val_loader, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            
        if step % 50 == 0:
            torch.save(model, 'saved/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
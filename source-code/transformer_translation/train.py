import math
import time
import torch
from torch import nn, optim
from data import Dataset
from datasets import load_dataset
from transformer import Transformer


MAX_LEN = 256
HID_DIM = 512
ENC_LAYERS = 6
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
BATCH_SIZE = 128
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_EPOCHS = 20
CLIP = 1
LEARNING_RATE = 0.0001


dataset = Dataset(huggingface_dataset_name='bentrevett/multi30k', max_len=MAX_LEN)
dataset.make_iterator()

src_pad_idx = dataset.tokenizer_src.pad_token_id
trg_pad_idx = dataset.tokenizer_trg.pad_token_id
trg_bos_idx = dataset.tokenizer_trg.cls_token_id
trg_eos_idx = dataset.tokenizer_trg.sep_token_id

INPUT_DIM = len(dataset.tokenizer_src)
OUTPUT_DIM = len(dataset.tokenizer_trg)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = Transformer(INPUT_DIM, OUTPUT_DIM, src_pad_idx, trg_pad_idx, max_length=MAX_LEN, hid_dim=HID_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch['src'].to(DEVICE)
        trg = batch['trg'].to(DEVICE)
        # print(dataset.tokenizer_trg.decode(trg[0]))
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
                
        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch['src'].to(DEVICE)
            trg = batch['trg'].to(DEVICE)

            output = model(src, trg[:, :-1])
            
            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def run(total_epoch):
    best_valid_loss = float('inf')
    for epoch in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, dataset.train_loader, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, dataset.val_loader, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'saved/model-{epoch}-{valid_loss}.pt')

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

if __name__ == '__main__':
    run(total_epoch=N_EPOCHS)
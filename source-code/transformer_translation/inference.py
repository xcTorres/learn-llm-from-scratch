import math
import torch
from torch import nn
from transformer import Transformer
from transformers import BertTokenizer
from data import Dataset
from train import evaluate

MAX_LEN = 256
HID_DIM = 512
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = Dataset(huggingface_dataset_name='bentrevett/multi30k', max_len=MAX_LEN)
dataset.make_iterator() 

INPUT_DIM = len(dataset.tokenizer_src)
OUTPUT_DIM = len(dataset.tokenizer_trg)
src_pad_idx = dataset.tokenizer_src.pad_token_id
trg_pad_idx = dataset.tokenizer_trg.pad_token_id

model = Transformer(INPUT_DIM, OUTPUT_DIM, src_pad_idx, trg_pad_idx, max_length=MAX_LEN, hid_dim=HID_DIM).to(DEVICE)
model.load_state_dict(torch.load('saved/model-19-1.4270771145820618.pt'))


tokenizer_src = dataset.tokenizer_src
tokenizer_trg = dataset.tokenizer_trg
source_lang = 'en'
target_lang  = 'de'

def transformer_predict(src, model, tokenizer_src, tokenizer_trg):
    source_indices = tokenizer_src(src, padding='max_length', max_length=MAX_LEN, truncation=True).input_ids
    source_indices = torch.LongTensor(source_indices).unsqueeze(0).to(DEVICE)
    trg_indices = model.predict(source_indices, tokenizer_trg.cls_token_id, tokenizer_trg.sep_token_id)
    return tokenizer_trg.decode(trg_indices.squeeze(), skip_special_tokens=True)

# Set the model to evaluation mode (important for inference)
model.eval()
# criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
# test_loss  = evaluate(model, dataset.test_loader, criterion, verbose=True)
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

for i in range(50):
    translation = dataset.dataset['train'][i]
    source_raw = translation[source_lang]
    target_raw = translation[target_lang]
    print('source: ', source_raw)
    print('target: ', target_raw)
    output = transformer_predict(source_raw, model, tokenizer_src, tokenizer_trg)
    print('predicted: ', output)
    print('*'*50)
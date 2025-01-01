import torch
from transformers import BertTokenizer
from config import *

model = torch.load('saved/model-4.3619584441185.pt', map_location=device)

# Set the model to evaluation mode (important for inference)
model.eval()
input = 'A woman with a pink purse is sitting on a bench.'

tokenizer_src = BertTokenizer.from_pretrained("bert-base-cased")
tokenizer_trg = BertTokenizer.from_pretrained("bert-base-german-cased")

# Tokenize the input sentence
input_tokens = tokenizer_src(input, padding='max_length', max_length=max_len, truncation=True)  # Custom tokenization logic
input_tensor = torch.tensor(input_tokens['input_ids']).unsqueeze(0).to(device)
input_mask = torch.tensor(input_tokens['attention_mask']).unsqueeze(0).to(device)

# Forward pass through the encoder
with torch.no_grad():
    enc_src = model.encoder(input_tensor, input_mask)  # Encoder output (hidden states)

    start_token = torch.tensor([tokenizer_trg.cls_token_id]).unsqueeze(0).to(device)
    generated_sequence = start_token


    # Decoder generates tokens autoregressively
    for _ in range(max_len):
        trg_mask = model.make_trg_mask(generated_sequence)
        decoder_output = model.decoder(generated_sequence, enc_src, trg_mask, input_mask)
        next_token_logits = decoder_output[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
        
        generated_sequence = torch.cat([generated_sequence, next_token], dim=1)
        
        # Stop if end token is generated
        if next_token == tokenizer_trg.sep_token_id:
            break

    # Decode the generated sequence into text
    translated_text = tokenizer_trg.convert_ids_to_tokens(generated_sequence.squeeze().cpu().numpy(), skip_special_tokens=True)  # Custom decoding logic
    print("Translated Text:", ' '.join(translated_text))
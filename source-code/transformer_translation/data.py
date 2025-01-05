from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset

class Dataset:
    def __init__(self, huggingface_dataset_name, max_len):

        self.dataset = load_dataset(huggingface_dataset_name, data_dir="")
        self.train_loader = None
        self.val_loader   = None
        self.test_loader  = None

        if huggingface_dataset_name == 'bentrevett/multi30k':
            self.source_lang, self.target_lang = 'en', 'de'
            self.tokenizer_src = BertTokenizer.from_pretrained("bert-base-cased")
            self.tokenizer_trg = BertTokenizer.from_pretrained("bert-base-cased")

            self.dataset = self.dataset.map(lambda x: self.encode(x, self.source_lang, self.target_lang, self.tokenizer_src, self.tokenizer_trg, max_len))
            self.dataset.set_format("torch")
            
    def encode(self, x, source_lang, target_lang, tokenizer_src, tokenizer_trg, max_len):
        """
        Encode the raw text into numerical token ids. Creating two new fields
        ``source_ids`` and ``target_ids``.
        Also append the init token and prepend eos token to the sentence.
        """
        source_raw = x[source_lang]
        target_raw = x[target_lang]
        source_encoded = tokenizer_src(source_raw, padding='max_length', max_length=max_len, truncation=True).input_ids
        target_encoded = tokenizer_trg(target_raw, padding='max_length', max_length=max_len, truncation=True).input_ids
        x['src'] = source_encoded
        x['trg'] = target_encoded
        return x
    
    def make_iterator(self, batch_size=128):
        self.train_loader = DataLoader(self.dataset['train'].remove_columns([self.source_lang, self.target_lang]), batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(self.dataset['validation'].remove_columns([self.source_lang, self.target_lang]), batch_size=batch_size)
        self.test_loader  = DataLoader(self.dataset['test'].remove_columns([self.source_lang, self.target_lang]), batch_size=batch_size)

        # for i, batch in enumerate(self.train_loader):
        #     print(batch['src'][0])
        #     print(batch['trg'][0])
        #     print(self.tokenizer_src.decode(batch['src'][0]))
        #     print(self.tokenizer_trg.decode(batch['trg'][0]))

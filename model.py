from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import pandas as pd
# from apex import amp


# BertModel
class GPReviewDataset_val(Dataset):
    def __init__(self, reviews, tokenizer, max_len):
        self.reviews = reviews
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }


def create_data_loader_val(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset_val(
        reviews=df.x.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size
    )


class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("rubert-base-cased-sentence")
        self.batchNorm1d_1 = nn.BatchNorm1d(self.bert.config.hidden_size)
        self.drop_1 = nn.Dropout(p=0.3)
        self.Linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.SiLU = nn.SiLU(self.bert.config.hidden_size)
        self.batchNorm1d_2 = nn.BatchNorm1d(self.bert.config.hidden_size)
        self.drop_2 = nn.Dropout(p=0.1)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        batchNorm1d_1 = self.batchNorm1d_1(pooled_output)
        drop = self.drop_1(batchNorm1d_1)
        Linear = self.Linear(drop)
        SiLU = self.SiLU(Linear)
        batchNorm1d_2 = self.batchNorm1d_2(SiLU)
        output = self.drop_2(batchNorm1d_2)
        out = self.out(output)
        return out


def get_predictions(model, device, data_loader, threshold=0):
    model.eval()
    predictions = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"]#.to(device)
            attention_mask = d["attention_mask"]#.to(device)
            y_pred = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            if threshold:
                predictions.extend(y_pred.flatten() > 0.5)
            else:
                predictions.extend(y_pred.flatten())
    predictions = torch.stack(predictions).cpu()
    return predictions


def model_loader_bert(path, device, fp16):
    tokenizer = BertTokenizer.from_pretrained('rubert-base-cased-sentence')
    model = SentimentClassifier()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    # model.to(device)
    # if fp16:
    #     model = amp.initialize(model, opt_level="O1")

    return model, tokenizer


def model_predictions_bert(model, tokenizer, device, news, data, threshold=0):
    list_candidates = np.array(data)
    list_length = list_candidates.shape[0]
    list_candidates = pd.DataFrame({'x': list_candidates})
    list_candidates['x'] = list_candidates['x'] + '\t' + news
    test_data_loader = create_data_loader_val(list_candidates, tokenizer, 512, list_length)
    predictions = get_predictions(model, device, test_data_loader, threshold)
    return predictions


# GPT3
MAX_LENGTH = int(10000)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def model_loader_gpt(path, device, fp16):
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    model = GPT2LMHeadModel.from_pretrained(path)
    model#.to(device)
    # if fp16:
    #     model = amp.initialize(model, opt_level="O1")
    return model, tokenizer


def model_predictions_gpt(model, tokenizer, device, max_len, prompts, temperature=0.9, k=50, p=0.95,
                          repetition_penalty=1.0, num_return_sequences=3, num_beams=1, early_stopping=True):
    length = adjust_length_to_model(max_len, max_sequence_length=model.config.max_position_embeddings)
    generated_sequences = []

    for prompt_text in tqdm(prompts):
        encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt  #.to(device)

        output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=length + len(encoded_prompt[0]),
            temperature=temperature,
            top_k=k,
            top_p=p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
            early_stopping=early_stopping
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences.append([])

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            # print("ruGPT:".format(generated_sequence_idx + 1))
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                # prompt_text +
                text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
            )

            generated_sequences[-1].append(total_sequence.split('<pad>', maxsplit=1)[0])
    return generated_sequences[0]

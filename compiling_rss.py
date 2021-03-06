import datetime
import argparse
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from apex import amp
import re

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_names = ['Bert_ru_512_title.bin', 'Bert_ru_512_descr.bin']


def KMP(t, s):
    x = -1
    v = [0] * len(s)
    for i in range(1, len(s)):
        k = v[i - 1]
        while k > 0 and s[k] != s[i]:
            k = v[k - 1]
        if s[k] == s[i]:
            k = k + 1
        v[i] = k
    k = 0
    for i in range(len(t)):
        while k > 0 and s[k] != t[i]:
            k = v[k - 1]
        if s[k] == t[i]:
            k = k + 1
        if k == len(s):
            x = i - len(s) + 1 + 1
            break
    return x


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
        self.bert = BertModel.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
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


def get_predictions(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            y_pred = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # predictions.extend(y_pred.flatten() > 0.5)
            predictions.extend(y_pred.flatten())
    predictions = torch.stack(predictions).cpu()
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rss_input", default=None, type=str, required=True,
                        help="Path to rss list file")
    parser.add_argument("--path_out", default=None, type=str, required=True,
                        help="Path to output rss news")
    parser.add_argument("--len_data", default=5, type=int)
    parser.add_argument("--max_len", default=256, type=int)
    parser.add_argument("--definition_of_quality", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="fp16")

    args = parser.parse_args()

    rss_input = args.rss_input
    Path_out = args.path_out
    len_data = args.len_data + 3

    names = []
    for root, dirs, files in os.walk(rss_input):
        for f in files:
            if f[-3:] == 'rss':
                names.append(f'{root}/{f}')

    data = []
    for name in names:
        with open(name, 'r', encoding='UTF-8') as f:
            data.append(f.readlines())

    for i in range(len(names)):
        a = names[i].split("/")[-1][:-4]
        if KMP(a, 'title') != -1:
            names[i] = [f'<container type="title" model="{a}">\n', 0]  # [container, № model]
        elif KMP(a, 'text') != -1:
            names[i] = [f'<container type="text" model="{a}">\n', 2]  # not
        else:
            names[i] = [f'<container type="description" model="{a}">\n', 1]

    with open(f'{rss_input}/rss_links.txt', 'r', encoding='UTF-8') as f:
        links = f.readlines()

    date_now = datetime.datetime.now().date()

    filter_if = 0
    filter_bert = 0

    with open(f'{Path_out}/compiled_rss_{date_now}.xml', 'w', encoding='UTF-8') as f:
        f.write(
            '<rss xmlns:yandex="http://news.yandex.ru" xmlns:media="http://search.yahoo.com/mrss/" version="2.0">\n'
            '<channel>\n'
            '<title>FEFU GPT-3</title>\n'
            '<description>Title and description</description>\n'
            '<link>...</link>\n'
        )

        if args.definition_of_quality:
            model = []
            tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence')
            for model_name in model_names:
                model.append(SentimentClassifier())
                model[-1].load_state_dict(torch.load(f'{rss_input}/{model_name}'))
                model[-1].to(device)
                if args.fp16:
                    model[-1] = amp.initialize(model[-1], opt_level="O1")

        for l_d in tqdm(range(0, len(data[0]), len_data)):
            f.write('<item>\n')
            f.write(f'<link>{links[(l_d + 1) // len_data][:-1]}</link>\n')
            f.write(f'<yandex:full-text>{data[0][l_d][:-4]}</yandex:full-text>\n')

            nums = re.findall(r'\d*\.\d+|\d+', data[0][l_d][:-4])  # проверка чисел
            nums = set([float(i) for i in nums])

            for d in range(len(data)):
                f.write(names[d][0])  # container
                if args.definition_of_quality:
                    list_candidates = []
                    for l_d_item in range(l_d + 2, l_d + len_data - 1):  # +2 компенсация 'title', -1 компенсация '----'
                        list_candidates.append(data[d][l_d_item][:-1])
                    list_candidates = np.array(list_candidates)
                    list_length = list_candidates.shape[0]
                    list_candidates = pd.DataFrame({'x': list_candidates})
                    list_candidates['x'] = list_candidates['x'] + '\t' + data[0][l_d][:-4]
                    test_data_loader = create_data_loader_val(list_candidates, tokenizer, args.max_len, list_length)
                    y_pred = get_predictions(model[names[d][1]], test_data_loader)
                a = 0
                for l_d_item in range(l_d + 2, l_d + len_data - 1):  # +2 компенсация 'title', -1 компенсация '----'
                    fl_nums = 1

                    if not re.search('=>', data[d][l_d_item][:-1]) and len(data[d][l_d_item][:-1]) > 15 and max(
                            [len(i) for i in data[d][l_d_item][:-1].split(' ')]) < 0.6 * len(data[d][l_d_item][:-1]) and \
                            y_pred[a] > 6:
                        # проверка на => \ на длину \ длина самого большого слова не должна быть больше 60% общей дляны
                        # \ и бал классификатора

                        if re.search(r"\d*\.\d+|\d+", data[d][l_d_item][:-1]) and fl_nums:
                            generation_numbers = set([float(i) for i in re.findall(r'\d*\.\d+|\d+', data[d][l_d_item][
                                                                                                    :-1])])  # получение уникальных чисел из генерации
                            for i in generation_numbers:
                                if i not in nums:
                                    fl_nums = 0
                                    break
                    else:
                        fl_nums = 0

                    if args.definition_of_quality:

                        if not fl_nums:
                            filter_if += 1
                        if y_pred[a] < 6:
                            filter_bert += 1

                        f.write(f'<p>({bool(fl_nums)}) {y_pred[a]}: {data[d][l_d_item][:-1]}</p>\n')
                        a += 1
                    else:
                        f.write(f'<p>{data[d][l_d_item][:-1]}</p>\n')
                f.write(f'</container>\n')
            f.write('</item>\n')

        f.write(
            '</channel>\n'
            '</rss>'
        )

        print(f'filter_if: {filter_if} \t filter_bert: {filter_bert}')


if __name__ == "__main__":
    main()

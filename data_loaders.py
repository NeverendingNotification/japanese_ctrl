import random

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler

import sentencepiece as sp



class PadCsvSpDataset(Dataset):
    def __init__(self, csv_files, ctrl_codes, col_name, sp_file, 
                max_doc_length=1000, max_token_size=256, pad_code=3, eos_code=2):
        self.sp = sp.SentencePieceProcessor()
        self.sp.load(sp_file)
        
        codes = []
        texts = []
        codes_ids = []
        ids_codes = {}
        for code, csv_file in zip(ctrl_codes, csv_files):
            df = pd.read_csv(csv_file)
            code_id = self.sp.piece_to_id(code)

            codes_ids.append(code_id)
            ids_codes[code_id] = code
            print(csv_file, " 制御コード ", code_id)
            docs = [] 
            for doc in df[col_name].values:
                if isinstance(doc, float):
                    continue
                docs.append(doc)
            texts.extend(docs)
            codes.extend([code_id]*len(docs))
        self.texts = texts
        self.codes = codes
        self.ctrl_codes = ctrl_codes
        self.max_len = max_token_size
        self.n_len = len(self.codes)
        self.codes_ids = codes_ids
        self.ids_codes = ids_codes
        self.pad_code = pad_code
        self.eos_code = eos_code
        
    def __len__(self):
        return self.n_len

    def __getitem__(self, idx):
        text = self.sp.encode_as_ids(self.texts[idx])
        if len(text) < self.max_len - 1:
            text.append(self.eos_code)

        n_text = len(text)
        if n_text > self.max_len - 1:
            text = text[:self.max_len-1]
        else:
            text = text + [self.pad_code] * (self.max_len - 1 - n_text)

        text = [self.codes[idx]] + text
        return torch.Tensor(text).long()


class CsvSpDataset(Dataset):
    def __init__(self, csv_files, ctrl_codes, col_name, sp_file, 
                max_doc_length=1000, max_token_size=256, pad_code=2):
        self.sp = sp.SentencePieceProcessor()
        self.sp.load(sp_file)
        
        codes = []
        texts = []
        codes_ids = []
        ids_codes = {}
        for code, csv_file in zip(ctrl_codes, csv_files):
            df = pd.read_csv(csv_file)
            code_id = self.sp.piece_to_id(code)

            codes_ids.append(code_id)
            ids_codes[code_id] = code
            print(csv_file, " 制御コード ", code_id)
            docs = [] 
            for doc in df[col_name].values:
                if isinstance(doc, float):
                    continue
                n_split = (len(doc) + max_doc_length - 1) // max_doc_length
                doc_piece = []
                for n in range(n_split):
                    doc_piece.append(doc[n*max_doc_length:(n+1)*max_doc_length])
                docs.extend(doc_piece)
            texts.extend(docs)
            codes.extend([code_id]*len(docs))
        self.texts = texts
        self.codes = codes
        self.ctrl_codes = ctrl_codes
        self.max_len = max_token_size
        self.n_len = len(self.codes)
        self.codes_ids = codes_ids
        self.ids_codes = ids_codes
        self.pad_code = pad_code
        
    def __len__(self):
        return self.n_len

    def __getitem__(self, idx):
        text = self.sp.encode_as_ids(self.texts[idx])
        n_text = len(text)
        if n_text > self.max_len - 1:
            i = random.randint(0,  n_text - self.max_len + 1)
            text = text[i:i+self.max_len-1]
        else:
            text = text + [self.pad_code] * (self.max_len - 1 - n_text)
        text = [self.codes[idx]] + text
        return torch.Tensor(text).long()


class CodeBalancedBatchSampler(BatchSampler):
    def __init__(self, codes, codes_ids, weights, batch_size):
        codes = np.array(codes)
        id_lists = {}

        n_dats = []
        n_batchs = []
        w_batch_sizes = []
        for weight, code_id in zip(weights, codes_ids):
            targ = codes == code_id
            id_lists[code_id] = np.where(targ)[0]
            n_data = np.sum(targ)
            n_dats.append(n_data)
            w_batch_size = int(batch_size * weight)
            w_batch_sizes.append(w_batch_size)
            n_batchs.append((n_data + w_batch_size - 1)// w_batch_size)
        self.n_iter = max(n_batchs)
        self.epoch_counts =[(self.n_iter + b -1)//b for b in n_batchs]
        self.code_ids = codes_ids
        self.id_lists = id_lists
        self.w_batch_sizes = w_batch_sizes
        print("loop count per epoch", self.epoch_counts)
        print("number of texts ", n_dats)

    def __len__(self):
        return self.n_iter


    def __iter__(self):
        epoch_ids = {}
        for c, code_id in enumerate(self.code_ids):
            cids = self.id_lists[code_id]
            ids = []
            for _ in range(self.epoch_counts[c]):
                ids.append(np.random.permutation(cids))
            epoch_ids[code_id] = list(np.concatenate(ids))

        for n in range(self.n_iter):
            indices = []
            for w_batch_size, code_id in zip(self.w_batch_sizes, self.code_ids):
                indices.extend(epoch_ids[code_id][n*w_batch_size:(n+1)*w_batch_size])
            random.shuffle(indices)
            yield indices





def get_loaders(params):
    data_params = params["data_params"]
    csv_files = data_params["csv_files"]
    ctrl_codes = data_params["ctrl_codes"]
    sp_file = data_params["sp_file"]
    col_name = data_params["col_name"]
    weights = data_params["weights"]
    batch_size = data_params["batch_size"]

    other_params = data_params.get("other_params", {})
    csv_type = data_params.get("csv_type", "single")

    if csv_type == "single":
        train_dataset = PadCsvSpDataset(csv_files, ctrl_codes,
                                    col_name, sp_file, **other_params)
    elif csv_type == "split":
        train_dataset = CsvSpDataset(csv_files, ctrl_codes,
                                    col_name, sp_file, **other_params)
    else:
        raise NotImplementedError(csv_type) 

    sampler = CodeBalancedBatchSampler(
        train_dataset.codes, train_dataset.codes_ids,
        weights, batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler)

    datasets = {}
    datasets["train"] = train_loader
    return datasets


if __name__ == "__main__":
    csv_files = ["aozora.csv", "wiki.csv", "jesc.csv"]
    ctrl_codes = ["青空", "辞書", "訳"]
    sp_file = "sp.model"
    col_name =  "text"
    weights = [0.4, 0.5, 0.1]
    batch_size = 32

    csv_dataset = CsvSpDataset(csv_files, ctrl_codes, col_name, sp_file)

    print(len(csv_dataset))
    # ids = csv_dataset[0]
    # print(ids)
    # print(csv_dataset.sp.decode_ids(ids))

    sampler = CodeBalancedBatchSampler(
        csv_dataset.codes, csv_dataset.codes_ids,
        weights, batch_size)
    batch = iter(sampler).__next__()
    print(batch)
    print(len(batch))

    train_loader = DataLoader(csv_dataset, batch_sampler=sampler)

    batch = iter(train_loader).__next__()
    print(len(batch))
    print(type(batch))
    print(batch.shape)

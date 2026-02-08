import torch 
from torch.utils.data import Dataset
from data_loader import build_input
from data_loader import get_schema

class SpiderDataset(Dataset):
    def __init__(self, data, schema, tokenizer, max_length=512):
        self.data = data
        self.schema = schema
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]
        question = sample["question"]
        sql = sample["query"]
        db_id = sample["db_id"]

        schema = get_schema(db_id, self.schema)
        if schema is None:
            raise ValueError(f"Schema not found for db_id: {db_id}")

        input_text = build_input(question, schema)

        encoding = self.tokenizer(
            input_text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        
        target_encoding = self.tokenizer(
            sql,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": target_encoding["input_ids"].squeeze(0),
        }

import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
from spider_class import SpiderDataset
model_name = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


with open("../data/spider/train_spider.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("../data/spider/tables.json", "r", encoding="utf-8") as f:
    
    schemas = json.load(f)

data = data[:100] 

dataset = SpiderDataset(data, schemas, tokenizer)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch in loader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    loss = outputs.loss
    print(f"Loss: {loss.item()}")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    for epoch in range(3):
        for batch in loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    model.save_pretrained("./t5-small-spider")
    tokenizer.save_pretrained("./t5-small-spider")

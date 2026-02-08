import json

with open("../data/spider/train_spider.json") as f:
    data = json.load(f)

with open("../data/spider/tables.json") as f:
    table = json.load(f)


def get_schema(db_id, schemas):
    """
    Get the schema of a table.
    """
    for db in schemas:
        if db["db_id"] == db_id:
            tables = db["table_names_original"]
            columns = db["column_names_original"]
            
            schema = {}
            for table_id, column_name in columns:
                if table_id == -1:
                    continue
                
                table_name = tables[table_id]
                schema.setdefault(table_name, []).append(column_name)
            return schema

def build_input(question, schema):
    schema_text = ""
    for table, columns in schema.items():
        schema_text += f"- {table}({', '.join(columns)})\n"

    input_text = f"""
Question: {question}
Tables:
{schema_text}
"""
    return input_text.strip()

sample = data[0]

question = sample["question"]
sql = sample["query"]
db_id = sample["db_id"]

schema = get_schema(db_id, table)
input_text = build_input(question, schema)

print(input_text)
print("SQL:", sql)

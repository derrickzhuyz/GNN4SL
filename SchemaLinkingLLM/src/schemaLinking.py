from openai import OpenAI
import os
import json
import prompts
from tqdm import tqdm
import time

PROMPT_INSTRUCTION = '''### Returns the Schema used for the question in json format only and with no explanation.
### Example:{example}
### Given the following SQLite database schema: 
{db_schema}
### Question: {question}
###'''

def get_response(prompt):
    client = OpenAI(
        # api_key=os.getenv("OPENAI_API_KEY"), # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        # base_url=os.getenv("OPENAI_BASE_URL"),  # 填写DashScope服务的base_url
        api_key="Bearer eaec587b008d486f9a579a172ef2de559c7383a64c404f8d9d67f4a6c61746b4",
        base_url="https://gpt-api.hkust-gz.edu.cn/v1"
    )
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo", #'gpt-4o-mini'
        #model="gpt-4", #'gpt-4o-2024-08-06'
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                  {'role': 'user', 'content': prompt},]
        )
    return completion.choices[0].message.content


def remove_code_block_tags(text):
    # 检查字符串是否以 ```json 开头并以 ``` 结尾
    if text.startswith("```json") and text.endswith("```"):
        # 去除开头和结尾的指定部分
        return text[7:-3].strip()  # 去除开头的 ```json 和结尾的 ```
    return text

if __name__ == '__main__':
    # 读取 JSON 文件
    with open('../data/gold_schema_linking/spider_dev_gold_schema_linking.json', 'r', encoding='utf-8') as file:
        data = json.load(file)  # 加载 JSON 数据为 Python 对象

    
    with open("linked_schema.json", "a", encoding="utf-8") as file:
        file.write("[")

    # 遍历每一个对象
    cnt = 0
    for item in tqdm(data, desc="Schema Linking Progress" ):
        database = item.get('database')
        question = item.get('question')
        #print(f"Database: {database}, Question: {question}")

        with open('../data/db_schema/spider_schemas.json', 'r', encoding='utf-8') as file:
            schemas = json.load(file)

        # 查找具有指定 'database' 值的 item
        schema = None
        for item in schemas:
            if item.get('database') == database:
                schema = item
                break  

        if not schema:
            raise ValueError("Schema not found.")
        formatted_prompt = PROMPT_INSTRUCTION.format(example=prompts.EXAMPLE, db_schema=schema, question=question)

        #print(formatted_prompt)
        linked_schema = remove_code_block_tags(get_response(formatted_prompt))

        with open("linked_schema.json", "a", encoding="utf-8") as file:
            file.write(linked_schema)
            file.write(",")
        cnt+=1
        print(f"成功{cnt}条")
        #break

    with open("linked_schema.json", "a", encoding="utf-8") as file:
        file.write("]")
    # print(get_response())
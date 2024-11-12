import os
import json
import time
from openai import OpenAI
from tqdm import tqdm
from loguru import logger
from prompts import *
#import prompts

logger.add("logs/llm_api.log", rotation="1 MB", level="INFO", 
           format="{time} {level} {message}", compression="zip")


"""
Get the response from the LLM by calling API
:param prompt: The prompt to send to the LLM
:return: The response from the LLM
"""
def get_response(prompt):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo", #'gpt-4o-mini'
        # model="gpt-4", #'gpt-4o-2024-08-06'
        messages=[{'role': 'system', 'content': 'You are an expert in database.'},
                  {'role': 'user', 'content': prompt},]
        )
    return completion#.choices[0].message.content


"""
Remove the code block tags from the response
:param text: The response from the LLM
:return: The response without the code block tags
"""
def remove_code_block_tags(text):
    # Check that the string starts with '''' JSON and ends with '''
    if text.startswith("```json") and text.endswith("```"):
        logger.info("[i] Handled the json formatting issue")
        return text[7:-3].strip()  # Remove the ''' json at the beginning and the ''' at the end
    return text

def remove_braces(text):
    # Check that the string starts with '''' JSON and ends with '''
    if text.startswith("{") and text.endswith("}"):
        logger.info("[i] Handled the braces formatting issue")
        return text[1:-1].strip()  # Remove the ''' json at the beginning and the ''' at the end
    return text


if __name__ == '__main__':
    # Read the JSON file
    with open('data/gold_schema_linking/bird_dev_gold_schema_linking.json', 'r', encoding='utf-8') as file:
        data = json.load(file)  # Load the JSON data as a Python object

    
    with open("linked_schema.json", "a", encoding="utf-8") as file:
        file.write("[")

    # Iterate over each object
    cnt = 0
    cost = 0.0
    for item in tqdm(data, desc="Schema Linking Progress" ):
        database = item.get('database')
        question = item.get('question')

        with open('data/db_schema/bird_dev_schemas.json', 'r', encoding='utf-8') as file:
            schemas = json.load(file)

        # Find the item with the specified 'database' value
        schema = None
        for item in schemas:
            if item.get('database') == database:
                schema = item
                break  

        if not schema:
            logger.error("[! Error] Schema not found.")
        formatted_prompt = PROMPT_INSTRUCTION.format(example=EXAMPLE, db_schema=schema, question=question)

        #print(formatted_prompt)
        response = get_response(formatted_prompt)
        cost += response.consume
        linked_schema = remove_braces(remove_code_block_tags(response.choices[0].message.content))


        with open("linked_schema.json", "a", encoding="utf-8") as file:
            file.write("{\n")
            file.write(f'"database": "{database}",\n')
            file.write(linked_schema)
            file.write(",\n")
            file.write(f'"question": "{question.replace("\"", "\\\"")}"\n')
            file.write('},\n')
        cnt+=1
        logger.info(f"[+] Successfully processed {cnt} items, cost {cost} CNY")
        

    with open("linked_schema.json", "r+", encoding="utf-8") as file:
        content = file.read()
        stripped_content = content.rstrip()
        if stripped_content and stripped_content[-1] == ",":
            new_content = stripped_content[:-1] + "]"
            # Move the file pointer to the beginning of the file and write the modified content
            file.seek(0)
            file.write(new_content)
            file.truncate()  # Remove redundant characters
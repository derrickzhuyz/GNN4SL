""" Inspired by: https://github.com/IBM/few-shot-schema-linking """

import re
import json
import sqlglot
import sqlglot.expressions as exp
import sqlglot.optimizer.qualify as qualify
from typing import Tuple, List, Dict


"""
Extract the tables and columns from a SQL query
:param sql_query: The SQL query
:param schema: The schema information of the database
:param dialect: The SQL dialect
:return: A dictionary of gold schema linking
"""
def extract_tables_and_columns(
    sql_query: str, curr_line_no: int, schema: Dict = None, dialect: str = "sqlite"
) -> Dict:
    
    original_sql_query = sql_query
    remarks = ""

    # Extract the database name from the SQL query (in the end of sql)
    sql_query_parts = sql_query.rsplit(maxsplit=1)
    if len(sql_query_parts) > 1 and sql_query_parts[-1].isidentifier():
        database = sql_query_parts[-1]
        sql_query = sql_query_parts[0]
    else:
        database = "database_name_error"

    # Preprocess the SQL query
    sql_query = preprocess_sql_query(sql_query)

    # Use try-except to catch potential parsing error
    try:
        expression = sqlglot.parse_one(sql_query, read=dialect)
        try:
            # Apply qualify to ensure all column names are fully qualified
            expression = qualify.qualify(expression, schema=schema)
        except sqlglot.errors.OptimizeError as qualification_e:
            print(f"[* Warning] Qualification failed on line {curr_line_no}: {qualification_e}")
            remarks = f"Qualification failed: {qualification_e}."
    except sqlglot.errors.ParseError as parse_sql_e:
        print(f"[! Error] Failed to parse on line {curr_line_no}. SQL query: {original_sql_query}")
        return {
            "database": database,
            "tables": [{"table": "extract_gold_schema_error", "columns": ["extract_schema_error"]}],
            "gold_sql": original_sql_query,
            "remarks": f"Failed to parse SQL: {parse_sql_e}."
        }
    
    
    cte_aliases = [cte.alias for cte in expression.find_all(exp.CTE)]
    sub_queries = list(expression.find_all((exp.Subquery, exp.CTE), bfs=False))
    sub_queries.reverse()
    sub_queries.append(expression)

    all_tables = {}
    
    for sub_query in sub_queries:
        sub_tables, sub_columns = get_subquery_tables_and_columns(sub_query, cte_aliases, schema)
        sub_query.pop()
        
        for table in sub_tables:
            if table not in all_tables:
                all_tables[table] = []
            all_tables[table].extend(sub_columns.get(table, []))

    # Remove duplicate columns
    for table in all_tables:
        all_tables[table] = list(set(all_tables[table]))

    # Return the result as a JSON object
    result = {
        "database": database,
        "tables": [{"table": table, "columns": columns} for table, columns in all_tables.items()],
        "gold_sql": original_sql_query,
        "remarks": remarks
    }

    return result


"""
Preprocess the SQL query to normalize it
:param sql_query: The original SQL query
:return: The normalized SQL query
"""
def preprocess_sql_query(sql_query: str) -> str:
    """Preprocess the SQL query to normalize it."""
    # Convert null and empty string comparisons to IS NULL and IS NOT NULL
    sql_query = re.sub(r"\s*(=|<>|!=)\s*('')", " IS NULL", sql_query)
    sql_query = re.sub(r"\s*IS\s+NOT\s+('')", " IS NOT NULL", sql_query)

    # Remove nested quotes (e.g., 'string "nested" string' or "string 'nested' string")
    sql_query = re.sub(r"'([^']*\"[^\"]*\")*[^']*'", lambda m: m.group(0).replace('"', ''), sql_query)
    sql_query = re.sub(r'"([^"]*)\'([^\']*)\'([^"]*)"', lambda m: m.group(0).replace("'", ''), sql_query)

    # Remove special characters from the SQL query to avoid errors
    # sql_query = sql_query.replace('`', '"').replace(";", "")
    sql_query = sql_query.replace('"',"'")

    return sql_query


"""
Extract tables and columns from a subquery when CTEs are involved
:param expression: The SQL expression
:param cte_aliases: The CTE aliases
:param schema: The schema information of the database
:return: A tuple of tables and columns
"""
def get_subquery_tables_and_columns(expression, cte_aliases, schema):
    tables = [
        t.name.lower()
        for t in expression.find_all(exp.Table)
        if not t.name.lower() in cte_aliases
    ]
    
    table_aliases = {
        t.alias.lower(): t.name.lower()
        for t in expression.find_all(exp.Table)
        if t.alias != ""
    }
    
    columns_dict = {}
    
    for c in expression.find_all(exp.Column):
        column_name = c.name.lower()
        table_name_or_alias = c.table.lower()
        
        if table_name_or_alias == "":
            if len(tables) == 1:
                table_name = tables[0]
            else:
                table_name = ""
                if schema:
                    for table in schema["schema_items"]:
                        if (
                            column_name in table["column_names"]
                            and table["table_name"] in tables
                        ):
                            table_name = table["table_name"]
                            break
                if table_name == "":
                    continue
        elif table_name_or_alias in table_aliases:
            table_name = table_aliases[table_name_or_alias]
        elif table_name_or_alias in tables:
            table_name = table_name_or_alias
        else:
            continue
        
        if table_name not in columns_dict:
            columns_dict[table_name] = []
        columns_dict[table_name].append(column_name)

    return tables, columns_dict
    

"""
Process gold SQL files, extract save the gold schema linking to a JSON file
:param input_file: The path of the input SQL file
:param output_file: The path of the output JSON file
:return: None
"""
def process_sql_file(input_file: str, output_file: str) -> None:
    results = []
    with open(input_file, 'r') as f:
        for idx, sql_query in enumerate(f):
            sql_query = sql_query.strip()
            if sql_query:
                try:
                    result = extract_tables_and_columns(sql_query, curr_line_no=idx + 1)
                    result["id"] = idx
                    results.append(result)
                except ValueError as e:
                    print(f"[! Error] Failed to process SQL on line {idx + 1}: {e} (id: {idx})")
                    

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        print("--------------------------------------------------------------------------------------------")
        print(f"[i] Gold schema linking has been saved to {output_file}")
        print("--------------------------------------------------------------------------------------------")



if __name__ == "__main__":
    with open('config.json') as config_file:
        config = json.load(config_file)

    spider_train_gold_sql = config['spider_paths']['full_train_gold_sql']
    spider_train_gold_schema_linking = config['gold_schema_linking_paths']['spider_train']
    process_sql_file(spider_train_gold_sql, spider_train_gold_schema_linking)

    spider_dev_gold_sql = config['spider_paths']['dev_gold_sql']
    spider_dev_gold_schema_linking = config['gold_schema_linking_paths']['spider_dev']
    process_sql_file(spider_dev_gold_sql, spider_dev_gold_schema_linking)
    
    bird_train_gold_sql = config['bird_paths']['train_gold_sql']
    bird_train_gold_schema_linking = config['gold_schema_linking_paths']['bird_train']
    process_sql_file(bird_train_gold_sql, bird_train_gold_schema_linking)

    bird_dev_gold_sql = config['bird_paths']['dev_gold_sql']
    bird_dev_gold_schema_linking = config['gold_schema_linking_paths']['bird_dev']
    process_sql_file(bird_dev_gold_sql, bird_dev_gold_schema_linking)


    #TODO: manual check the output file, especailly remarks are not null
    #TODO: fix issues

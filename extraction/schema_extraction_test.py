from typing import Tuple, List, Dict
import sqlglot
import sqlglot.expressions as exp
import sqlglot.optimizer.qualify as qualify
import json
import re


def extract_tables_and_columns(
    sql_query: str, curr_line_no: int, schema: Dict = None, dialect: str = "sqlite"
) -> Dict:
    
    original_sql_query = sql_query

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
        except sqlglot.errors.OptimizeError as e:
            print(f"[! Warning] Qualification failed on line {curr_line_no}: {e}")
    except sqlglot.errors.ParseError:
        print(f"[! Error] Failed to parse on line {curr_line_no}. SQL query: {original_sql_query}")
        return {
            "database": database,
            "tables": [{"table": "extract_gold_schema_error", "columns": ["extract_schema_error"]}],
            "gold_sql": original_sql_query
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
        "gold_sql": original_sql_query
    }

    return result


def preprocess_sql_query(sql_query: str) -> str:
    """Preprocess the SQL query to normalize it."""
    # Convert null and empty string comparisons to IS NULL and IS NOT NULL
    sql_query = re.sub(r"\s*(=|<>|!=)\s*('')", " IS NULL", sql_query)
    sql_query = re.sub(r"\s*IS\s+NOT\s+('')", " IS NOT NULL", sql_query)

    # Remove nested quotes (e.g., 'string "nested" string' or "string 'nested' string")
    sql_query = re.sub(r"'([^']*\"[^\"]*\")*[^']*'", lambda m: m.group(0).replace('"', ''), sql_query)
    sql_query = re.sub(r'"([^"]*)\'([^\']*)\'([^"]*)"', lambda m: m.group(0).replace("'", ''), sql_query)

    # Remove special characters from the SQL query to avoid errors
    sql_query = sql_query.replace('"',"'")

    return sql_query


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
    

def process_sql_file(input_file: str, output_file: str):
    results = []
    with open(input_file, 'r') as f:
        for idx, sql_query in enumerate(f):
            sql_query = sql_query.strip()
            if sql_query:
                try:
                    result = extract_tables_and_columns(sql_query, curr_line_no=idx + 1)
                    result["id"] = idx + 1
                    results.append(result)
                except ValueError as e:
                    print(f"[! Error] Failed to process SQL on line {idx + 1}: {e}")
                    

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[i] Gold schema linking has been saved to {output_file}")


if __name__ == "__main__":

    sql_query_1 = """SELECT name FROM head WHERE born_state == 'California'  geo
    """

    sql_query_2 = """
    WITH ManagerCTE AS (
        SELECT EmployeeID, EmployeeName, ManagerID
        FROM Employees
    )
    SELECT 
        e.EmployeeName AS 'Employee',
        m.EmployeeName AS 'Manager'
    FROM 
        ManagerCTE e
    LEFT JOIN 
        ManagerCTE m ON e.ManagerID = m.EmployeeID; hhh
    """

    sql_query_3 = """SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1;   www
    """

    sql_query_4 = """SELECT count(*) FROM singer\tconcert_singer
    """

    sql_query_5 = """SELECT country FROM singer WHERE age  >  40 INTERSECT SELECT country FROM singer WHERE age  <  30\tconcert_singer
    """

    sql_query_6 = """WITH SubQuery AS (SELECT DISTINCT T1.atom_id, T1.element, T1.molecule_id, T2.label FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.molecule_id = 'TR006') SELECT CAST(COUNT(CASE WHEN element = 'h' THEN atom_id ELSE NULL END) AS REAL) / (CASE WHEN COUNT(atom_id) = 0 THEN NULL ELSE COUNT(atom_id) END) AS ratio, label FROM SubQuery GROUP BY label\ttoxicology
    """

    sql_query_7 = """SELECT count(*) FROM head WHERE age  >  56	department_management"""

    # Qualification failed cases
    sql_query_8 = """SELECT template_type_code FROM Templates EXCEPT SELECT template_type_code FROM Templates AS T1 JOIN Documents AS T2 ON T1.template_id  =  T2.template_id\tcre_Doc_Template_Mgt
    """

    sql_query_9 = """SELECT winner_name FROM matches WHERE YEAR  =  2013 INTERSECT SELECT winner_name FROM matches WHERE YEAR  =  2016\twta_1"""

    sql_query_10 = """SELECT template_id FROM Templates EXCEPT SELECT template_id FROM Documents\tcre_Doc_Template_Mgt"""

    sql_query_11 = """SELECT professional_id ,  last_name ,  cell_number FROM Professionals WHERE state  =  'Indiana' UNION SELECT T1.professional_id ,  T1.last_name ,  T1.cell_number FROM Professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id  =  T2.professional_id GROUP BY T1.professional_id HAVING count(*)  >  2\tdog_kennels
    """

    sql_query_12 = """
    WITH SubQuery AS (SELECT DISTINCT T1.atom_id, T1.element, T1.molecule_id, T2.label FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.molecule_id = 'TR006') SELECT CAST(COUNT(CASE WHEN element = 'h' THEN atom_id ELSE NULL END) AS REAL) / (CASE WHEN COUNT(atom_id) = 0 THEN NULL ELSE COUNT(atom_id) END) AS ratio, label FROM SubQuery GROUP BY label\ttoxicology
    """

    result = extract_tables_and_columns(sql_query_12, curr_line_no = 1)
    print(json.dumps(result, indent=2, ensure_ascii=False)) 
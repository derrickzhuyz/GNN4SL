import json


"""
Convert the score-based predictions to top-k tables and columns predictions.
:param data_path: Path to the JSON file containing the score-based predictions.
:param top_n_tables: Number of top tables to select.
:param top_n_columns: Number of top columns to select.
"""
def select_top_tables_and_columns(data_path: str, top_n_tables: int = 4, top_n_columns: int = 5) -> None:
    with open(data_path, 'r') as f:
        data = json.load(f)
    for item in data:
        # Sort tables by score and get top 4
        tables = item['tables']
        tables.sort(key=lambda x: x['score'], reverse=True)
        top_tables = tables[:top_n_tables]
        
        # Mark top tables as relevant
        for table in tables:
            table['relevant'] = table in top_tables
            
            # Sort columns by score and get top 5
            columns = table['columns']
            columns.sort(key=lambda x: x['score'], reverse=True)
            top_columns = columns[:top_n_columns]
            
            # Mark top columns as relevant
            for column in columns:
                column['relevant'] = column in top_columns and table in top_tables

    # Save the modified data to a new JSON file
    output_path = data_path.replace('.json', f'_top_{top_n_tables}_{top_n_columns}.json')
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f'[i] Processed and saved to {output_path}')


if __name__ == '__main__':
    # select_top_tables_and_columns('gnn/results/link_level/spider_dev_predictions_link_level_model_combined_20241125_132315.json', top_n_tables=4, top_n_columns=5)
    select_top_tables_and_columns('gnn/results/link_level/bird_dev_predictions_link_level_model_combined_20241125_132315.json', top_n_tables=4, top_n_columns=5)

    pass


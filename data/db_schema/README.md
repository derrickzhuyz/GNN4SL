The entire schema of each database is extracted from .sqlite file and stored in this dir.

Example of database schema:

```json
  {
    "database": "body_builder",
    "tables": [
      {
        "table": "body_builder",
        "columns": [
          "Body_Builder_ID",
          "People_ID",
          "Snatch",
          "Clean_Jerk",
          "Total"
        ],
        "primary_keys": [
          "Body_Builder_ID"
        ]
      },
      {
        "table": "people",
        "columns": [
          "People_ID",
          "Name",
          "Height",
          "Weight",
          "Birth_Date",
          "Birth_Place"
        ],
        "primary_keys": [
          "People_ID"
        ]
      }
    ],
    "foreign_keys": [
      {
        "table": [
          "body_builder",
          "people"
        ],
        "column": [
          "People_ID",
          "People_ID"
        ]
      }
    ],
    "foreign_key_count": 1,
    "table_count": 2,
    "total_column_count": 11
  },
  {
    ...
  }
```
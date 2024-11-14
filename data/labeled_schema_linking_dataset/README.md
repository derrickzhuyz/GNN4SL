Labeled schema linking dataset: for each example, label with the relevance of each table and column based on gold schema linking

Example labeled with gold schema linking:

```json
  {
    "database": "concert_singer",
    "question": "How many singers do we have?",
    "id": 0,
    "gold_sql": "SELECT count(*) FROM singer\tconcert_singer",
    "remarks": "",
    "tables": [
      {
        "name": "stadium",
        "relevant": 0,
        "columns": [
          {
            "name": "Stadium_ID",
            "relevant": 0
          },
          {
            "name": "Location",
            "relevant": 0
          },
          {
            "name": "Name",
            "relevant": 0
          },
          {
            "name": "Capacity",
            "relevant": 0
          },
          {
            "name": "Highest",
            "relevant": 0
          },
          {
            "name": "Lowest",
            "relevant": 0
          },
          {
            "name": "Average",
            "relevant": 0
          }
        ]
      },
      {
        "name": "singer",
        "relevant": 1,
        "columns": [
          {
            "name": "Singer_ID",
            "relevant": 0
          },
          {
            "name": "Name",
            "relevant": 0
          },
          {
            "name": "Country",
            "relevant": 0
          },
          {
            "name": "Song_Name",
            "relevant": 0
          },
          {
            "name": "Song_release_year",
            "relevant": 0
          },
          {
            "name": "Age",
            "relevant": 0
          },
          {
            "name": "Is_male",
            "relevant": 0
          }
        ]
      },
      {
        "name": "concert",
        "relevant": 0,
        "columns": [
          {
            "name": "concert_ID",
            "relevant": 0
          },
          {
            "name": "concert_Name",
            "relevant": 0
          },
          {
            "name": "Theme",
            "relevant": 0
          },
          {
            "name": "Stadium_ID",
            "relevant": 0
          },
          {
            "name": "Year",
            "relevant": 0
          }
        ]
      },
      {
        "name": "singer_in_concert",
        "relevant": 0,
        "columns": [
          {
            "name": "concert_ID",
            "relevant": 0
          },
          {
            "name": "Singer_ID",
            "relevant": 0
          }
        ]
      }
    ],
    "foreign_keys": [
      {
        "table": [
          "concert",
          "stadium"
        ],
        "column": [
          "Stadium_ID",
          "Stadium_ID"
        ]
      },
      {
        "table": [
          "singer_in_concert",
          "singer"
        ],
        "column": [
          "Singer_ID",
          "Singer_ID"
        ]
      },
      {
        "table": [
          "singer_in_concert",
          "concert"
        ],
        "column": [
          "concert_ID",
          "concert_ID"
        ]
      }
    ],
    "table_count": 4,
    "total_column_count": 21,
    "foreign_key_count": 3,
    "involved_table_count": 1,
    "involved_column_count": 0
  },
  {
    ......
  }
```
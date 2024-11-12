EXAMPLE = '''## Given the following SQLite database schema: 
  {
    "database": "concert_singer",
    "tables": [
      {
        "table": "stadium",
        "columns": [
          "Stadium_ID",
          "Location",
          "Name",
          "Capacity",
          "Highest",
          "Lowest",
          "Average"
        ],
        "primary_keys": [
          "Stadium_ID"
        ]
      },
      {
        "table": "singer",
        "columns": [
          "Singer_ID",
          "Name",
          "Country",
          "Song_Name",
          "Song_release_year",
          "Age",
          "Is_male"
        ],
        "primary_keys": [
          "Singer_ID"
        ]
      },
      {
        "table": "concert",
        "columns": [
          "concert_ID",
          "concert_Name",
          "Theme",
          "Stadium_ID",
          "Year"
        ],
        "primary_keys": [
          "concert_ID"
        ]
      },
      {
        "table": "singer_in_concert",
        "columns": [
          "concert_ID",
          "Singer_ID"
        ],
        "primary_keys": [
          "concert_ID",
          "Singer_ID"
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
    "foreign_key_count": 3,
    "table_count": 4,
    "total_column_count": 21
  }
## Quesion: "What are the names, countries, and ages for every singer in descending order of age?"
##
    "tables": [
      {
        "table": "singer",
        "columns": [
          "name",
          "age",
          "country"
        ]
      }
    ]
    '''

PROMPT_INSTRUCTION = '''### Returns the Schema used for the question in json format only and with no explanation.
### Example:{example}
### Given the following SQLite database schema: 
{db_schema}
### Quesion: {question}
###'''
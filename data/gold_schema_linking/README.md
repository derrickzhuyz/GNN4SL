The schema of each example (NL, SQL) pair is extracted from gold SQL and stored in this dir.

Example for gold schema linking:

```json
{
    "database": "concert_singer",
    "tables": [
      {
        "table": "concert",
        "columns": [
          "stadium_id",
          "year"
        ]
      },
      {
        "table": "stadium",
        "columns": [
          "stadium_id",
          "capacity",
          "name"
        ]
      }
    ],
    "gold_sql": "SELECT T2.name ,  T2.capacity FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.year  >=  2014 GROUP BY T2.stadium_id ORDER BY count(*) DESC LIMIT 1\tconcert_singer",
    "remarks": "",
    "id": 25,
    "question": "What is the name and capacity of the stadium  with the most concerts after 2013?",
    "involved_table_count": 2,
    "involved_column_count": 4
},
{
......
}
```
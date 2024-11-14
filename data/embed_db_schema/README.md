To store database schema with embeddings of table and column names.

Example:

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
      ],
      "table_name_embedding": [
        0.38914465144999866,
        0.8132362252453139,
        0.27417797137679645
      ],
      "column_name_embeddings": {
        "Body_Builder_ID": [
          0.9358401318602013,
          0.7542914919168114,
          0.12337282796814675
        ],
        "People_ID": [
          0.3372112723409929,
          0.008846404188221246,
          0.1634358251257736
        ],
        "Snatch": [
          0.03483923589444193,
          0.6939027822855781,
          0.2494291754282969
        ],
        "Clean_Jerk": [
          0.7166522348858981,
          0.9491461126304741,
          0.40667610548844335
        ],
        "Total": [
          0.20385345748448047,
          0.8887649318494109,
          0.5530436485163128
        ]
      }
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
      ],
      "table_name_embedding": [
        0.04282544159424995,
        0.25375683256924975,
        0.9616950856087195
      ],
      "column_name_embeddings": {
        "People_ID": [
          0.3815252270105369,
          0.9547299535088725,
          0.2779015867226353
        ],
        "Name": [
          0.7586455848839778,
          0.8305970829305139,
          0.6156037000395168
        ],
        "Height": [
          0.3374833157586401,
          0.4419609950516864,
          0.5932775606383603
        ],
        "Weight": [
          0.9792599384815707,
          0.6234432793880265,
          0.8746375578564196
        ],
        "Birth_Date": [
          0.7307797774538874,
          0.8598225588348808,
          0.36782514433831615
        ],
        "Birth_Place": [
          0.7007840376441065,
          0.22168921971764788,
          0.06067329063701632
        ]
      }
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
  "total_column_count": 11,
  "database_name_embedding": [
    0.7714080972042239,
    0.983439063710305,
    0.7012135361207671
  ]
},
{
  ......
}
```
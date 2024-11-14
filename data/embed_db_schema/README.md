To store database schema with table and column name embedded.

Example:

```
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
        ],
        "table_embedding": [
          0.14163017671691447,
          0.059977153780528814,
          0.37587119965258975
        ],
        "column_embeddings": {
          "Stadium_ID": [
            0.9598121842610345,
            0.876209230359925,
            0.6214717516015649
          ],
          "Location": [
            0.46147301179723865,
            0.9423509301653539,
            0.5400036415231201
          ],
          "Name": [
            0.1929948758088802,
            0.047508354761691796,
            0.8593871686207779
          ],
          "Capacity": [
            0.010227495368304362,
            0.05636102437112156,
            0.2710962541159454
          ],
          "Highest": [
            0.4754155545921136,
            0.5421896546167827,
            0.5711959322935362
          ],
          "Lowest": [
            0.5849270269700312,
            0.2958014121741076,
            0.2617687122058441
          ],
          "Average": [
            0.4351719740134724,
            0.10516213180610523,
            0.36116316578456165
          ]
        }
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
        ],
        "table_embedding": [
          0.03566350297858656,
          0.987600279752835,
          0.5365064391775425
        ],
        "column_embeddings": {
          "Singer_ID": [
            0.8317641351946311,
            0.7373170573074839,
            0.4786752285636847
          ],
          "Name": [
            0.26785500641089066,
            0.3310877845037554,
            0.460108395062955
          ],
          "Country": [
            0.87194620347083,
            0.4426993904643246,
            0.24226349362060484
          ],
          "Song_Name": [
            0.6412787985527955,
            0.375914004719375,
            0.333344276922921
          ],
          "Song_release_year": [
            0.8628548338301291,
            0.23908199428728005,
            0.4737911135040729
          ],
          "Age": [
            0.9529990623896498,
            0.12273936372848304,
            0.25472111124974384
          ],
          "Is_male": [
            0.6782111877046142,
            0.23786062915841222,
            0.5269237940197532
          ]
        }
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
        ],
        "table_embedding": [
          0.32129904693090494,
          0.8146375214845064,
          0.4493960841789786
        ],
        "column_embeddings": {
          "concert_ID": [
            0.9641202317834177,
            0.6338754334494504,
            0.6346403252215804
          ],
          "concert_Name": [
            0.9534903890717882,
            0.31435370978910426,
            0.5677963623947332
          ],
          "Theme": [
            0.5303665214064651,
            0.8925714648355085,
            0.6941507854830421
          ],
          "Stadium_ID": [
            0.6428093830404324,
            0.4321838559322183,
            0.18101800222354092
          ],
          "Year": [
            0.10570069522948167,
            0.8097656022353649,
            0.3636834026638508
          ]
        }
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
        ],
        "table_embedding": [
          0.385140747071151,
          0.3217034503830273,
          0.13163174453312598
        ],
        "column_embeddings": {
          "concert_ID": [
            0.14828885128865432,
            0.24808607311702802,
            0.15775996875024156
          ],
          "Singer_ID": [
            0.6959359335138362,
            0.8955553990798085,
            0.4272487918165131
          ]
        }
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
    "total_column_count": 21,
    "database_embedding": [
      0.19010250394877581,
      0.811525982996565,
      0.6471325023800776
    ]
},
......
```
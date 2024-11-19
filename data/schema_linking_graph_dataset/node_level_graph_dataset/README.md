Node-level graph datasets: for each dataset, the number of generated graphs equals to the number of examples (or questions)

Inspection Example:

```json

=== Inspecting Node-level Graph: 0 ===

Graph Structure:
Data(x=[25, 3], edge_index=[2, 24], y=[25], node_names=[25], node_types=[25])

Graph Statistics:
Number of nodes: 25
Number of edges: 24
Average degree: 1.92
Number of relevant nodes: 1
Percentage of relevant nodes: 4.00%

Node Information:
Node 0: stadium (table)
  - Relevant: 0
  - Embedding (first three values): [0.07861121 0.48702663 0.6872855 ]
Node 1: stadium_id (column)
  - Relevant: 0
  - Embedding (first three values): [0.33337724 0.79151    0.1838022 ]
Node 2: location (column)
  - Relevant: 0
  - Embedding (first three values): [0.5894001  0.5754917  0.70753515]
Node 3: name (column)
  - Relevant: 0
  - Embedding (first three values): [0.2785914  0.308952   0.08608443]
Node 4: capacity (column)
  - Relevant: 0
  - Embedding (first three values): [0.3890879  0.5893352  0.34179825]
Node 5: highest (column)
  - Relevant: 0
  - Embedding (first three values): [0.67118883 0.01473221 0.78720355]
Node 6: lowest (column)
  - Relevant: 0
  - Embedding (first three values): [0.26065654 0.5880603  0.6401594 ]
Node 7: average (column)
  - Relevant: 0
  - Embedding (first three values): [0.921932  0.2910665 0.8168265]
Node 8: singer (table)
  - Relevant: 1
  - Embedding (first three values): [0.54271084 0.64325535 0.11860269]
Node 9: singer_id (column)
  - Relevant: 0
  - Embedding (first three values): [0.1333736  0.09488335 0.8006504 ]
Node 10: name (column)
  - Relevant: 0
  - Embedding (first three values): [0.2785914  0.308952   0.08608443]
Node 11: country (column)
  - Relevant: 0
  - Embedding (first three values): [0.58892316 0.5464688  0.3507285 ]
Node 12: song_name (column)
  - Relevant: 0
  - Embedding (first three values): [0.340062   0.96974266 0.55370545]
Node 13: song_release_year (column)
  - Relevant: 0
  - Embedding (first three values): [0.6664307 0.9498801 0.7546454]
Node 14: age (column)
  - Relevant: 0
  - Embedding (first three values): [0.31199104 0.24097204 0.07128235]
Node 15: is_male (column)
  - Relevant: 0
  - Embedding (first three values): [0.9242103  0.17472635 0.7158627 ]
Node 16: concert (table)
  - Relevant: 0
  - Embedding (first three values): [0.5070624  0.6325287  0.92787063]
Node 17: concert_id (column)
  - Relevant: 0
  - Embedding (first three values): [0.7032236  0.14594649 0.97615886]
Node 18: concert_name (column)
  - Relevant: 0
  - Embedding (first three values): [0.06982073 0.91768193 0.9711458 ]
Node 19: theme (column)
  - Relevant: 0
  - Embedding (first three values): [0.7929897  0.99647254 0.507424  ]
Node 20: stadium_id (column)
  - Relevant: 0
  - Embedding (first three values): [0.33337724 0.79151    0.1838022 ]
Node 21: year (column)
  - Relevant: 0
  - Embedding (first three values): [0.24793766 0.51838195 0.20411338]
Node 22: singer_in_concert (table)
  - Relevant: 0
  - Embedding (first three values): [0.9673047  0.6608301  0.43312776]
Node 23: concert_id (column)
  - Relevant: 0
  - Embedding (first three values): [0.7032236  0.14594649 0.97615886]
Node 24: singer_id (column)
  - Relevant: 0
  - Embedding (first three values): [0.1333736  0.09488335 0.8006504 ]

Edge Information:
Edge: 0(stadium) <-> 1(stadium_id)
Edge: 0(stadium) <-> 2(location)
Edge: 0(stadium) <-> 3(name)
Edge: 0(stadium) <-> 4(capacity)
Edge: 0(stadium) <-> 5(highest)
Edge: 0(stadium) <-> 6(lowest)
Edge: 0(stadium) <-> 7(average)
Edge: 1(stadium_id) <-> 20(stadium_id)
Edge: 8(singer) <-> 9(singer_id)
Edge: 8(singer) <-> 10(name)
Edge: 8(singer) <-> 11(country)
Edge: 8(singer) <-> 12(song_name)
Edge: 8(singer) <-> 13(song_release_year)
Edge: 8(singer) <-> 14(age)
Edge: 8(singer) <-> 15(is_male)
Edge: 9(singer_id) <-> 24(singer_id)
Edge: 16(concert) <-> 17(concert_id)
Edge: 16(concert) <-> 18(concert_name)
Edge: 16(concert) <-> 19(theme)
Edge: 16(concert) <-> 20(stadium_id)
Edge: 16(concert) <-> 21(year)
Edge: 17(concert_id) <-> 23(concert_id)
Edge: 22(singer_in_concert) <-> 23(concert_id)
Edge: 22(singer_in_concert) <-> 24(singer_id)

Graph Properties:
Is connected: True
Number of connected components: 1
```

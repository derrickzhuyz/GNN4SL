# Graph Data (Schema-to-Graph)

Convert entire database schema to heterogenous graph, and labeled with relevant table(s) and column(s) based on gold schema linking of each question.

Two kinds of nodes: table, clolumn

Two kinds of edges: contain, foreign key

Check results example:

```
HeteroData(
  table={
    x=[4, 4],
    y=[4],
  },
  column={
    x=[21, 21],
    y=[21],
  },
  (table, contains, column)={ edge_index=[2, 21] },
  (column, foreign_key, column)={ edge_index=[2, 3] }
)

Number of foreign key edges: 3
Foreign key edges (source <-> target):
  Edge 0: concert.Stadium_ID <-> stadium.Stadium_ID
  Edge 1: singer_in_concert.Singer_ID <-> singer.Singer_ID
  Edge 2: singer_in_concert.concert_ID <-> concert.concert_ID

Number of contains edges: 21
Contains edges (table -> column):
  Edge 0: stadium -> stadium.Stadium_ID
  Edge 1: stadium -> stadium.Location
  Edge 2: stadium -> stadium.Name
  Edge 3: stadium -> stadium.Capacity
  Edge 4: stadium -> stadium.Highest
  Edge 5: stadium -> stadium.Lowest
  Edge 6: stadium -> stadium.Average
  Edge 7: singer -> singer.Singer_ID
  Edge 8: singer -> singer.Name
  Edge 9: singer -> singer.Country
  Edge 10: singer -> singer.Song_Name
  Edge 11: singer -> singer.Song_release_year
  Edge 12: singer -> singer.Age
  Edge 13: singer -> singer.Is_male
  Edge 14: concert -> concert.concert_ID
  Edge 15: concert -> concert.concert_Name
  Edge 16: concert -> concert.Theme
  Edge 17: concert -> concert.Stadium_ID
  Edge 18: concert -> concert.Year
  Edge 19: singer_in_concert -> singer_in_concert.concert_ID
  Edge 20: singer_in_concert -> singer_in_concert.Singer_ID
```
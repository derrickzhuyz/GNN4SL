# Heterogeneous Graph Data (Schema-to-Graph)

Convert entire database schema to heterogenous graph, and labeled with relevant table(s) and column(s) based on gold schema linking of each question.

Two kinds of nodes: table, clolumn

Two kinds of edges: contain, foreign key

Results example:

```
=== Inspecting Heterogeneous Graph: 0 ===

Graph Structure:
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

Table Nodes:
Number of tables: 4
Table labels (1=relevant, 0=not relevant):
Table 0: label=0
Table 1: label=1
Table 2: label=0
Table 3: label=0

Column Nodes:
Number of columns: 21
Column labels (1=relevant, 0=not relevant):
Column 0: label=0
Column 1: label=0
Column 2: label=0
Column 3: label=0
Column 4: label=0
Column 5: label=0
Column 6: label=0
Column 7: label=0
Column 8: label=0
Column 9: label=0
Column 10: label=0
Column 11: label=0
Column 12: label=0
Column 13: label=0
Column 14: label=0
Column 15: label=0
Column 16: label=0
Column 17: label=0
Column 18: label=0
Column 19: label=0
Column 20: label=0

Contains Edges (table -> column):
Table 0 contains Column 0
Table 0 contains Column 1
Table 0 contains Column 2
Table 0 contains Column 3
Table 0 contains Column 4
Table 0 contains Column 5
Table 0 contains Column 6
Table 1 contains Column 7
Table 1 contains Column 8
Table 1 contains Column 9
Table 1 contains Column 10
Table 1 contains Column 11
Table 1 contains Column 12
Table 1 contains Column 13
Table 2 contains Column 14
Table 2 contains Column 15
Table 2 contains Column 16
Table 2 contains Column 17
Table 2 contains Column 18
Table 3 contains Column 19
Table 3 contains Column 20

Foreign Key Edges (column <-> column):
Column 17 <-> Column 0
Column 20 <-> Column 7
Column 19 <-> Column 14
```
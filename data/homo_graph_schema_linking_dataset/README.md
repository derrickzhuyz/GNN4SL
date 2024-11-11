# Homogeneous Graph Data (Schema-to-Graph)

Convert entire database schema to homogeneous graph, and labeled with relevant table(s) and column(s) based on gold schema linking of each question.

One kind of nodes: no matter table or clolumn

One kind of edges: contain or foreign key

Results example:

```
=== Inspecting Homogeneous Graph: 0 ===

Graph Structure:
Data(x=[25, 25], edge_index=[2, 48], y=[25])

Number of nodes: 25
Node feature dimension: 25

Node labels (1=relevant, 0=not relevant):
Node 0: label=0
Node 1: label=0
Node 2: label=0
Node 3: label=0
Node 4: label=0
Node 5: label=0
Node 6: label=0
Node 7: label=0
Node 8: label=1
Node 9: label=0
Node 10: label=0
Node 11: label=0
Node 12: label=0
Node 13: label=0
Node 14: label=0
Node 15: label=0
Node 16: label=0
Node 17: label=0
Node 18: label=0
Node 19: label=0
Node 20: label=0
Node 21: label=0
Node 22: label=0
Node 23: label=0
Node 24: label=0

Edges:
Total number of edges: 48

First 10 edges:
Node 0 -> Node 1
Node 1 -> Node 0
Node 0 -> Node 2
Node 2 -> Node 0
Node 0 -> Node 3
Node 3 -> Node 0
Node 0 -> Node 4
Node 4 -> Node 0
Node 0 -> Node 5
Node 5 -> Node 0

Graph Statistics:
Average node degree: 1.92
Number of relevant nodes: 1
Percentage of relevant nodes: 4.00%
```
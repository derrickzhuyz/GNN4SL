<h1 align="center">GNN4SL: Semantic-Aware Graph Learning <br> for Schema Linking in Natural Language to SQL</h1>

Schema linking plays a pivotal role in bridging the gap between natural language queries and database schemas in Natural Language to SQL (NL2SQL) tasks. GNN4SL leverages graph neural networks (GNNs) to enhance schema linking by reformulating it as a link prediction problem. This approach integrates the structural insights of graph learning with the semantic understanding capabilities of large language models.

## Key Contributions

1. **Reformulation of Schema Linking**:
   The schema linking task is treated as a link prediction problem within a graph learning framework. Here, nodes represent schema elements (e.g., tables, columns) and natural language (NL) query elements, while edges signify relationships between these nodes.
2. **Semantic-Aware Embeddings**:
   By utilizing large language models, semantic vector embeddings are generated to capture the rich contextual and semantic information inherent in NL queries and schema elements.
3. **Graph Construction and Dataset Preparation**:
   Graph datasets are constructed from the Spider and BIRD benchmarks. Each database is modeled as a graph where tables and columns are represented as nodes, and NL queries are incorporated as query nodes connected to relevant schema nodes. This structure enables the GNN model to learn schema structural patterns.
4. **Link-Level Graph Neural Networks**:
   Link-level GNNs are trained to predict connections between query nodes and schema nodes, thereby solving the schema linking task. We support GCN, GAT, and RGAT models.

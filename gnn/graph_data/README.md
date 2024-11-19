## Documentation for graph data construction

To construct graph datasets for schema linking tasks

* nl_embedder: a class to embed natrual language like names or questions into vectors, which are used as the node features (initial features to be upgraded by GNN)
* nl_embedding_processor: to process and conduct embedding, and store properly
* node_level_graph_dataset:

  * treat schema linking as node-level classification task, each node (table/column) has a label denoting if it is relevant to current question;
  * construct graph datasets for node-level task;
  * each example in datasets (spider or bird) corresponds to a node-labeled graph;
  * for each dataset, the number of generated graphs equals to the number of examples (or to say questions).
* inspect_node_level_pt: to visualize and inspect obtained node-level graph datasets in .pt format, check if these graphs meet our requirements
* link_level_graph_dataset:

  * treat schema linking as link-level classification task, alongside edges denoting contain and foreign key, if a table/column node is related to the current question, there will be a edge betweem them;
  * contruct graph datasets for link-level task: link prediction;
  * there will be many questions for one database, so in the graph for a database, there will be many question nodes;
  * for each dataset, the number of generated graphs equals to the number of databases.
* inspect_link_level_pt: to visualize and inspect obtained link-level graph datasets in .pt format, check if these graphs meet our requirements

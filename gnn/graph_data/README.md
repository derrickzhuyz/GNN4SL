To construct graph datasets for schema linking tasks

* nl_embedder: a class to embed natrual language like names or questions into vectors
* nl_embedding_processor: to process and conduct embedding, and store properly
* node_level_graph_dataset:

  * treat schema linking as node-level classification task, each node (table/column) has a label denoting if it is relevant to current question
  * construct graph datasets for node-level task
  * each example in datasets (spider or bird) corresponds to a node-labeled graph
* inspect_pt: to visualize and inspect obtained graph datasets in .pt format, check if these graphs meet our requirements

#!/bin/bash

# Set the environment variables if needed
# export VARIABLE_NAME=value

# Set the visible GPU(s)
export CUDA_VISIBLE_DEVICES='1,2'

# Train the link-level model
echo "Starting training of the link-level model..."
python -m gnn.model.link_level_train

# # Test the link-level model
# echo "Starting evaluation of the link-level model..."
# python -m gnn.model.link_level_evaluate


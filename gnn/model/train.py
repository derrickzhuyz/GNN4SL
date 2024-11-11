# import torch
# import torch.nn.functional as F
# from torch_geometric.loader import DataLoader
# from model import SchemaLinkingGNN
# from graph_dataset import SchemaLinkingGraphDataset
# from sklearn.metrics import f1_score
# import numpy as np

# def train_epoch(model, loader, optimizer, device):
#     model.train()
#     total_loss = 0
    
#     for batch in loader:
#         batch = batch.to(device)
#         optimizer.zero_grad()
        
#         # Forward pass
#         table_out, column_out = model(batch.x_dict, batch.edge_index_dict)
        
#         # Calculate loss for both table and column nodes
#         table_loss = F.binary_cross_entropy(table_out, batch['table'].y.float())
#         column_loss = F.binary_cross_entropy(column_out, batch['column'].y.float())
#         loss = table_loss + column_loss
        
#         # Backward pass
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
    
#     return total_loss / len(loader)

# @torch.no_grad()
# def evaluate(model, loader, device):
#     model.eval()
#     table_preds, table_labels = [], []
#     column_preds, column_labels = [], []
    
#     for batch in loader:
#         batch = batch.to(device)
#         table_out, column_out = model(batch.x_dict, batch.edge_index_dict)
        
#         # Convert predictions to binary (0 or 1)
#         table_pred = (table_out > 0.5).float()
#         column_pred = (column_out > 0.5).float()
        
#         table_preds.append(table_pred.cpu())
#         table_labels.append(batch['table'].y.cpu())
#         column_preds.append(column_pred.cpu())
#         column_labels.append(batch['column'].y.cpu())
    
#     # Calculate F1 scores
#     table_preds = torch.cat(table_preds, dim=0).numpy()
#     table_labels = torch.cat(table_labels, dim=0).numpy()
#     column_preds = torch.cat(column_preds, dim=0).numpy()
#     column_labels = torch.cat(column_labels, dim=0).numpy()
    
#     table_f1 = f1_score(table_labels, table_preds, average='binary')
#     column_f1 = f1_score(column_labels, column_preds, average='binary')
    
#     return table_f1, column_f1

# def main():
#     # Hyperparameters
#     hidden_channels = 256
#     num_layers = 3
#     batch_size = 32
#     num_epochs = 100
#     lr = 0.001
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Load datasets
#     spider_train = SchemaLinkingGraphDataset(root='data/', dataset_type='spider', split='train')
#     spider_dev = SchemaLinkingGraphDataset(root='data/', dataset_type='spider', split='dev')
#     bird_train = SchemaLinkingGraphDataset(root='data/', dataset_type='spider', split='train')
#     bird_dev = SchemaLinkingGraphDataset(root='data/', dataset_type='bird', split='dev')
    
#     # Create data loaders
#     train_loader = DataLoader(spider_train + bird_train, batch_size=batch_size, shuffle=True)
#     spider_dev_loader = DataLoader(spider_dev, batch_size=batch_size)
#     bird_dev_loader = DataLoader(bird_dev, batch_size=batch_size)
    
#     # Initialize model
#     model = SchemaLinkingGNN(hidden_channels, num_layers).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
#     # Training loop
#     best_f1 = 0
#     for epoch in range(num_epochs):
#         # Train
#         loss = train_epoch(model, train_loader, optimizer, device)
        
#         # Evaluate
#         spider_table_f1, spider_column_f1 = evaluate(model, spider_dev_loader, device)
#         bird_table_f1, bird_column_f1 = evaluate(model, bird_dev_loader, device)
        
#         # Calculate average F1 score
#         avg_f1 = (spider_table_f1 + spider_column_f1 + bird_table_f1 + bird_column_f1) / 4
        
#         print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
#         print(f'Spider Dev - Table F1: {spider_table_f1:.4f}, Column F1: {spider_column_f1:.4f}')
#         print(f'BIRD Dev - Table F1: {bird_table_f1:.4f}, Column F1: {bird_column_f1:.4f}')
        
#         # Save best model
#         if avg_f1 > best_f1:
#             best_f1 = avg_f1
#             torch.save(model.state_dict(), 'best_model.pt')

# if __name__ == '__main__':
#     main()
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from tqdm import tqdm
import os
import chardet
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader as GeometricDataLoader
import time

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    return result['encoding']

def load_and_preprocess_data(gt_file, include_labels):
    encoding = detect_encoding(gt_file)
    gt_df = pd.read_csv(gt_file, encoding=encoding)
    gt_df = gt_df.drop_duplicates()
    gt_df = gt_df[gt_df['fine_grained_label'].isin(include_labels)]
    return gt_df

def process_tables(gt_df, data_folder):
    files_data = {}
    continuous_cols, all_values, file_names, labels_list, additional_features, headers = [], [], [], [], [], []
    label_counts = gt_df['fine_grained_label'].value_counts()

    for file_name in tqdm(gt_df['fileName'].unique(), desc="Processing tables"):
        path = os.path.join(data_folder, file_name)
        try:
            files_data[file_name] = pd.read_csv(path)
        except Exception as e:
            print(f"Failed to read {file_name}: {e}")
            continue

        table = files_data[file_name]
        for _, row in gt_df[gt_df['fileName'] == file_name].iterrows():
            column_name = row['fine_grained_label']
            column_index = table.columns.get_loc(row['colName']) if row['colName'] in table.columns else -1
            if column_index == -1 or label_counts.get(column_name, 0) < 2:
                continue

            selected_column = table.iloc[:, column_index].replace([np.inf, -np.inf], np.nan).dropna()
            if selected_column.empty or not pd.api.types.is_numeric_dtype(selected_column):
                continue

            continuous_cols.append(selected_column.values.reshape(-1, 1))
            all_values.extend(selected_column)
            file_names.append((file_name, column_index))
            labels_list.append(column_name)
            headers.append(row['colName'])
            additional_features.append([
                selected_column.nunique(),
                selected_column.mean(),
                np.std(selected_column) / selected_column.mean() if selected_column.mean() != 0 else 0,
                entropy(selected_column.value_counts(normalize=True), base=2),
                float(selected_column.max()) - float(selected_column.min()),
                np.percentile(selected_column, 10),
                np.percentile(selected_column, 90)
            ])

    return continuous_cols, all_values, file_names, labels_list, additional_features, headers

def encode_headers_with_sbert(column_headers):
    start_time = time.time()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"SBERT model loaded in {time.time() - start_time:.2f} seconds.")
    
    start_time = time.time()
    header_embeddings = model.encode(column_headers, convert_to_tensor=True)
    print(f"Headers encoded in {time.time() - start_time:.2f} seconds.")
    
    return header_embeddings

# Pythagoras model class
class PythagorasModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, gnn_layers=2):
        super(PythagorasModel, self).__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(gnn_layers)])
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = self.feature_encoder(x)
        for conv in self.convs:
            h = conv(h, edge_index)
        hg = global_mean_pool(h, batch)
        return self.classifier(hg), h

def create_graph_representation(continuous_cols, column_headers, additional_features, encoded_labels):
    header_embeddings = encode_headers_with_sbert(column_headers)
    embedding_dim = header_embeddings.shape[1]
    additional_features_tensor = torch.tensor(additional_features, dtype=torch.float)
    
    data_list = []
    for i in tqdm(range(len(column_headers)), desc='Creating graph representations'):
        node_features = []
        edge_index = []

        # Column header node
        column_node_features = torch.cat([header_embeddings[i].unsqueeze(0), additional_features_tensor[i].unsqueeze(0)], dim=1)
        node_features.append(column_node_features.squeeze(0))

       
        edge_index = torch.empty((2, 0), dtype=torch.long)

        node_features = torch.stack(node_features)

        data = Data(x=node_features, edge_index=edge_index, y=torch.tensor([encoded_labels[i]], dtype=torch.long))
        data_list.append(data)

    return data_list

def precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix):
    output_data = []  
    precision_recall_data = []  
    unique_labels = sorted(set(labels_list))

    total_avg_precision = 0.0  

    for iteration, label in tqdm(enumerate(unique_labels, start=1), desc="Analyzing precision and recall", total=len(unique_labels)):
        label_indices = [i for i, x in enumerate(labels_list) if x == label]

        total_tp, total_fp, total_fn = 0, 0, 0
        total_precision, total_recall = 0.0, 0.0
        label_output_data = []

        for selected_column_index in label_indices:
            selected_column = file_names[selected_column_index]
            similarities = cosine_sim_matrix[selected_column_index]
            sorted_indices = np.argsort(-similarities)
            top_k_indices = [idx for idx in sorted_indices[1:] if idx < len(labels_list)]

            tp = sum(labels_list[idx] == label for idx in top_k_indices)
            fp = len(top_k_indices) - tp
            fn = label_counts[label] - tp

            total_tp += tp
            total_fp += fp
            total_fn += fn

            if tp + fp > 0:
                instance_precision = tp / (tp + fp)
                total_precision += instance_precision

            if tp + fn > 0:
                instance_recall = tp / (tp + fn)
                total_recall += instance_recall

            label_output_data.extend([
                (iteration, selected_column[0], selected_column[1], label, 
                 file_names[idx][0], file_names[idx][1], labels_list[idx], similarities[idx]) 
                for idx in top_k_indices
            ])

        avg_precision = total_precision / len(label_indices) if label_indices else 0
        avg_recall = total_recall / len(label_indices) if label_indices else 0
        precision_recall_data.append([iteration, label, avg_precision, avg_recall, total_tp, total_fp, total_fn])
        output_data.extend(label_output_data)

        total_avg_precision += avg_precision  # Accumulate average precision

    overall_avg_precision = total_avg_precision / len(unique_labels) if unique_labels else 0
    print(f"Overall Average Precision: {overall_avg_precision:.4f}")

    return output_data, precision_recall_data

def perform_analysis(file_names, labels_list, label_counts, latent_space, analysis_name):
    # Compute cosine similarity matrix for latent space
    cosine_sim_matrix = cosine_similarity(latent_space)
    
    # Perform precision-recall analysis
    output_data, precision_recall_data = precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix)
    
    # Save results to CSV files
    neighbors_df = pd.DataFrame(output_data, columns=['Iteration', 'Selected_File', 'Selected_Column', 'Selected_Label', 
                                                      'Neighbor_File', 'Neighbor_Column', 'Neighbor_Label', 'Similarity'])
    neighbors_df.to_csv(f'top_neighbors_{analysis_name}.csv', index=False)
    
    precision_recall_df = pd.DataFrame(precision_recall_data, columns=['Iteration', 'Label', 'Precision', 'Recall', 
                                                                       'True_Positives', 'False_Positives', 'False_Negatives'])
    precision_recall_df.to_csv(f'precision_recall_{analysis_name}.csv', index=False)

    total_avg_precision = sum([row[2] for row in precision_recall_data]) / len(precision_recall_data) if precision_recall_data else 0
    print(f"Overall Average Precision for {analysis_name}: {total_avg_precision:.4f}")



# Main 
# Main execution settings
include_labels = [
    'Year', '..'
]

data_folder = 'tables/'
gt_file = 'updated_column_gt.csv'

start_time = time.time()
gt_df = load_and_preprocess_data(gt_file, include_labels)
label_counts = gt_df['fine_grained_label'].value_counts()
print(f"Data loaded and preprocessed in {time.time() - start_time:.2f} seconds.")

start_time = time.time()
continuous_cols, all_values, processed_file_names, labels_list, additional_features, column_headers = process_tables(gt_df, data_folder)
print(f"Tables processed in {time.time() - start_time:.2f} seconds.")

print(f"Total number of selected columns: {len(continuous_cols)}")
print(f"Total cluster count (unique 'annotation_label' count): {len(set(labels_list))}")

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels_list)

# 
assert len(processed_file_names) == len(labels_list) == len(encoded_labels), "Mismatch in lengths of processed data structures"

# Create graph representation for GNN
start_time = time.time()
data_list = create_graph_representation(continuous_cols, column_headers, additional_features, encoded_labels)
print(f"Graph representations created in {time.time() - start_time:.2f} seconds.")

# Define dimensions
input_dim = data_list[0].x.shape[1]
hidden_dim = 768
output_dim = len(set(encoded_labels))

# Initialize and train Pythagoras model
pythagoras_model = PythagorasModel(input_dim, hidden_dim, output_dim, gnn_layers=4)
pythagoras_model.train()

# Create DataLoader for training
geometric_dataloader = GeometricDataLoader(data_list, batch_size=64, shuffle=True)

# Train the model
optimizer = optim.Adam(pythagoras_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

start_time = time.time()
for epoch in tqdm(range(50), desc='Training epochs'):  # Example epoch count
    for data in tqdm(geometric_dataloader, desc=f'Epoch {epoch+1}', leave=False):
        optimizer.zero_grad()
        outputs, _ = pythagoras_model(data)
        loss = criterion(outputs, data.y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/50, Loss: {loss.item()}")
print(f"Model trained in {time.time() - start_time:.2f} seconds.")

# Extract latent space
latent_spaces = []
pythagoras_model.eval()
with torch.no_grad():
    for data in tqdm(geometric_dataloader, desc='Extracting latent space'):
        _, latent_space = pythagoras_model(data)
        latent_spaces.append(latent_space)

latent_space = torch.cat(latent_spaces, dim=0)

# Perform analysis
perform_analysis(processed_file_names, labels_list, label_counts, latent_space.numpy(), 'headers_values_transformer')

import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os

def load_and_preprocess_data(data_folder, gt_file, include_labels):
    """
    Load the ground truth data and filter based on specified labels.
    """
    try:
        gt_df = pd.read_csv(os.path.join(data_folder, gt_file), encoding='ISO-8859-1')
    except UnicodeDecodeError:
        gt_df = pd.read_csv(os.path.join(data_folder, gt_file), encoding='utf-8')
    return gt_df[gt_df['ColumnLabel'].isin(include_labels)]

def process_tables(gt_df, data_folder):
    """
    Process each table to extract continuous columns and their values for further processing.
    """
    continuous_cols, all_values, file_names, labels_list = [], [], [], []
    label_counts = gt_df['ColumnLabel'].value_counts()

    for file_name in tqdm(gt_df['fileName'].unique(), desc='Processing tables'):
        path = os.path.join(data_folder, file_name)
        try:
            
            for chunk in pd.read_csv(path, encoding='ISO-8859-1', chunksize=10000):
                table = chunk
        except Exception as e:
            print(f"Failed to read {file_name}: {e}")
            continue

        for _, row in gt_df[gt_df['fileName'] == file_name].iterrows():
            column_name = row['ColumnLabel']
            column_index = table.columns.get_loc(row['colName']) if row['colName'] in table.columns else -1
            if column_index == -1 or label_counts.get(column_name, 0) < 2:
                continue

            # Select and clean the column
            selected_column = table.iloc[:, column_index].replace([np.inf, -np.inf], np.nan).dropna()
            if selected_column.empty or not pd.api.types.is_numeric_dtype(selected_column):
                continue

           
            continuous_cols.append(selected_column.values.reshape(-1, 1))
            all_values.extend(selected_column)
            file_names.append((file_name, column_index))
            labels_list.append(column_name)

    return continuous_cols, all_values, file_names, labels_list

# Squashing function with vectorization
def squashing_function(x):
    x = np.asarray(x)
    pos_mask = x > 1
    neg_mask = x < -1
    x[pos_mask] = np.log(x[pos_mask]) + 1
    x[neg_mask] = -np.log(-x[neg_mask]) - 1
    return x

def prototype_induction_som(all_values, num_prototypes=50):
    """
    Train a Self-Organizing Map (SOM) and extract prototypes from the data.
    """
    som = MiniSom(1, num_prototypes, 1)
    som.train(np.array(all_values).reshape(-1, 1), 100)
    return som.get_weights().flatten()

def similarity_function(n, prototypes, beta=1.0):
    """
    Calculate similarity based on the squashed values and prototypes.
    """
    g_n = squashing_function(n)
    g_prototypes = squashing_function(prototypes)
    differences = np.abs(g_n - g_prototypes)
    differences = np.where(differences == 0, 1e-10, differences)  # Avoid division by zero
    return np.power(differences, -beta)

def calculate_embeddings(continuous_cols, prototypes):
    """
    Calculate embeddings for each column based on the similarity function and prototypes.
    """
    dim = len(prototypes)
    embeddings = np.zeros((len(continuous_cols), dim))

    # Vectorized loop for more efficient calculation
    for i, col in tqdm(enumerate(continuous_cols), total=len(continuous_cols), desc='Calculating embeddings'):
        col = np.asarray(col).flatten()
        valid_vals = col[~np.isnan(col) & (col != 0)]
        if valid_vals.size > 0:
            similarities = similarity_function(valid_vals[:, None], prototypes)
            normalized_similarities = similarities / np.sum(similarities, axis=1, keepdims=True)
            embeddings[i, :] = np.mean(normalized_similarities, axis=0)

    return embeddings

def precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix):
    """
    Perform precision-recall analysis and calculate precision/recall metrics for the data.
    """
    output_data, precision_recall_data = [], []
    unique_labels = sorted(set(labels_list))
    total_avg_precision = 0.0  # To accumulate average precision for all labels

    for iteration, label in tqdm(enumerate(unique_labels, start=1), desc="Analyzing precision and recall", total=len(unique_labels)):
        label_indices = [i for i, x in enumerate(labels_list) if x == label]

        total_tp, total_fp, total_fn = 0, 0, 0
        total_precision, total_recall = 0.0, 0.0
        label_output_data = []

        for selected_column_index in label_indices:
            selected_column = file_names[selected_column_index]
            similarities = cosine_sim_matrix[selected_column_index]
            sorted_indices = np.argsort(-similarities)
            top_k_indices = sorted_indices[1:label_counts[label] + 1]

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
        precision_recall_data.append([iteration, avg_precision, avg_recall, label, label_counts[label], total_tp, total_fp, total_fn])
        output_data.extend(label_output_data)

        total_avg_precision += avg_precision  # Accumulate average precision

    overall_avg_precision = total_avg_precision / len(unique_labels) if unique_labels else 0
    print(f"Overall Average Precision: {overall_avg_precision:.4f}")

    return output_data, precision_recall_data

# Main execution settings
include_labels = [
    'Year', '..'
]

# Main 
data_folder = 'tables/'
gt_file = 'updated_column_gt.csv'
gt_df = load_and_preprocess_data(data_folder, gt_file, include_labels)
label_counts = gt_df['ColumnLabel'].value_counts()

continuous_cols, all_values, file_names, labels_list = process_tables(gt_df, data_folder)
print(f"Total number of selected columns: {len(continuous_cols)}")
print(f"Total cluster count (unique 'annotation_label' count): {len(set(labels_list))}")

# Use SOM for prototype induction
prototypes = prototype_induction_som(all_values, num_prototypes=50)
embeddings = calculate_embeddings(continuous_cols, prototypes)

print(f"Embedding matrix shape: {embeddings.shape}")

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels_list)
np.savetxt('labels.txt', encoded_labels, fmt='%d')

cosine_sim_matrix = cosine_similarity(embeddings)
output_data, precision_recall_data = precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix)

# Save results to CSV files
neighbors_df = pd.DataFrame(output_data, columns=['Iteration', 'Selected_File', 'Selected_Column', 'Selected_Label', 
                                                  'Neighbor_File', 'Neighbor_Column', 'Neighbor_Label', 'Similarity'])
neighbors_df.to_csv('top_neighbors.csv', index=False)

precision_recall_df = pd.DataFrame(precision_recall_data, columns=['Iteration', 'Avg_Precision', 'Avg_Recall', 'Selected_Label', 
                                                                   'K', 'True_Positives', 'False_Positives', 'False_Negatives'])
precision_recall_df.to_csv('precision_recall.csv', index=False)

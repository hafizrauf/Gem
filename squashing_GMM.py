import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os

# Load and preprocess data
def load_and_preprocess_data(data_folder, gt_file, include_labels):
    try:
        gt_df = pd.read_csv(os.path.join(data_folder, gt_file), encoding='ISO-8859-1')
    except UnicodeDecodeError:
        gt_df = pd.read_csv(os.path.join(data_folder, gt_file), encoding='utf-8')
    gt_df = gt_df[gt_df['ColumnLabel'].isin(include_labels)]
    return gt_df

# Process tables
def process_tables(gt_df, data_folder):
    continuous_cols, all_values, file_names, labels_list = [], [], [], []
    label_counts = gt_df['ColumnLabel'].value_counts()

    for file_name in tqdm(gt_df['fileName'].unique(), desc='Processing tables'):
        path = os.path.join(data_folder, file_name)
        try:
            table = pd.read_csv(path, encoding='ISO-8859-1')
        except Exception as e:
            print(f"Failed to read {file_name}: {e}")
            continue

        for _, row in gt_df[gt_df['fileName'] == file_name].iterrows():
            column_name = row['ColumnLabel']
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

    return continuous_cols, all_values, file_names, labels_list

# Squashing function
def squashing_function(x):
    if x > 1:
        return np.log(x) + 1
    elif x < -1:
        return -np.log(-x) - 1
    else:
        return x

# Prototype induction using GMM
def prototype_induction(all_values, num_prototypes=5):
    gmm = GaussianMixture(n_components=num_prototypes, random_state=0)
    gmm.fit(np.array(all_values).reshape(-1, 1))
    prototypes = gmm.means_.flatten()
    return prototypes

def similarity_function(n, prototypes, beta=1.0):
    g_n = squashing_function(n)
    g_prototypes = np.array([squashing_function(p) for p in prototypes])
    differences = np.abs(g_n - g_prototypes)
    differences = np.where(differences == 0, 1e-10, differences)
    return np.power(differences, -beta)

def calculate_embeddings(continuous_cols, prototypes):
    dim = len(prototypes)
    embeddings = np.zeros((len(continuous_cols), dim))

    for i, col in tqdm(enumerate(continuous_cols), total=len(continuous_cols), desc='Calculating embeddings'):
        col_embeddings = []
        for n in col:
            if not np.isnan(n[0]) and n[0] != 0:  # Skip NaN or zero values
                similarities = similarity_function(n[0], prototypes)
                normalized_similarities = similarities / np.sum(similarities)
                col_embeddings.append(normalized_similarities)
        
        if col_embeddings:
            embeddings[i, :] = np.nanmean(col_embeddings, axis=0)  # Average non-NaN embeddings

    return embeddings

def precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix):
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

# Main execution starts here
data_folder = 'tables/'
gt_file = 'updated_column_gt.csv'

# Load and preprocess data using the include list
gt_df = load_and_preprocess_data(data_folder, gt_file, include_labels)
label_counts = gt_df['ColumnLabel'].value_counts()

# Process tables
continuous_cols, all_values, file_names, labels_list = process_tables(gt_df, data_folder)
print(f"Total number of selected columns: {len(continuous_cols)}")
print(f"Total cluster count (unique 'annotation_label' count): {len(set(labels_list))}")

# Prototype induction using Gaussian Mixture Model
prototypes = prototype_induction(all_values, num_prototypes=5)
embeddings = calculate_embeddings(continuous_cols, prototypes)

print(f"Embedding matrix shape: {embeddings.shape}")

# Label encoding and similarity computation
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels_list)

cosine_sim_matrix = cosine_similarity(embeddings)
output_data, precision_recall_data = precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix)

# Save results to CSV files
neighbors_df = pd.DataFrame(output_data, columns=['Iteration', 'Selected_File', 'Selected_Column', 'Selected_Label', 
                                                  'Neighbor_File', 'Neighbor_Column', 'Neighbor_Label', 'Similarity'])
neighbors_df.to_csv('top_neighbors.csv', index=False)

precision_recall_df = pd.DataFrame(precision_recall_data, columns=['Iteration', 'Avg_Precision', 'Avg_Recall', 'Selected_Label', 
                                                                   'K', 'True_Positives', 'False_Positives', 'False_Negatives'])
precision_recall_df.to_csv('precision_recall.csv', index=False)

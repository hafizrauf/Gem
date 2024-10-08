import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os

def piecewise_linear_encoding(column, num_bins):
    # Calculate the quantiles including duplicates
    quantiles = np.quantile(column, np.linspace(0, 1, num_bins + 1))
    bins = quantiles  # Use exact quantiles as bins
    encoded_values = np.zeros((len(column), num_bins))

    for i, value in enumerate(column):
        for t in range(1, len(bins)):
            if value < bins[t]:
                bin_width = max(bins[t] - bins[t - 1], np.finfo(float).eps)
                encoded_values[i, t - 1] = (value - bins[t - 1]) / bin_width
                break
            elif value >= bins[-1]:
                encoded_values[i, -1] = 1
                break
            else:
                encoded_values[i, t - 1] = 1

    return encoded_values

def process_tables_with_ple(gt_df, data_folder, num_bins=10):
    encoded_matrices, file_names, labels_list = [], [], []
    label_counts = gt_df['annotation_label'].value_counts()
    for _, row in tqdm(gt_df.iterrows(), desc='Processing tables with PLE'):
        table_id = row['table_id']
        target_column = f'col{row["target_column"]}'
        annotation_label = row['annotation_label']

        if label_counts[annotation_label] < 10:
            continue

        try:
            table = pd.read_csv(os.path.join(data_folder, f'{table_id}.csv'))
            if pd.api.types.is_numeric_dtype(table[target_column]):
                selected_column = table[target_column].replace([np.inf, -np.inf], np.nan).dropna()
                if len(selected_column) > 0:
                    encoded_matrix = piecewise_linear_encoding(selected_column, num_bins)
                    if np.isnan(encoded_matrix).any():
                        print(f"NaN values found in encoded matrix for column: {target_column} in file: {table_id}. Skipping this column.")
                    else:
                        encoded_matrices.append(encoded_matrix.mean(axis=0))
                        file_names.append((table_id, target_column))
                        labels_list.append(annotation_label)
        except Exception as e:
            print(f'Error processing "{table_id}". Error message: {e}')
    return encoded_matrices, file_names, labels_list

def load_and_preprocess_data(data_folder, gt_file, exclude_labels):
    gt_df = pd.read_csv(gt_file)
    gt_df = gt_df[~gt_df['annotation_label'].isin(exclude_labels)]
    return gt_df

def precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix):
    output_data = []  # This will store detailed comparison results.
    precision_recall_data = []  # This stores summary statistics.
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
        precision_recall_data.append([iteration, label, avg_precision, avg_recall, total_tp, total_fp, total_fn])
        output_data.extend(label_output_data)

        total_avg_precision += avg_precision  # Accumulate average precision

    overall_avg_precision = total_avg_precision / len(unique_labels) if unique_labels else 0
    print(f"Overall Average Precision: {overall_avg_precision:.4f}")

    return output_data, precision_recall_data

# Main execution starts here
data_folder = 'tables/'
gt_file = 'dbpedia_gt.csv'
exclude_labels = ['id', 'year', 'url', 'date', 'serial number', 'postal code', 'email']

gt_df = load_and_preprocess_data(data_folder, gt_file, exclude_labels)
label_counts = gt_df['annotation_label'].value_counts()

num_bins = 50
encoded_matrices, file_names, labels_list = process_tables_with_ple(gt_df, data_folder, num_bins)

print(f"Total number of selected columns: {len(encoded_matrices)}")
print(f"Total cluster count (unique 'annotation_label' count): {len(set(labels_list))}")

encoded_matrix = np.vstack(encoded_matrices)
cosine_sim_matrix = cosine_similarity(encoded_matrix)
np.savetxt('embedding.txt', cosine_sim_matrix)
np.savetxt('labels_text.txt', np.array(labels_list).reshape(-1, 1), fmt='%s')

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels_list)
np.savetxt('labels.txt', encoded_labels, fmt='%d')

output_data, precision_recall_data = precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix)

neighbors_df = pd.DataFrame(output_data, columns=['Iteration', 'Selected_File', 'Selected_Column', 'Selected_Label', 
                                                  'Neighbor_File', 'Neighbor_Column', 'Neighbor_Label', 'Similarity'])
neighbors_df.to_csv('top_neighbors.csv', index=False)

precision_recall_df = pd.DataFrame(precision_recall_data, columns=['Iteration', 'Label', 'Precision', 'Recall', 'True_Positives', 'False_Positives', 'False_Negatives'])
precision_recall_df.to_csv('precision_recall.csv', index=False)

# find similar pairs
def find_similar_pairs(cosine_sim_matrix, encoded_labels, file_names, threshold_similarity=0.80, max_pairs=100):
    similar_pairs = []
    used_columns = set()  # Set to track used columns

    for i in range(len(cosine_sim_matrix)):
        if i in used_columns:
            continue  # Skip if this column has already been used

        for j in range(i + 1, len(cosine_sim_matrix)):
            if j in used_columns:
                continue  # Skip if the other column has already been used

            if cosine_sim_matrix[i, j] > threshold_similarity and encoded_labels[i] != encoded_labels[j]:
                similar_pairs.append((i, j))
                used_columns.update([i, j])  # Mark both columns as used

                if len(similar_pairs) >= max_pairs:
                    return similar_pairs

    return similar_pairs

# save misclassifications
def save_top_misclassifications_with_column_values(similar_pairs, file_names, labels_list, cosine_sim_matrix, data_folder, folder='missclassifications/', top_n=100):
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Sort pairs by similarity in decreasing order and select top 100
    sorted_pairs = sorted(similar_pairs, key=lambda x: cosine_sim_matrix[x[0], x[1]], reverse=True)[:top_n]

    # Collect data for each pair including full column values
    misclassification_data = []
    for pair in sorted_pairs:
        file1_info, file2_info = file_names[pair[0]], file_names[pair[1]]
        similarity_score = cosine_sim_matrix[pair[0], pair[1]]

        # Load the full columns from the CSV files
        file1_path = os.path.join(data_folder, file1_info[0])
        file2_path = os.path.join(data_folder, file2_info[0])
        
        if not os.path.exists(file1_path):
            print(f'File not found: {file1_path}')
            continue
        if not os.path.exists(file2_path):
            print(f'File not found: {file2_path}')
            continue
        
        column1_values = pd.read_csv(file1_path).iloc[:, int(file1_info[1][3:])].tolist()
        column2_values = pd.read_csv(file2_path).iloc[:, int(file2_info[1][3:])].tolist()

        misclassification_data.append([
            file1_info[0], file1_info[1], labels_list[pair[0]], column1_values,
            file2_info[0], file2_info[1], labels_list[pair[1]], column2_values,
            similarity_score
        ])

    # Create DataFrame and save to CSV
    columns = ['File1_Name', 'Column1_Index', 'Label1', 'Column1_Values', 
               'File2_Name', 'Column2_Index', 'Label2', 'Column2_Values', 'Similarity']
    misclassifications_df = pd.DataFrame(misclassification_data, columns=columns)
    misclassifications_df.to_csv(f'{folder}top_misclassifications_full.csv', index=False)

# Generate similar pairs
similar_pairs = find_similar_pairs(cosine_sim_matrix, encoded_labels, file_names)

# Example usage
save_top_misclassifications_with_column_values(similar_pairs, file_names, labels_list, cosine_sim_matrix, data_folder)

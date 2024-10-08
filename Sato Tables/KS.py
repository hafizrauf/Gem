import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, norm, uniform, expon, beta, gamma, lognorm, logistic, entropy
from tqdm import tqdm
import os

#  load and preprocess data
def load_and_preprocess_data(data_folder, gt_file, exclude_labels):
    gt_df = pd.read_csv(gt_file)
    gt_df = gt_df[~gt_df['Column_Name'].isin(exclude_labels)]
    return gt_df

#  process each table
def process_tables(gt_df, data_folder):
    # Read data once and store in a dictionary
    files_data = {}
    label_counts = gt_df['Column_Name'].value_counts()

    for file_name in gt_df['File_Name'].unique():
        try:
            files_data[file_name] = pd.read_csv(os.path.join(data_folder, file_name))
        except Exception as e:
            print(f"Failed to read {file_name}: {e}")

    continuous_cols, all_values, file_names, labels_list = [], [], [], []

    for _, row in tqdm(gt_df.iterrows(), desc='Processing tables'):
        file_name, column_name, column_index = row['File_Name'], row['Column_Name'], row['Column_Index']

        if label_counts[column_name] < 10:
            continue

        table = files_data.get(file_name)
        if table is None or not pd.api.types.is_numeric_dtype(table.iloc[:, column_index]):
            continue

        selected_column = table.iloc[:, column_index].replace([np.inf, -np.inf], np.nan).dropna()
        if selected_column.empty:
            continue

        continuous_cols.append(selected_column.values.reshape(-1, 1))
        all_values.extend(selected_column)
        file_names.append((file_name, column_index))
        labels_list.append(column_name)

    return continuous_cols, all_values, file_names, labels_list

#  calculate KS matrix
def calculate_ks_matrix(continuous_cols):
    if len(continuous_cols) == 0:
        raise ValueError("No valid continuous columns found for KS matrix calculation.")

    ks_matrix = np.zeros((len(continuous_cols), 7))  # 7 distributions: normal, uniform, exponential, beta, gamma, lognormal, logistic

    for i, column in tqdm(enumerate(continuous_cols), desc="Calculating KS matrix", total=len(continuous_cols)):
        if column.size > 0:
            # KS statistic against different distributions
            ks_stat_norm, _ = ks_2samp(column.flatten(), norm.rvs(size=10000))
            ks_stat_uniform, _ = ks_2samp(column.flatten(), uniform.rvs(size=10000))
            ks_stat_expon, _ = ks_2samp(column.flatten(), expon.rvs(size=10000))
            ks_stat_beta, _ = ks_2samp(column.flatten(), beta.rvs(a=2, b=5, size=10000))
            ks_stat_gamma, _ = ks_2samp(column.flatten(), gamma.rvs(a=2, size=10000))
            ks_stat_lognorm, _ = ks_2samp(column.flatten(), lognorm.rvs(s=0.954, size=10000))
            ks_stat_logistic, _ = ks_2samp(column.flatten(), logistic.rvs(size=10000))

            ks_matrix[i, :] = [ks_stat_norm, ks_stat_uniform, ks_stat_expon, ks_stat_beta, ks_stat_gamma, ks_stat_lognorm, ks_stat_logistic]
        else:
            ks_matrix[i, :] = 0  # If column is empty, set KS stats to 0

    # Normalize the matrix
    ks_matrix = ks_matrix / np.linalg.norm(ks_matrix, ord=1, axis=1, keepdims=True)

    # Replace NaN values (resulting from division by zero) with zeros
    ks_matrix = np.nan_to_num(ks_matrix)

    print(f"Shape of ks_matrix: {ks_matrix.shape}")
    return ks_matrix

# Optimized precision-recall analysis
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
gt_file = 'GT.csv'
exclude_labels = ['id', 'url', 'date', 'serial number', 'postal code', 'email', 'rank', 'ranking', 'fileSize', 'code', '']

gt_df = load_and_preprocess_data(data_folder, gt_file, exclude_labels)
label_counts = gt_df['Column_Name'].value_counts()

continuous_cols, all_values, file_names, labels_list = process_tables(gt_df, data_folder)

print(f"Total number of selected columns: {len(continuous_cols)}")
print(f"Total cluster count (unique 'annotation_label' count): {len(set(labels_list))}")

# Calculate KS statistics matrix
ks_matrix = calculate_ks_matrix(continuous_cols)

# Calculate cosine similarity matrix based on KS matrix
cosine_sim_matrix = cosine_similarity(ks_matrix)

# Calculate distance matrix
distance_matrix = cosine_distances(ks_matrix)
np.savetxt('embedding.txt', distance_matrix)
np.savetxt('labels_text.txt', np.array(labels_list).reshape(-1, 1), fmt='%s')

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels_list)
np.savetxt('labels.txt', encoded_labels, fmt='%d')

output_data, precision_recall_data = precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix)

# Save results to CSV files
neighbors_df = pd.DataFrame(output_data, columns=['Iteration', 'Selected_File', 'Selected_Column', 'Selected_Label', 
                                                  'Neighbor_File', 'Neighbor_Column', 'Neighbor_Label', 'Similarity'])
neighbors_df.to_csv('top_neighbors.csv', index=False)

precision_recall_df = pd.DataFrame(precision_recall_data, columns=['Iteration', 'Label', 'Precision', 'Recall', 'True_Positives', 'False_Positives', 'False_Negatives'])
precision_recall_df.to_csv('precision_recall.csv', index=False)

#  find similar pairs
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

#  save misclassifications
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
        column1_values = pd.read_csv(os.path.join(data_folder, file1_info[0])).iloc[:, file1_info[1]].tolist()
        column2_values = pd.read_csv(os.path.join(data_folder, file2_info[0])).iloc[:, file2_info[1]].tolist()

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

# Plot and save figure for KS statistic component counts
def plot_ks_component_counts(proba_matrix, n_components, filename='Fig1.png'):
    component_counts = np.zeros(n_components)
    for i in range(len(proba_matrix)):
        for j in range(n_components):
            if proba_matrix[i, j] > 0:
                component_counts[j] += 1

    plt.figure(figsize=(10, 5))
    plt.bar(range(n_components), component_counts)
    plt.xlabel('Component')
    plt.ylabel('Number of columns')
    plt.title('Number of columns sharing each KS statistic component')
    plt.xticks(range(n_components))
    plt.savefig(filename, dpi=600)
    plt.close()

plot_ks_component_counts(ks_matrix, ks_matrix.shape[1])

# Plot and save histogram with KS statistic components
def plot_histogram_ks(all_values, ks_matrix, filename='Fig2.png'):
    plt.figure(figsize=(10, 5))
    plt.hist(all_values, bins=50, density=True, alpha=0.6, color='g')

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 50)
    for i in range(ks_matrix.shape[1]):
        p = ks_matrix[:, i]
        plt.plot(x, p, linewidth=2)

    plt.title('KS Statistic Components')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.savefig(filename, dpi=600)
    plt.close()

plot_histogram_ks(all_values, ks_matrix)

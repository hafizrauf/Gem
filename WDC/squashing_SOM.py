import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor

def load_and_preprocess_data(data_folder, gt_file, exclude_labels):
    # Read the CSV file and filter labels
    try:
        gt_df = pd.read_csv(os.path.join(data_folder, gt_file), encoding='ISO-8859-1')
    except UnicodeDecodeError:
        gt_df = pd.read_csv(os.path.join(data_folder, gt_file), encoding='utf-8')
    return gt_df[~gt_df['ColumnLabel'].isin(exclude_labels)]

def process_single_file(file_name, data_folder):
    # Helper function to read a single file
    path = os.path.join(data_folder, file_name)
    try:
        return pd.read_csv(path, encoding='ISO-8859-1')
    except Exception as e:
        print(f"Failed to read {file_name}: {e}")
        return None

def process_tables(gt_df, data_folder):
    label_counts = gt_df['ColumnLabel'].value_counts()
    continuous_cols, all_values, file_names, labels_list, column_headers = [], [], [], [], []

    # Process files concurrently to speed up reading large datasets
    with ThreadPoolExecutor() as executor:
        file_data = {file_name: executor.submit(process_single_file, file_name, data_folder)
                     for file_name in gt_df['fileName'].unique()}

    for _, row in tqdm(gt_df.iterrows(), desc='Processing tables', total=len(gt_df)):
        file_name, column_name, col_name = row['fileName'], row['ColumnLabel'], row['colName']

        if label_counts.get(column_name, 0) < 2:
            continue

        table = file_data[file_name].result()  # Retrieve result from ThreadPoolExecutor
        if table is None:
            continue

        for column_index in range(table.shape[1]):
            selected_column = table.iloc[:, column_index].replace([np.inf, -np.inf], np.nan).dropna()
            if selected_column.empty or not pd.api.types.is_numeric_dtype(selected_column):
                continue

            if selected_column.dtype == np.bool_:
                selected_column = selected_column.astype(int)

            continuous_cols.append(selected_column.values.reshape(-1, 1))
            all_values.extend(selected_column)
            file_names.append((file_name, column_index))
            labels_list.append(column_name)
            column_headers.append(col_name)

    return continuous_cols, all_values, file_names, labels_list, column_headers
def precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix):
    output_data, precision_recall_data = [], []
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

        total_avg_precision += avg_precision

    overall_avg_precision = total_avg_precision / len(unique_labels) if unique_labels else 0
    print(f"Overall Average Precision: {overall_avg_precision:.4f}")

    return output_data, precision_recall_data

def squashing_function(x):
    return np.where(
        x > 1, np.log(x) + 1, 
        np.where(x < -1, -np.log(-x) - 1, x)
    )

def prototype_induction_som(all_values, num_prototypes=50):
    som = MiniSom(1, num_prototypes, 1)
    som.train(np.array(all_values).reshape(-1, 1), 100)
    return som.get_weights().flatten()

def similarity_function(n, prototypes, beta=1.0):
    g_n = squashing_function(n)
    g_prototypes = squashing_function(prototypes)
    differences = np.abs(g_n - g_prototypes)
    return np.power(np.where(differences == 0, 1e-10, differences), -beta)

def calculate_embeddings(continuous_cols, prototypes):
    dim = len(prototypes)
    embeddings = np.zeros((len(continuous_cols), dim))

    for i, col in tqdm(enumerate(continuous_cols), total=len(continuous_cols), desc='Calculating embeddings'):
        similarities = np.array([similarity_function(n[0], prototypes) for n in col if not np.isnan(n[0]) and n[0] != 0])
        if similarities.size:
            embeddings[i, :] = np.nanmean(similarities, axis=0)

    return embeddings

def precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix):
    output_data, precision_recall_data = [], []
    unique_labels = sorted(set(labels_list))

    for iteration, label in tqdm(enumerate(unique_labels, start=1), desc="Analyzing precision and recall", total=len(unique_labels)):
        label_indices = [i for i, x in enumerate(labels_list) if x == label]
        total_tp, total_fp, total_fn = 0, 0, 0
        total_precision, total_recall = 0.0, 0.0

        for selected_column_index in label_indices:
            selected_column = file_names[selected_column_index]
            similarities = cosine_sim_matrix[selected_column_index]
            top_k_indices = np.argsort(-similarities)[1:label_counts[label] + 1]

            tp = sum(labels_list[idx] == label for idx in top_k_indices)
            fp, fn = len(top_k_indices) - tp, label_counts[label] - tp
            total_tp, total_fp, total_fn = total_tp + tp, total_fp + fp, total_fn + fn

            total_precision += tp / (tp + fp) if tp + fp > 0 else 0
            total_recall += tp / (tp + fn) if tp + fn > 0 else 0

            output_data.extend([
                (iteration, selected_column[0], selected_column[1], label,
                 file_names[idx][0], file_names[idx][1], labels_list[idx], similarities[idx])
                for idx in top_k_indices
            ])

        precision_recall_data.append([iteration, label, total_precision / len(label_indices), total_recall / len(label_indices), total_tp, total_fp, total_fn])

    return output_data, precision_recall_data



data_folder = 'tables/'
gt_file = 'column_gt.csv'
exclude_labels = [
    'Airport', 'Book','..']

gt_df = load_and_preprocess_data(data_folder, gt_file, exclude_labels)


label_counts = gt_df['ColumnLabel'].value_counts()

continuous_cols, all_values, file_names, labels_list, column_headers = process_tables(gt_df, data_folder)
prototypes = prototype_induction_som(all_values, num_prototypes=50) 
embeddings = calculate_embeddings(continuous_cols, prototypes)

np.savetxt('embedding.txt', embeddings)
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels_list)
cosine_sim_matrix = cosine_similarity(embeddings)

# Precision and recall analysis
output_data, precision_recall_data = precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix)

# Calculate and print the overall average precision
precision_values = [entry[2] for entry in precision_recall_data]  # Extract precision from precision_recall_data
average_precision = np.mean(precision_values)
print(f"Overall Average Precision: {average_precision:.4f}")

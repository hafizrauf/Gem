import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import chardet
import torch
from concurrent.futures import ThreadPoolExecutor

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    return result['encoding']

def load_and_preprocess_data(gt_file, exclude_labels):
    encoding = detect_encoding(gt_file)
    gt_df = pd.read_csv(gt_file, encoding=encoding)
    gt_df = gt_df.drop_duplicates()
    gt_df = gt_df[~gt_df['fine_grained_label'].isin(exclude_labels)]
    return gt_df

def read_csv_parallel(file_name, data_folder):
    path = os.path.join(data_folder, file_name)
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read {file_name}: {e}")
        return None

def process_tables(gt_df, data_folder):
    files_data = {}
    label_counts = gt_df['fine_grained_label'].value_counts()

    file_names_unique = gt_df['fileName'].unique()
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(read_csv_parallel, file_names_unique, [data_folder]*len(file_names_unique)), desc='Loading tables', total=len(file_names_unique)))
    
    for file_name, table in zip(file_names_unique, results):
        if table is not None:
            files_data[file_name] = table

    values_list, file_names, labels_list, column_headers = [], [], [], []

    for _, row in tqdm(gt_df.iterrows(), desc='Processing tables', total=len(gt_df)):
        file_name, column_name, col_name = row['fileName'], row['fine_grained_label'], row['colName']
        
        if label_counts.get(column_name, 0) < 2:
            continue

        table = files_data.get(file_name)
        if table is None:
            continue

        for column_index in range(table.shape[1]):
            if pd.api.types.is_numeric_dtype(table.iloc[:, column_index]):
                selected_column = table.iloc[:, column_index].replace([np.inf, -np.inf], np.nan).dropna()
                if selected_column.empty:
                    continue

                if selected_column.dtype == np.bool_:
                    selected_column = selected_column.astype(int)

                if selected_column.dtype.kind not in 'biufc':
                    continue

                values_list.append(' '.join(selected_column.astype(str).values))
                file_names.append((file_name, column_index))
                labels_list.append(column_name)
                column_headers.append(col_name)

    return values_list, file_names, labels_list, column_headers

def encode_with_sbert_in_batches(data_list, model, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(data_list), batch_size), desc='Encoding with SBERT'):
        batch = data_list[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings)

def precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix):
    precision_recall_data = []
    unique_labels = sorted(set(labels_list))

    for iteration, label in tqdm(enumerate(unique_labels, start=1), desc='Precision-recall analysis', total=len(unique_labels)):
        label_indices = [i for i, x in enumerate(labels_list) if x == label]

        total_tp, total_fp, total_fn = 0, 0, 0
        total_precision, total_recall = 0.0, 0.0
        for selected_column_index in label_indices:
            selected_column = file_names[selected_column_index]
            similarities = cosine_sim_matrix[selected_column_index]
            sorted_indices = np.argsort(-similarities)
            top_k_indices = sorted_indices[1:label_counts[label] + 1]  

            tp = sum(labels_list[idx] == label for idx in top_k_indices)
            fp = len(top_k_indices) - tp
            fn = label_counts[label] - tp

            if tp + fp > 0:
                instance_precision = tp / (tp + fp)
                total_precision += instance_precision
                total_tp += tp
                total_fp += fp

            if tp + fn > 0:
                instance_recall = tp / (tp + fn)
                total_recall += instance_recall
                total_fn += fn

        avg_precision = total_precision / len(label_indices) if label_indices else 0
        avg_recall = total_recall / len(label_indices) if label_indices else 0

        precision_recall_data.append([iteration, avg_precision, avg_recall, label, label_counts[label], total_tp, total_fp, total_fn])

    return precision_recall_data



exclude_labels = [
    'Airport', 'Book','..']

data_folder = 'tables/'
gt_file = 'updated_column_gt.csv'

gt_df = load_and_preprocess_data(gt_file, exclude_labels)
label_counts = gt_df['fine_grained_label'].value_counts()

values_list, file_names, labels_list, column_headers = process_tables(gt_df, data_folder)

print(f"Total number of selected columns: {len(values_list)}")
print(f"Total cluster count (unique 'annotation_label' count): {len(set(labels_list))}")

# Load SBERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Encode headers and values with SBERT in batches
header_embeddings = encode_with_sbert_in_batches(column_headers, model)
value_embeddings = encode_with_sbert_in_batches(values_list, model)


header_embeddings_normalized = header_embeddings / header_embeddings.norm(dim=1)[:, None]
value_embeddings_normalized = value_embeddings / value_embeddings.norm(dim=1)[:, None]

# Combine headers and values
combined_embeddings = torch.cat([header_embeddings_normalized, value_embeddings_normalized], dim=1).cpu().numpy()


cosine_sim_matrix_combined = cosine_similarity(combined_embeddings)
precision_recall_data_combined = precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix_combined)

# Save results to CSV files
precision_recall_df_combined = pd.DataFrame(precision_recall_data_combined, columns=['Iteration', 'Precision', 'Recall', 'Selected_Label', 
                                                                                     'K', 'True_Positives', 'False_Positives', 'False_Negatives'])
precision_recall_df_combined.to_csv('precision_recall_combined.csv', index=False)

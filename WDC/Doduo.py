import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from tqdm import tqdm
import argparse
import os
import chardet
from sentence_transformers import SentenceTransformer
from doduo.doduo import Doduo  


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


def process_tables(gt_df, data_folder):
    files_data = {}
    label_counts = gt_df['fine_grained_label'].value_counts()

    for file_name in gt_df['fileName'].unique():
        path = os.path.join(data_folder, file_name)
        try:
            files_data[file_name] = pd.read_csv(path)
        except Exception as e:
            print(f"Failed to read {file_name}: {e}")

    continuous_cols, all_values, file_names, labels_list, additional_features, column_headers = [], [], [], [], [], []

    for _, row in tqdm(gt_df.iterrows(), desc='Processing tables'):
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

                continuous_cols.append(selected_column.values.reshape(-1, 1))
                all_values.extend(selected_column)
                file_names.append((file_name, column_index))
                labels_list.append(column_name)
                column_headers.append(col_name)

                feature_dict = {
                    'unique_count': selected_column.nunique(),
                    'mean_val': selected_column.mean(),
                    'cv': np.std(selected_column) / selected_column.mean() if selected_column.mean() != 0 else 0,
                    'data_entropy': entropy(selected_column.value_counts(normalize=True), base=2),
                    'range': float(selected_column.max()) - float(selected_column.min()),
                    '10th_percentile': np.percentile(selected_column.astype(float), 10),
                    '90th_percentile': np.percentile(selected_column.astype(float), 90)
                }
                additional_features.append(list(feature_dict.values()))

    return continuous_cols, all_values, file_names, labels_list, additional_features, column_headers


def apply_doduo_embeddings(headers, continuous_cols):

    
    args = argparse.Namespace(
        model="wikitable",  
        shortcut_name="bert-base-uncased",  
        batch_size=16,  
        colpair=False  
    )
    doduo = Doduo(args=args)

    combined_embeddings = []

    
    for col_data, header in tqdm(zip(continuous_cols, headers), desc="Generating Doduo Embeddings"):
        
        col_df = pd.DataFrame({header: col_data.flatten()})
        annotated_data = doduo.annotate_columns(col_df)  
        combined_embeddings.append(np.array(annotated_data.colemb))

    return np.vstack(combined_embeddings)  


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
    return output_data, precision_recall_data, overall_avg_precision



data_folder = 'tables/'
gt_file = 'updated_column_gt.csv'
exclude_labels = [
    'Airport', 
]


gt_df = load_and_preprocess_data(gt_file, exclude_labels)
label_counts = gt_df['fine_grained_label'].value_counts()


continuous_cols, all_values, file_names, labels_list, additional_features, column_headers = process_tables(gt_df, data_folder)

print(f"Total number of selected columns: {len(continuous_cols)}")
print(f"Total cluster count (unique 'annotation_label' count): {len(set(labels_list))}")


final_embeddings = apply_doduo_embeddings(column_headers, continuous_cols)


final_embeddings = final_embeddings / np.linalg.norm(final_embeddings, axis=1, keepdims=True)


np.savetxt('final_embeddings.txt', final_embeddings)


cosine_sim_matrix = cosine_similarity(final_embeddings)


output_data, precision_recall_data, avg_precision = precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix)


neighbors_df = pd.DataFrame(output_data, columns=['Iteration', 'Selected_File', 'Selected_Column', 'Selected_Label', 
                                                  'Neighbor_File', 'Neighbor_Column', 'Neighbor_Label', 'Similarity'])
neighbors_df.to_csv('top_neighbors.csv', index=False)

precision_recall_df = pd.DataFrame(precision_recall_data, columns=['Iteration', 'Label', 'Precision', 'Recall', 'True_Positives', 'False_Positives', 'False_Negatives'])
precision_recall_df.to_csv('precision_recall.csv', index=False)


print(f"Overall Average Precision: {avg_precision:.4f}")

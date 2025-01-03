import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os

# Load and preprocess data
def load_and_preprocess_data(data_folder, gt_file, exclude_labels):
    try:
        gt_df = pd.read_csv(os.path.join(data_folder, gt_file), encoding='ISO-8859-1')
    except UnicodeDecodeError:
        gt_df = pd.read_csv(os.path.join(data_folder, gt_file), encoding='utf-8')
    gt_df = gt_df[~gt_df['ColumnLabel'].isin(exclude_labels)]
    return gt_df

# Process tables
def process_tables(gt_df, data_folder):
    files_data = {}
    label_counts = gt_df['ColumnLabel'].value_counts()

    for file_name in gt_df['fileName'].unique():
        path = os.path.join(data_folder, file_name)
        try:
            files_data[file_name] = pd.read_csv(path, encoding='ISO-8859-1')
        except Exception as e:
            print(f"Failed to read {file_name}: {e}")

    continuous_cols, all_values, file_names, labels_list, column_headers = [], [], [], [], []

    for _, row in tqdm(gt_df.iterrows(), desc='Processing tables'):
        file_name, column_name, col_name = row['fileName'], row['ColumnLabel'], row['colName']

        if label_counts.get(column_name, 0) < 2:
            continue

        table = files_data.get(file_name)
        if table is None:
            continue

        for column_index in range(table.shape[1]):
            selected_column = table.iloc[:, column_index].replace([np.inf, -np.inf], np.nan).dropna()
            if selected_column.empty:
                continue

            if pd.api.types.is_numeric_dtype(selected_column):
                if selected_column.dtype == np.bool_:
                    selected_column = selected_column.astype(int)

                continuous_cols.append(selected_column.values.reshape(-1, 1))
                all_values.extend(selected_column)
                file_names.append((file_name, column_index))
                labels_list.append(column_name)
                column_headers.append(col_name)

    return continuous_cols, all_values, file_names, labels_list, column_headers

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
            if not np.isnan(n[0]) and n[0] != 0:
                similarities = similarity_function(n[0], prototypes)
                normalized_similarities = similarities / np.sum(similarities)
                col_embeddings.append(normalized_similarities)
        
        if col_embeddings:
            embeddings[i, :] = np.nanmean(col_embeddings, axis=0)

    return embeddings

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

def find_similar_pairs(cosine_sim_matrix, encoded_labels, file_names, threshold_similarity=0.80, max_pairs=100):
    similar_pairs = []
    used_columns = set()

    for i in range(len(cosine_sim_matrix)):
        if i in used_columns:
            continue

        for j in range(i + 1, len(cosine_sim_matrix)):
            if j in used_columns:
                continue

            if cosine_sim_matrix[i, j] > threshold_similarity and encoded_labels[i] != encoded_labels[j]:
                similar_pairs.append((i, j))
                used_columns.update([i, j])

                if len(similar_pairs) >= max_pairs:
                    return similar_pairs

    return similar_pairs

def save_top_misclassifications_with_column_values(similar_pairs, file_names, labels_list, cosine_sim_matrix, data_folder, folder='missclassifications/', top_n=100):
    if not os.path.exists(folder):
        os.makedirs(folder)

    sorted_pairs = sorted(similar_pairs, key=lambda x: cosine_sim_matrix[x[0], x[1]], reverse=True)[:top_n]

    misclassification_data = []
    for pair in sorted_pairs:
        file1_info, file2_info = file_names[pair[0]], file_names[pair[1]]
        similarity_score = cosine_sim_matrix[pair[0], pair[1]]

        column1_values = pd.read_csv(os.path.join(data_folder, file1_info[0])).iloc[:, file1_info[1]].tolist()
        column2_values = pd.read_csv(os.path.join(data_folder, file2_info[0])).iloc[:, file2_info[1]].tolist()

        misclassification_data.append([
            file1_info[0], file1_info[1], labels_list[pair[0]], column1_values,
            file2_info[0], file2_info[1], labels_list[pair[1]], column2_values,
            similarity_score
        ])

    columns = ['File1_Name', 'Column1_Index', 'Label1', 'Column1_Values', 
               'File2_Name', 'Column2_Index', 'Label2', 'Column2_Values', 'Similarity']
    misclassifications_df = pd.DataFrame(misclassification_data, columns=columns)
    misclassifications_df.to_csv(f'{folder}top_misclassifications_full.csv', index=False)


data_folder = 'tables/'
gt_file = 'column_gt.csv'
exclude_labels = [
    'Airport', 'Book','..']

gt_df = load_and_preprocess_data(data_folder, gt_file, exclude_labels)
label_counts = gt_df['ColumnLabel'].value_counts()

continuous_cols, all_values, file_names, labels_list, column_headers = process_tables(gt_df, data_folder)
print(f"Total number of selected columns: {len(continuous_cols)}")
print(f"Total cluster count (unique 'annotation_label' count): {len(set(labels_list))}")

prototypes = prototype_induction(all_values, num_prototypes=50)
embeddings = calculate_embeddings(continuous_cols, prototypes)

print(f"Embedding matrix shape: {embeddings.shape}")

np.savetxt('embedding.txt', embeddings)
np.savetxt('labels_text.txt', np.array(labels_list).reshape(-1, 1), fmt='%s')

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels_list)
np.savetxt('labels.txt', encoded_labels, fmt='%d')

cosine_sim_matrix = cosine_similarity(embeddings)
output_data, precision_recall_data = precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix)

neighbors_df = pd.DataFrame(output_data, columns=['Iteration', 'Selected_File', 'Selected_Column', 'Selected_Label', 
                                                  'Neighbor_File', 'Neighbor_Column', 'Neighbor_Label', 'Similarity'])
neighbors_df.to_csv('top_neighbors.csv', index=False)

precision_recall_df = pd.DataFrame(precision_recall_data, columns=['Iteration', 'Label', 'Precision', 'Recall', 'True_Positives', 'False_Positives', 'False_Negatives'])
precision_recall_df.to_csv('precision_recall.csv', index=False)

similar_pairs = find_similar_pairs(cosine_sim_matrix, encoded_labels, file_names)

save_top_misclassifications_with_column_values(similar_pairs, file_names, labels_list, cosine_sim_matrix, data_folder)

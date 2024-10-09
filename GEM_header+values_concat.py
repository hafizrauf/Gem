import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
from tqdm import tqdm
import os
import chardet
from sentence_transformers import SentenceTransformer

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

def fit_gaussian_mixture(all_values, n_components=5, n_init=10):
    gmm = GaussianMixture(n_components=n_components, n_init=n_init)
    all_values_array = np.array(all_values).reshape(-1, 1)
    gmm.fit(all_values_array)
    return gmm

def calculate_probability_matrix(continuous_cols, gmm, additional_features):
    scaler = StandardScaler()
    additional_features = np.array(additional_features)
    scaled_features = scaler.fit_transform(additional_features)

    print(f"Shape of additional_features: {additional_features.shape}")
    print(f"Shape of scaled_features: {scaled_features.shape}")
    
    proba_matrix = np.zeros((len(continuous_cols), gmm.n_components + scaled_features.shape[1]))
    for i, column in tqdm(enumerate(continuous_cols), desc="Calculating probability matrix", total=len(continuous_cols)):
        probabilities = gmm.predict_proba(column)
        mean_probabilities = np.mean(probabilities, axis=0)
        augmented_features = np.hstack([mean_probabilities, scaled_features[i]])
        proba_matrix[i, :] = augmented_features / np.linalg.norm(augmented_features, ord=1)
    
    print(f"Shape of proba_matrix: {proba_matrix.shape}")
    return proba_matrix

def precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix):
    output_data = []  
    precision_recall_data = []  
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

# Function to find similar pairs
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


include_labels = [
    'Year', '..'
]

# Main execution
n_components = 50
n_init = 10
data_folder = 'tables/'
gt_file = 'updated_column_gt.csv'

# Load and preprocess data
gt_df = load_and_preprocess_data(gt_file, include_labels)
label_counts = gt_df['fine_grained_label'].value_counts()
print(f"Number of unique labels before cleaning: {len(label_counts)}")

continuous_cols, all_values, file_names, labels_list, additional_features, headers = process_tables(gt_df, data_folder)

print(f"Number of columns selected after cleaning: {len(continuous_cols)}")
print(f"Number of unique labels after cleaning: {len(set(labels_list))}")

# Fit Gaussian Mixture Model
gmm = fit_gaussian_mixture(all_values, n_components=n_components, n_init=n_init)

# Calculate probability matrix
proba_matrix = calculate_probability_matrix(continuous_cols, gmm, additional_features)

np.savetxt('values_only.txt', proba_matrix, fmt='%f')

# Load pre-trained SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate header embeddings using SBERT
header_embeddings = model.encode(headers)

# Normalize header embeddings
header_embeddings = header_embeddings / np.linalg.norm(header_embeddings, axis=1, keepdims=True)

# Save header-only embeddings
np.savetxt('headers_only.txt', header_embeddings)

# Calculate cosine similarity matrix for header-only embeddings
header_cosine_sim_matrix = cosine_similarity(header_embeddings)

# Precision and recall analysis for headers only
header_output_data, header_precision_recall_data = precision_recall_analysis(file_names, labels_list, label_counts, header_cosine_sim_matrix)

# Save precision and recall results for headers only to CSV
header_neighbors_df = pd.DataFrame(header_output_data, columns=['Iteration', 'Selected_File', 'Selected_Column', 'Selected_Label', 
                                                                'Neighbor_File', 'Neighbor_Column', 'Neighbor_Label', 'Similarity'])
header_neighbors_df.to_csv('top_neighbors_headers_only.csv', index=False)

header_precision_recall_df = pd.DataFrame(header_precision_recall_data, columns=['Iteration', 'Label', 'Precision', 'Recall', 'True_Positives', 'False_Positives', 'False_Negatives'])
header_precision_recall_df.to_csv('precision_recall_headers_only.csv', index=False)

# Print average precision for headers only
total_avg_precision_headers_only = sum([row[2] for row in header_precision_recall_data]) / len(header_precision_recall_data) if header_precision_recall_data else 0
print(f"Overall Average Precision for Headers Only: {total_avg_precision_headers_only:.4f}")

# Concatenate GMM, additional features, and header embeddings
final_embeddings = np.hstack((proba_matrix, header_embeddings))

# Normalize final embeddings
final_embeddings = final_embeddings / np.linalg.norm(final_embeddings, axis=1, keepdims=True)

# Save final embeddings
np.savetxt('headers+values.txt', final_embeddings)

# Compute cosine similarity matrix for final embeddings
cosine_sim_matrix = cosine_similarity(final_embeddings)

# Save results
np.savetxt('final_embeddings.txt', final_embeddings)
np.savetxt('labels_text.txt', np.array(labels_list).reshape(-1, 1), fmt='%s')

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels_list)
np.savetxt('labels.txt', encoded_labels, fmt='%d')

# Precision and recall analysis for combined embeddings
output_data, precision_recall_data = precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix)

# Save precision and recall results to CSV
neighbors_df = pd.DataFrame(output_data, columns=['Iteration', 'Selected_File', 'Selected_Column', 'Selected_Label', 
                                                  'Neighbor_File', 'Neighbor_Column', 'Neighbor_Label', 'Similarity'])
neighbors_df.to_csv('top_neighbors.csv', index=False)

precision_recall_df = pd.DataFrame(precision_recall_data, columns=['Iteration', 'Label', 'Precision', 'Recall', 'True_Positives', 'False_Positives', 'False_Negatives'])
precision_recall_df.to_csv('precision_recall.csv', index=False)

# Print average precision for combined embeddings
total_avg_precision_combined = sum([row[2] for row in precision_recall_data]) / len(precision_recall_data) if precision_recall_data else 0
print(f"Overall Average Precision for Combined: {total_avg_precision_combined:.4f}")

# Find similar pairs
similar_pairs = find_similar_pairs(cosine_sim_matrix, encoded_labels, file_names)

# Save top misclassifications with column values
save_top_misclassifications_with_column_values(similar_pairs, file_names, labels_list, cosine_sim_matrix, data_folder)

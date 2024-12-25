import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from scipy.stats import norm, entropy
from tqdm import tqdm
import os
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

def load_and_preprocess_biodivtab(gt_file):
    
    gt_df = pd.read_csv(gt_file, header=None, sep=",", names=['Table_ID', 'Column_Index', 'Wikidata_Entities'])
    gt_df['Wikidata_Entities'] = gt_df['Wikidata_Entities'].astype(str)
    gt_df = gt_df.explode('Wikidata_Entities')
    
    label_encoder = LabelEncoder()
    gt_df['Numeric_Label'] = label_encoder.fit_transform(gt_df['Wikidata_Entities'])

    label_counts = gt_df['Numeric_Label'].value_counts()
    valid_labels = label_counts[label_counts >= 5].index
    gt_df = gt_df[gt_df['Numeric_Label'].isin(valid_labels)]

    return gt_df, label_encoder


def process_biodivtab_tables(gt_df, data_folder):
    label_counts = gt_df['Numeric_Label'].value_counts()
    valid_labels = label_counts[label_counts >= 5].index
    filtered_gt_df = gt_df[gt_df['Numeric_Label'].isin(valid_labels)]

    # Initialize storage for processed data
    continuous_cols, all_values, file_names, labels_list, additional_features = [], [], [], [], []

    for idx, row in tqdm(filtered_gt_df.iterrows(), desc='Processing tables', total=len(filtered_gt_df)):
        try:
            
            if pd.isna(row['Table_ID']) or pd.isna(row['Column_Index']):
                continue  
            table_id = row['Table_ID']
            column_index = int(row['Column_Index']) 
            label = row['Numeric_Label']
            table_path = os.path.join(data_folder, f'{table_id}.csv')
            if not os.path.exists(table_path):
                print(f'File not found: {table_path}')
                continue
            table = pd.read_csv(table_path)
            if column_index >= len(table.columns):
                print(f"Invalid column index: {column_index} for table {table_id}")
                continue         
            column_name = table.columns[column_index]
            if pd.api.types.is_numeric_dtype(table[column_name]):
           
                selected_column = table[column_name].replace([np.inf, -np.inf], np.nan).dropna()
                if selected_column.empty:
                    continue
                features = [
                    selected_column.nunique(),
                    selected_column.mean(),
                    np.std(selected_column) / selected_column.mean() if selected_column.mean() != 0 else 0,
                    entropy(selected_column.value_counts(normalize=True), base=2),
                    float(selected_column.max()) - float(selected_column.min()),
                    np.percentile(selected_column, 10),
                    np.percentile(selected_column, 90)
                ]
                continuous_cols.append(selected_column.values.reshape(-1, 1))
                all_values.extend(selected_column)
                file_names.append((table_id, column_index))
                labels_list.append(label)
                additional_features.append(features)
            else:
                print(f"Column {column_name} in table {table_id} is not numeric.")
        except Exception as e:
            print(f'Error processing table {row["Table_ID"]}, column {row["Column_Index"]}: {e}')

   
    assert len(continuous_cols) == len(file_names) == len(labels_list) == len(additional_features), \
        f"Data lists are not synchronized! continuous_cols: {len(continuous_cols)}, file_names: {len(file_names)}, labels_list: {len(labels_list)}, additional_features: {len(additional_features)}"


    print(f"Total numeric columns processed: {len(continuous_cols)}")
    print(f"Unique labels: {len(set(labels_list))}")
    print(f"Additional features collected: {len(additional_features)}")

    return continuous_cols, all_values, file_names, labels_list, additional_features

# Fit GNN
def fit_gaussian_mixture(all_values, n_components=5, n_init=10):
    gmm = GaussianMixture(n_components=n_components, n_init=n_init)
    all_values_array = np.array(all_values).reshape(-1, 1)
    gmm.fit(all_values_array)
    return gmm

# Calculate probability matrix
def calculate_probability_matrix(continuous_cols, gmm, additional_features):
    scaler = StandardScaler()
    additional_features = np.array(additional_features)
    scaled_features = scaler.fit_transform(additional_features)
    if len(continuous_cols) != len(scaled_features):
        print(f"Mismatch: {len(continuous_cols)} continuous columns vs. {len(scaled_features)} additional features.")
        return None

    proba_matrix = np.zeros((len(continuous_cols), gmm.n_components + scaled_features.shape[1]))
    for i, column in tqdm(enumerate(continuous_cols), desc="Calculating probability matrix", total=len(continuous_cols)):
        probabilities = gmm.predict_proba(column)
        mean_probabilities = np.mean(probabilities, axis=0)
        augmented_features = np.hstack([mean_probabilities, scaled_features[i]])
        proba_matrix[i, :] = augmented_features / np.linalg.norm(augmented_features, ord=1)

    return proba_matrix

# Precision and Recall analysis
def precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix):
    output_data = []  
    precision_recall_data = []  
    unique_labels = sorted(set(labels_list))
    total_avg_precision = 0.0  
    for iteration, label in tqdm(enumerate(unique_labels, start=1), desc="Analyzing precision and recall", total=len(unique_labels)):
        label_indices = [i for i, x in enumerate(labels_list) if x == label]
        if not label_indices:
            continue
        total_tp, total_fp, total_fn = 0, 0, 0
        total_precision, total_recall = 0.0, 0.0
        label_output_data = []
        for selected_column_index in label_indices:
            if selected_column_index >= len(cosine_sim_matrix):
                print(f"Skipping index {selected_column_index} out of bounds for cosine_sim_matrix with size {len(cosine_sim_matrix)}.")
                continue
            selected_column = file_names[selected_column_index]
            similarities = cosine_sim_matrix[selected_column_index]
            sorted_indices = np.argsort(-similarities)
            top_k_indices = sorted_indices[1:label_counts[label] + 1]
            tp = sum(labels_list[idx] == label for idx in top_k_indices if idx < len(labels_list))
            fp = len([idx for idx in top_k_indices if idx < len(labels_list) and labels_list[idx] != label])
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
                for idx in top_k_indices if idx < len(file_names)
            ])

        avg_precision = total_precision / len(label_indices) if label_indices else 0
        avg_recall = total_recall / len(label_indices) if label_indices else 0
        precision_recall_data.append([iteration, label, avg_precision, avg_recall, total_tp, total_fp, total_fn])
        output_data.extend(label_output_data)
        total_avg_precision += avg_precision  
    overall_avg_precision = total_avg_precision / len(unique_labels) if unique_labels else 0
    print(f"Overall Average Precision: {overall_avg_precision:.4f}")

    return output_data, precision_recall_data
# nDCG analysis
def ndcg_analysis(file_names, labels_list, label_counts, cosine_sim_matrix):
    """
    Compute and export the Normalized Discounted Cumulative Gain (nDCG) for each cluster.

    Parameters:
        file_names (list): List of tuples (table_id, target_column) for each data column.
        labels_list (list): List of labels corresponding to the columns.
        label_counts (pd.Series): Count of occurrences for each label.
        cosine_sim_matrix (np.array): Cosine similarity matrix.

    Returns:
        pd.DataFrame: DataFrame containing nDCG scores for each label.
    """
    unique_labels = sorted(set(labels_list))
    ndcg_results = []

    for label in tqdm(unique_labels, desc="Calculating nDCG"):
        
        label_indices = [i for i, x in enumerate(labels_list) if x == label]
        
        if not label_indices:
            continue

        ndcg_scores = []

        for selected_column_index in label_indices:
            similarities = cosine_sim_matrix[selected_column_index]
            sorted_indices = np.argsort(-similarities)  
            top_k = label_counts[label]
            top_k_indices = sorted_indices[1:top_k + 1]
            retrieved_relevance = [
                1 if labels_list[idx] == label else 0 for idx in top_k_indices
            ]
            ground_truth_relevance = [1] * label_counts[label]
            ndcg = compute_ndcg(ground_truth_relevance, retrieved_relevance, top_k)
            ndcg_scores.append(ndcg)
        avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
        ndcg_results.append([label, avg_ndcg])
    ndcg_df = pd.DataFrame(ndcg_results, columns=["Label", "nDCG"])
    overall_avg_ndcg = ndcg_df["nDCG"].mean()
    print(f"Overall Average nDCG: {overall_avg_ndcg:.4f}")
    ndcg_df.to_csv("ndcg.csv", index=False, sep=",")
    print(f"nDCG results saved to 'ndcg.csv'")
    return ndcg_df


def compute_ndcg(ground_truth_relevance, retrieved_relevance, top_k):
    """
    Compute Normalized Discounted Cumulative Gain (nDCG).

    Parameters:
        ground_truth_relevance (list): List of relevance scores for the ideal ranking.
        retrieved_relevance (list): List of relevance scores for the retrieved ranking.
        top_k (int): Number of top results to consider.

    Returns:
        float: nDCG value.
    """
    
    ideal_relevance = np.sort(ground_truth_relevance)[::-1]
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))
    dcg = np.sum(retrieved_relevance / np.log2(np.arange(2, len(retrieved_relevance) + 2)))
    return dcg / idcg if idcg > 0 else 0.0

if __name__ == "__main__":
    data_folder = "tables/"
    gt_file = "CTA_biodivtab_2021_gt.csv"
    gt_df, label_encoder = load_and_preprocess_biodivtab(gt_file)
    label_counts = gt_df['Numeric_Label'].value_counts()
    continuous_cols, all_values, file_names, labels_list, additional_features = process_biodivtab_tables(gt_df, data_folder)
    print(f"Total numeric columns processed: {len(continuous_cols)}")
    print(f"Unique labels: {len(set(labels_list))}")
    if len(all_values) == 0:
        print("No valid numeric columns found. Please check your input data and filtering criteria.")
        exit()
    gmm = fit_gaussian_mixture(all_values)
    proba_matrix = calculate_probability_matrix(continuous_cols, gmm, additional_features)
    cosine_sim_matrix = cosine_similarity(proba_matrix)
    output_data, precision_recall_data = precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix)
    # Save precision-recall results
    precision_recall_df = pd.DataFrame(precision_recall_data, columns=['Iteration', 'Label', 'Precision', 'Recall', 'True Positives', 'False Positives', 'False Negatives'])
    precision_recall_df.to_csv("precision_recall.csv", index=False)
    
    neighbors_df = pd.DataFrame(output_data, columns=[
        'Iteration', 'Selected_File', 'Selected_Column', 'Selected_Label', 
        'Neighbor_File', 'Neighbor_Column', 'Neighbor_Label', 'Similarity'
    ])
    neighbors_df.to_csv('top_neighbors.csv', index=False)

    # nDCG analysis
    ndcg_results = ndcg_analysis(file_names, labels_list, label_counts, cosine_sim_matrix)

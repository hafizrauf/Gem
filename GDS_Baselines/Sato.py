import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from collections import OrderedDict
from sherlock.features.stats_helper import compute_stats
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity

# Feature extraction for numerical columns (Stat)
def sato_feature_extraction(numeric_column):
    features = OrderedDict()
    numeric_column = numeric_column.replace([np.inf, -np.inf], np.nan).dropna()

    if numeric_column.empty:
        return [0] * 8

    # Compute basic statistics
    _mean, _variance, _skew, _kurtosis, _min, _max, _sum = compute_stats(numeric_column)
    _median = np.median(numeric_column)

    # Assign statistical features
    features['mean'] = _mean
    features['variance'] = _variance
    features['skewness'] = _skew
    features['kurtosis'] = _kurtosis
    features['min'] = _min
    features['max'] = _max
    features['median'] = _median
    features['sum'] = _sum

    return list(features.values())

# Extract paragraph-level embeddings using SBERT
def extract_embeddings(headers):
    # Use SBERT to get paragraph embeddings from the headers
    para_model = SentenceTransformer('all-MiniLM-L6-v2')
    para_embeddings = para_model.encode(headers)  # Use headers for paragraph embeddings

    return para_embeddings

# Combine extracted features (Stat + Para)
def combine_features(continuous_cols, para_embeddings):
    combined_features = []
    
    for idx in range(len(continuous_cols)):
        combined = continuous_cols[idx] + list(para_embeddings[idx])
        combined_features.append(combined)

    return np.array(combined_features)

# Build the Sato-based model for embeddings
def build_sato_model(input_dim, num_classes):
    input_combined = Input(shape=(input_dim,))
    
    dense_combined = Dense(256, activation='relu')(input_combined)
    dropout_combined = Dropout(0.3)(dense_combined)

    dense_topic = Dense(128, activation='relu')(dropout_combined)
    dropout_topic = Dropout(0.3)(dense_topic)

    output = Dense(num_classes, activation='softmax')(dropout_topic)

    model = Model(inputs=input_combined, outputs=output)
    return model

# Training
def train_and_get_embeddings(combined_features, y_train):
    input_dim = combined_features.shape[1]
    num_classes = len(set(y_train))

    sato_model = build_sato_model(input_dim, num_classes)

    optimizer = Adam(learning_rate=0.001)
    sato_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    sato_model.fit(combined_features, y_train, epochs=50, batch_size=32, validation_split=0.3)

    embedding_model = Model(inputs=sato_model.input, outputs=sato_model.layers[-2].output)
    embeddings = embedding_model.predict(combined_features)

    return embeddings

# Process tables and extract features 
def process_tables_with_sato(gt_df, data_folder, include_labels):
    continuous_cols, file_names, labels_list, headers = [], [], [], []
    
    for file_name in tqdm(gt_df['fileName'].unique(), desc="Processing tables"):
        path = os.path.join(data_folder, file_name)
        try:
            table = pd.read_csv(path)
        except:
            continue

        for _, row in gt_df[gt_df['fileName'] == file_name].iterrows():
            column_name = row['fine_grained_label']
            column_index = table.columns.get_loc(row['colName']) if row['colName'] in table.columns else -1

            if column_index == -1 or column_name not in include_labels:
                continue

            selected_column = table.iloc[:, column_index].replace([np.inf, -np.inf], np.nan).dropna()

            if selected_column.empty or not pd.api.types.is_numeric_dtype(selected_column):
                continue

            sato_features = sato_feature_extraction(selected_column)
            continuous_cols.append(sato_features)
            file_names.append((file_name, column_index))
            labels_list.append(column_name)
            headers.append(row['colName'])

    return continuous_cols, file_names, labels_list, headers

# Precision and recall analysis for final embeddings
def precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix):
    output_data = []  # detailed comparison results.
    precision_recall_data = []  # stores summary statistics.
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

# Main execution
gt_file = 'updated_column_gt.csv'
data_folder = 'tables/'
# Main execution settings
include_labels = [
    'Year', '..'
]

gt_df = pd.read_csv(gt_file)
continuous_cols, file_names, labels_list, headers = process_tables_with_sato(gt_df, data_folder, include_labels)

para_embeddings = extract_embeddings(headers)

combined_features = combine_features(continuous_cols, para_embeddings)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(labels_list)

final_embeddings = train_and_get_embeddings(combined_features, y_train_encoded)
cosine_sim_matrix = cosine_similarity(final_embeddings)

# Precision and recall analysis
output_data, precision_recall_data = precision_recall_analysis(file_names, labels_list, gt_df['fine_grained_label'].value_counts(), cosine_sim_matrix)

# Save precision and recall results to CSV
neighbors_df = pd.DataFrame(output_data, columns=['Iteration', 'Selected_File', 'Selected_Column', 'Selected_Label',
                                                  'Neighbor_File', 'Neighbor_Column', 'Neighbor_Label', 'Similarity'])
neighbors_df.to_csv('top_neighbors.csv', index=False)

precision_recall_df = pd.DataFrame(precision_recall_data, columns=['Iteration', 'Label', 'Precision', 'Recall', 'True_Positives', 'False_Positives', 'False_Negatives'])
precision_recall_df.to_csv('precision_recall.csv', index=False)

# Print overall average precision for combined embeddings
total_avg_precision_combined = sum([row[2] for row in precision_recall_data]) / len(precision_recall_data) if precision_recall_data else 0
print(f"Overall Average Precision for Combined: {total_avg_precision_combined:.4f}")

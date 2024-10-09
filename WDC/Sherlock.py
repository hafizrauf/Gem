import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import chardet
from sherlock.features.stats_helper import compute_stats
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Input



from collections import OrderedDict

# Detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    return result['encoding']

# Load and preprocess data
def load_and_preprocess_data(gt_file, exclude_labels):
    encoding = detect_encoding(gt_file)
    gt_df = pd.read_csv(gt_file, encoding=encoding)
    gt_df = gt_df.drop_duplicates()
    gt_df = gt_df[~gt_df['fine_grained_label'].isin(exclude_labels)]
    return gt_df

# Sherlock Feature Extraction for numerical columns
def sherlock_feature_extraction(numeric_column):
    features = OrderedDict()
    numeric_column = numeric_column.replace([np.inf, -np.inf], np.nan).dropna()

    if numeric_column.empty:
        features = {'mean': 0, 'variance': 0, 'skewness': 0, 'kurtosis': 0, 'min': 0, 'max': 0, 'median': 0, 'sum': 0}
    else:
        _mean, _variance, _skew, _kurtosis, _min, _max, _sum = compute_stats(numeric_column)
        _median = np.median(numeric_column)
        features = {
            'mean': _mean, 'variance': _variance, 'skewness': _skew,
            'kurtosis': _kurtosis, 'min': _min, 'max': _max, 'median': _median, 'sum': _sum
        }
    
    return list(features.values())


def process_tables_with_sherlock(gt_df, data_folder):
    files_data = {}
    label_counts = gt_df['fine_grained_label'].value_counts()

    for file_name in gt_df['fileName'].unique():
        path = os.path.join(data_folder, file_name)
        try:
            files_data[file_name] = pd.read_csv(path)
        except Exception as e:
            print(f"Failed to read {file_name}: {e}")

    continuous_cols, all_values, file_names, labels_list, headers = [], [], [], [], []

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

                
                sherlock_features = sherlock_feature_extraction(selected_column)
                continuous_cols.append(sherlock_features)

                all_values.extend(selected_column)
                file_names.append((file_name, column_index))
                labels_list.append(column_name)
                headers.append(col_name)

    return continuous_cols, all_values, file_names, labels_list, headers



def extract_header_embeddings(headers):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    header_embeddings = model.encode(headers)
    header_embeddings = header_embeddings / np.linalg.norm(header_embeddings, axis=1, keepdims=True)
    return header_embeddings

# Combine numerical features and header embeddings
def combine_features(continuous_cols, header_embeddings):
    continuous_cols_array = np.array(continuous_cols)
    if continuous_cols_array.ndim == 1:
        continuous_cols_array = continuous_cols_array.reshape(-1, 1)
    
    if continuous_cols_array.shape[0] != header_embeddings.shape[0]:
        raise ValueError("Number of rows in numerical features and header embeddings must match.")
    
    combined_embeddings = np.hstack((continuous_cols_array, header_embeddings))
    return combined_embeddings

# Define neural submodules for each feature set and concatenate them
def build_sherlock_model(input_dim, num_classes):
    input_numerical = Input(shape=(input_dim['numerical'],))
    input_headers = Input(shape=(input_dim['headers'],))
    
    dense_num = Dense(300, activation='relu')(input_numerical)
    dropout_num = Dropout(0.35)(dense_num)

    dense_header = Dense(300, activation='relu')(input_headers)
    dropout_header = Dropout(0.35)(dense_header)
    
    concatenated = Concatenate()([dropout_num, dropout_header])
    dense_concat = Dense(500, activation='relu')(concatenated)
    dropout_concat = Dropout(0.35)(dense_concat)
    
    output = Dense(num_classes, activation='softmax')(dropout_concat)
    
    model = Model(inputs=[input_numerical, input_headers], outputs=output)
    return model

# Training
def train_and_get_embeddings(X_num, X_header, y_train):
    input_dim = {'numerical': X_num.shape[1], 'headers': X_header.shape[1]}
    num_classes = len(set(y_train))
    
    sherlock_model = build_sherlock_model(input_dim, num_classes)
    sherlock_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    sherlock_model.fit([X_num, X_header], y_train, epochs=50, batch_size=32, validation_split=0.3)

    embedding_model = Model(inputs=sherlock_model.input, outputs=sherlock_model.layers[-2].output)
    embeddings = embedding_model.predict([X_num, X_header])
    
    return embeddings

# Precision and recall analysis for final embeddings
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
    'Airport', 'Book','..']

# Load and preprocess the new dataset
gt_df = load_and_preprocess_data(gt_file, exclude_labels)
label_counts = gt_df['fine_grained_label'].value_counts()

# Process tables with Sherlock feature extraction
continuous_cols, all_values, file_names, labels_list, headers = process_tables_with_sherlock(gt_df, data_folder)

# Print number of columns and labels
print(f"Number of columns selected: {len(continuous_cols)}")
print(f"Number of unique labels: {len(set(labels_list))}")

# Extract header embeddings using SBERT
header_embeddings = extract_header_embeddings(headers)
combined_features = combine_features(continuous_cols, header_embeddings)

# Convert labels to numeric values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(labels_list)

# Train Sherlock model and get embeddings
final_embeddings = train_and_get_embeddings(np.array(continuous_cols), header_embeddings, y_train_encoded)
cosine_sim_matrix = cosine_similarity(final_embeddings)

# Precision and recall analysis for final embeddings
output_data, precision_recall_data, combined_avg_precision = precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix)

# Save precision and recall results
neighbors_df = pd.DataFrame(output_data, columns=['Iteration', 'Selected_File', 'Selected_Column', 'Selected_Label', 
                                                  'Neighbor_File', 'Neighbor_Column', 'Neighbor_Label', 'Similarity'])
neighbors_df.to_csv('top_neighbors.csv', index=False)

precision_recall_df = pd.DataFrame(precision_recall_data, columns=['Iteration', 'Label', 'Precision', 'Recall', 'True_Positives', 'False_Positives', 'False_Negatives'])
precision_recall_df.to_csv('precision_recall.csv', index=False)

# Print average precision
print(f"Overall Average Precision for Combined: {combined_avg_precision:.4f}")

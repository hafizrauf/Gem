import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os
import chardet
from collections import OrderedDict
from sherlock.features.stats_helper import compute_stats
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Input


# Detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    return result['encoding']


# Load and preprocess data
def load_and_preprocess_data(gt_file, include_labels):
    encoding = detect_encoding(gt_file)
    gt_df = pd.read_csv(gt_file, encoding=encoding)
    gt_df = gt_df.drop_duplicates()
    gt_df = gt_df[gt_df['fine_grained_label'].isin(include_labels)]
    return gt_df


# Sherlock Feature Extraction for numerical columns
def sherlock_feature_extraction(numeric_column):
    features = OrderedDict()
    
    
    numeric_column = numeric_column.replace([np.inf, -np.inf], np.nan).dropna()

    
    if numeric_column.empty:
        
        features['mean'] = 0
        features['variance'] = 0
        features['skewness'] = 0
        features['kurtosis'] = 0
        features['min'] = 0
        features['max'] = 0
        features['median'] = 0
        features['sum'] = 0
        return list(features.values())

    # Sherlock statistical feature extraction
    _mean, _variance, _skew, _kurtosis, _min, _max, _sum = compute_stats(numeric_column)
    _median = np.median(numeric_column)
    features['mean'] = _mean
    features['variance'] = _variance
    features['skewness'] = _skew
    features['kurtosis'] = _kurtosis
    features['min'] = _min
    features['max'] = _max
    features['median'] = _median
    features['sum'] = _sum
    
    # Return features as a list of values
    return list(features.values())

# Process tables using Sherlock feature extraction for numerical values
def process_tables_with_sherlock(gt_df, data_folder):
    files_data = {}
    continuous_cols, all_values, file_names, labels_list, headers = [], [], [], [], []
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

            # Extract Sherlock features and append to continuous_cols
            sherlock_features = sherlock_feature_extraction(selected_column)
            continuous_cols.append(sherlock_features)

            
            all_values.extend(selected_column)
            file_names.append((file_name, column_index))
            labels_list.append(column_name)
            headers.append(row['colName'])

    return continuous_cols, all_values, file_names, labels_list, headers






def extract_header_embeddings(headers):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    header_embeddings = model.encode(headers)
    
    
    header_embeddings = header_embeddings / np.linalg.norm(header_embeddings, axis=1, keepdims=True)
    return header_embeddings



def combine_features(continuous_cols, header_embeddings):
    
    continuous_cols_array = np.array(continuous_cols)
    
    
    if continuous_cols_array.ndim == 1:
        continuous_cols_array = continuous_cols_array.reshape(-1, 1)
    
   
    if continuous_cols_array.shape[0] != header_embeddings.shape[0]:
        raise ValueError("Number of rows in numerical features and header embeddings must match.")
    
    # Concatenate continuous_cols_array and header_embeddings
    combined_embeddings = np.hstack((continuous_cols_array, header_embeddings))
    
    return combined_embeddings



def build_sherlock_model(input_dim, num_classes):
    input_numerical = Input(shape=(input_dim['numerical'],))
    input_headers = Input(shape=(input_dim['headers'],))
    
    # Process numerical features
    dense_num = Dense(300, activation='relu')(input_numerical)
    dropout_num = Dropout(0.35)(dense_num)

    # Process header embeddings
    dense_header = Dense(300, activation='relu')(input_headers)
    dropout_header = Dropout(0.35)(dense_header)
    
    # Concatenate numerical and header submodules
    concatenated = Concatenate()([dropout_num, dropout_header])
    dense_concat = Dense(500, activation='relu')(concatenated)
    dropout_concat = Dropout(0.35)(dense_concat)
    
    # Final layer to classify embeddings
    output = Dense(num_classes, activation='softmax')(dropout_concat)
    
    model = Model(inputs=[input_numerical, input_headers], outputs=output)
    
    return model



# Training
def train_and_get_embeddings(X_num, X_header, y_train):
    input_dim = {
        'numerical': X_num.shape[1],
        'headers': X_header.shape[1]
    }
    
    num_classes = len(set(y_train))  
    
    # Build the Sherlock model
    sherlock_model = build_sherlock_model(input_dim, num_classes)
    sherlock_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    sherlock_model.fit([X_num, X_header], y_train, epochs=50, batch_size=32, validation_split=0.3)

    
    embedding_model = Model(inputs=sherlock_model.input, outputs=sherlock_model.layers[-2].output)
    
    
    embeddings = embedding_model.predict([X_num, X_header])
    
    return embeddings


# Precision and recall analysis for final embeddings
def precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix):
    output_data = [] 
    precision_recall_data = []  
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

# Load and preprocess data
gt_df = load_and_preprocess_data(gt_file, include_labels)
label_counts = gt_df['fine_grained_label'].value_counts()
print(f"Number of unique labels before cleaning: {len(label_counts)}")

# Process tables with Sherlock feature extraction
continuous_cols, all_values, file_names, labels_list, headers = process_tables_with_sherlock(gt_df, data_folder)

print(f"Number of columns selected after cleaning: {len(continuous_cols)}")
print(f"Number of unique labels after cleaning: {len(set(labels_list))}")

# Extract header embeddings using SBERT
header_embeddings = extract_header_embeddings(headers)

# Combine numerical features and header embeddings
combined_features = combine_features(continuous_cols, header_embeddings)


# Convert string labels to numeric labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(labels_list)


final_embeddings = train_and_get_embeddings(np.array(continuous_cols), header_embeddings, y_train_encoded)

# 
cosine_sim_matrix = cosine_similarity(final_embeddings)

# Precision and recall analysis
output_data, precision_recall_data = precision_recall_analysis(file_names, labels_list, label_counts, cosine_sim_matrix)


neighbors_df = pd.DataFrame(output_data, columns=['Iteration', 'Selected_File', 'Selected_Column', 'Selected_Label', 
                                                  'Neighbor_File', 'Neighbor_Column', 'Neighbor_Label', 'Similarity'])
neighbors_df.to_csv('top_neighbors.csv', index=False)

precision_recall_df = pd.DataFrame(precision_recall_data, columns=['Iteration', 'Label', 'Precision', 'Recall', 'True_Positives', 'False_Positives', 'False_Negatives'])
precision_recall_df.to_csv('precision_recall.csv', index=False)

# Print average precision for combined embeddings
total_avg_precision_combined = sum([row[2] for row in precision_recall_data]) / len(precision_recall_data) if precision_recall_data else 0
print(f"Overall Average Precision for Combined: {total_avg_precision_combined:.4f}")

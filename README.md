
# GEM: Gaussian Mixture Model Embeddings for Numerical Feature Distributions

This repository contains an implementation of **(GEM)**. The results can be reproduced using the following datasets:
- GDS
- WDC
- Sato Tables
- Git Tables

By default, the repository is set to run experiments on the **GDS** dataset.

## Overview

```
├── GDS_Baselines/          # Contains baseline models (headers + values) for the GDS dataset
├── GitTables/              # Contains all Gem's settings,  baselines and data for Git Tables
├── Sato Tables/            # Contains all Gem's settings,  baselines and data for Sato Tables
├── WDC/                    # Contains all Gem's settings,  baselines and data for WDC
├── GEM_header+values_concat.py  # Script for GEM embedding generation (headers + values)
├── GEM_values.py           # Script for GEM embedding generation (values only)
├── KS.py                   # values only baseline
├── PAF.py                  # values only baseline
├── PLE.py                  # values only baseline
├── squashing_GMM.py        # values only baseline
├── squashing_SOM.py        # values only baseline
├── Table_P1.zip            # Part 1 of the dataset files
├── Table_P2.zip            # Part 2 of the dataset files
├── include_labels.txt      # Preprocessed ground truth data (numeric-only columns)
├── updated_column_gt.csv   # Contains coarse and fine-grained labels for ground truth
├── README.md               # The README file
```

## Installation Requirements

Ensure you have Python 3.8+ and the following libraries installed before running `Gem.py`:

- pandas
- numpy
- scikit-learn
- scipy
- tqdm
- chardet

## Dataset Preparation

1. **Extract the Dataset Files**:
   - Unzip both `Table_P1.zip` and `Table_P2.zip`, and merge their contents into a directory called `Tables/`.

2. **Ground Truth Labels**:
   - The `include_labels.txt` file contains preprocessed numeric-only labels.
   - The `updated_column_gt.csv` file has both coarse-grained and fine-grained labels:
     - Use `ColumnLabel` for coarse-grained labels.
     - Use `fine_grained_label` for fine-grained labels.

To use coarse-grained labels, replace `gt_df['fine_grained_label']` with `gt_df['ColumnLabel']` in `Gem.py` and the relevant baseline scripts.

## Running GEM

To replicate the results using the GEM method, follow these steps:

1. **Execute the GEM Script**:

```bash
python Gem.py
```

The script will:
- Load and preprocess the ground truth labels from `updated_column_gt.csv`.
- Process the relevant tables in the `Tables/` directory by selecting columns.
- Train a Gaussian Mixture Model (GMM) on the numerical values of the selected columns.
- Calculate a probability matrix by combining GMM predictions with additional features.
- Use the probability matrix to compute a cosine similarity matrix.

## Breakdown of the Steps in `Gem.py`

1. **Data Loading and Preprocessing**:
   - The function `load_and_preprocess_data()` reads the ground truth file and filters the rows.

2. **Table Processing**:
   - The function `process_tables()` handles table columns and calculates additional statistics.

3. **Fitting a Gaussian Mixture Model**:
   - The `fit_gaussian_mixture()` function trains the GMM.

4. **Computing the Probability Matrix**:
   - The `calculate_probability_matrix()` function uses GMM predictions and additional features to build the probability matrix.

5. **Precision and Recall Calculation**:
   - The `precision_recall_analysis()` function calculates precision and recall using the cosine similarity matrix.

6. **Finding Similar Pairs**:
   - The `find_similar_pairs()` function identifies similar column pairs based on the cosine similarity threshold.

7. **Saving Results**:
   - The results, including the embeddings, nearest neighbors, precision-recall data, and similar pairs, will be saved in CSV files.

## Output Files

After running `Gem.py`, results are saved as:
- `final_embeddings.txt`: Final GEM embeddings for the columns.
- `top_neighbors.csv`: The most similar columns (nearest neighbors).
- `precision_recall.csv`: Precision and recall statistics for each label.
- `labels_text.txt` and `labels.txt`: Label files (text and encoded formats).

## Customizing Parameters

1. **Adjust the Number of GMM Components**:
   - By default, GMM uses 50 components. You can change this by updating the `n_components` parameter in `Gem.py`:
     ```bash
     n_components = 100  # Example for using 100 components
     ```

2. **Switch Between Coarse and Fine-Grained Labels**:
   - Replace all instances of `gt_df['fine_grained_label']` with `gt_df['ColumnLabel']` for coarse labels.

3. **Modify the Number of Nearest Neighbors**:
   - You can adjust the number of neighbors by updating the precision-recall or nearest neighbor functions.

## Troubleshooting

- **Encoding Issues**:
  The script uses `chardet` to detect CSV file encodings. If you encounter encoding errors, ensure the encoding is correctly detected.

- **Missing Columns**:
  If tables are missing columns, the script skips them and logs an error. Ensure all necessary columns are present in the `Tables/` directory.

# ML Folder Technical Specification

## Overview
The `ML` folder contains machine learning scripts and notebooks for the IndirectTax project. It is responsible for data preprocessing, feature engineering, model training, evaluation, and predictions related to journal and reconciliation data.

## Contents
- `main.py`: Main script for running ML tasks.
- `reclass-model.py`: Script for feature engineering, training a RandomForest model, and making predictions for city reclassification.
- `journal.ipynb`, `reclass.ipynb`, `reconcillation.ipynb`: Jupyter notebooks for exploratory data analysis, model development, and results visualization.
- `requirements.txt`: Python dependencies for running the ML scripts and notebooks.
- `README.MD`: High-level documentation for the ML folder.

## Data Files
- `../Data/journal_synthetic.csv`: Synthetic journal data used for training and feature engineering.
- `../Data/reconcillation_synthetic.csv`: Synthetic reconciliation data used for testing and predictions.
- `../Data/Journal.xlsx`: Excel version of journal data.

## Key Features
- **Feature Engineering**: Creation of new features such as ratios, log transforms, and categorical encodings.
- **Model Training**: Uses `RandomForestClassifier` from scikit-learn for city reclassification tasks.
- **Prediction**: Predicts city labels for test data, ensuring predictions are consistent with region and county constraints.
- **Correlation Analysis**: (Optional) Can output feature correlation matrices to CSV for analysis.
- **Output**: Saves prediction results to `predicted_output.csv`.

## Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn

Install dependencies with:
```
pip install -r requirements.txt
```

## Usage
- Run `reclass-model.py` to perform feature engineering, train the model, and generate predictions.
- Notebooks can be used for interactive analysis and development.

## Notes
- Ensure data files are present in the `../Data/` directory relative to the ML folder.
- Outputs such as correlation matrices and predictions are saved in the ML folder by default.

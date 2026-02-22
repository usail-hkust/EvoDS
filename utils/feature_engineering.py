import pandas as pd
import numpy as np
from typing import Union, List
from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from itertools import combinations
from scipy.special import expit
import warnings 
import re
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.util import write_program, execute_script
warnings.filterwarnings("ignore")


# FEATURE_ENGINEER_SYS_PROMPT = """You are a data science expert specializing in feature engineering tasks. You have access to a set of tools that can help solve these tasks. When given a dataset path and a feature engineering task description, your first step is to check if the provided tools can directly solve the task. If they can, use the appropriate tool to perform the feature engineering. The tool will automatically save the processed dataset.
# 
# **If the tools cannot directly solve the task, you should not call any tool and directly write Python code that can address the task according to the given task description**. After processing, ensure that the processed data is saved to the appropriate path.
# 
# For example:
# ```python
# def feature_engineering_tool(dataset_path):
#     # feature engineering code here
# 
# # Perform the feature engineering task on the dataset. Note: You should use a real dataset path instead of the example path 'dataset.csv'
# feature_engineering_tool('dataset.csv')
# ```
# """


FEATURE_ENGINEER_SYS_PROMPT = """You are a data science expert specializing in feature engineering tasks. You have access to a set of tools that can help solve these tasks. When given a dataset path and a feature engineering task description, your first step is to check if the provided tools can directly solve the task. If they can, use the appropriate tool to perform the feature engineering. The tool will automatically save the processed dataset.

If the tools cannot directly solve the task, you should use the `feature_engineering_tool_creation` tool to create a new tool to address the task based on the description provided.

The created tool should strictly follow the format below:
```python
def tool_name(parameters):
    # detail of the code

# execute the tool
if __name__ == '__main__':
    parameters = {...}
    tool_name(parameters)
```
"""


FEATURE_ENGINEERING_PROMPT = """
You are given a dataset located at {dataset_file}. Your task is to process the dataset according to the following requirements:

# FEATURE ENGINEERING TASK #
{task}

After processing, save the processed dataset to {saved_dataset_file}.
"""


def one_hot_encode(dataset_path: str, 
                   columns: Union[str, List[str]], 
                   drop_original: bool = False, 
                   handle_unknown: str = 'error',
                   work_dir: str = None,
                   index: int = None) -> str:
    """
    Perform different types of encoding (one-hot, label, frequency, or target encoding) 
    on specified columns in the dataset.

    Args:
        dataset_path (str): The path to the dataset file.
        columns (str or List[str]): Column label or list of column labels to encode.
        drop_original (bool, optional): If True, drop original columns. Defaults to False.
        handle_unknown (str, optional): How to handle unknown categories for one-hot encoding. Options are 'error' or 'ignore'. Defaults to 'error'.

    Returns:
        str: The execution result message.
    """
    try:
        # Get file extension
        if work_dir:
            dataset_path = os.path.join(work_dir, os.path.normpath(dataset_path))
        file_extension = dataset_path.split('.')[-1].lower()

        # Read the dataset based on the file extension
        if file_extension == 'csv':
            data = pd.read_csv(dataset_path)
        elif file_extension == 'xlsx':
            data = pd.read_excel(dataset_path)
        elif file_extension == 'json':
            data = pd.read_json(dataset_path)
        else:
            return f"Feature engineering task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        if isinstance(columns, str):
            columns = [columns]

        # Check if specified columns exist
        missing_columns = set(columns) - set(data.columns)
        if missing_columns:
            return f"Feature engineering task failed. The reason is: Columns {missing_columns} not found in the dataset."

        result = data.copy()

        # One-Hot Encoding
        encoder = OneHotEncoder(sparse_output=False, handle_unknown=handle_unknown)
        encoded = encoder.fit_transform(data[columns])
        new_columns = [f"{col}_{val}" for col, vals in zip(columns, encoder.categories_) for val in vals]
        encoded_df = pd.DataFrame(encoded, columns=new_columns, index=data.index)
        result = pd.concat([data, encoded_df], axis=1)
        if drop_original:
            result = result.drop(columns, axis=1)

        # Define the path for saving the cleaned dataset
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + "_feature_engineered_{}.".format(index) + dataset_path.split(".")[-1]
        saved_dataset_file = saved_dataset_path.split("/")[-1]

        # Save the cleaned data to the new file
        if file_extension == 'csv':
            result.to_csv(saved_dataset_path, index=False)
        elif file_extension == 'xlsx':
            result.to_excel(saved_dataset_path, index=False)
        elif file_extension == 'json':
            result.to_json(saved_dataset_path, orient='records', lines=True)

        return f"Feature engineering task completed successfully. One-Hot encoding applied to columns {columns} and saved the result in '{saved_dataset_file}'."

    except Exception as e:
        return f"Feature engineering task failed. The reason is: Error occurred during encoding: {e}"


def label_encode(dataset_path: str, 
                 columns: Union[str, List[str]], 
                 work_dir: str = None,
                 index: int = None) -> str:
    """
    Perform different types of encoding (one-hot, label, frequency, or target encoding) 
    on specified columns in the dataset.

    Args:
        dataset_path (str): The path to the dataset file.
        columns (str or List[str]): Column label or list of column labels to encode.
    Returns:
        str: The execution result message.
    """
    try:
        # Get file extension
        if work_dir:
            dataset_path = os.path.join(work_dir, os.path.normpath(dataset_path))
        file_extension = dataset_path.split('.')[-1].lower()

        # Read the dataset based on the file extension
        if file_extension == 'csv':
            data = pd.read_csv(dataset_path)
        elif file_extension == 'xlsx':
            data = pd.read_excel(dataset_path)
        elif file_extension == 'json':
            data = pd.read_json(dataset_path)
        else:
            return f"Feature engineering task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        if isinstance(columns, str):
            columns = [columns]

        # Check if specified columns exist
        missing_columns = set(columns) - set(data.columns)
        if missing_columns:
            return f"Feature engineering task failed. The reason is: Columns {missing_columns} not found in the dataset."

        result = data.copy()

        # Label Encoding
        for col in columns:
            col_data = data[col]
            if pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
                encoder = LabelEncoder()
                encoded_col_name = f"{col}_encoded"
                result[encoded_col_name] = encoder.fit_transform(col_data.astype(str))
            else:
                return f"Feature engineering task failed. Column '{col}' is not categorical."

        # Define the path for saving the cleaned dataset
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + "_feature_engineered_{}.".format(index) + dataset_path.split(".")[-1]
        saved_dataset_file = saved_dataset_path.split("/")[-1]

        # Save the cleaned data to the new file
        if file_extension == 'csv':
            result.to_csv(saved_dataset_path, index=False)
        elif file_extension == 'xlsx':
            result.to_excel(saved_dataset_path, index=False)
        elif file_extension == 'json':
            result.to_json(saved_dataset_path, orient='records', lines=True)

        return f"Feature engineering task completed successfully. Label encoding applied to columns {columns} and saved the result in '{saved_dataset_file}'."

    except Exception as e:
        return f"Feature engineering task failed. The reason is: Error occurred during encoding: {e}"


def frequency_encode(dataset_path: str, 
                   columns: Union[str, List[str]], 
                   drop_original: bool = False, 
                   work_dir: str = None,
                   index: int = None) -> str:
    """
    Perform different types of encoding (one-hot, label, frequency, or target encoding) 
    on specified columns in the dataset.

    Args:
        dataset_path (str): The path to the dataset file.
        columns (str or List[str]): Column label or list of column labels to encode.
        drop_original (bool, optional): If True, drop original columns. Defaults to False.
    Returns:
        str: The execution result message.
    """
    try:
        # Get file extension
        if work_dir:
            dataset_path = os.path.join(work_dir, os.path.normpath(dataset_path))
        file_extension = dataset_path.split('.')[-1].lower()

        # Read the dataset based on the file extension
        if file_extension == 'csv':
            data = pd.read_csv(dataset_path)
        elif file_extension == 'xlsx':
            data = pd.read_excel(dataset_path)
        elif file_extension == 'json':
            data = pd.read_json(dataset_path)
        else:
            return f"Feature engineering task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        if isinstance(columns, str):
            columns = [columns]

        # Check if specified columns exist
        missing_columns = set(columns) - set(data.columns)
        if missing_columns:
            return f"Feature engineering task failed. The reason is: Columns {missing_columns} not found in the dataset."

        result = data.copy()

        # Frequency Encoding
        for col in columns:
            col_data = data[col]
            frequency = col_data.value_counts(normalize=True)
            encoded_col_name = f"{col}_freq"
            result[encoded_col_name] = col_data.map(frequency)
        if drop_original:
            result = result.drop(columns, axis=1)

        # Define the path for saving the cleaned dataset
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + "_feature_engineered_{}.".format(index) + dataset_path.split(".")[-1]
        saved_dataset_file = saved_dataset_path.split("/")[-1]

        # Save the cleaned data to the new file
        if file_extension == 'csv':
            result.to_csv(saved_dataset_path, index=False)
        elif file_extension == 'xlsx':
            result.to_excel(saved_dataset_path, index=False)
        elif file_extension == 'json':
            result.to_json(saved_dataset_path, orient='records', lines=True)

        return f"Feature engineering task completed successfully. Frequency encoding applied to columns {columns} and saved the result in '{saved_dataset_file}'."

    except Exception as e:
        return f"Feature engineering task failed. The reason is: Error occurred during encoding: {e}"
    

def target_encode(dataset_path: str, 
                   columns: Union[str, List[str]], 
                   target: str, 
                   min_samples_leaf: int = 1, 
                   smoothing: float = 1.0, 
                   work_dir: str = None,
                   index: int = None) -> str:
    """
    Perform different types of encoding (one-hot, label, frequency, or target encoding) 
    on specified columns in the dataset.

    Args:
        dataset_path (str): The path to the dataset file.
        columns (str or List[str]): Column label or list of column labels to encode.
        target (str, optional): The name of the target column for target encoding (only required for 'target' encoding).
        min_samples_leaf (int, optional): Minimum samples to take category average into account for target encoding. Defaults to 1.
        smoothing (float, optional): Smoothing effect for target encoding. Defaults to 1.0.
    Returns:
        str: The execution result message.
    """
    try:
        # Get file extension
        if work_dir:
            dataset_path = os.path.join(work_dir, os.path.normpath(dataset_path))
        file_extension = dataset_path.split('.')[-1].lower()

        # Read the dataset based on the file extension
        if file_extension == 'csv':
            data = pd.read_csv(dataset_path)
        elif file_extension == 'xlsx':
            data = pd.read_excel(dataset_path)
        elif file_extension == 'json':
            data = pd.read_json(dataset_path)
        else:
            return f"Feature engineering task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        if isinstance(columns, str):
            columns = [columns]

        # Check if specified columns exist
        missing_columns = set(columns) - set(data.columns)
        if missing_columns:
            return f"Feature engineering task failed. The reason is: Columns {missing_columns} not found in the dataset."

        result = data.copy()

        # Target Encoding
        if target not in data.columns:
            return f"Feature engineering task failed. The reason is: Target column '{target}' not found in the dataset."

        if min_samples_leaf < 0:
            return f"Feature engineering task failed. The reason is: min_samples_leaf should be non-negative."

        if smoothing <= 0:
            return f"Feature engineering task failed. The reason is: smoothing should be positive."

        prior = data[target].mean()
        for col in columns:
            col_data = data[col]
            averages = data.groupby(col)[target].agg(["count", "mean"])
            smoothing_factor = expit((averages["count"] - min_samples_leaf) / smoothing)
            averages["smooth"] = prior * (1 - smoothing_factor) + averages["mean"] * smoothing_factor
            encoded_col_name = f"{col}_target_enc"
            result[encoded_col_name] = col_data.map(averages["smooth"]).fillna(prior)

        # Define the path for saving the cleaned dataset
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + "_feature_engineered_{}.".format(index) + dataset_path.split(".")[-1]
        saved_dataset_file = saved_dataset_path.split("/")[-1]

        # Save the cleaned data to the new file
        if file_extension == 'csv':
            result.to_csv(saved_dataset_path, index=False)
        elif file_extension == 'xlsx':
            result.to_excel(saved_dataset_path, index=False)
        elif file_extension == 'json':
            result.to_json(saved_dataset_path, orient='records', lines=True)

        return f"Feature engineering task completed successfully. Target encoding applied to columns {columns} and saved the result in '{saved_dataset_file}'."

    except Exception as e:
        return f"Feature engineering task failed. The reason is: Error occurred during encoding: {e}"


def correlation_feature_selection(dataset_path: str,
                                  target: str, 
                                  method: str = 'pearson',
                                  threshold: float = 0.5, 
                                  work_dir: str = None,
                                  index: int = None) -> str:
    """
    Perform feature selection based on correlation or variance analysis, and return the selected features.

    Args:
        dataset_path (str): The path to the dataset file.
        target (str): The target column for correlation-based selection.
        method (str, optional): The correlation method to use. 
            Options: 'pearson', 'spearman', 'kendall'. Defaults to 'pearson'.
        threshold (float, optional): The threshold for selection. For 'correlation', features with absolute correlation 
                                     greater than this value will be selected. Defaults to 0.5.
    Returns:
        str: The execution result message.
    """
    try:
        # Get file extension
        if work_dir:
            dataset_path = os.path.join(work_dir, os.path.normpath(dataset_path))
        file_extension = dataset_path.split('.')[-1].lower()

        # Read the dataset based on the file extension
        if file_extension == 'csv':
            data = pd.read_csv(dataset_path)
        elif file_extension == 'xlsx':
            data = pd.read_excel(dataset_path)
        elif file_extension == 'json':
            data = pd.read_json(dataset_path)
        else:
            return f"Feature engineering task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        # Perform correlation-based feature selection
        if target is None:
            return "Feature engineering task failed. The reason is: 'target' must be specified for correlation-based feature selection."
        
        if target not in data.columns:
            return f"Feature engineering task failed. The reason is: Target column '{target}' not found in the dataset."

        # Separate features and target
        X = data.drop(columns=[target])
        y = data[target]

        # Calculate correlation
        if method == 'spearman':
            corr_matrix, _ = spearmanr(X, y)
            corr_with_target = pd.Series(corr_matrix[-1][:-1], index=X.columns)
        else:
            corr_with_target = X.apply(lambda x: x.corr(y, method=method))

        # Select features based on threshold
        selected_features = corr_with_target[abs(corr_with_target) > threshold]

        return f"Feature engineering task completed successfully. Correlation-based feature selection applied with threshold {threshold}. Selected features: {selected_features.index.tolist()}"

    except Exception as e:
        return f"Feature engineering task failed. The reason is: Error occurred during feature selection: {e}"


def variance_feature_selection(dataset_path: str, 
                               threshold: float = 0.0, 
                               columns: Union[str, List[str]] = None,
                               work_dir: str = None,
                               index: int = None) -> str:
    """
    Perform feature selection based on correlation or variance analysis, and return the selected features.

    Args:
        dataset_path (str): The path to the dataset file.
        threshold (float, optional): The threshold for selection. For 'correlation', features with absolute correlation 
                                     greater than this value will be selected. For 'variance', features with a 
                                     variance lower than this value will be removed. Defaults to 0.5 for correlation and 0.0 for variance.
        columns (str or List[str], optional): Column label or sequence of labels to consider. If None, all columns are used. Defaults to None.

    Returns:
        str: The execution result message.
    """
    try:
        # Get file extension
        if work_dir:
            dataset_path = os.path.join(work_dir, os.path.normpath(dataset_path))
        file_extension = dataset_path.split('.')[-1].lower()

        # Read the dataset based on the file extension
        if file_extension == 'csv':
            data = pd.read_csv(dataset_path)
        elif file_extension == 'xlsx':
            data = pd.read_excel(dataset_path)
        elif file_extension == 'json':
            data = pd.read_json(dataset_path)
        else:
            return f"Feature engineering task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        if columns is None:
            columns = data.columns
        elif isinstance(columns, str):
            columns = [columns]

        # Select specified columns
        X = data[columns]

        # Initialize VarianceThreshold
        selector = VarianceThreshold(threshold=threshold)

        # Fit the selector
        selector.fit(X)

        # Get the mask of selected features
        feature_mask = selector.get_support()

        # # Get the variances
        # variances = selector.variances_

        return f"Feature engineering task completed successfully. Variance-based feature selection applied with threshold {threshold}. Selected features: {X.columns[feature_mask].tolist()}"

    except Exception as e:
        return f"Feature engineering task failed. The reason is: Error occurred during feature selection: {e}"


def scale_features(dataset_path: str, 
                   columns: Union[str, List[str]], 
                   method: str = 'standard', 
                   copy: bool = True,
                   work_dir: str = None,
                   index: int = None) -> str:
    """
    Scale numerical features in the specified columns of the dataset located at `dataset_path` and save the resulting dataset to a new file.

    Args:
        dataset_path (str): The path to the dataset file.
        columns (str or List[str]): Column label or sequence of labels of numerical features to scale.
        method (str, optional): The scaling method to use. 
            Options: 'standard' for StandardScaler, 
                     'minmax' for MinMaxScaler, 
                     'robust' for RobustScaler. 
            Defaults to 'standard'.
        copy (bool, optional): If False, try to avoid a copy and do inplace scaling instead. 
            This is not guaranteed to always work inplace. Defaults to True.

    Returns:
        str: The execution result message.
    """
    try:
        # Get file extension
        if work_dir:
            dataset_path = os.path.join(work_dir, os.path.normpath(dataset_path))
        file_extension = dataset_path.split('.')[-1].lower()

        # Read the dataset based on the file extension
        if file_extension == 'csv':
            data = pd.read_csv(dataset_path)
        elif file_extension == 'xlsx':
            data = pd.read_excel(dataset_path)
        elif file_extension == 'json':
            data = pd.read_json(dataset_path)
        else:
            return f"Feature engineering task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        if isinstance(columns, str):
            columns = [columns]

        # Check if specified columns exist
        missing_columns = set(columns) - set(data.columns)
        if missing_columns:
            return f"Feature engineering task failed. The reason is: Columns {missing_columns} not found in the dataset."

        # Handle duplicate columns
        unique_columns = []
        for col in columns:
            if col in unique_columns:
                continue
            col_data = data[col]
            if isinstance(col_data, pd.DataFrame):
                # Check if all duplicate columns are identical
                if col_data.nunique().eq(1).all():
                    # print(f"Warning: Duplicate identical columns found for '{col}'. Only one instance will be scaled.")
                    unique_columns.append(col)
                else:
                    return f"Feature engineering task failed. The reason is: Duplicate non-identical columns found for '{col}'. Please resolve this before scaling."
            else:
                unique_columns.append(col)

        # Check if all specified columns are numerical
        non_numeric_cols = [col for col in unique_columns if not pd.api.types.is_numeric_dtype(data[col])]
        if non_numeric_cols:
            return f"Feature engineering task failed. The reason is: The following columns are not numerical: {non_numeric_cols}. Please only specify numerical columns for scaling."

        # Select the appropriate scaler
        if method == 'standard':
            scaler = StandardScaler(copy=copy)
        elif method == 'minmax':
            scaler = MinMaxScaler(copy=copy)
        elif method == 'robust':
            scaler = RobustScaler(copy=copy)
        else:
            return "Feature engineering task failed. The reason is: Invalid method. Choose 'standard', 'minmax', or 'robust'."

        # Create a copy of the dataframe if required
        if copy:
            data = data.copy()

        # Fit and transform the selected columns
        scaled_data = scaler.fit_transform(data[unique_columns])

        # Replace the original columns with scaled data
        data[unique_columns] = scaled_data

        # Define the path for saving the cleaned dataset
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + "_feature_engineered_{}.".format(index) + dataset_path.split(".")[-1]
        saved_dataset_file = saved_dataset_path.split("/")[-1]

        # Save the result to the new file
        if file_extension == 'csv':
            data.to_csv(saved_dataset_path, index=False)
        elif file_extension == 'xlsx':
            data.to_excel(saved_dataset_path, index=False)
        elif file_extension == 'json':
            data.to_json(saved_dataset_path, orient='records', lines=True)

        return f"Feature engineering task completed successfully. Scaled columns {columns} using '{method}' method and saved the result in '{saved_dataset_file}'."
    
    except Exception as e:
        return f"Feature engineering task failed. The reason is: Error occurred during feature scaling: {e}"


def perform_pca(dataset_path: str, 
                n_components: Union[int, float, str] = 0.95, 
                columns: Union[str, List[str]] = None, 
                scale: bool = True,
                work_dir: str = None,
                index: int = None) -> str:
    """
    Perform Principal Component Analysis (PCA) on the specified columns of the dataset located at `dataset_path` 
    and save the resulting dataset to a new file.

    Args:
        dataset_path (str): The path to the dataset file.
        n_components (int, float, or str, optional): Number of components to keep.
            If int, it represents the exact number of components.
            If float between 0 and 1, it represents the proportion of variance to be retained.
            If 'mle', Minka's MLE is used to guess the dimension.
            Defaults to 0.95 (95% of variance).
        columns (str or List[str], optional): Column label or sequence of labels to consider.
            If None, use all columns. Defaults to None.
        scale (bool, optional): Whether to scale the data before applying PCA.
            Recommended when features are not on the same scale. Defaults to True.

    Returns:
        str: The execution result message.
    """
    try:
        # Get file extension
        if work_dir:
            dataset_path = os.path.join(work_dir, os.path.normpath(dataset_path))
        file_extension = dataset_path.split('.')[-1].lower()

        # Read the dataset based on the file extension
        if file_extension == 'csv':
            data = pd.read_csv(dataset_path)
        elif file_extension == 'xlsx':
            data = pd.read_excel(dataset_path)
        elif file_extension == 'json':
            data = pd.read_json(dataset_path)
        else:
            return f"Feature engineering task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        if columns is None:
            columns = data.columns
        elif isinstance(columns, str):
            columns = [columns]

        X = data[columns]

        # Check for non-numeric data types
        non_numeric_cols = X.select_dtypes(exclude=['number']).columns
        if not non_numeric_cols.empty:
            return f"Feature engineering task failed. The reason is: Non-numeric data types detected in columns: {list(non_numeric_cols)}. Please ensure all features are properly encoded and scaled before applying PCA."

        # Warn if data doesn't seem to be scaled
        # if (X.std() > 10).any():
        #     return f"Feature engineering task failed. Some features have high standard deviations. Consider scaling your data before applying PCA for better results."

        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(X)

        # Create a DataFrame with PCA results
        pca_df = pd.DataFrame(
            data=pca_result,
            columns=[f'PC{i+1}' for i in range(pca_result.shape[1])]
        )

        # Define the path for saving the PCA results
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + "_feature_engineered_{}.".format(index) + dataset_path.split(".")[-1]
        saved_dataset_file = saved_dataset_path.split("/")[-1]

        # Save the result to the new file
        if file_extension == 'csv':
            pca_df.to_csv(saved_dataset_path, index=False)
        elif file_extension == 'xlsx':
            pca_df.to_excel(saved_dataset_path, index=False)
        elif file_extension == 'json':
            pca_df.to_json(saved_dataset_path, orient='records', lines=True)

        return f"Feature engineering task completed successfully. PCA results saved to '{saved_dataset_file}'."

    except Exception as e:
        return f"Feature engineering task failed. The reason is: Error occurred during PCA: {e}"


def perform_rfe(dataset_path: str, 
                target: Union[str, pd.Series], 
                n_features_to_select: Union[int, float] = 0.5, 
                step: int = 1, 
                estimator: str = 'auto', 
                columns: Union[str, List[str]] = None,
                work_dir: str = None,
                index: int = None) -> str:
    """
    Perform Recursive Feature Elimination (RFE) on the specified columns of the dataset located at `dataset_path`
    and save the resulting dataset to a new file.

    Args:
        dataset_path (str): The path to the dataset file.
        target (str or pd.Series): The target variable. If string, it should be the name of the target column in data.
        n_features_to_select (int or float, optional): Number of features to select.
            If int, it represents the exact number of features.
            If float between 0 and 1, it represents the proportion of features to select.
            Defaults to 0.5 (50% of features).
        step (int, optional): Number of features to remove at each iteration. Defaults to 1.
        estimator (str, optional): The estimator to use for feature importance ranking.
            Options: 'auto', 'logistic', 'rf', 'linear', 'rf_regressor'.
            'auto' will automatically choose based on the target variable type.
            Defaults to 'auto'.
        columns (str or List[str], optional): Column label or sequence of labels to consider.
            If None, use all columns except the target (if target is a column name in data).
            Defaults to None.

    Returns:
        str: The execution result message.
    """
    try:
        # Get file extension
        if work_dir:
            dataset_path = os.path.join(work_dir, os.path.normpath(dataset_path))
        file_extension = dataset_path.split('.')[-1].lower()

        # Read the dataset based on the file extension
        if file_extension == 'csv':
            data = pd.read_csv(dataset_path)
        elif file_extension == 'xlsx':
            data = pd.read_excel(dataset_path)
        elif file_extension == 'json':
            data = pd.read_json(dataset_path)
        else:
            return f"Feature engineering task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        # Prepare the feature matrix and target vector
        if isinstance(target, str):
            y = data[target]
            X = data.drop(columns=[target])
        else:
            y = target
            X = data

        # Select columns if specified
        if columns:
            if isinstance(columns, str):
                columns = [columns]
            X = X[columns]

        # Determine the number of features to select
        if isinstance(n_features_to_select, float):
            n_features_to_select = max(1, int(n_features_to_select * X.shape[1]))

        # Determine if the target is continuous or discrete
        is_continuous = np.issubdtype(y.dtype, np.number) and len(np.unique(y)) > 10

        # Choose the estimator
        if estimator == 'auto':
            estimator = 'linear' if is_continuous else 'logistic'

        if estimator == 'logistic':
            est = LogisticRegression(random_state=42)
        elif estimator == 'rf':
            est = RandomForestClassifier(random_state=42)
        elif estimator == 'linear':
            est = LinearRegression()
        elif estimator == 'rf_regressor':
            est = RandomForestRegressor(random_state=42)
        else:
            return "Feature engineering task failed. The reason is: Invalid estimator. Choose 'auto', 'logistic', 'rf', 'linear', or 'rf_regressor'."

        # Perform RFE
        rfe = RFE(estimator=est, n_features_to_select=n_features_to_select, step=step)
        rfe.fit(X, y)

        # Get selected features
        selected_features = X.columns[rfe.support_].tolist()

        # Save the resulting dataset with selected features
        result_data = data[selected_features]

        # Define the path for saving the feature-engineered dataset
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + "_feature_engineered_{}.".format(index) + dataset_path.split(".")[-1]
        saved_dataset_file = saved_dataset_path.split("/")[-1]

        # Save the result to the new file
        if file_extension == 'csv':
            result_data.to_csv(saved_dataset_path, index=False)
        elif file_extension == 'xlsx':
            result_data.to_excel(saved_dataset_path, index=False)
        elif file_extension == 'json':
            result_data.to_json(saved_dataset_path, orient='records', lines=True)

        return f"Feature engineering task completed successfully. Selected features saved to '{saved_dataset_file}'."

    except Exception as e:
        return f"Feature engineering task failed. The reason is: Error occurred during RFE: {e}"


def create_polynomial_features(dataset_path: str, 
                               columns: Union[str, List[str]],
                               degree: int = 2, 
                               interaction_only: bool = False, 
                               include_bias: bool = False, 
                               work_dir: str = None,
                               index: int = None) -> str:
    """
    Create feature from specified columns of the dataset located at `dataset_path`.
    The function can create polynomial features or feature combinations (multiplication or addition).
    
    Args:
        dataset_path (str): The path to the dataset file.
        columns (str or List[str]): Column label or list of column labels to use for feature creation.
        degree (int, optional): The degree of polynomial features (only for 'polynomial' method). Defaults to 2.
        interaction_only (bool, optional): If True, only interaction features are produced for polynomial features. Defaults to False.
        include_bias (bool, optional): If True, includes a bias column (all 1s) for polynomial features. Defaults to False.
    Returns:
        str: The execution result message.
    """
    try:
        # Get file extension
        if work_dir:
            dataset_path = os.path.join(work_dir, os.path.normpath(dataset_path))
        file_extension = dataset_path.split('.')[-1].lower()

        # Read the dataset based on the file extension
        if file_extension == 'csv':
            data = pd.read_csv(dataset_path)
        elif file_extension == 'xlsx':
            data = pd.read_excel(dataset_path)
        elif file_extension == 'json':
            data = pd.read_json(dataset_path)
        else:
            return f"Feature engineering task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        if isinstance(columns, str):
            columns = [columns]

        # Check if specified columns exist
        missing_columns = set(columns) - set(data.columns)
        if missing_columns:
            return f"Feature engineering task failed. The reason is: Columns {missing_columns} not found in the DataFrame."

        # Check if all specified columns are numeric
        non_numeric_cols = [col for col in columns if not pd.api.types.is_numeric_dtype(data[col])]
        if non_numeric_cols:
            return f"Feature engineering task failed. The reason is: Columns {non_numeric_cols} are not numeric. Feature engineering requires numeric data."

        # Polynomial feature creation
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
        X = data[columns]
        poly_features = poly.fit_transform(X)

        # Generate feature names
        feature_names = poly.get_feature_names_out(columns)
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=data.index)

        # Remove duplicate columns (original features)
        poly_df = poly_df.loc[:, ~poly_df.columns.duplicated()]

        result = pd.concat([data, poly_df], axis=1)

        # Define the path for saving the feature-engineered dataset
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + "_feature_engineered_{}.".format(index) + dataset_path.split(".")[-1]
        saved_dataset_file = saved_dataset_path.split("/")[-1]

        # Save the result to the new file
        if file_extension == 'csv':
            result.to_csv(saved_dataset_path, index=False)
        elif file_extension == 'xlsx':
            result.to_excel(saved_dataset_path, index=False)
        elif file_extension == 'json':
            result.to_json(saved_dataset_path, orient='records', lines=True)

        return f"Feature engineering task completed successfully. The feature-engineered dataset is saved in '{saved_dataset_file}'."

    except Exception as e:
        return f"Feature engineering task failed. The reason is: Error occurred during feature engineering: {e}"


def create_feature_combinations(dataset_path: str, 
                                columns: Union[str, List[str]], 
                                combination_type: str = 'multiplication', 
                                max_combination_size: int = 2,
                                work_dir: str = None,
                                index: int = None) -> str:
    """
    Create feature from specified columns of the dataset located at `dataset_path`.
    The function can create polynomial features or feature combinations (multiplication or addition).
    
    Args:
        dataset_path (str): The path to the dataset file.
        columns (str or List[str]): Column label or list of column labels to use for feature creation.
        combination_type (str, optional): The type of feature combination. Options: 'multiplication' or 'addition' (only for 'combination' method). Defaults to 'multiplication'.
        max_combination_size (int, optional): Maximum number of features to combine (only for 'combination' method). Defaults to 2.

    Returns:
        str: The execution result message.
    """
    try:
        # Get file extension
        if work_dir:
            dataset_path = os.path.join(work_dir, os.path.normpath(dataset_path))
        file_extension = dataset_path.split('.')[-1].lower()

        # Read the dataset based on the file extension
        if file_extension == 'csv':
            data = pd.read_csv(dataset_path)
        elif file_extension == 'xlsx':
            data = pd.read_excel(dataset_path)
        elif file_extension == 'json':
            data = pd.read_json(dataset_path)
        else:
            return f"Feature engineering task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        if isinstance(columns, str):
            columns = [columns]

        # Check if specified columns exist
        missing_columns = set(columns) - set(data.columns)
        if missing_columns:
            return f"Feature engineering task failed. The reason is: Columns {missing_columns} not found in the DataFrame."

        # Check if all specified columns are numeric
        non_numeric_cols = [col for col in columns if not pd.api.types.is_numeric_dtype(data[col])]
        if non_numeric_cols:
            return f"Feature engineering task failed. The reason is: Columns {non_numeric_cols} are not numeric. Feature engineering requires numeric data."

        # Feature combination (multiplication or addition)
        if max_combination_size < 2:
            return "Feature engineering task failed. The reason is: max_combination_size must be at least 2."

        if combination_type not in ['multiplication', 'addition']:
            return "Feature engineering task failed. The reason is: combination_type must be either 'multiplication' or 'addition'."

        result = data.copy()

        # Generate feature combinations
        for r in range(2, min(len(columns), max_combination_size) + 1):
            for combo in combinations(columns, r):
                if combination_type == 'multiplication':
                    new_col = result[list(combo)].prod(axis=1)
                    new_col_name = ' * '.join(combo)
                else:  # addition
                    new_col = result[list(combo)].sum(axis=1)
                    new_col_name = ' + '.join(combo)

                result[new_col_name] = new_col

        # Define the path for saving the feature-engineered dataset
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + "_feature_engineered_{}.".format(index) + dataset_path.split(".")[-1]
        saved_dataset_file = saved_dataset_path.split("/")[-1]

        # Save the result to the new file
        if file_extension == 'csv':
            result.to_csv(saved_dataset_path, index=False)
        elif file_extension == 'xlsx':
            result.to_excel(saved_dataset_path, index=False)
        elif file_extension == 'json':
            result.to_json(saved_dataset_path, orient='records', lines=True)

        return f"Feature engineering task completed successfully. The feature-engineered dataset is saved in '{saved_dataset_file}'."

    except Exception as e:
        return f"Feature engineering task failed. The reason is: Error occurred during feature engineering: {e}"
    

def feature_engineering_tool_creation(code, script_file_name, dataset_path, work_dir, index):
    script_file_name = os.path.normpath(script_file_name)
    dataset_path = os.path.normpath(dataset_path)
    saved_dataset_file = ".".join(dataset_path.split(".")[:-1]) + "_feature_engineered_{}.".format(index) + dataset_path.split(".")[-1]
    write_program(code, os.path.join(work_dir, script_file_name))
    success, output = execute_script(script_file_name, work_dir)
    if success:
        if output is None or output == '':
            return f"Feature engineering task completed successfully. The relevant code is saved in '{script_file_name}'. The modified data is saved in '{saved_dataset_file}'."
        return f"Feature engineering task completed successfully. The relevant code is saved in '{script_file_name}'. The modified data is saved in '{saved_dataset_file}'. The output is as follows: \n{output}"
    else:
        return f"Feature engineering task failed. The relevant code is as follows: \n{code}\nThe reason is: {output}"


feature_engineering_tools = [
    {
        "type": "function",
        "function": {
            "name": "one_hot_encode",
            "description": "Perform one-hot encoding on specified categorical columns. The processed dataset will be saved and the filename will be returned.",
            "parameters": {
                "type": "object",
                "required": ["dataset_path", "columns"],
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    },
                    "columns": {
                        "type": "list",
                        "description": "Columns to perform encoding on.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "drop_original": {
                        "type": "bool",
                        "description": "If True, drop original columns after encoding. Defaults to False. (Applicable for 'one_hot', 'label', and 'frequency' encoding)"
                    },
                    "handle_unknown": {
                        "type": "string",
                        "description": "How to handle unknown categories during encoding. Options: 'error', 'ignore'. Defaults to 'error'. (Only for 'one_hot' encoding)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "label_encode",
            "description": "Perform label encoding on specified categorical columns. The processed dataset will be saved and the filename will be returned.",
            "parameters": {
                "type": "object",
                "required": ["dataset_path", "columns"],
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    },
                    "columns": {
                        "type": "list",
                        "description": "Columns to perform encoding on.",
                        "items": {
                            "type": "string"
                        }
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "frequency_encode",
            "description": "Perform frequency encoding on specified categorical columns. The processed dataset will be saved and the filename will be returned.",
            "parameters": {
                "type": "object",
                "required": ["dataset_path", "columns"],
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    },
                    "columns": {
                        "type": "list",
                        "description": "Columns to perform encoding on.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "drop_original": {
                        "type": "bool",
                        "description": "If True, drop original columns after encoding. Defaults to False. (Applicable for 'one_hot', 'label', and 'frequency' encoding)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "target_encode",
            "description": "Perform target encoding on specified categorical columns. The processed dataset will be saved and the filename will be returned.",
            "parameters": {
                "type": "object",
                "required": ["dataset_path", "columns", "target"],
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    },
                    "columns": {
                        "type": "list",
                        "description": "Columns to perform encoding on.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "target": {
                        "type": "string",
                        "description": "The target column for encoding."
                    },
                    "min_samples_leaf": {
                        "type": "int",
                        "description": "Minimum number of samples required to include category in the encoding. Default is 1. (Only for 'target' encoding)"
                    },
                    "smoothing": {
                        "type": "float",
                        "description": "Smoothing factor to balance categorical average vs. global prior. Default is 1.0. (Only for 'target' encoding)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "correlation_feature_selection",
            "description": "Perform feature selection based on correlation with the target column. The selected features will be returned.",
            "parameters": {
                "type": "object",
                "required": ["dataset_path", "target"],
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    },
                    "target": {
                        "type": "string",
                        "description": "The target column to calculate correlation."
                    },
                    "method": {
                        "type": "string",
                        "description": "The correlation method to use. Options: 'pearson', 'spearman', 'kendall'. Defaults to 'pearson'."
                    },
                    "threshold": {
                        "type": "float",
                        "description": "The threshold for feature selection. Features with absolute correlation greater than this value will be selected. Defaults to 0.5."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "variance_feature_selection",
            "description": "Perform feature selection based on correlation with the target column or variance analysis. The selected features will be returned.",
            "parameters": {
                "type": "object",
                "required": ["dataset_path"],
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    },
                    "threshold": {
                        "type": "float",
                        "description": "The threshold for feature selection. Features with variance lower than this threshold will be removed. Defaults to 0.0."
                    },
                    "columns": {
                        "type": "list",
                        "description": "Columns to consider for variance-based feature selection. If None, use all columns. Defaults to None.",
                        "items": {
                            "type": "string"
                        }
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scale_features",
            "description": "Scale numerical features in specified columns using the selected scaling method. The processed dataset will be saved and the filename will be returned.",
            "parameters": {
                "type": "object",
                "required": ["dataset_path", "columns"],
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    },
                    "columns": {
                        "type": "list",
                        "description": "Columns to scale."
                    },
                    "method": {
                        "type": "string",
                        "description": "The scaling method. Options: 'standard', 'minmax', 'robust'. Defaults to 'standard'."
                    },
                    "copy": {
                        "type": "bool",
                        "description": "Whether to perform scaling in place or copy the data. Defaults to True."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "perform_pca",
            "description": "Perform Principal Component Analysis (PCA) on the dataset. The processed dataset will be saved and the filename will be returned.",
            "parameters": {
                "type": "object",
                "required": ["dataset_path", "n_components"],
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    },
                    "n_components": {
                        "type": "int or float",
                        "description": "If int, it represents the exact number of components. If float between 0 and 1, it represents the proportion of variance to be retained."
                    },
                    "columns": {
                        "type": "list",
                        "description": "Columns to apply PCA to. Defaults to None (all columns)."
                    },
                    "scale": {
                        "type": "bool",
                        "description": "Whether to scale the data before PCA. Defaults to True."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "perform_rfe",
            "description": "Perform Recursive Feature Elimination (RFE) to select important features using the specified estimator. The processed dataset will be saved and the filename will be returned.",
            "parameters": {
                "type": "object",
                "required": ["dataset_path", "target", "n_features_to_select"],
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    },
                    "target": {
                        "type": "string",
                        "description": "The target column."
                    },
                    "n_features_to_select": {
                        "type": "int or float",
                        "description": "If int, it represents the exact number of features to select. If float between 0 and 1, it represents the proportion of features to select. Defaults to 0.5."
                    },
                    "step": {
                        "type": "int",
                        "description": "Number of features to remove at each iteration. Defaults to 1."
                    },
                    "estimator": {
                        "type": "string",
                        "description": "The estimator for feature ranking. Options: 'auto', 'logistic', 'rf', 'linear', 'rf_regressor'. Defaults to 'auto'."
                    },
                    "columns": {
                        "type": "list",
                        "description": "Columns to consider for RFE. If None, use all columns except the target. Defaults to None."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_polynomial_features",
            "description": "Create polynomial features based on specified columns. The processed dataset will be saved and the filename will be returned.",
            "parameters": {
                "type": "object",
                "required": ["dataset_path", "columns"],
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    },
                    "columns": {
                        "type": "list",
                        "description": "The columns (or a list of columns) to use for feature creation.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "degree": {
                        "type": "int",
                        "description": "Degree of the polynomial features (only used for 'polynomial' method). Defaults to 2."
                    },
                    "interaction_only": {
                        "type": "bool",
                        "description": "If True, only interaction features will be generated (only used for 'polynomial' method). Defaults to False."
                    },
                    "include_bias": {
                        "type": "bool",
                        "description": "If True, include the bias (intercept) term (only used for 'polynomial' method). Defaults to False."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_feature_combinations",
            "description": "Create feature combinations (multiplication or addition) based on specified columns. The processed dataset will be saved and the filename will be returned.",
            "parameters": {
                "type": "object",
                "required": ["dataset_path", "columns"],
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    },
                    "columns": {
                        "type": "list",
                        "description": "The columns (or a list of columns) to use for feature creation.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "combination_type": {
                        "type": "string",
                        "description": "The type of feature combination. Options: 'multiplication' or 'addition' (only used for 'combination' method). Defaults to 'multiplication'."
                    },
                    "max_combination_size": {
                        "type": "int",
                        "description": "Maximum number of features to combine (only used for 'combination' method). Defaults to 2."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "feature_engineering_tool_creation",
            "description": "Create a **general-purpose, reusable feature engineering tool** when necessary to solve the current task. The generated tool should be **dataset-agnostic** and applicable to a wide range of feature engineering scenarios rather than being tailored to a specific dataset or variable. The tool will be saved as a Python script, executed immediately, and return the execution results, including any printed outputs or errors. The created tool is expected to be reusable in future feature engineering tasks within an autonomous data science pipeline.",
            "parameters": {
                "type": "object",
                "required": ["code", "script_file_name", "dataset_path"],
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code of the newly created feature engineering tool. The tool must follow the strict format requirement and be general enough for reuse as: ```python\ndef tool_name(parameters):\n    # detail of the code\n\n# execute the tool\nif __name__ == '__main__':\n    parameters = {...}\n    tool_name(parameters)\n```"
                    },
                    "script_file_name": {
                        "type": "string",
                        "description": "The name of the Python script of the newly created tool will be saved and executed."
                    },
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    }
                }
            }
        }
    }
]

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from scripts.async_llm import LLMsConfig
# from scripts.async_llm import create_llm_instance
# models_config = LLMsConfig.default()
# import asyncio
# import json
# async def run_loop(messages, tools):
#     choice = await llm.generate(messages, tools=tools)
#     return choice
# 
# llm_config = models_config.get('deepseek-ai/DeepSeek-V3.1-Terminus_v32')
# llm = create_llm_instance(llm_config)
# tool_map = {
#     "one_hot_encode": one_hot_encode,
#     "label_encode": label_encode,
#     "frequency_encode": frequency_encode,
#     "target_encode": target_encode,
#     "correlation_feature_selection": correlation_feature_selection,
#     "variance_feature_selection": variance_feature_selection,
#     "scale_features": scale_features,
#     "perform_pca": perform_pca,
#     "perform_rfe": perform_rfe,
#     "create_polynomial_features": create_polynomial_features,
#     "create_feature_combinations": create_feature_combinations,
#     "feature_engineering_tool_creation": feature_engineering_tool_creation
# }

# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("verl/Qwen3-8B")

# work_dir = './test'
# count = 1
# task = 'use the create_feature_combinations tool to process the dataset with columns age and score'
# prompt = FEATURE_ENGINEERING_PROMPT.format(dataset_file='synthetic_dataset.csv', task=task, saved_dataset_file='dataset_data_processed_{}.csv'.format(count))
# messages = [{"role": "system", "content": FEATURE_ENGINEER_SYS_PROMPT},
#             {"role": "user", "content": prompt}]
# choice = asyncio.run(run_loop(messages, feature_engineering_tools))
# finish_reason = choice.finish_reason
# if finish_reason == "tool_calls":
#     messages.append(dict(choice.message))
#     for tool_call in choice.message.tool_calls:
#         tool_call_name = tool_call['function']['name']
#         try:
#             tool_call_arguments = json.loads(tool_call['function']['arguments'])
#             tool_function = tool_map[tool_call_name]
#             tool_result = tool_function(**tool_call_arguments, work_dir=work_dir, index=count)
#         except Exception as e:
#             tool_result = f"Tool call failed due to {str(e)}"
#         print("tool_result:", tool_result)
# else:
#     content = choice.message.content
#     match = re.search(r"```python(.*?)```", content, re.DOTALL)
#     if match:
#         code = match.group(1).strip()
#         with open(os.path.join(work_dir, f"feature_engineering_{count}.py"), "w+", encoding="utf-8") as f:
#             f.write(code)
#         success, output = execute_script(f"feature_engineering_{count}.py", work_dir)
#     else:
#         result = content.strip()
#     print(1)
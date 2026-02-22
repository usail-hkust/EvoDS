# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from uuid import uuid4
from typing import Any, Optional, Union, List
import subprocess
from verl.utils.rollout_trace import rollout_trace_op
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.experimental.agent_loop.data_science_agent_loop import AgentData
from verl.tools.utils.tool_registry import initialize_tools_from_config
import re
import tiktoken
import asyncio
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from itertools import combinations
import torch
import sys
from scipy.special import expit
import warnings 
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


async def write_program(assistant_output, out_fname):
    match = re.search(r"```python(.*?)```", assistant_output, re.DOTALL)
    if match:
        result = match.group(1).strip()
    else:
        result = assistant_output.strip()

    with open(out_fname, "w+", encoding="utf-8") as f:
        f.write(result)

async def execute_script(script_path: str, work_dir: str):
    device = 0
    python = sys.executable
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES") or str(np.arange(torch.cuda.device_count()).tolist())[1:-1]
    cmd_list = [python, "-u", script_path]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd_list,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
            env=my_env
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=900)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return False, "Timeout"
    except Exception as e:
        raise print(f"Something went wrong in executing {script_path}: {e}. Please check if it is ready to be executed.")

    stdout = stdout_bytes.decode(errors='replace')
    stderr = stderr_bytes.decode(errors='replace')

    if proc.returncode != 0:
        raw = stderr
    else:
        raw = stdout

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(raw)
    if len(tokens) >= 1000:
        head = enc.decode(tokens[:500])
        tail = enc.decode(tokens[-500:])
        output = f"{head}\n...\n{tail}"
    else:
        output = raw

    return proc.returncode == 0, output


async def one_hot_encode(dataset_path: str, 
                   columns: Union[str, List[str]], 
                   drop_original: bool = False, 
                   handle_unknown: str = 'error',
                   work_dir: str = None) -> str:
    """
    Perform one-hot encoding on specified columns in the dataset.

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
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + "_feature_engineered." + dataset_path.split(".")[-1]
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
    

async def label_encode(dataset_path: str, 
                   columns: Union[str, List[str]], 
                   work_dir: str = None) -> str:
    """
    Perform label encoding on specified columns in the dataset.

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
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + "_feature_engineered." + dataset_path.split(".")[-1]
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
    

async def frequency_encode(dataset_path: str, 
                   columns: Union[str, List[str]], 
                   drop_original: bool = False, 
                   work_dir: str = None) -> str:
    """
    Perform frequency encoding on specified columns in the dataset.

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
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + "_feature_engineered." + dataset_path.split(".")[-1]
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

    
async def target_encode(dataset_path: str, 
                   columns: Union[str, List[str]], 
                   target: str = None, 
                   min_samples_leaf: int = 1, 
                   smoothing: float = 1.0, 
                   work_dir: str = None) -> str:
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
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + "_feature_engineered." + dataset_path.split(".")[-1]
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
    

async def correlation_feature_selection(dataset_path: str, 
                                        target: str, 
                                        method: str = 'pearson', 
                                        threshold: float = 0.5, 
                                        work_dir: str = None) -> str:
    """
    Perform feature selection based on correlation, and return the selected features.

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


async def variance_feature_selection(dataset_path: str, 
                      threshold: float = 0.0, 
                      columns: Union[str, List[str]] = None,
                      work_dir: str = None) -> str:
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

        # Perform variance-based feature selection
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

        # Get the variances
        variances = selector.variances_

        return f"Feature engineering task completed successfully. Variance-based feature selection applied with threshold {threshold}. Selected features: {X.columns[feature_mask].tolist()}"

    except Exception as e:
        return f"Feature engineering task failed. The reason is: Error occurred during feature selection: {e}"


async def scale_features(dataset_path: str, 
                   columns: Union[str, List[str]], 
                   method: str = 'standard', 
                   copy: bool = True,
                   work_dir: str = None) -> str:
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
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + "_feature_engineered." + dataset_path.split(".")[-1]
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


async def perform_pca(dataset_path: str, 
                n_components: Union[int, float, str] = 0.95, 
                columns: Union[str, List[str]] = None, 
                scale: bool = True,
                work_dir: str = None) -> str:
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
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + "_feature_engineered." + dataset_path.split(".")[-1]
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


async def perform_rfe(dataset_path: str, 
                target: Union[str, pd.Series], 
                n_features_to_select: Union[int, float] = 0.5, 
                step: int = 1, 
                estimator: str = 'auto', 
                columns: Union[str, List[str]] = None,
                work_dir: str = None) -> str:
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
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + "_feature_engineered." + dataset_path.split(".")[-1]
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


async def create_polynomial_features(dataset_path: str, 
                     columns: Union[str, List[str]], 
                     degree: int = 2, 
                     interaction_only: bool = False, 
                     include_bias: bool = False,
                     work_dir: str = None) -> str:
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
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + "_feature_engineered." + dataset_path.split(".")[-1]
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


async def create_feature_combinations(dataset_path: str, 
                     columns: Union[str, List[str]], 
                     combination_type: str = 'multiplication', 
                     max_combination_size: int = 2,
                     work_dir: str = None) -> str:
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
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + "_feature_engineered." + dataset_path.split(".")[-1]
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


async def feature_engineering_tool_creation(code, script_file_path, dataset_path, work_dir):
    script_file_path = os.path.normpath(script_file_path)
    dataset_path = os.path.normpath(dataset_path)
    saved_dataset_file = ".".join(dataset_path.split(".")[:-1]) + "_feature_engineered." + dataset_path.split(".")[-1]
    try:
        await write_program(code, os.path.join(work_dir, script_file_path))
        success, output = await execute_script(script_file_path, work_dir)
    except Exception as e:
        return f"Feature engineering task failed. The reason is: Error occurred while executing the script: {str(e)}"
    if success:
        if output is None or output == '':
            return f"Feature engineering task completed successfully. The relevant code is saved in '{script_file_path}'. The modified data is saved in '{saved_dataset_file}'."
        return f"Feature engineering task completed successfully. The relevant code is saved in '{script_file_path}'. The modified data is saved in '{saved_dataset_file}'. The output is as follows: \n{output}"
    else:
        return f"Feature engineering task failed. The relevant code is as follows: \n\n```python\n{code}\n```\n\nThe failed reason is: {output}"


class OneHotEncodingTool(BaseTool):
    """Tool for encoding columns using one-hot encoding."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["dataset_path", "columns", "work_dir"]

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"status": "initialized"}
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, AgentLoopOutput]:
        missing_list = [k for k in self.required_keys if parameters.get(k) is None]
        if missing_list:
            return ToolResponse(text=f"Encoding task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        columns = parameters.get("columns")
        drop_original = parameters.get("drop_original", False)
        handle_unknown = parameters.get("handle_unknown", 'error')
        work_dir = parameters.get("work_dir")

        result_message = await one_hot_encode(dataset_path, columns, drop_original, handle_unknown, work_dir)
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

class LabelEncodingTool(BaseTool):
    """Tool for encoding columns using label encoding."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["dataset_path", "columns", "work_dir"]

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"status": "initialized"}
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, AgentLoopOutput]:
        missing_list = [k for k in self.required_keys if parameters.get(k) is None]
        if missing_list:
            return ToolResponse(text=f"Encoding task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        columns = parameters.get("columns")
        work_dir = parameters.get("work_dir")

        result_message = await label_encode(dataset_path, columns, work_dir)
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

class FrequencyEncodingTool(BaseTool):
    """Tool for encoding columns using frequency encoding."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["dataset_path", "columns", "work_dir"]

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"status": "initialized"}
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, AgentLoopOutput]:
        missing_list = [k for k in self.required_keys if parameters.get(k) is None]
        if missing_list:
            return ToolResponse(text=f"Encoding task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        columns = parameters.get("columns")
        drop_original = parameters.get("drop_original", False)
        work_dir = parameters.get("work_dir")

        result_message = await frequency_encode(dataset_path, columns, drop_original, work_dir)
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

class TargetEncodingTool(BaseTool):
    """Tool for encoding columns using target encoding."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["dataset_path", "columns", "target", "work_dir"]

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"status": "initialized"}
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, AgentLoopOutput]:
        missing_list = [k for k in self.required_keys if parameters.get(k) is None]
        if missing_list:
            return ToolResponse(text=f"Encoding task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        columns = parameters.get("columns")
        target = parameters.get("target", None)
        min_samples_leaf = parameters.get("min_samples_leaf", 1)
        smoothing = parameters.get("smoothing", 1.0)
        work_dir = parameters.get("work_dir")

        result_message = await target_encode(dataset_path, columns, target, min_samples_leaf, smoothing, work_dir)
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class CorrelationFeatureSelectionTool(BaseTool):
    """Tool for performing feature selection using correlation or variance methods."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["dataset_path", "target", "work_dir"]

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"status": "initialized"}
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, AgentLoopOutput]:
        missing_list = [k for k in self.required_keys if parameters.get(k) is None]
        if missing_list:
            return ToolResponse(text=f"Feature selection task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        target = parameters.get("target", None)
        method = parameters.get("method", 'pearson')
        threshold = parameters.get("threshold", 0.5)
        work_dir = parameters.get("work_dir")

        result_message = await correlation_feature_selection(dataset_path, target, method, threshold, work_dir)
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class VarianceFeatureSelectionTool(BaseTool):
    """Tool for performing feature selection using correlation or variance methods."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["dataset_path", "work_dir"]

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"status": "initialized"}
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, AgentLoopOutput]:
        missing_list = [k for k in self.required_keys if parameters.get(k) is None]
        if missing_list:
            return ToolResponse(text=f"Feature selection task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        threshold = parameters.get("threshold", 0)
        columns = parameters.get("columns", None)
        work_dir = parameters.get("work_dir")

        result_message = await variance_feature_selection(dataset_path, threshold, columns, work_dir)
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class ScaleFeaturesTool(BaseTool):
    """Tool for scaling features using methods like standard, minmax, or robust scaling."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["dataset_path", "columns", "work_dir"]

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"status": "initialized"}
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, AgentLoopOutput]:
        missing_list = [k for k in self.required_keys if parameters.get(k) is None]
        if missing_list:
            return ToolResponse(text=f"Scaling task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        columns = parameters.get("columns")
        method = parameters.get("method", 'standard')
        copy = parameters.get("copy", True)
        work_dir = parameters.get("work_dir")

        result_message = await scale_features(dataset_path, columns, method, copy, work_dir)
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

class PerformPCATool(BaseTool):
    """Tool for performing Principal Component Analysis (PCA) on specified columns of the dataset."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["dataset_path", "n_components", "work_dir"]

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"status": "initialized"}
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, AgentLoopOutput]:
        missing_list = [k for k in self.required_keys if parameters.get(k) is None]
        if missing_list:
            return ToolResponse(text=f"PCA task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        n_components = parameters.get("n_components", 0.95)
        columns = parameters.get("columns", None)
        scale = parameters.get("scale", True)
        work_dir = parameters.get("work_dir")

        result_message = await perform_pca(dataset_path, n_components, columns, scale, work_dir)
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

class PerformRFETool(BaseTool):
    """Tool for performing Recursive Feature Elimination (RFE) on the dataset."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["dataset_path", "target", "n_features_to_select", "work_dir"]

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"status": "initialized"}
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, AgentLoopOutput]:
        missing_list = [k for k in self.required_keys if parameters.get(k) is None]
        if missing_list:
            return ToolResponse(text=f"RFE task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        target = parameters.get("target")
        n_features_to_select = parameters.get("n_features_to_select", 0.5)
        step = parameters.get("step", 1)
        estimator = parameters.get("estimator", 'auto')
        columns = parameters.get("columns", None)
        work_dir = parameters.get("work_dir")

        result_message = await perform_rfe(dataset_path, target, n_features_to_select, step, estimator, columns, work_dir)
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class PolynomialFeatureCreationTool(BaseTool):
    """Tool for creating new features using methods like polynomial or combination of features."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["dataset_path", "columns", "work_dir"]

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"status": "initialized"}
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, AgentLoopOutput]:
        missing_list = [k for k in self.required_keys if parameters.get(k) is None]
        if missing_list:
            return ToolResponse(text=f"Feature creation task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        columns = parameters.get("columns")
        degree = parameters.get("degree", 2)
        interaction_only = parameters.get("interaction_only", False)
        include_bias = parameters.get("include_bias", False)
        work_dir = parameters.get("work_dir")

        result_message = await create_polynomial_features(dataset_path, columns, degree, interaction_only, include_bias, work_dir)
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

class CombinationFeatureCreationTool(BaseTool):
    """Tool for creating new features using methods like polynomial or combination of features."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["dataset_path", "columns", "work_dir"]

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"status": "initialized"}
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, AgentLoopOutput]:
        missing_list = [k for k in self.required_keys if parameters.get(k) is None]
        if missing_list:
            return ToolResponse(text=f"Feature creation task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        columns = parameters.get("columns")
        combination_type = parameters.get("combination_type", 'multiplication')
        max_combination_size = parameters.get("max_combination_size", 2)
        work_dir = parameters.get("work_dir")

        result_message = await create_feature_combinations(dataset_path, columns, combination_type, max_combination_size, work_dir)
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

class FeatureEngineeringToolCreationTool(BaseTool):
    """Tool for creating feature engineering scripts and executing them."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["code", "script_file_name", "dataset_path", "work_dir"]

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"status": "initialized"}
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, AgentLoopOutput]:
        missing_list = [k for k in self.required_keys if parameters.get(k) is None]
        if missing_list:
            return ToolResponse(text=f"Feature engineering script creation failed. The following parameters are missing: {', '.join(missing_list)}")

        code = parameters.get("code")
        script_file_name = parameters.get("script_file_name")
        dataset_path = parameters.get("dataset_path")
        work_dir = parameters.get("work_dir")

        result_message = await feature_engineering_tool_creation(code, script_file_name, dataset_path, work_dir)
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.util import write_program, execute_script
from typing import Any, Union, List
import re
import warnings
warnings.filterwarnings("ignore")
import os


DATA_CLEARNER_SYS_PROMPT = """You are a data science expert specializing in data cleaning tasks. You are working as a sub-agent in a multi-agent system.

# GLOBAL CONTEXT #
The global context contains the overall task objective. You should use this information to understand the broader goal and maintain consistency with the overall workflow.

However, your responsibility is only to solve the assigned data cleaning subtask rather than the entire pipeline.

You have access to a set of tools that can help solve data cleaning tasks. When given a dataset path and a data cleaning task description, your first step is to determine whether the provided tools can directly solve the assigned subtask. If they can, use the appropriate tool to perform the cleaning. The tool will automatically save the cleaned dataset.

If the existing tools cannot directly solve the subtask, use the `data_cleaning_tool_creation` tool to create a new tool based on the provided task description.

The created tool must strictly follow the format below:
```python
def tool_name(parameters):
    # detail of the code

# execute the tool
if __name__ == '__main__':
    parameters = {...}
    tool_name(parameters)
````
"""

DATA_CLEANING_PROMPT = """
You are given a dataset located at {dataset_file}.

{global_task}

Your primary objective is to solve the assigned data cleaning subtask below.

# DATA CLEANING SUBTASK #
{task}

After cleaning, save the cleaned dataset to {saved_dataset_file}.
"""


def fill_missing_values(dataset_path: str, columns: Union[str, List[str]], method: str = 'auto', fill_value: Any = None, work_dir=None, index=None) -> str:
    """
    Fill missing values in specified columns of a DataFrame and save the cleaned dataset to a new file.

    Args:
        dataset_path (str): The path to the dataset file.
        columns (str or List[str]): The name(s) of the column(s) to fill missing values.
        method (str, optional): The method to use for filling missing values. 
            Options: 'auto', 'mean', 'median', 'mode', 'constant'. Defaults to 'auto'.
        fill_value (Any, optional): The value to use when method is 'constant'. Defaults to None.

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
            return "Data cleaning task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        if isinstance(columns, str):
            columns = [columns]

        # Fill missing values based on the specified method
        for column in columns:
            if method == 'auto':
                if pd.api.types.is_numeric_dtype(data[column]):
                    data[column].fillna(data[column].mean(), inplace=True)
                else:
                    data[column].fillna(data[column].mode()[0], inplace=True)
            elif method == 'mean':
                data[column].fillna(data[column].mean(), inplace=True)
            elif method == 'median':
                data[column].fillna(data[column].median(), inplace=True)
            elif method == 'mode':
                data[column].fillna(data[column].mode()[0], inplace=True)
            elif method == 'constant':
                data[column].fillna(fill_value, inplace=True)
            else:
                return "Data cleaning task failed. The reason is: Invalid method. Choose from 'auto', 'mean', 'median', 'mode', or 'constant'."

        # Define the path for saving the cleaned dataset
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + f"_cleaned_{index}." + dataset_path.split(".")[-1]
        saved_dataset_file = saved_dataset_path.split("/")[-1]

        # Save the cleaned data to the new file
        if file_extension == 'csv':
            data.to_csv(saved_dataset_path, index=False)
        elif file_extension == 'xlsx':
            data.to_excel(saved_dataset_path, index=False)
        elif file_extension == 'json':
            data.to_json(saved_dataset_path, orient='records', lines=True)

        return f"Data cleaning task completed successfully. Using method '{method}' to fill missing values in columns {columns}. The cleaned data is saved in '{saved_dataset_file}'."
    except Exception as e:
        return f"Data cleaning task failed. The reason is: Error occurred while filling missing values: {e}"
    

def remove_columns_with_missing_data(dataset_path: str, columns: Union[str, List[str]] = None, thresh: float = 0.5, work_dir=None, index=None) -> str:
    """
    Remove columns containing missing values from a DataFrame based on a threshold.

    Args:
        dataset_path (str): The path to the dataset file.
        thresh (float, optional): The minimum proportion of missing values required to drop a column. 
                                    Should be between 0 and 1. Defaults to 0.5.
        columns (str or List[str], optional): Labels of columns to consider. Defaults to None.

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
            return f"Data cleaning task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        if not 0 <= thresh <= 1:
            return "Data cleaning task failed. The reason is: thresh must be between 0 and 1."

        if columns is not None:
            if isinstance(columns, str):
                columns = [columns]
            data_subset = data[columns]
        else:
            data_subset = data

        # Calculate the number of missing values allowed based on the threshold
        max_missing = int(thresh * len(data_subset))

        # Identify columns to keep
        columns_to_keep = data_subset.columns[data_subset.isna().sum() < max_missing]

        # If columns was specified, add back other columns not in the subset
        if columns is not None:
            columns_to_keep = columns_to_keep.union(data.columns.difference(columns))

        # Filter the dataset to keep only the identified columns
        data_cleaned = data[columns_to_keep]

        # Define the path for saving the cleaned dataset
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + f"_cleaned_{index}." + dataset_path.split(".")[-1]
        saved_dataset_file = saved_dataset_path.split("/")[-1]

        # Save the cleaned data to the new file
        if file_extension == 'csv':
            data_cleaned.to_csv(saved_dataset_path, index=False)
        elif file_extension == 'xlsx':
            data_cleaned.to_excel(saved_dataset_path, index=False)
        elif file_extension == 'json':
            data_cleaned.to_json(saved_dataset_path, orient='records', lines=True)

        return f"Data cleaning task completed successfully. Columns with more than {thresh*100}% missing values were removed. The cleaned data is saved in '{saved_dataset_file}'."
    except Exception as e:
        return f"Data cleaning task failed. The reason is: Error occurred while removing columns with missing data: {e}"
        

def detect_and_handle_outliers_zscore(dataset_path: str, 
                                      columns: Union[str, List[str]], 
                                      threshold: float = 3.0, 
                                      method: str = 'clip',
                                      work_dir=None,
                                      index=None) -> str:
    """
    Detect and handle outliers in specified columns using either the Z-score or IQR method.

    Args:
        dataset_path (str): The path to the dataset file.
        columns (str or List[str]): The name(s) of the column(s) to check for outliers.
        method_type (str, optional): The method to detect outliers. Options: 'zscore', 'iqr'. Defaults to 'zscore'.
        threshold (float, optional): The threshold to identify outliers. Defaults to 3.0.
        method (str, optional): The method to handle outliers. Options: 'clip', 'remove'. Defaults to 'clip'.

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
            return f"Data cleaning task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        if isinstance(columns, str):
            columns = [columns]

        for column in columns:
            if not pd.api.types.is_numeric_dtype(data[column]):
                return f"Data cleaning task failed. The reason is: Column '{column}' must be numeric."

            # Z-score method
            mean = data[column].mean()
            std = data[column].std()
            z_scores = (data[column] - mean) / std

            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
                
            if method == 'clip':
                data.loc[z_scores > threshold, column] = upper_bound
                data.loc[z_scores < -threshold, column] = lower_bound
            elif method == 'remove':
                data = data[abs(z_scores) <= threshold]
            else:
                return f"Data cleaning task failed. The reason is: Invalid handling method. Choose from 'clip' or 'remove'."

        # Define the path for saving the cleaned dataset
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + f"_cleaned_{index}." + dataset_path.split(".")[-1]
        saved_dataset_file = saved_dataset_path.split("/")[-1]

        # Save the cleaned data to the new file
        if file_extension == 'csv':
            data.to_csv(saved_dataset_path, index=False)
        elif file_extension == 'xlsx':
            data.to_excel(saved_dataset_path, index=False)
        elif file_extension == 'json':
            data.to_json(saved_dataset_path, orient='records', lines=True)

        return f"Data cleaning task completed successfully. Outliers were handled using zscore method with '{method}' handling in columns {columns}. The cleaned data is saved in '{saved_dataset_file}'."
    
    except Exception as e:
        return f"Data cleaning task failed. The reason is: Error occurred while handling outliers: {e}"


def detect_and_handle_outliers_iqr(dataset_path: str, 
                                   columns: Union[str, List[str]], 
                                   factor: float = 1.5, 
                                   method: str = 'clip',
                                   work_dir=None,
                                   index=None) -> str:
    """
    Detect and handle outliers in specified columns using either the Z-score or IQR method.

    Args:
        dataset_path (str): The path to the dataset file.
        columns (str or List[str]): The name(s) of the column(s) to check for outliers.
        factor (float, optional): The threshold to identify outliers. Defaults to 1.5.
        method (str, optional): The method to handle outliers. Options: 'clip', 'remove'. Defaults to 'clip'.

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
            return f"Data cleaning task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        if isinstance(columns, str):
            columns = [columns]

        for column in columns:
            if not pd.api.types.is_numeric_dtype(data[column]):
                return f"Data cleaning task failed. The reason is: Column '{column}' must be numeric."

            # IQR method
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR

            if method == 'clip':
                data[column] = data[column].clip(lower_bound, upper_bound)
            elif method == 'remove':
                data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
            else:
                return f"Data cleaning task failed. The reason is: Invalid handling method. Choose from 'clip' or 'remove'."

        # Define the path for saving the cleaned dataset
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + f"_cleaned_{index}." + dataset_path.split(".")[-1]
        saved_dataset_file = saved_dataset_path.split("/")[-1]

        # Save the cleaned data to the new file
        if file_extension == 'csv':
            data.to_csv(saved_dataset_path, index=False)
        elif file_extension == 'xlsx':
            data.to_excel(saved_dataset_path, index=False)
        elif file_extension == 'json':
            data.to_json(saved_dataset_path, orient='records', lines=True)

        return f"Data cleaning task completed successfully. Outliers were handled using iqr method with '{method}' handling in columns {columns}. The cleaned data is saved in '{saved_dataset_file}'."
    
    except Exception as e:
        return f"Data cleaning task failed. The reason is: Error occurred while handling outliers: {e}"


def remove_duplicates(dataset_path: str, columns: Union[str, List[str]] = None, keep: str = 'first', work_dir: str = None, index=None) -> str:
    """
    Remove duplicate rows from a dataset.

    Args:
        dataset_path (str): The path to the dataset file.
        columns (str or List[str], optional): Column label or sequence of labels to consider for identifying duplicates. 
                                                If None, use all columns. Defaults to None.
        keep (str, optional): Determines which duplicates (if any) to keep.
            - 'first' : Drop duplicates except for the first occurrence.
            - 'last' : Drop duplicates except for the last occurrence.
            - False : Drop all duplicates.
            Defaults to 'first'.

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
            return f"Data cleaning task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        # Validate arguments
        if columns is not None and not isinstance(columns, (str, list)):
            return f"Data cleaning task failed. The reason is: The 'columns' argument must be a string, list of strings, or None."

        if keep not in ['first', 'last', False]:
            return f"Data cleaning task failed. The reason is: The 'keep' argument must be 'first', 'last', or False."
        
        # Remove duplicates
        data.drop_duplicates(subset=columns, keep=keep, inplace=True)

        # Define the path for saving the cleaned dataset
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + f"_cleaned_{index}." + dataset_path.split(".")[-1]
        saved_dataset_file = saved_dataset_path.split("/")[-1]

        # Save the cleaned data to the new file
        if file_extension == 'csv':
            data.to_csv(saved_dataset_path, index=False)
        elif file_extension == 'xlsx':
            data.to_excel(saved_dataset_path, index=False)
        elif file_extension == 'json':
            data.to_json(saved_dataset_path, orient='records', lines=True)

        return f"Data cleaning task completed successfully. Duplicates were handled using method '{keep}' in columns {columns}. The cleaned data is saved in '{saved_dataset_file}'."
    
    except Exception as e:
        return f"Data cleaning task failed. The reason is: Error occurred while removing duplicates: {e}"


def convert_data_types(dataset_path: str, columns: Union[str, List[str]], target_type: str, work_dir: str = None, index=None) -> str:
    """
    Convert the data type of specified columns in a dataset.

    Args:
        dataset_path (str): The path to the dataset file.
        columns (str or List[str]): Column label or sequence of labels to convert.
        target_type (str): The target data type to convert to. 
                            Options: 'int', 'float', 'str', 'bool', 'datetime'.

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
            return f"Data cleaning task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        if isinstance(columns, str):
            columns = [columns]

        for column in columns:
            if column not in data.columns:
                return f"Data cleaning task failed. The reason is: Column '{column}' not found in the dataset."

            if target_type == 'int':
                data[column] = pd.to_numeric(data[column], errors='coerce').astype('Int64')
            elif target_type == 'float':
                data[column] = pd.to_numeric(data[column], errors='coerce')
            elif target_type == 'str':
                data[column] = data[column].astype(str)
            elif target_type == 'bool':
                data[column] = data[column].astype(bool)
            elif target_type == 'datetime':
                data[column] = pd.to_datetime(data[column], errors='coerce')
            else:
                return f"Data cleaning task failed. The reason is: Invalid target_type. Choose from 'int', 'float', 'str', 'bool', or 'datetime'."

        # Define the path for saving the cleaned dataset
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + f"_cleaned_{index}." + dataset_path.split(".")[-1]
        saved_dataset_file = saved_dataset_path.split("/")[-1]

        # Save the cleaned data to the new file
        if file_extension == 'csv':
            data.to_csv(saved_dataset_path, index=False)
        elif file_extension == 'xlsx':
            data.to_excel(saved_dataset_path, index=False)
        elif file_extension == 'json':
            data.to_json(saved_dataset_path, orient='records', lines=True)

        return f"Data cleaning task completed successfully. Data types were converted in columns {columns}. The cleaned data is saved in '{saved_dataset_file}'."
    except Exception as e:
        return f"Data cleaning task failed. The reason is: Error occurred while converting data types: {e}"


def format_datetime(dataset_path: str, columns: Union[str, List[str]], format: str = '%Y-%m-%d %H:%M:%S', errors: str = 'coerce', work_dir: str = None, index=None) -> str:
    """
    Format datetime columns in a dataset to a specified format.

    Args:
        dataset_path (str): The path to the dataset file.
        columns (str or List[str]): Column label or sequence of labels to format.
        format (str, optional): The desired output format for datetime. 
                                Defaults to '%Y-%m-%d %H:%M:%S'.
        errors (str, optional): How to handle parsing errors. 
                                Options: 'raise', 'coerce', 'ignore'. Defaults to 'coerce'.

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
            return f"Data cleaning task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        if isinstance(columns, str):
            columns = [columns]

        for column in columns:
            if column not in data.columns:
                return f"Data cleaning task failed. The reason is: Column '{column}' not found in the dataset."

            # First, ensure the column is in datetime format
            data[column] = pd.to_datetime(data[column], errors=errors)

            # Then, format the datetime column
            data[column] = data[column].dt.strftime(format)

        # Define the path for saving the cleaned dataset
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + f"_cleaned_{index}." + dataset_path.split(".")[-1]
        saved_dataset_file = saved_dataset_path.split("/")[-1]

        # Save the cleaned data to the new file
        if file_extension == 'csv':
            data.to_csv(saved_dataset_path, index=False)
        elif file_extension == 'xlsx':
            data.to_excel(saved_dataset_path, index=False)
        elif file_extension == 'json':
            data.to_json(saved_dataset_path, orient='records', lines=True)

        return f"Data cleaning task completed successfully. Datetime columns were formatted in columns {columns}. The cleaned data is saved in '{saved_dataset_file}'."
    except Exception as e:
        return f"Data cleaning task failed. The reason is: Error occurred while formatting datetime columns: {e}"


def data_cleaning_tool_creation(code, script_file_name, dataset_path, work_dir, index):
    script_file_name = os.path.normpath(script_file_name)
    dataset_path = os.path.normpath(dataset_path)
    saved_dataset_file = ".".join(dataset_path.split(".")[:-1]) + "_cleaned_{}.".format(index) + dataset_path.split(".")[-1]
    write_program(code, os.path.join(work_dir, script_file_name))
    success, output = execute_script(script_file_name, work_dir)
    if success:
        if output is None or output == '':
            return f"Data cleaning task completed successfully. The relevant code is saved in '{script_file_name}'. The modified data is saved in '{saved_dataset_file}'."
        return f"Data cleaning task completed successfully. The relevant code is saved in '{script_file_name}'. The modified data is saved in '{saved_dataset_file}'. The output is as follows: \n{output}"
    else:
        return f"Data cleaning task failed. The relevant code is as follows: \n{code}\nThe reason is: {output}"


data_cleaning_tools = [
    {
        "type": "function",
        "function": {
            "name": "fill_missing_values",
            "description": "Fill missing values in specified columns of a dataset using the chosen method. The cleaned dataset will be saved and the filename will be returned.",
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
                        "description": "The columns (or a single column) to fill missing values for.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "method": {
                        "type": "string",
                        "description": "The method for filling missing values. Options: 'auto', 'mean', 'median', 'mode', 'constant'. Defaults to 'auto'."
                    },
                    "fill_value": {
                        "type": "any",
                        "description": "The value to use when the method is 'constant'. Defaults to None."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_columns_with_missing_data",
            "description": "Remove columns from the dataset that contain excessive missing values, based on a given threshold. The cleaned dataset will be saved and the filename will be returned.",
            "parameters": {
                "type": "object",
                "required": ["dataset_path"],
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    },
                    "columns": {
                        "type": "list",
                        "description": "The columns to consider for removal. Defaults to None (all columns).",
                        "items": {
                            "type": "string"
                        }
                    },
                    "thresh": {
                        "type": "float",
                        "description": "The minimum proportion of missing values required to drop a column (value between 0 and 1). Defaults to 0.5."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_and_handle_outliers_zscore",
            "description": "Detect and handle outliers using the Z-score method in specified columns of a dataset. The cleaned dataset will be saved and the filename will be returned.",
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
                        "description": "The columns (or a single column) to check for outliers.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "threshold": {
                        "type": "float",
                        "description": "The threshold used to determine outlier thresholds. Defaults to 3.0."
                    },
                    "method": {
                        "type": "string",
                        "description": "The method to handle outliers. Options: 'clip' (clip the values) or 'remove' (remove the outliers). Defaults to 'clip'."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_and_handle_outliers_iqr",
            "description": "Detect and handle outliers using the IQR method in specified columns of a dataset. The cleaned dataset will be saved and the filename will be returned.",
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
                        "description": "The columns (or a single column) to check for outliers.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "factor": {
                        "type": "float",
                        "description": "The factor used to determine outlier thresholds. Defaults to 1.5."
                    },
                    "method": {
                        "type": "string",
                        "description": "The method to handle outliers. Options: 'clip' (clip the values) or 'remove' (remove the outliers). Defaults to 'clip'."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_duplicates",
            "description": "Remove duplicate rows from the dataset based on specified columns. The cleaned dataset will be saved and the filename will be returned.",
            "parameters": {
                "type": "object",
                "required": ["dataset_path"],
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    },
                    "columns": {
                        "type": "list",
                        "description": "The columns (or a single column) to consider for identifying duplicates. Defaults to None (all columns).",
                        "items": {
                            "type": "string"
                        }
                    },
                    "keep": {
                        "type": "string",
                        "description": "Determines which duplicates to keep. Options: 'first', 'last', or False (drop all duplicates). Defaults to 'first'."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_data_types",
            "description": "Convert the data type of specified columns in the dataset. The cleaned dataset will be saved and the filename will be returned.",
            "parameters": {
                "type": "object",
                "required": ["dataset_path", "columns", "target_type"],
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    },
                    "columns": {
                        "type": "list",
                        "description": "The columns (or a single column) to convert.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "target_type": {
                        "type": "string",
                        "description": "The target data type to convert to. Options: 'int', 'float', 'str', 'bool', 'datetime'."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "format_datetime",
            "description": "Format datetime columns in the dataset to a specified format. The cleaned dataset will be saved and the filename will be returned.",
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
                        "description": "The columns (or a single column) to format as datetime.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "format": {
                        "type": "string",
                        "description": "The desired output format for datetime. Defaults to '%Y-%m-%d %H:%M:%S'."
                    },
                    "errors": {
                        "type": "string",
                        "description": "How to handle parsing errors. Options: 'raise', 'coerce', 'ignore'. Defaults to 'coerce'."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "data_cleaning_tool_creation",
            "description": "Create a **general-purpose, reusable data cleaning tool** when necessary to solve the current task. The generated tool should be **dataset-agnostic** and applicable to a wide range of data cleaning scenarios rather than being tailored to a specific dataset or variable. The tool will be saved as a Python script, executed immediately, and return the execution results, including any printed outputs or errors. The created tool is expected to be reusable in future data cleaning tasks within an autonomous data science pipeline.",
            "parameters": {
                "type": "object",
                "required": ["code", "script_file_name", "dataset_path"],
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code of the newly created data cleaning tool. The tool must follow the strict format requirement and be general enough for reuse as: ```python\ndef tool_name(parameters):\n    # detail of the code\n\n# execute the tool\nif __name__ == '__main__':\n    parameters = {...}\n    tool_name(parameters)\n```"
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


# Tool_Configuration_Extraction_PROMPT = """You are a **Tool Configuration Extraction Assistant**.
# 
# You are provided with the source code of a single function as:
# {code}
# 
# Your task is to analyze the function definition and implementation, and extract a corresponding tool configuration strictly following the format below:
# 
# ```json
# {{
#   "type": "function",
#   "function": {{
#     "name": "tool name",
#     "description": "tool description.",
#     "parameters": {{
#       "type": "object",
#       "required": ["required parameters"],
#       "properties": {{
#         "parameter name": {{
#           "type": "parameter type",
#           "description": "parameter description"
#         }}
#       }}
#     }}
#   }}
# }}
# ```
# 
# **Extraction Rules**
# The tool name must exactly match the function name.
# The tool description must concisely describe the purpose and behavior of the function.
# Parameters must be inferred from the function signature.
# Required parameters are those without default values.
# Optional parameters are those with default values.
# 
# **Output Requirements**
# Output **only** the tool configuration in valid JSON as:
# ```json
# content.
# ```
# Do NOT include explanations, comments, or markdown.
# Do NOT include any content outside the JSON object.
# Follow the exact schema and field names shown above.
# 
# **Additional Constraints**
# Do NOT invent parameters that are not present in the function.
# Do NOT omit parameters defined in the function signature.
# If the function takes no parameters, use:
# 
# ```
# "required": [],
# "properties": {{}}
# ```
# 
# Generate the tool configuration now."""
# 
# 
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from scripts.async_llm import LLMsConfig
# from scripts.async_llm import create_llm_instance
# models_config = LLMsConfig.default()
# import asyncio
# import json
# async def run_loop(messages, tools=None):
#     choice = await llm.generate(messages, tools=tools)
#     return choice
# 
# llm_config = models_config.get('deepseek-ai/DeepSeek-V3.1-Terminus_v32')
# llm = create_llm_instance(llm_config)
# tool_map = {
#     "fill_missing_values": fill_missing_values,
#     "remove_columns_with_missing_data": remove_columns_with_missing_data,
#     "detect_and_handle_outliers_zscore": detect_and_handle_outliers_zscore,
#     "detect_and_handle_outliers_iqr": detect_and_handle_outliers_iqr,
#     "remove_duplicates": remove_duplicates,
#     "convert_data_types": convert_data_types,
#     "format_datetime": format_datetime,
#     "data_cleaning_tool_creation": data_cleaning_tool_creation
# }
# 
# add_tools = []
# 
# import ast
# import re
# from pathlib import Path
# from heapq import nlargest
# import json5
# 
# def extract_tool_details(source_code):
#     """
#     从代码中提取 import 语句和函数定义，并将它们组合在一起。
#     """
#     try:
#         tree = ast.parse(source_code)
#         lines = source_code.splitlines()
#         
#         import_statements = []
#         function_nodes = []
# 
#         # 第一次遍历：识别所有的 import 和 函数定义
#         for node in tree.body:
#             # 处理 import os 或 from math import ...
#             if isinstance(node, (ast.Import, ast.ImportFrom)):
#                 import_code = "\n".join(lines[node.lineno - 1 : node.end_lineno])
#                 import_statements.append(import_code)
#             
#             # 处理函数定义
#             elif isinstance(node, ast.FunctionDef):
#                 function_nodes.append(node)
# 
#         # 合并所有的 import 语句为一个字符串
#         full_imports = "\n".join(import_statements)
#         
#         results = []
#         # 第二次遍历：为每个函数生成包含 import 的完整代码块
#         for node in function_nodes:
#             func_name = node.name
#             func_body_code = "\n".join(lines[node.lineno - 1 : node.end_lineno])
#             
#             # 组合：Import 部分 + 函数部分
#             # 这样提取出的 tool_function 可以直接独立运行
#             combined_code = f"{full_imports}\n\n{func_body_code}" if full_imports else func_body_code
#             
#             results.append({
#                 "tool_name": func_name,
#                 "tool_function": combined_code
#             })
#         
#         return results
#     except Exception as e:
#         return {"error": str(e)}
# 
# 
# def validate_tool_config(tool_config: dict) -> None:
#     """
#     Validate whether the given tool_config is a valid function-style tool configuration.
# 
#     Raises:
#         ValueError: if the tool_config is invalid.
#     """
# 
#     # ---------- Top-level checks ----------
#     if not isinstance(tool_config, dict):
#         raise ValueError("Tool config must be a dictionary.")
# 
#     if tool_config.get("type") != "function":
#         raise ValueError("Tool config must have type == 'function'.")
# 
#     if "function" not in tool_config:
#         raise ValueError("Tool config must contain a 'function' field.")
# 
#     function_block = tool_config["function"]
#     if not isinstance(function_block, dict):
#         raise ValueError("'function' must be a dictionary.")
# 
#     # ---------- Function metadata ----------
#     name = function_block.get("name")
#     if not isinstance(name, str) or not name:
#         raise ValueError("Function name must be a non-empty string.")
# 
#     description = function_block.get("description")
#     if not isinstance(description, str) or not description:
#         raise ValueError("Function description must be a non-empty string.")
# 
#     # ---------- Parameters block ----------
#     parameters = function_block.get("parameters")
#     if not isinstance(parameters, dict):
#         raise ValueError("'parameters' must be a dictionary.")
# 
#     if parameters.get("type") != "object":
#         raise ValueError("'parameters.type' must be 'object'.")
# 
#     required = parameters.get("required")
#     properties = parameters.get("properties")
# 
#     if not isinstance(required, list):
#         raise ValueError("'parameters.required' must be a list.")
# 
#     if not isinstance(properties, dict):
#         raise ValueError("'parameters.properties' must be a dictionary.")
# 
#     # ---------- Properties validation ----------
#     for param_name, param_spec in properties.items():
#         if not isinstance(param_name, str) or not param_name:
#             raise ValueError("Parameter names must be non-empty strings.")
# 
#         if not isinstance(param_spec, dict):
#             raise ValueError(f"Specification for parameter '{param_name}' must be a dictionary.")
# 
#         param_type = param_spec.get("type")
#         if not isinstance(param_type, str) or not param_type:
#             raise ValueError(f"Parameter '{param_name}' must have a valid 'type' field.")
# 
#         param_desc = param_spec.get("description")
#         if not isinstance(param_desc, str) or not param_desc:
#             raise ValueError(f"Parameter '{param_name}' must have a non-empty 'description'.")
# 
#     # ---------- Required parameters consistency ----------
#     for req_param in required:
#         if req_param not in properties:
#             raise ValueError(
#                 f"Required parameter '{req_param}' is not defined in properties."
#             )
# 
#     # ---------- Passed all checks ----------
#     return None
# 
# 
# def execute_tool(code, tool_name, arguments, work_dir):
#     current_dir = Path.cwd()
#     try:
#         workdir_path = Path(work_dir).resolve()
#         os.chdir(workdir_path)
# 
#         exec_context = {}
#         exec(code, exec_context, exec_context)
#         my_tool_func = exec_context[tool_name]
#         output = my_tool_func(arguments)
#         return f"Data cleaning task completed successfully. The output is as follows: \n{output}"
#     except:
#         return f"Data cleaning task failed. Try again."
#     finally:
#         os.chdir(current_dir)
# 
# 
# def update_tools(code, domain):
#     # add tools
#     if os.path.exists(f"utils/created_tools/{domain}_tools.json"):    
#         with open(f"utils/created_tools/{domain}_tools.json", "r", encoding='utf-8') as f:
#             tools = json.load(f)
#     else:
#         tools = {}
#     if os.path.exists(f"utils/created_tools/{domain}_tool_count.json"):    
#         with open(f"utils/created_tools/{domain}_tool_count.json", "r", encoding='utf-8') as f:
#             tool_count = json.load(f)
#     else:
#         tool_count = {}
#     results = extract_tool_details(code)
#     if type(results) == dict:
#         return
#     else:
#         for result in results:
#             messages = [{"role": "user", "content": Tool_Configuration_Extraction_PROMPT.format(code=result["tool_function"])}]
#             choice = asyncio.run(run_loop(messages))
#             content = choice.message.content
#             match = re.search(r"```json(.*?)```", content, re.DOTALL)
#             if match:
#                 tool_config = match.group(1).strip()
#             else:
#                 tool_config = content.strip()
#             tool_config = json5.loads(tool_config)
#             try:
#                 validate_tool_config(tool_config)
#             except ValueError as e:
#                 print(f"Invalid tool config: {e}")
#                 return
# 
#             tool_name = result["tool_name"]
#             if tool_name in tools:
#                 tool_count[tool_name] += 1
#             else:
#                 tool_count[tool_name] = 1
#             # tool may not the same, update in the future
#             tools[tool_name] = {"code": result["tool_function"], "tool_config": tool_config}
# 
#             top10 = [k for k, _ in nlargest(10, tool_count.items(), key=lambda x: x[1]) if tool_count[k] >= 3]
#             add_tools = [tools[tool_name]['tool_config'] for tool_name in top10]
#         with open(f"utils/created_tools/{domain}_tools.json", "w", encoding='utf-8') as f:
#             json.dump(tools, f, indent=4, ensure_ascii=False)
#         with open(f"utils/created_tools/{domain}_tool_count.json", "w", encoding='utf-8') as f:
#             json.dump(tool_count, f, indent=4, ensure_ascii=False)
#     
#     return add_tools

# work_dir = './test'
# count = 1
# task = 'create a new tool to clean the dataset'
# prompt = DATA_CLEANING_PROMPT.format(dataset_file='synthetic_dataset.csv', task=task, saved_dataset_file='dataset_data_processed_{}.csv'.format(count))
# messages = [{"role": "system", "content": DATA_CLEARNER_SYS_PROMPT},
#             {"role": "user", "content": prompt}]
# tools = data_cleaning_tools + add_tools
# choice = asyncio.run(run_loop(messages, data_cleaning_tools))
# finish_reason = choice.finish_reason
# if finish_reason == "tool_calls":
#     messages.append(dict(choice.message))
#     for tool_call in choice.message.tool_calls:
#         tool_call_name = tool_call['function']['name']
#         try:
#             tool_call_arguments = json.loads(tool_call['function']['arguments'])
#             if tool_call_name in tool_map:
#                 tool_function = tool_map[tool_call_name]
#                 tool_result = tool_function(**tool_call_arguments, work_dir=work_dir, index=count)
#             else:
#                 with open(f"utils/created_tools/data_clean_tools.json", "r", encoding='utf-8') as f:
#                     created_tools = json.load(f)
#                 tool_code = created_tools[tool_call_name]['code']
#                 tool_result = execute_tool(tool_code, tool_call_name, tool_call_arguments, work_dir=work_dir)
#         except Exception as e:
#             tool_result = f"Tool call failed due to {str(e)}"
#         if tool_call_name == "data_cleaning_tool_creation" and tool_result.startswith('Data cleaning task completed successfully.'):
#             try:
#                 add_tools = update_tools(tool_call_arguments['code'], 'data_clean')
#             except Exception as e:
#                 print("Update tool failed due to: ", e)
#         print("tool_result:", tool_result)
# else:
#     content = choice.message.content
#     match = re.search(r"```python(.*?)```", content, re.DOTALL)
#     if match:
#         code = match.group(1).strip()
#         with open(os.path.join(work_dir, f"data_cleaning_{count}.py"), "w+", encoding="utf-8") as f:
#             f.write(code)
#         success, output = execute_script(f"data_cleaning_{count}.py", work_dir)
#     else:
#         result = content.strip()
#     print(1)
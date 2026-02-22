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
import warnings
import numpy as np
import torch
import sys
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

async def fill_missing_values(dataset_path: str, columns: Union[str, List[str]], method: str = 'auto', fill_value: Any = None, work_dir=None) -> str:
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
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + f"_cleaned." + dataset_path.split(".")[-1]
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
    

async def remove_columns_with_missing_data(dataset_path: str, columns: Union[str, List[str]] = None, thresh: float = 0.5, work_dir=None) -> str:
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
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + f"_cleaned." + dataset_path.split(".")[-1]
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
        

async def detect_and_handle_outliers_zscore(dataset_path: str, 
                                            columns: Union[str, List[str]], 
                                            threshold: float = 3.0, 
                                            method: str = 'clip',
                                            work_dir=None) -> str:
    """
    Detect and handle outliers in specified columns using either the Z-score or IQR method.

    Args:
        dataset_path (str): The path to the dataset file.
        columns (str or List[str]): The name(s) of the column(s) to check for outliers.
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
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + f"_cleaned." + dataset_path.split(".")[-1]
        saved_dataset_file = saved_dataset_path.split("/")[-1]

        # Save the cleaned data to the new file
        if file_extension == 'csv':
            data.to_csv(saved_dataset_path, index=False)
        elif file_extension == 'xlsx':
            data.to_excel(saved_dataset_path, index=False)
        elif file_extension == 'json':
            data.to_json(saved_dataset_path, orient='records', lines=True)

        return f"Data cleaning task completed successfully. Outliers were handled using Z-score method with '{method}' handling in columns {columns}. The cleaned data is saved in '{saved_dataset_file}'."
    
    except Exception as e:
        return f"Data cleaning task failed. The reason is: Error occurred while handling outliers: {e}"


async def detect_and_handle_outliers_iqr(dataset_path: str, 
                                         columns: Union[str, List[str]], 
                                         factor: float = 1.5, 
                                         method: str = 'clip',
                                         work_dir=None) -> str:
    """
    Detect and handle outliers in specified columns using the IQR method.

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
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + f"_cleaned." + dataset_path.split(".")[-1]
        saved_dataset_file = saved_dataset_path.split("/")[-1]

        # Save the cleaned data to the new file
        if file_extension == 'csv':
            data.to_csv(saved_dataset_path, index=False)
        elif file_extension == 'xlsx':
            data.to_excel(saved_dataset_path, index=False)
        elif file_extension == 'json':
            data.to_json(saved_dataset_path, orient='records', lines=True)

        return f"Data cleaning task completed successfully. Outliers were handled using IQR method with '{method}' handling in columns {columns}. The cleaned data is saved in '{saved_dataset_file}'."
    
    except Exception as e:
        return f"Data cleaning task failed. The reason is: Error occurred while handling outliers: {e}"
    

async def remove_duplicates(dataset_path: str, columns: Union[str, List[str]] = None, keep: str = 'first', work_dir: str = None) -> str:
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
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + f"_cleaned." + dataset_path.split(".")[-1]
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


async def convert_data_types(dataset_path: str, columns: Union[str, List[str]], target_type: str, work_dir: str = None) -> str:
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
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + f"_cleaned." + dataset_path.split(".")[-1]
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


async def format_datetime(dataset_path: str, columns: Union[str, List[str]], format: str = '%Y-%m-%d %H:%M:%S', errors: str = 'coerce', work_dir: str = None) -> str:
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
        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + f"_cleaned." + dataset_path.split(".")[-1]
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


async def data_cleaning_tool_creation(code, script_file_path, dataset_path, work_dir):
    script_file_path = os.path.normpath(script_file_path)
    dataset_path = os.path.normpath(dataset_path)
    saved_dataset_file = ".".join(dataset_path.split(".")[:-1]) + "_cleaned." + dataset_path.split(".")[-1]
    try:
        await write_program(code, os.path.join(work_dir, script_file_path))
        success, output = await execute_script(script_file_path, work_dir)
    except Exception as e:
        return f"Data cleaning task failed. The reason is: Error occurred while executing the script: {str(e)}"
    if success:
        if output is None or output == '':
            return f"Data cleaning task completed successfully. The relevant code is saved in '{script_file_path}'. The modified data is saved in '{saved_dataset_file}'."
        return f"Data cleaning task completed successfully. The relevant code is saved in '{script_file_path}'. The modified data is saved in '{saved_dataset_file}'. The output is as follows: \n{output}"
    else:
        return f"Data cleaning task failed. The relevant code is as follows: \n\n```python\n{code}\n```\n\nThe failed reason is: {output}"


class FillMissingValuesTool(BaseTool):
    """Tool for filling missing values in specified columns of a dataset."""

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

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, AgentLoopOutput]:
        missing_list = [k for k in self.required_keys if parameters.get(k) is None]
        if missing_list:
            return ToolResponse(text=f"Fill missing values task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        columns = parameters.get("columns")
        method = parameters.get("method", 'auto')
        fill_value = parameters.get("fill_value", None)
        work_dir = parameters.get("work_dir")

        result_message = await fill_missing_values(dataset_path, columns, method, fill_value, work_dir)
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class RemoveColumnsWithMissingDataTool(BaseTool):
    """Tool for removing columns with missing data based on a threshold."""

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

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, AgentLoopOutput]:
        missing_list = [k for k in self.required_keys if parameters.get(k) is None]
        if missing_list:
            return ToolResponse(text=f"Remove columns task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        columns = parameters.get("columns", None)
        thresh = parameters.get("thresh", 0.5)
        work_dir = parameters.get("work_dir")

        result_message = await remove_columns_with_missing_data(dataset_path, columns, thresh, work_dir)
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

    
class DetectAndHandleOutliersZScoreTool(BaseTool):
    """Tool for detecting and handling outliers in specified columns using zscore."""

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
            return ToolResponse(text=f"Outlier detection task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        columns = parameters.get("columns")
        threshold = parameters.get("threshold", 3.0)
        method = parameters.get("method", 'clip')
        work_dir = parameters.get("work_dir")

        result_message = await detect_and_handle_outliers_zscore(dataset_path, columns, threshold, method, work_dir)
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class DetectAndHandleOutliersIQRTool(BaseTool):
    """Tool for detecting and handling outliers in specified columns."""

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
            return ToolResponse(text=f"Outlier detection task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        columns = parameters.get("columns")
        factor = parameters.get("factor", 1.5)
        method = parameters.get("method", 'clip')
        work_dir = parameters.get("work_dir")

        result_message = await detect_and_handle_outliers_iqr(dataset_path, columns, factor, method, work_dir)
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class RemoveDuplicatesTool(BaseTool):
    """Tool for removing duplicate rows from a dataset."""

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
            return ToolResponse(text=f"Remove duplicates task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        columns = parameters.get("columns", None)
        keep = parameters.get("keep", 'first')
        work_dir = parameters.get("work_dir")

        result_message = await remove_duplicates(dataset_path, columns, keep, work_dir)
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class ConvertDataTypesTool(BaseTool):
    """Tool for converting the data types of specified columns."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["dataset_path", "columns", "target_type", "work_dir"]

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
            return ToolResponse(text=f"Data type conversion task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        columns = parameters.get("columns")
        target_type = parameters.get("target_type")
        work_dir = parameters.get("work_dir")

        result_message = await convert_data_types(dataset_path, columns, target_type, work_dir)
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class FormatDatetimeTool(BaseTool):
    """Tool for formatting datetime columns in a dataset."""

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
            return ToolResponse(text=f"Datetime formatting task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        columns = parameters.get("columns")
        format = parameters.get("format", '%Y-%m-%d %H:%M:%S')
        errors = parameters.get("errors", "coerce")
        work_dir = parameters.get("work_dir")

        result_message = await format_datetime(dataset_path, columns, format, errors, work_dir)
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class DataCleaningToolCreationTool(BaseTool):
    """Tool for running data cleaning tasks based on a provided script."""

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
            return ToolResponse(text=f"Data cleaning tool creation task failed. The following parameters are missing: {', '.join(missing_list)}")

        code = parameters.get("code")
        script_file_name = parameters.get("script_file_name")
        dataset_path = parameters.get("dataset_path")
        work_dir = parameters.get("work_dir")

        result_message = await data_cleaning_tool_creation(code, script_file_name, dataset_path, work_dir)
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
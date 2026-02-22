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
import matplotlib.pyplot as plt
import seaborn as sns
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


async def plot_line(dataset_path: str,
                    columns: list,
                    save_filename: str,
                    title: str = None,
                    xlabel: str = None,
                    ylabel: str = None,
                    legend_title: str = None,
                    xtick_labels: list = None,
                    ytick_labels: list = None,
                    color: str = 'blue',
                    figsize: tuple = (10, 6),
                    work_dir: str = None) -> str:
    """
    Generate line plot and save the output to a specified file.

    Args:
        dataset_path (str): Path to the dataset file.
        columns (list): List of columns to be used for the plot.
        save_filename (str): File name to save the plot.
        title (str, optional): Title for the plot.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis (default is 'Frequency').
        legend_title (str, optional): Title for the legend (optional).
        xtick_labels (list, optional): Custom labels for the x-axis ticks (optional).
        ytick_labels (list, optional): Custom labels for the y-axis ticks (optional).
        color (str, optional): Color to use for the plot.
        figsize (tuple, optional): Size of the plot.

    Returns:
        str: Result message, including success or failure information.
    """

    # Load the dataset
    try:
        # Get file extension
        dataset_path = os.path.normpath(dataset_path)
        save_filename = os.path.normpath(save_filename)
        dataset_path = os.path.join(work_dir, dataset_path)
        save_file_path = os.path.join(work_dir, save_filename)
        file_extension = dataset_path.split('.')[-1].lower()

        # Read the dataset based on the file extension
        if file_extension == 'csv':
            data = pd.read_csv(dataset_path)
        elif file_extension == 'xlsx':
            data = pd.read_excel(dataset_path)
        elif file_extension == 'json':
            data = pd.read_json(dataset_path)
        else:
            return f"Visualization task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        if len(columns) == 1:
            y_column = columns[0]
            x_column = 'index'
            x_column_value = range(len(data[y_column]))
        elif len(columns) == 2:
            x_column, y_column = columns
            x_column_value = data[x_column]
        else:
            return f"Visualization task failed. The reason is: For line plot, either one or two columns must be provided."

        # Plot the line plot
        plt.figure(figsize=figsize)
        plt.plot(x_column_value, data[y_column], color=color, label=f'{y_column} vs {x_column}')
        plt.title(title if title else f'Line plot between {x_column} and {y_column}')
        plt.xlabel(xlabel if xlabel else x_column)
        plt.ylabel(ylabel if ylabel else y_column)
        if xtick_labels:
            plt.xticks(xtick_labels)
        if ytick_labels:
            plt.yticks(ytick_labels)
        if legend_title:
            plt.legend(title=legend_title)
        plt.savefig(save_file_path)
        plt.close()
        return f"Visualization task completed successfully. Line plot saved as '{save_filename}'."
    except Exception as e:
        return f"Visualization task failed. The reason is: {e}"

async def plot_bar(dataset_path: str,
                   columns: list,
                   save_filename: str,
                   title: str = None,
                   xlabel: str = None,
                   ylabel: str = None,
                   legend_title: str = None,
                   xtick_labels: list = None,
                   ytick_labels: list = None,
                   color: str = 'blue',
                   figsize: tuple = (10, 6),
                   work_dir: str = None) -> str:
    """
    Generate bar plot and save the output to a specified file.

    Args:
        dataset_path (str): Path to the dataset file.
        columns (list): List of columns to be used for the plot.
        save_filename (str): File name to save the plot.
        title (str, optional): Title for the plot.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis (default is 'Frequency').
        legend_title (str, optional): Title for the legend (optional).
        xtick_labels (list, optional): Custom labels for the x-axis ticks (optional).
        ytick_labels (list, optional): Custom labels for the y-axis ticks (optional).
        color (str, optional): Color to use for the plot.
        figsize (tuple, optional): Size of the plot.

    Returns:
        str: Result message, including success or failure information.
    """

    # Load the dataset
    try:
        # Get file extension
        dataset_path = os.path.normpath(dataset_path)
        save_filename = os.path.normpath(save_filename)
        dataset_path = os.path.join(work_dir, dataset_path)
        save_file_path = os.path.join(work_dir, save_filename)
        file_extension = dataset_path.split('.')[-1].lower()

        # Read the dataset based on the file extension
        if file_extension == 'csv':
            data = pd.read_csv(dataset_path)
        elif file_extension == 'xlsx':
            data = pd.read_excel(dataset_path)
        elif file_extension == 'json':
            data = pd.read_json(dataset_path)
        else:
            return f"Visualization task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        if len(columns) == 1:
            y_column = columns[0]
            x_column = 'index'
            x_column_value = range(len(data[y_column]))
        elif len(columns) == 2:
            x_column, y_column = columns
            x_column_value = data[x_column]
        else:
            return f"Visualization task failed. The reason is: For bar plot, either one or two columns must be provided."

        # Plot the bar plot
        plt.figure(figsize=figsize)
        plt.bar(x_column_value, data[y_column], color=color, label=f'{y_column} vs {x_column}')
        plt.title(title if title else f'Bar plot between {x_column} and {y_column}')
        plt.xlabel(xlabel if xlabel else x_column)
        plt.ylabel(ylabel if ylabel else y_column)
        if xtick_labels:
            plt.xticks(xtick_labels)
        if ytick_labels:
            plt.yticks(ytick_labels)
        if legend_title:
            plt.legend(title=legend_title)
        plt.savefig(save_file_path)
        plt.close()
        return f"Visualization task completed successfully. Bar plot saved as '{save_filename}'."
    except Exception as e:
        return f"Visualization task failed. The reason is: {e}"
    
async def plot_histogram(dataset_path: str,
                         column: str,
                         save_filename: str,
                         title: str = None,
                         xlabel: str = None,
                         ylabel: str = None,
                         xtick_labels: list = None,
                         ytick_labels: list = None,
                         bins: int = 10,
                         color: str = 'blue',
                         figsize: tuple = (10, 6),
                         work_dir: str = None) -> str:
    """
    Generate histogram plot type and save the output to a specified file.

    Args:
        dataset_path (str): Path to the dataset file.
        column (str): Name of column to be used for the plot.
        save_filename (str): File name to save the plot.
        title (str, optional): Title for the plot.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis (default is 'Frequency').
        xtick_labels (list, optional): Custom labels for the x-axis ticks (optional).
        ytick_labels (list, optional): Custom labels for the y-axis ticks (optional).
        bins (int, optional): Number of bins to use for histogram.
        color (str, optional): Color to use for the plot.
        figsize (tuple, optional): Size of the plot.

    Returns:
        str: Result message, including success or failure information.
    """

    # Load the dataset
    try:
        # Get file extension
        dataset_path = os.path.normpath(dataset_path)
        save_filename = os.path.normpath(save_filename)
        dataset_path = os.path.join(work_dir, dataset_path)
        save_file_path = os.path.join(work_dir, save_filename)
        file_extension = dataset_path.split('.')[-1].lower()

        # Read the dataset based on the file extension
        if file_extension == 'csv':
            data = pd.read_csv(dataset_path)
        elif file_extension == 'xlsx':
            data = pd.read_excel(dataset_path)
        elif file_extension == 'json':
            data = pd.read_json(dataset_path)
        else:
            return f"Visualization task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        # Plot the histogram
        plt.figure(figsize=figsize)
        plt.hist(data[column], bins=bins, color=color)
        plt.title(title if title else f'Histogram of {column}')
        plt.xlabel(xlabel if xlabel else column)
        plt.ylabel(ylabel)
        if xtick_labels:
            plt.xticks(xtick_labels)
        if ytick_labels:
            plt.yticks(ytick_labels)
        plt.savefig(save_file_path)
        plt.close()
        return f"Visualization task completed successfully. Histogram saved as '{save_filename}'."
    
    except Exception as e:
        return f"Visualization task failed. The reason is: {e}"
    
async def plot_boxplot(dataset_path: str,
                       column: str,
                       save_filename: str,
                       title: str = None,
                       xtick_labels: list = None,
                       ytick_labels: list = None,
                       group_by: str = None,
                       color: str = 'blue',
                       figsize: tuple = (10, 6),
                       work_dir: str = None) -> str:
    """
    Generate boxplot and save the output to a specified file.

    Args:
        dataset_path (str): Path to the dataset file.
        column (str): Name of column to be used for the plot.
        save_filename (str): File name to save the plot.
        title (str, optional): Title for the plot.
        xtick_labels (list, optional): Custom labels for the x-axis ticks (optional).
        ytick_labels (list, optional): Custom labels for the y-axis ticks (optional).
        group_by (str, optional): Column to group the data by for boxplot.
        color (str, optional): Color to use for the plot.
        figsize (tuple, optional): Size of the plot.

    Returns:
        str: Result message, including success or failure information.
    """

    # Load the dataset
    try:
        # Get file extension
        dataset_path = os.path.normpath(dataset_path)
        save_filename = os.path.normpath(save_filename)
        dataset_path = os.path.join(work_dir, dataset_path)
        save_file_path = os.path.join(work_dir, save_filename)
        file_extension = dataset_path.split('.')[-1].lower()

        # Read the dataset based on the file extension
        if file_extension == 'csv':
            data = pd.read_csv(dataset_path)
        elif file_extension == 'xlsx':
            data = pd.read_excel(dataset_path)
        elif file_extension == 'json':
            data = pd.read_json(dataset_path)
        else:
            return f"Visualization task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        # Plot the boxplot
        plt.figure(figsize=figsize)
        if group_by:
            sns.boxplot(x=group_by, y=column, data=data, palette=color)
        else:
            sns.boxplot(y=data[column], color=color)
        plt.title(title if title else f'Boxplot of {column}')
        if xtick_labels:
            plt.xticks(xtick_labels)
        if ytick_labels:
            plt.yticks(ytick_labels)
        plt.savefig(save_file_path)
        plt.close()
        return f"Visualization task completed successfully. Boxplot saved as '{save_filename}'."
    except Exception as e:
        return f"Visualization task failed. The reason is: {e}"
    
async def plot_scatter(dataset_path: str,
                 columns: list,
                 save_filename: str,
                 title: str = None,
                 xlabel: str = None,
                 ylabel: str = None,
                 legend_title: str = None,
                 xtick_labels: list = None,
                 ytick_labels: list = None,
                 color: str = 'blue',
                 figsize: tuple = (10, 6),
                 hue: str = None,
                 size: str = None,
                 work_dir: str = None) -> str:
    """
    Generate scatter plot and save the output to a specified file.

    Args:
        dataset_path (str): Path to the dataset file.
        columns (list): List of columns to be used for the plot.
        save_filename (str): File name to save the plot.
        title (str, optional): Title for the plot.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis (default is 'Frequency').
        legend_title (str, optional): Title for the legend (optional).
        xtick_labels (list, optional): Custom labels for the x-axis ticks (optional).
        ytick_labels (list, optional): Custom labels for the y-axis ticks (optional).
        color (str, optional): Color to use for the plot.
        figsize (tuple, optional): Size of the plot.
        hue (str, optional): Column to color the points by.
        size (str, optional): Column to determine the size of scatter points.

    Returns:
        str: Result message, including success or failure information.
    """

    # Load the dataset
    try:
        # Get file extension
        dataset_path = os.path.normpath(dataset_path)
        save_filename = os.path.normpath(save_filename)
        dataset_path = os.path.join(work_dir, dataset_path)
        save_file_path = os.path.join(work_dir, save_filename)
        file_extension = dataset_path.split('.')[-1].lower()

        # Read the dataset based on the file extension
        if file_extension == 'csv':
            data = pd.read_csv(dataset_path)
        elif file_extension == 'xlsx':
            data = pd.read_excel(dataset_path)
        elif file_extension == 'json':
            data = pd.read_json(dataset_path)
        else:
            return f"Visualization task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        if len(columns) == 1:
            y_column = columns[0]
            x_column = 'index'
            x_column_value = range(len(data[y_column]))
        elif len(columns) == 2:
            x_column, y_column = columns
            x_column_value = data[x_column]
        else:
            return f"Visualization task failed. The reason is: For scatter plot, either one or two columns must be provided."

        # Plot the scatter plot
        plt.figure(figsize=figsize)
        plt.scatter(x_column_value, data[y_column], c=data[hue] if hue else color, s=data[size] if size else 50, alpha=0.7)
        plt.title(title if title else f'Scatter plot between {x_column} and {y_column}')
        plt.xlabel(xlabel if xlabel else x_column)
        plt.ylabel(ylabel if ylabel else y_column)
        if xtick_labels:
            plt.xticks(xtick_labels)
        if ytick_labels:
            plt.yticks(ytick_labels)
        if legend_title:
            plt.legend(title=legend_title)
        plt.savefig(save_file_path)
        plt.close()
        return f"Visualization task completed successfully. Scatter plot saved as '{save_filename}'."
    
    except Exception as e:
        return f"Visualization task failed. The reason is: {e}"
    

async def plot_heatmap(dataset_path: str,
                 columns: list,
                 save_filename: str,
                 title: str = None,
                 legend_title: str = None,
                 xtick_labels: list = None,
                 ytick_labels: list = None,
                 figsize: tuple = (10, 6),
                 annot: bool = True,
                 cmap: str = 'coolwarm',
                 work_dir: str = None) -> str:
    """
    Generate heatmap plot and save the output to a specified file.

    Args:
        dataset_path (str): Path to the dataset file.
        columns (list): List of columns to be used for the plot.
        save_filename (str): File name to save the plot.
        title (str, optional): Title for the plot.
        legend_title (str, optional): Title for the legend (optional).
        xtick_labels (list, optional): Custom labels for the x-axis ticks (optional).
        ytick_labels (list, optional): Custom labels for the y-axis ticks (optional).
        figsize (tuple, optional): Size of the plot.
        annot (bool, optional): Whether to annotate the heatmap with correlation values.
        cmap (str, optional): Colormap to use for the heatmap.

    Returns:
        str: Result message, including success or failure information.
    """

    # Load the dataset
    try:
        # Get file extension
        dataset_path = os.path.normpath(dataset_path)
        save_filename = os.path.normpath(save_filename)
        dataset_path = os.path.join(work_dir, dataset_path)
        save_file_path = os.path.join(work_dir, save_filename)
        file_extension = dataset_path.split('.')[-1].lower()

        # Read the dataset based on the file extension
        if file_extension == 'csv':
            data = pd.read_csv(dataset_path)
        elif file_extension == 'xlsx':
            data = pd.read_excel(dataset_path)
        elif file_extension == 'json':
            data = pd.read_json(dataset_path)
        else:
            return f"Visualization task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        if len(columns) < 2:
            return f"Visualization task failed. The reason is: For heatmap, at least two columns are required."

        # Calculate correlation matrix
        corr_matrix = data[columns].corr()

        # Plot the heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=annot, cmap=cmap, fmt='.2f', linewidths=0.5)
        plt.title(title if title else 'Correlation Heatmap')
        if xtick_labels:
            plt.xticks(xtick_labels)
        if ytick_labels:
            plt.yticks(ytick_labels)
        if legend_title:
            plt.legend(title=legend_title)
        plt.savefig(save_file_path)
        plt.close()
        return f"Visualization task completed successfully. Heatmap saved as '{save_filename}'."
    
    except Exception as e:
        return f"Visualization task failed. The reason is: {e}"
    

async def plot_pie(dataset_path: str,
             column: str,
             save_filename: str,
             title: str = None,
             figsize: tuple = (10, 6),
             autopct: str = '%1.1f%%',
             work_dir: str = None) -> str:
    """
    Generate pie plot and save the output to a specified file.

    Args:
        dataset_path (str): Path to the dataset file.
        column (str): Name of column to be used for the plot.
        save_filename (str): File name to save the plot.
        title (str, optional): Title for the plot.
        figsize (tuple, optional): Size of the plot.
        autopct (str, optional): String format for displaying percentages on pie chart.

    Returns:
        str: Result message, including success or failure information.
    """

    # Load the dataset
    try:
        # Get file extension
        dataset_path = os.path.normpath(dataset_path)
        save_filename = os.path.normpath(save_filename)
        dataset_path = os.path.join(work_dir, dataset_path)
        save_file_path = os.path.join(work_dir, save_filename)
        file_extension = dataset_path.split('.')[-1].lower()

        # Read the dataset based on the file extension
        if file_extension == 'csv':
            data = pd.read_csv(dataset_path)
        elif file_extension == 'xlsx':
            data = pd.read_excel(dataset_path)
        elif file_extension == 'json':
            data = pd.read_json(dataset_path)
        else:
            return f"Visualization task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        # Get category counts
        category_counts = data[column].value_counts()

        # Plot the pie chart
        plt.figure(figsize=figsize)
        plt.pie(category_counts, autopct=autopct)
        plt.title(title if title else f'Pie Chart of {column}')
        plt.savefig(save_file_path)
        plt.close()
        return f"Visualization task completed successfully. Pie chart saved as '{save_filename}'."
    
    except Exception as e:
        return f"Visualization task failed. The reason is: {e}"


async def plot_pairplot(dataset_path: str,
                  columns: list,
                  save_filename: str,
                  title: str = None,
                  hue: str = None,
                  palette: str = 'muted',
                  work_dir: str = None) -> str:
    """
    Generate pairplot and save the output to a specified file.

    Args:
        dataset_path (str): Path to the dataset file.
        columns (list): List of columns to be used for the plot.
        save_filename (str): File name to save the plot.
        title (str, optional): Title for the plot.
        hue (str, optional): Column to color the points by.
        palette (str, optional): Color palette to use for pairplot.

    Returns:
        str: Result message, including success or failure information.
    """

    # Load the dataset
    try:
        # Get file extension
        dataset_path = os.path.normpath(dataset_path)
        save_filename = os.path.normpath(save_filename)
        dataset_path = os.path.join(work_dir, dataset_path)
        save_file_path = os.path.join(work_dir, save_filename)
        file_extension = dataset_path.split('.')[-1].lower()

        # Read the dataset based on the file extension
        if file_extension == 'csv':
            data = pd.read_csv(dataset_path)
        elif file_extension == 'xlsx':
            data = pd.read_excel(dataset_path)
        elif file_extension == 'json':
            data = pd.read_json(dataset_path)
        else:
            return f"Visualization task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        if len(columns) < 2:
            return f"Visualization task failed. The reason is: For pairplot, at least two columns are required."

        # Plot the pairplot
        if hue:
            sns.pairplot(data[columns], hue=hue, palette=palette)
        else:
            sns.pairplot(data[columns])
        plt.title(title if title else 'Pairplot of Features')
        plt.savefig(save_file_path)
        plt.close()
        return f"Visualization task completed successfully. Pairplot saved as '{save_filename}'."

    except Exception as e:
        return f"Visualization task failed. The reason is: {e}"
    

async def visualization_tool_creation(code, script_file_path, save_filename, work_dir):
    script_file_path = os.path.normpath(script_file_path)
    save_filename = os.path.normpath(save_filename)
    try:
        await write_program(code, os.path.join(work_dir, script_file_path))
        success, output = await execute_script(script_file_path, work_dir)
    except Exception as e:
        return f"Visualization task failed. The reason is: Error occurred while executing the script: {str(e)}"
    if success:
        if output is None or output == '':
            return f"Visualization task completed successfully. The relevant code is saved in '{script_file_path}'. The modified data is saved in '{save_filename}'."
        return f"Visualization task completed successfully. The relevant code is saved in '{script_file_path}'. The modified data is saved in '{save_filename}'. The output is as follows: \n{output}"
    else:
        return f"Visualization task failed. The relevant code is as follows: \n\n```python\n{code}\n```\n\nThe failed reason is: {output}"


class LinePlotTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["dataset_path", "columns", "save_filename", "work_dir"]

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
            return ToolResponse(text=f"Visualization task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        columns = parameters.get("columns")
        save_filename = parameters.get("save_filename")
        title = parameters.get("title", None)
        xlabel = parameters.get("xlabel", None)
        ylabel = parameters.get("ylabel", None)
        legend_title = parameters.get("legend_title", None)
        xtick_labels = parameters.get("xtick_labels", None)
        ytick_labels = parameters.get("ytick_labels", None)
        color = parameters.get("color", 'blue')
        figsize = parameters.get("figsize", (10, 6))
        work_dir = parameters.get("work_dir")

        result_message = await plot_line(
            dataset_path, columns, save_filename, title, xlabel, ylabel,
            legend_title, xtick_labels, ytick_labels, color, figsize, work_dir
        )
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class BarPlotTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["dataset_path", "columns", "save_filename", "work_dir"]

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
            return ToolResponse(text=f"Visualization task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        columns = parameters.get("columns")
        save_filename = parameters.get("save_filename")
        title = parameters.get("title", None)
        xlabel = parameters.get("xlabel", None)
        ylabel = parameters.get("ylabel", None)
        legend_title = parameters.get("legend_title", None)
        xtick_labels = parameters.get("xtick_labels", None)
        ytick_labels = parameters.get("ytick_labels", None)
        color = parameters.get("color", 'blue')
        figsize = parameters.get("figsize", (10, 6))
        work_dir = parameters.get("work_dir")

        result_message = await plot_bar(
            dataset_path, columns, save_filename, title, xlabel, ylabel,
            legend_title, xtick_labels, ytick_labels, color, figsize, work_dir
        )
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class HistogramPlotTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["dataset_path", "column", "save_filename", "work_dir"]

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
            return ToolResponse(text=f"Visualization task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        column = parameters.get("column")
        save_filename = parameters.get("save_filename")
        title = parameters.get("title", None)
        xlabel = parameters.get("xlabel", None)
        ylabel = parameters.get("ylabel", None)
        legend_title = parameters.get("legend_title", None)
        xtick_labels = parameters.get("xtick_labels", None)
        ytick_labels = parameters.get("ytick_labels", None)
        bins = parameters.get("bins", 10)
        color = parameters.get("color", 'blue')
        figsize = parameters.get("figsize", (10, 6))
        work_dir = parameters.get("work_dir")

        result_message = await plot_histogram(
            dataset_path, column, save_filename, title, xlabel, ylabel,
            legend_title, xtick_labels, ytick_labels, bins, color, figsize, work_dir
        )
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

class BoxPlotTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["dataset_path", "column", "save_filename", "work_dir"]

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
            return ToolResponse(text=f"Visualization task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        column = parameters.get("column")
        save_filename = parameters.get("save_filename")
        title = parameters.get("title", None)
        xtick_labels = parameters.get("xtick_labels", None)
        ytick_labels = parameters.get("ytick_labels", None)
        group_by = parameters.get("group_by", None)
        color = parameters.get("color", 'blue')
        figsize = parameters.get("figsize", (10, 6))
        work_dir = parameters.get("work_dir")

        result_message = await plot_boxplot(
            dataset_path, column, save_filename, title, xtick_labels, ytick_labels, group_by, color, figsize, work_dir
        )
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

class ScatterPlotTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["dataset_path", "columns", "save_filename", "work_dir"]

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
            return ToolResponse(text=f"Visualization task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        columns = parameters.get("columns")
        save_filename = parameters.get("save_filename")
        title = parameters.get("title", None)
        xlabel = parameters.get("xlabel", None)
        ylabel = parameters.get("ylabel", None)
        legend_title = parameters.get("legend_title", None)
        xtick_labels = parameters.get("xtick_labels", None)
        ytick_labels = parameters.get("ytick_labels", None)
        color = parameters.get("color", 'blue')
        figsize = parameters.get("figsize", (10, 6))
        hue = parameters.get("hue", None)
        size = parameters.get("size", None)
        work_dir = parameters.get("work_dir")

        result_message = await plot_scatter(
            dataset_path, columns, save_filename, title, xlabel, ylabel,
            legend_title, xtick_labels, ytick_labels, color, figsize,
            hue, size, work_dir
        )
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

class HeatmapPlotTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["dataset_path", "columns", "save_filename", "work_dir"]

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
            return ToolResponse(text=f"Visualization task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        columns = parameters.get("columns")
        save_filename = parameters.get("save_filename")
        title = parameters.get("title", None)
        legend_title = parameters.get("legend_title", None)
        xtick_labels = parameters.get("xtick_labels", None)
        ytick_labels = parameters.get("ytick_labels", None)
        figsize = parameters.get("figsize", (10, 6))
        annot = parameters.get("annot", True)
        cmap = parameters.get("cmap", 'coolwarm')
        work_dir = parameters.get("work_dir")

        result_message = await plot_heatmap(
            dataset_path, columns, save_filename, title, 
            legend_title, xtick_labels, ytick_labels, figsize, annot, cmap, work_dir
        )
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

class PiePlotTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["dataset_path", "column", "save_filename", "work_dir"]

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
            return ToolResponse(text=f"Visualization task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        column = parameters.get("column")
        save_filename = parameters.get("save_filename")
        title = parameters.get("title", None)
        figsize = parameters.get("figsize", (10, 6))
        autopct = parameters.get("autopct", '%1.1f%%')
        work_dir = parameters.get("work_dir")

        result_message = await plot_pie(
            dataset_path, column, save_filename, title, figsize, autopct, work_dir
        )
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

class PairPlotTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["dataset_path", "columns", "save_filename", "work_dir"]

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
            return ToolResponse(text=f"Visualization task failed. The following parameters are missing: {', '.join(missing_list)}")

        dataset_path = parameters.get("dataset_path")
        columns = parameters.get("columns")
        save_filename = parameters.get("save_filename")
        title = parameters.get("title", None)
        hue = parameters.get("hue", None)
        palette = parameters.get("palette", 'muted')
        work_dir = parameters.get("work_dir")

        result_message = await plot_pairplot(
            dataset_path, columns, save_filename, title, hue, palette, work_dir
        )
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

class VisualizationToolCreationTool(BaseTool):
    """Tool for creating and executing a visualization tool script."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["code", "script_file_name", "save_filename", "work_dir"]

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
            return ToolResponse(text=f"Visualization task failed. The following parameters are missing: {', '.join(missing_list)}")

        code = parameters.get("code")
        script_file_name = parameters.get("script_file_name")
        save_filename = parameters.get("save_filename")
        work_dir = parameters.get("work_dir")

        result_message = await visualization_tool_creation(code, script_file_name, save_filename, work_dir)
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

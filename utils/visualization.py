import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.util import write_program, execute_script
import os


VISUALIZER_SYS_PROMPT = """You are a data science expert specializing in data visualization tasks. You have access to a set of tools that can help solve these tasks. When given a dataset path and a data visualization task description, your first step is to check if the provided tools can directly solve the task. If they can, use the appropriate tool to perform the data visualization. The tool will automatically save the visualization plot.

If the tools cannot directly solve the task, you should use the `visualization_tool_creation` tool to create a new tool to address the task based on the description provided.

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

VISUALIZATION_PROMPT = """
You are given a dataset located at {dataset_file}. Your task is to process the dataset according to the following requirements:

# VISUALIZATION TASK #
{task}

After visualization, save the visualization plot to {saved_plot_file}.
"""


def plot_line(dataset_path: str,
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
    Generate various visualizations based on the specified plot type and save the output to a specified file.

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

        # Check the number of columns and ensure they are valid for the selected plot type
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


def plot_bar(dataset_path: str,
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
    Generate a bar plot and save the output to a specified file.

    Args:
        dataset_path (str): Path to the dataset file.
        columns (list): List of columns to be used for the plot.
                        - One column: value counts bar plot
                        - Two columns: x-y bar plot
        save_filename (str): File name to save the plot.
        title (str, optional): Title for the plot.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        legend_title (str, optional): Title for the legend (optional).
        xtick_labels (list, optional): Custom labels for the x-axis ticks (optional).
        ytick_labels (list, optional): Custom labels for the y-axis ticks (optional).
        color (str, optional): Color to use for the bars.
        figsize (tuple, optional): Size of the plot.

    Returns:
        str: Result message, including success or failure information.
    """

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

        # Check the number of columns and ensure they are valid for the selected plot type
        if len(columns) == 1:
            y_column = columns[0]
            x_column = 'index'
            x_column_value = range(len(data[y_column]))
        elif len(columns) == 2:
            x_column, y_column = columns
            x_column_value = data[x_column]
        else:
            return f"Visualization task failed. The reason is: For bar plot, either one or two columns must be provided."

        # Plot the line plot
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
    

def plot_histogram(dataset_path: str,
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
    Generate various visualizations based on the specified plot type and save the output to a specified file.

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
        return f"Visualization task completed successfully. Histogram plot saved as '{save_filename}'."

    except Exception as e:
        return f"Visualization task failed. The reason is: {e}"


def plot_boxplot(dataset_path: str,
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
    Generate various visualizations based on the specified plot type and save the output to a specified file.

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
    

def plot_scatter(dataset_path: str,
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
    Generate various visualizations based on the specified plot type and save the output to a specified file.

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
    

def plot_heatmap(dataset_path: str,
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
    Generate various visualizations based on the specified plot type and save the output to a specified file.

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
    

def plot_pie(dataset_path: str,
             column: str,
             save_filename: str,
             title: str = None,
             figsize: tuple = (10, 6),
             autopct: str = '%1.1f%%',
             work_dir: str = None) -> str:
    """
    Generate various visualizations based on the specified plot type and save the output to a specified file.

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


def plot_pairplot(dataset_path: str,
                  columns: list,
                  save_filename: str,
                  title: str = None,
                  hue: str = None,
                  palette: str = 'muted',
                  work_dir: str = None) -> str:
    """
    Generate various visualizations based on the specified plot type and save the output to a specified file.

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


def visualization_tool_creation(code, script_file_name, save_filename, work_dir):
    script_file_name = os.path.normpath(script_file_name)
    save_filename = os.path.normpath(save_filename)
    write_program(code, os.path.join(work_dir, script_file_name))
    success, output = execute_script(script_file_name, work_dir)
    if success:
        if output is None or output == '':
            return f"Visualization task completed successfully. The relevant code is saved in '{script_file_name}'. The modified data is saved in '{save_filename}'."
        return f"Visualization task completed successfully. The relevant code is saved in '{script_file_name}'. The modified data is saved in '{save_filename}'. The output is as follows: \n{output}"
    else:
        return f"Visualization task failed. The relevant code is as follows: \n{code}\nThe reason is: {output}"


visualization_tools = [
    {
        "type": "function",
        "function": {
            "name": "plot_line",
            "description": "Generate line plot and save the output to the specified file.",
            "parameters": {
                "type": "object",
                "required": ["dataset_path", "columns", "save_filename"],
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    },
                    "columns": {
                        "type": "list",
                        "description": "List of columns to be used for the plot. The number of columns should be 1 or 2."
                    },
                    "save_filename": {
                        "type": "string",
                        "description": "The filename to save the plot to."
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for the plot. Default is None."
                    },
                    "xlabel": {
                        "type": "string",
                        "description": "Label for the x-axis. Default is None."
                    },
                    "ylabel": {
                        "type": "string",
                        "description": "Label for the y-axis. Default is None."
                    },
                    "legend_title": {
                        "type": "string",
                        "description": "Title for the legend. Default is None."
                    },
                    "xtick_labels": {
                        "type": "list",
                        "description": "Custom labels for the x-axis ticks. Default is None."
                    },
                    "ytick_labels": {
                        "type": "list",
                        "description": "Custom labels for the y-axis ticks. Default is None."
                    },
                    "color": {
                        "type": "string",
                        "description": "Color to use for the plot. Default is 'blue'."
                    },
                    "figsize": {
                        "type": "tuple",
                        "description": "Size of the plot as a tuple (width, height). Default is (10, 6)."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_bar",
            "description": "Generate bar plot and save the output to the specified file.",
            "parameters": {
                "type": "object",
                "required": ["dataset_path", "columns", "save_filename"],
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    },
                    "columns": {
                        "type": "list",
                        "description": "List of columns to be used for the plot. The number of columns should be 1 or 2."
                    },
                    "save_filename": {
                        "type": "string",
                        "description": "The filename to save the plot to."
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for the plot. Default is None."
                    },
                    "xlabel": {
                        "type": "string",
                        "description": "Label for the x-axis. Default is None."
                    },
                    "ylabel": {
                        "type": "string",
                        "description": "Label for the y-axis. Default is None."
                    },
                    "legend_title": {
                        "type": "string",
                        "description": "Title for the legend. Default is None."
                    },
                    "xtick_labels": {
                        "type": "list",
                        "description": "Custom labels for the x-axis ticks. Default is None."
                    },
                    "ytick_labels": {
                        "type": "list",
                        "description": "Custom labels for the y-axis ticks. Default is None."
                    },
                    "color": {
                        "type": "string",
                        "description": "Color to use for the plot. Default is 'blue'."
                    },
                    "figsize": {
                        "type": "tuple",
                        "description": "Size of the plot as a tuple (width, height). Default is (10, 6)."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_histogram",
            "description": "Generate histogram plot and save the output to the specified file.",
            "parameters": {
                "type": "object",
                "required": ["dataset_path", "column", "save_filename"],
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    },
                    "column": {
                        "type": "str",
                        "description": "The name of column to be used for the plot."
                    },
                    "save_filename": {
                        "type": "string",
                        "description": "The filename to save the plot to."
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for the plot. Default is None."
                    },
                    "xlabel": {
                        "type": "string",
                        "description": "Label for the x-axis. Default is None."
                    },
                    "ylabel": {
                        "type": "string",
                        "description": "Label for the y-axis. Default is None."
                    },
                    "legend_title": {
                        "type": "string",
                        "description": "Title for the legend. Default is None."
                    },
                    "xtick_labels": {
                        "type": "list",
                        "description": "Custom labels for the x-axis ticks. Default is None."
                    },
                    "ytick_labels": {
                        "type": "list",
                        "description": "Custom labels for the y-axis ticks. Default is None."
                    },
                    "bins": {
                        "type": "int",
                        "description": "Number of bins to use for histogram. Default is 10."
                    },
                    "color": {
                        "type": "string",
                        "description": "Color to use for the plot. Default is 'blue'."
                    },
                    "figsize": {
                        "type": "tuple",
                        "description": "Size of the plot as a tuple (width, height). Default is (10, 6)."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_boxplot",
            "description": "Generate box plot and save the output to the specified file.",
            "parameters": {
                "type": "object",
                "required": ["dataset_path", "column", "save_filename"],
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    },
                    "column": {
                        "type": "str",
                        "description": "The column to be used for the plot."
                    },
                    "save_filename": {
                        "type": "string",
                        "description": "The filename to save the plot to."
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for the plot. Default is None."
                    },
                    "xtick_labels": {
                        "type": "list",
                        "description": "Custom labels for the x-axis ticks. Default is None."
                    },
                    "ytick_labels": {
                        "type": "list",
                        "description": "Custom labels for the y-axis ticks. Default is None."
                    },
                    "group_by": {
                        "type": "string",
                        "description": "Column to group the data by. Default is None."
                    },
                    "color": {
                        "type": "string",
                        "description": "Color to use for the plot. Default is 'blue'."
                    },
                    "figsize": {
                        "type": "tuple",
                        "description": "Size of the plot as a tuple (width, height). Default is (10, 6)."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_scatter",
            "description": "Generate scatter plot and save the output to the specified file.",
            "parameters": {
                "type": "object",
                "required": ["dataset_path", "columns", "save_filename"],
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    },
                    "columns": {
                        "type": "list",
                        "description": "List of columns to be used for the plot. The number of columns should be 1 or 2."
                    },
                    "save_filename": {
                        "type": "string",
                        "description": "The filename to save the plot to."
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for the plot. Default is None."
                    },
                    "xlabel": {
                        "type": "string",
                        "description": "Label for the x-axis. Default is None."
                    },
                    "ylabel": {
                        "type": "string",
                        "description": "Label for the y-axis. Default is None."
                    },
                    "legend_title": {
                        "type": "string",
                        "description": "Title for the legend. Default is None."
                    },
                    "xtick_labels": {
                        "type": "list",
                        "description": "Custom labels for the x-axis ticks. Default is None."
                    },
                    "ytick_labels": {
                        "type": "list",
                        "description": "Custom labels for the y-axis ticks. Default is None."
                    },
                    "color": {
                        "type": "string",
                        "description": "Color to use for the plot. Default is 'blue'."
                    },
                    "figsize": {
                        "type": "tuple",
                        "description": "Size of the plot as a tuple (width, height). Default is (10, 6)."
                    },
                    "hue": {
                        "type": "string",
                        "description": "Column to color the points by. Default is None."
                    },
                    "size": {
                        "type": "string",
                        "description": "Column to determine the size of scatter points. Default is None."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_heatmap",
            "description": "Generate heatmap and save the output to the specified file.",
            "parameters": {
                "type": "object",
                "required": ["dataset_path", "columns", "save_filename"],
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    },
                    "columns": {
                        "type": "list",
                        "description": "List of columns to be used for the plot. The number of columns should more than 1."
                    },
                    "save_filename": {
                        "type": "string",
                        "description": "The filename to save the plot to."
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for the plot. Default is None."
                    },
                    "legend_title": {
                        "type": "string",
                        "description": "Title for the legend. Default is None."
                    },
                    "xtick_labels": {
                        "type": "list",
                        "description": "Custom labels for the x-axis ticks. Default is None."
                    },
                    "ytick_labels": {
                        "type": "list",
                        "description": "Custom labels for the y-axis ticks. Default is None."
                    },
                    "figsize": {
                        "type": "tuple",
                        "description": "Size of the plot as a tuple (width, height). Default is (10, 6)."
                    },
                    "annot": {
                        "type": "bool",
                        "description": "Whether to annotate the heatmap with correlation values. Default is True."
                    },
                    "cmap": {
                        "type": "string",
                        "description": "Colormap to use for the heatmap. Default is 'coolwarm'."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_pie",
            "description": "Generate pie plot and save the output to the specified file.",
            "parameters": {
                "type": "object",
                "required": ["dataset_path", "column", "save_filename"],
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    },
                    "column": {
                        "type": "str",
                        "description": "The name of column to be used for the plot."
                    },
                    "save_filename": {
                        "type": "string",
                        "description": "The filename to save the plot to."
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for the plot. Default is None."
                    },
                    "figsize": {
                        "type": "tuple",
                        "description": "Size of the plot as a tuple (width, height). Default is (10, 6)."
                    },
                    "autopct": {
                        "type": "string",
                        "description": "String format for displaying percentages on the pie chart. Default is '%1.1f%%'."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_pairplot",
            "description": "Generate pairplot and save the output to the specified file.",
            "parameters": {
                "type": "object",
                "required": ["dataset_path", "columns", "save_filename"],
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset file to be processed."
                    },
                    "columns": {
                        "type": "list",
                        "description": "List of columns to be used for the plot. The number of columns should more than 1."
                    },
                    "save_filename": {
                        "type": "string",
                        "description": "The filename to save the plot to."
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for the plot. Default is None."
                    },
                    "hue": {
                        "type": "string",
                        "description": "Column to color the points by. Default is None."
                    },
                    "palette": {
                        "type": "string",
                        "description": "Color palette to use for pairplot. Default is 'muted'."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "visualization_tool_creation",
            "description": "Create a **general-purpose, reusable visualization tool** when necessary to solve the current task. The generated tool should be **dataset-agnostic** and applicable to a wide range of visualization scenarios rather than being tailored to a specific dataset or variable. The tool will be saved as a Python script, executed immediately, and return the execution results, including any printed outputs or errors. The created tool is expected to be reusable in future visualization tasks within an autonomous data science pipeline.",
            "parameters": {
                "type": "object",
                "required": ["code", "script_file_name", "save_filename"],
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code of the newly created visualization tool. The tool must follow the strict format requirement and be general enough for reuse as: ```python\ndef tool_name(parameters):\n    # detail of the code\n\n# execute the tool\nif __name__ == '__main__':\n    parameters = {...}\n    tool_name(parameters)\n```"
                    },
                    "script_file_name": {
                        "type": "string",
                        "description": "The name of the Python script of the newly created tool will be saved and executed."
                    },
                    "save_filename": {
                        "type": "string",
                        "description": "The filename to save the plot to."
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
#     "plot_line": plot_line,
#     "plot_histogram": plot_histogram,
#     "plot_boxplot": plot_boxplot,
#     "plot_scatter": plot_scatter,
#     "plot_heatmap": plot_heatmap,
#     "plot_pie": plot_pie,
#     "plot_pairplot": plot_pairplot,
#     "visualization_tool_creation": visualization_tool_creation
# }
# 
# task = 'use the plot_pie tool to visualize the age column'
# prompt = VISUALIZATION_PROMPT.format(dataset_file='synthetic_dataset.csv', task=task, saved_plot_file='data_visualization.png')
# messages = [{"role": "system", "content": VISUALIZER_SYS_PROMPT},
#             {"role": "user", "content": prompt}]
# choice = asyncio.run(run_loop(messages, visualization_tools))
# finish_reason = choice.finish_reason
# if finish_reason == "tool_calls":
#     messages.append(dict(choice.message))
#     for tool_call in choice.message.tool_calls:
#         tool_call_name = tool_call['function']['name']
#         try:
#             tool_call_arguments = json.loads(tool_call['function']['arguments'])
#             tool_function = tool_map[tool_call_name]
#             tool_result = tool_function(**tool_call_arguments, work_dir='./test')
#         except Exception as e:
#             tool_result = f"Tool call failed due to {str(e)}"
#         print("tool_result:", tool_result)
# else:
#     print(1)
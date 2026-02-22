


tools = [
  {
    "type": "function",
    "function": {
      "name": "bash",
      "description": "Execute a bash command. The result of the executed bash command will be returned.",
      "parameters": {
        "type": "object",
        "required": ["code"],
        "properties": {
          "code": {
            "type": "string",
            "description": "Bash command to be executed. This should be a complete and self-contained shell command, including any necessary arguments or options."
          }
        }
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "python",
      "description": "Save the provided Python code to a script file and execute it. The function will return the result of executing the Python script, including any printed outputs or errors.",
      "parameters": {
        "type": "object",
        "required": ["code", "file_path"],
        "properties": {
          "code": {
            "type": "string",
            "description": "Python code to be saved and executed. This should be a complete, self-contained script, including necessary imports, function definitions, and any other required elements."
          },
          "file_path": {
            "type": "string",
            "description": "Path where the Python code will be saved as a script file. This must be within the current directory, as saving to any other location is prohibited. The Python file will be created and executed in this directory."
          }
        }
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "sql",
      "description": "Execute a SQL command on the specified database file. This is used to interact with SQL databases, retrieve data, and manipulate database records. The result of the executed SQL query will be returned.",
      "parameters": {
        "type": "object",
        "required": ["code", "dataset_path", "output_file"],
        "properties": {
          "code": {
            "type": "string",
            "description": "SQL query to be executed. This should be a complete, valid SQL statement, such as SELECT, UPDATE, INSERT, or DELETE, and can include necessary clauses or conditions."
          },
          "dataset_path": {
            "type": "string",
            "description": "Path to the database file, typically a SQLite database."
          },
          "output_file": {
            "type": "string",
            "description": "If set to a CSV file path, the results are saved to this CSV file; if set to 'direct', results are displayed directly."
          }
        }
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "data_cleaning",
      "description": "Clean the provided dataset based on the given data cleaning task. The cleaned dataset will be saved and the filename will be returned.",
      "parameters": {
        "type": "object",
        "required": ["dataset_path", "task"],
        "properties": {
          "dataset_path": {
            "type": "string",
            "description": "Path to the dataset that needs to be cleaned."
          },
          "task": {
            "type": "string",
            "description": "Data cleaning task to be performed on the dataset. The task requires a detailed description."
          }
        }
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "feature_engineering",
      "description": "Perform feature engineering on the provided dataset according to the given feature engineering task. The transformed dataset will be saved and the filename will be returned.",
      "parameters": {
        "type": "object",
        "required": ["dataset_path", "task"],
        "properties": {
          "dataset_path": {
            "type": "string",
            "description": "Path to the dataset for feature engineering."
          },
          "task": {
            "type": "string",
            "description": "Feature engineering task to be performed on the dataset. The task requires a detailed description."
          }
        }
      }
    }
  },
  {
    "type": "function",
    "function": {
        "name": "model_development",
        "description": "Solve the provided machine learning task. The tool will return both the code used for machine learning and the execution results.",
        "parameters": {
            "type": "object",
            "required": ["train_dataset_path", "test_dataset_path", "task"],
            "properties": {
                "train_dataset_path": {
                    "type": "string",
                    "description": "Path to the dataset file used for model training."
                },
                "test_dataset_path": {
                    "type": "string",
                    "description": "Path to the dataset file used for model evaluation or prediction. This path could be identical to `train_dataset_path`."
                },
                "task": {
                    "type": "string",
                    "description": "Specific requirements of the machine learning task, including the dataset name, detailed task description, and the path where the prediction results should be saved."
                }
            }
        }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "visualization",
      "description": "Create visualizations according to the task requirements. The visualization result will be saved and the filename will be returned.",
      "parameters": {
        "type": "object",
        "required": ["dataset_path", "task", "saved_plot_path"],
        "properties": {
          "dataset_path": {
            "type": "string",
            "description": "Path to the dataset file used for visualization."
          },
          "task": {
            "type": "string",
            "description": "Specific requirements of the visualization task."
          },
          "saved_plot_path": {
            "type": "string",
            "description": "Path where the plot will be saved."
          }
        }
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "debugging",
      "description": "Debug the provided code for a task. This involves identifying and fixing errors in the code, ensuring that the code functions correctly. The debugged code and its execution results will be returned.",
      "parameters": {
        "type": "object",
        "required": ["task", "code"],
        "properties": {
          "task": {
            "type": "string",
            "description": "The task that the code is intended to solve."
          },
          "code": {
            "type": "string",
            "description": "The Python code that needs debugging. This can be either the full Python code or the path to a file containing the code. After debugging, the updated code will be returned."
          }
        }
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "context_summarize",
      "description": "Summarize and distill the current interaction context into a compact, structured memory. This tool could be invoked when the accumulated context becomes overly long or noisy, in order to retain only high-value, task-relevant information for subsequent planning, reasoning, and tool usage.",
      "parameters": {
        "type": "object",
        "required": [],
        "properties": {}
      }
  }
}
]
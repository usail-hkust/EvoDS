MODEL_AGENT_SYS_PROMPT = """You are a data science expert specializing in machine learning and deep learning tasks. You are working as a sub-agent in a multi-agent system.

# GLOBAL CONTEXT #
The global context contains the overall task objective. You should use this information to understand the broader goal and maintain consistency with the overall workflow.

However, your responsibility is only to solve the assigned modeling subtask rather than the entire pipeline.

You have access to a set of tools that can help solve machine learning and deep learning tasks. When given dataset paths and a modeling task description, your first step is to determine whether the provided tools can directly solve the assigned subtask. If they can, use the appropriate tool to complete the task. The tool will automatically save the submission file.

If the existing tools cannot directly solve the subtask, use the `machine_learning_tool_creation` tool to create a new tool based on the provided task description.

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

MODEL_DEVELOPMENT_PROMPT = """
You are given a training dataset located at {train_dataset_path} and a testing dataset located at {test_dataset_path}.

{global_task}

Your primary objective is to solve the assigned modeling subtask below.

# MODELING SUBTASK

{task}

If you create a new tool to solve the task, you must strictly follow the instructions below:

### Instructions:

1. **Load the Dataset**: Start by loading the dataset.
2. **Design the Model**: Based on the specified subtask, choose an appropriate machine learning or deep learning model.
3. **Train the Model**: Train the selected model using the provided data. Optimize and tune the model when necessary.
4. **Validate the Model**: Evaluate the model using suitable metrics (e.g., accuracy, F1 score, RMSE) on a validation set.
5. **Print Results**: Print the evaluation metrics clearly for inspection and further iteration.
6. **Make Predictions**: Use the trained model to generate predictions on the test dataset.
7. **Save the Results**: Save the prediction results in the required format.

Please ensure that:

* The model is appropriately designed and trained according to the assigned subtask.
* The predictions are saved in the correct format.
* The printed results are clear and useful for further iteration and debugging.
"""


import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.util import write_program, execute_script


def logistic_regression(train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir: str = None):
    """
    执行逻辑回归任务，包括数据加载、预处理、交叉验证、预测及保存结果。

    参数:
    train_dataset_path (str): 训练集 CSV 文件路径
    test_dataset_path (str): 测试集 CSV 文件路径
    X_columns (list): 特征列名列表 (需要是数值型特征)
    Y_column (str): 目标标签列名
    submission_columns (list): 提交文件中的列名列表 (如 ['id'])
    submission_file_path (str): 保存预测结果的 CSV 路径
    """

    try:
        # Get file extension
        if work_dir:
            train_dataset_path = os.path.join(work_dir, os.path.normpath(train_dataset_path))
            test_dataset_path = os.path.join(work_dir, os.path.normpath(test_dataset_path))

        file_extension = train_dataset_path.split('.')[-1].lower()

        # Read the dataset based on the file extension
        if file_extension == 'csv':
            train_data = pd.read_csv(train_dataset_path)
            test_data = pd.read_csv(test_dataset_path)
        elif file_extension == 'xlsx':
            train_data = pd.read_excel(train_dataset_path)
            test_data = pd.read_excel(test_dataset_path)
        elif file_extension == 'json':
            train_data = pd.read_json(train_dataset_path)
            test_data = pd.read_json(test_dataset_path)
        else:
            return f"Machine learning task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        X_train = train_data[X_columns]
        y_train = train_data[Y_column]
        X_test = test_data[X_columns]

        pipeline = make_pipeline(LogisticRegression(max_iter=1000, random_state=42))
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        submission = test_data[submission_columns].copy()
        submission[Y_column] = predictions

        test_data.to_csv(submission_file_path, index=False)
        return f"Machine learning task completed successfully. Avg validation Accuracy: {scores.mean():.4f}. Submission file was saved in '{submission_file_path}'."

    except Exception as e:
        return f"Machine learning task failed. The reason is: Error occurred during encoding: {e}"


def linear_regression(train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir: str = None):
    """
    执行线性回归任务。
    """
    try:
        if work_dir:
            train_dataset_path = os.path.join(work_dir, os.path.normpath(train_dataset_path))
            test_dataset_path = os.path.join(work_dir, os.path.normpath(test_dataset_path))

        file_extension = train_dataset_path.split('.')[-1].lower()

        if file_extension == 'csv':
            train_data = pd.read_csv(train_dataset_path)
            test_data = pd.read_csv(test_dataset_path)
        elif file_extension == 'xlsx':
            train_data = pd.read_excel(train_dataset_path)
            test_data = pd.read_excel(test_dataset_path)
        elif file_extension == 'json':
            train_data = pd.read_json(train_dataset_path)
            test_data = pd.read_json(test_dataset_path)
        else:
            return f"Machine learning task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        X_train = train_data[X_columns]
        y_train = train_data[Y_column]
        X_test = test_data[X_columns]

        pipeline = make_pipeline(LinearRegression())
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        submission = test_data[submission_columns].copy()
        submission[Y_column] = predictions

        submission.to_csv(submission_file_path, index=False)
        return f"Machine learning task completed successfully. Avg validation R2 score: {scores.mean():.4f}. Submission file was saved in '{submission_file_path}'."

    except Exception as e:
        return f"Machine learning task failed. The reason is: Error occurred during encoding: {e}"
    

def random_forest_regression(train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir: str = None):
    try:
        if work_dir:
            train_dataset_path = os.path.join(work_dir, os.path.normpath(train_dataset_path))
            test_dataset_path = os.path.join(work_dir, os.path.normpath(test_dataset_path))
        
        file_extension = train_dataset_path.split('.')[-1].lower()
        # (省略重复的读取逻辑，实际代码中应包含，此处简略示意结构一致)
        if file_extension == 'csv':
            train_data = pd.read_csv(train_dataset_path)
            test_data = pd.read_csv(test_dataset_path)
        # ... 其他格式支持同上 ...
        elif file_extension == 'xlsx':
            train_data = pd.read_excel(train_dataset_path)
            test_data = pd.read_excel(test_dataset_path)
        elif file_extension == 'json':
            train_data = pd.read_json(train_dataset_path)
            test_data = pd.read_json(test_dataset_path)
        else: 
            return f"Machine learning task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        X_train = train_data[X_columns]
        y_train = train_data[Y_column]
        X_test = test_data[X_columns]

        pipeline = make_pipeline(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        submission = test_data[submission_columns].copy()
        submission[Y_column] = predictions

        submission.to_csv(submission_file_path, index=False)
        return f"Machine learning task completed successfully. Avg validation R2 score: {scores.mean():.4f}. Submission file was saved in '{submission_file_path}'."

    except Exception as e:
        return f"Machine learning task failed. The reason is: Error occurred during encoding: {e}"
    

def random_forest_classification(train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir: str = None):
    try:
        if work_dir:
            train_dataset_path = os.path.join(work_dir, os.path.normpath(train_dataset_path))
            test_dataset_path = os.path.join(work_dir, os.path.normpath(test_dataset_path))

        # Data Loading
        file_extension = train_dataset_path.split('.')[-1].lower()
        if file_extension == 'csv':
            train_data = pd.read_csv(train_dataset_path)
            test_data = pd.read_csv(test_dataset_path)
        elif file_extension == 'xlsx':
            train_data = pd.read_excel(train_dataset_path)
            test_data = pd.read_excel(test_dataset_path)
        elif file_extension == 'json':
            train_data = pd.read_json(train_dataset_path)
            test_data = pd.read_json(test_dataset_path)
        else: 
            return f"Machine learning task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        X_train = train_data[X_columns]
        y_train = train_data[Y_column]
        X_test = test_data[X_columns]

        pipeline = make_pipeline(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        submission = test_data[submission_columns].copy()
        submission[Y_column] = predictions

        submission.to_csv(submission_file_path, index=False)
        return f"Machine learning task completed successfully. Avg validation Accuracy: {scores.mean():.4f}. Submission file was saved in '{submission_file_path}'."

    except Exception as e:
        return f"Machine learning task failed. The reason is: Error occurred during encoding: {e}"


def xgboost_regression(train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir: str = None):
    try:
        if work_dir:
            train_dataset_path = os.path.join(work_dir, os.path.normpath(train_dataset_path))
            test_dataset_path = os.path.join(work_dir, os.path.normpath(test_dataset_path))
            
        # Data Loading
        file_extension = train_dataset_path.split('.')[-1].lower()
        if file_extension == 'csv':
            train_data = pd.read_csv(train_dataset_path)
            test_data = pd.read_csv(test_dataset_path)
        elif file_extension == 'xlsx':
            train_data = pd.read_excel(train_dataset_path)
            test_data = pd.read_excel(test_dataset_path)
        elif file_extension == 'json':
            train_data = pd.read_json(train_dataset_path)
            test_data = pd.read_json(test_dataset_path)
        else: 
            return f"Machine learning task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        X_train = train_data[X_columns]
        y_train = train_data[Y_column]
        X_test = test_data[X_columns]

        pipeline = make_pipeline(XGBRegressor(random_state=42, n_jobs=-1))
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        submission = test_data[submission_columns].copy()
        submission[Y_column] = predictions

        submission.to_csv(submission_file_path, index=False)
        return f"Machine learning task completed successfully. Avg validation R2 score: {scores.mean():.4f}. Submission file was saved in '{submission_file_path}'."

    except Exception as e:
        return f"Machine learning task failed. The reason is: Error occurred during encoding: {e}"


def xgboost_classification(train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir: str = None):
    try:
        if work_dir:
            train_dataset_path = os.path.join(work_dir, os.path.normpath(train_dataset_path))
            test_dataset_path = os.path.join(work_dir, os.path.normpath(test_dataset_path))
            
        # Data Loading
        file_extension = train_dataset_path.split('.')[-1].lower()
        if file_extension == 'csv':
            train_data = pd.read_csv(train_dataset_path)
            test_data = pd.read_csv(test_dataset_path)
        elif file_extension == 'xlsx':
            train_data = pd.read_excel(train_dataset_path)
            test_data = pd.read_excel(test_dataset_path)
        elif file_extension == 'json':
            train_data = pd.read_json(train_dataset_path)
            test_data = pd.read_json(test_dataset_path)
        else: 
            return f"Machine learning task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        X_train = train_data[X_columns]
        y_train = train_data[Y_column]
        X_test = test_data[X_columns]

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)

        # use_label_encoder=False 防止新版 xgboost 警告
        pipeline = make_pipeline(XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'))
        scores = cross_val_score(pipeline, X_train, y_train_enc, cv=5, scoring='accuracy')
        
        pipeline.fit(X_train, y_train_enc)
        predictions = pipeline.predict(X_test)
        
        predictions = le.inverse_transform(predictions)
        
        submission = test_data[submission_columns].copy()
        submission[Y_column] = predictions

        submission.to_csv(submission_file_path, index=False)
        return f"Machine learning task completed successfully. Avg validation Accuracy: {scores.mean():.4f}. Submission file was saved in '{submission_file_path}'."

    except Exception as e:
        return f"Machine learning task failed. The reason is: Error occurred during encoding: {e}"

    
def lightgbm_regression(train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir: str = None):
    try:
        if work_dir:
            train_dataset_path = os.path.join(work_dir, os.path.normpath(train_dataset_path))
            test_dataset_path = os.path.join(work_dir, os.path.normpath(test_dataset_path))
            
        # Data Loading
        file_extension = train_dataset_path.split('.')[-1].lower()
        if file_extension == 'csv':
            train_data = pd.read_csv(train_dataset_path)
            test_data = pd.read_csv(test_dataset_path)
        elif file_extension == 'xlsx':
            train_data = pd.read_excel(train_dataset_path)
            test_data = pd.read_excel(test_dataset_path)
        elif file_extension == 'json':
            train_data = pd.read_json(train_dataset_path)
            test_data = pd.read_json(test_dataset_path)
        else: 
            return f"Machine learning task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        X_train = train_data[X_columns]
        y_train = train_data[Y_column]
        X_test = test_data[X_columns]

        # verbose=-1 抑制警告
        pipeline = make_pipeline(LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1))
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        submission = test_data[submission_columns].copy()
        submission[Y_column] = predictions

        submission.to_csv(submission_file_path, index=False)
        return f"Machine learning task completed successfully. Avg validation R2 score: {scores.mean():.4f}. Submission file was saved in '{submission_file_path}'."

    except Exception as e:
        return f"Machine learning task failed. The reason is: Error occurred during encoding: {e}"
    

def lightgbm_classification(train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir: str = None):
    try:
        if work_dir:
            train_dataset_path = os.path.join(work_dir, os.path.normpath(train_dataset_path))
            test_dataset_path = os.path.join(work_dir, os.path.normpath(test_dataset_path))
            
        # Data Loading
        file_extension = train_dataset_path.split('.')[-1].lower()
        if file_extension == 'csv':
            train_data = pd.read_csv(train_dataset_path)
            test_data = pd.read_csv(test_dataset_path)
        elif file_extension == 'xlsx':
            train_data = pd.read_excel(train_dataset_path)
            test_data = pd.read_excel(test_dataset_path)
        elif file_extension == 'json':
            train_data = pd.read_json(train_dataset_path)
            test_data = pd.read_json(test_dataset_path)
        else: 
            return f"Machine learning task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        X_train = train_data[X_columns]
        y_train = train_data[Y_column]
        X_test = test_data[X_columns]

        pipeline = make_pipeline(LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1))
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        submission = test_data[submission_columns].copy()
        submission[Y_column] = predictions

        submission.to_csv(submission_file_path, index=False)
        return f"Machine learning task completed successfully. Avg validation Accuracy: {scores.mean():.4f}. Submission file was saved in '{submission_file_path}'."

    except Exception as e:
        return f"Machine learning task failed. The reason is: Error occurred during encoding: {e}"
    

def catboost_regression(train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir: str = None):
    try:
        if work_dir:
            train_dataset_path = os.path.join(work_dir, os.path.normpath(train_dataset_path))
            test_dataset_path = os.path.join(work_dir, os.path.normpath(test_dataset_path))
            
        # Data Loading
        file_extension = train_dataset_path.split('.')[-1].lower()
        if file_extension == 'csv':
            train_data = pd.read_csv(train_dataset_path)
            test_data = pd.read_csv(test_dataset_path)
        elif file_extension == 'xlsx':
            train_data = pd.read_excel(train_dataset_path)
            test_data = pd.read_excel(test_dataset_path)
        elif file_extension == 'json':
            train_data = pd.read_json(train_dataset_path)
            test_data = pd.read_json(test_dataset_path)
        else: 
            return f"Machine learning task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        X_train = train_data[X_columns]
        y_train = train_data[Y_column]
        X_test = test_data[X_columns]

        # verbose=0 静默训练, allow_writing_files=False 防止生成 catboost_info 文件夹
        pipeline = make_pipeline(CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False))
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        submission = test_data[submission_columns].copy()
        submission[Y_column] = predictions

        submission.to_csv(submission_file_path, index=False)
        return f"Machine learning task completed successfully. Avg validation R2 score: {scores.mean():.4f}. Submission file was saved in '{submission_file_path}'."

    except Exception as e:
        return f"Machine learning task failed. The reason is: Error occurred during encoding: {e}"

    
def catboost_classification(train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir: str = None):
    try:
        if work_dir:
            train_dataset_path = os.path.join(work_dir, os.path.normpath(train_dataset_path))
            test_dataset_path = os.path.join(work_dir, os.path.normpath(test_dataset_path))
            
        # Data Loading
        file_extension = train_dataset_path.split('.')[-1].lower()
        if file_extension == 'csv':
            train_data = pd.read_csv(train_dataset_path)
            test_data = pd.read_csv(test_dataset_path)
        elif file_extension == 'xlsx':
            train_data = pd.read_excel(train_dataset_path)
            test_data = pd.read_excel(test_dataset_path)
        elif file_extension == 'json':
            train_data = pd.read_json(train_dataset_path)
            test_data = pd.read_json(test_dataset_path)
        else: 
            return f"Machine learning task failed. The reason is: Unsupported file format: {file_extension}. Supported formats are csv, xlsx, and json."

        X_train = train_data[X_columns]
        y_train = train_data[Y_column]
        X_test = test_data[X_columns]

        pipeline = make_pipeline(CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False))
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        submission = test_data[submission_columns].copy()
        submission[Y_column] = predictions.reshape(-1)

        submission.to_csv(submission_file_path, index=False)
        return f"Machine learning task completed successfully. Avg validation Accuracy: {scores.mean():.4f}. Submission file was saved in '{submission_file_path}'."

    except Exception as e:
        return f"Machine learning task failed. The reason is: Error occurred during encoding: {e}"


def machine_learning_tool_creation(code, script_file_name, work_dir):
    script_file_name = os.path.normpath(script_file_name)
    write_program(code, os.path.join(work_dir, script_file_name))
    success, output = execute_script(script_file_name, work_dir)
    if success:
        if output is None or output == '':
            return f"Machine learning task completed successfully. The relevant code is as follows: \n{code}."
        return f"Machine learning task completed successfully. The relevant code is as follows: \n{code}\nThe result is as follows: \n{output}."
    else:
        return f"Machine learning task failed. The relevant code is as follows: \n{code}\nThe reason is: {output}"


machine_learning_tools = [
    {
        "type": "function",
        "function": {
            "name": "logistic_regression",
            "description": "Execute a logistic regression model. This includes loading data (supports csv, xlsx, json), training the model with cross-validation, generating predictions on the test set, and saving the submission file.",
            "parameters": {
                "type": "object",
                "required": [
                    "train_dataset_path",
                    "test_dataset_path",
                    "X_columns",
                    "Y_column",
                    "submission_columns",
                    "submission_file_path"
                ],
                "properties": {
                    "train_dataset_path": {
                        "type": "string",
                        "description": "Path to the training dataset file."
                    },
                    "test_dataset_path": {
                        "type": "string",
                        "description": "Path to the testing dataset file."
                    },
                    "X_columns": {
                        "type": "list",
                        "description": "List of feature column names to be used for training (features must be numerical).",
                        "items": {
                            "type": "string"
                        }
                    },
                    "Y_column": {
                        "type": "string",
                        "description": "The name of the target label column."
                    },
                    "submission_columns": {
                        "type": "list",
                        "description": "List of columns from the test dataset to include in the submission file.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "submission_file_path": {
                        "type": "string",
                        "description": "File path where the prediction results (submission file) will be saved."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "linear_regression",
            "description": "Execute a linear regression model. This includes loading data (supports csv, xlsx, json), training the model with cross-validation, generating predictions on the test set, and saving the submission file.",
            "parameters": {
                "type": "object",
                "required": [
                    "train_dataset_path",
                    "test_dataset_path",
                    "X_columns",
                    "Y_column",
                    "submission_columns",
                    "submission_file_path"
                ],
                "properties": {
                    "train_dataset_path": {
                        "type": "string",
                        "description": "Path to the training dataset file."
                    },
                    "test_dataset_path": {
                        "type": "string",
                        "description": "Path to the testing dataset file."
                    },
                    "X_columns": {
                        "type": "list",
                        "description": "List of feature column names to be used for training (features must be numerical).",
                        "items": {
                            "type": "string"
                        }
                    },
                    "Y_column": {
                        "type": "string",
                        "description": "The name of the target label column."
                    },
                    "submission_columns": {
                        "type": "list",
                        "description": "List of columns from the test dataset to include in the submission file.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "submission_file_path": {
                        "type": "string",
                        "description": "File path where the prediction results (submission file) will be saved."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "random_forest_regression",
            "description": "Execute a random forest regression model. This includes loading data (supports csv, xlsx, json), training the model with cross-validation, generating predictions on the test set, and saving the submission file.",
            "parameters": {
                "type": "object",
                "required": [
                    "train_dataset_path",
                    "test_dataset_path",
                    "X_columns",
                    "Y_column",
                    "submission_columns",
                    "submission_file_path"
                ],
                "properties": {
                    "train_dataset_path": {
                        "type": "string",
                        "description": "Path to the training dataset file."
                    },
                    "test_dataset_path": {
                        "type": "string",
                        "description": "Path to the testing dataset file."
                    },
                    "X_columns": {
                        "type": "list",
                        "description": "List of feature column names to be used for training (features must be numerical).",
                        "items": {
                            "type": "string"
                        }
                    },
                    "Y_column": {
                        "type": "string",
                        "description": "The name of the target label column."
                    },
                    "submission_columns": {
                        "type": "list",
                        "description": "List of columns from the test dataset to include in the submission file.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "submission_file_path": {
                        "type": "string",
                        "description": "File path where the prediction results (submission file) will be saved."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "random_forest_classification",
            "description": "Execute a random forest classification model. This includes loading data (supports csv, xlsx, json), training the model with cross-validation, generating predictions on the test set, and saving the submission file.",
            "parameters": {
                "type": "object",
                "required": [
                    "train_dataset_path",
                    "test_dataset_path",
                    "X_columns",
                    "Y_column",
                    "submission_columns",
                    "submission_file_path"
                ],
                "properties": {
                    "train_dataset_path": {
                        "type": "string",
                        "description": "Path to the training dataset file."
                    },
                    "test_dataset_path": {
                        "type": "string",
                        "description": "Path to the testing dataset file."
                    },
                    "X_columns": {
                        "type": "list",
                        "description": "List of feature column names to be used for training (features must be numerical).",
                        "items": {
                            "type": "string"
                        }
                    },
                    "Y_column": {
                        "type": "string",
                        "description": "The name of the target label column."
                    },
                    "submission_columns": {
                        "type": "list",
                        "description": "List of columns from the test dataset to include in the submission file.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "submission_file_path": {
                        "type": "string",
                        "description": "File path where the prediction results (submission file) will be saved."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "xgboost_regression",
            "description": "Execute a xgboost regression model. This includes loading data (supports csv, xlsx, json), training the model with cross-validation, generating predictions on the test set, and saving the submission file.",
            "parameters": {
                "type": "object",
                "required": [
                    "train_dataset_path",
                    "test_dataset_path",
                    "X_columns",
                    "Y_column",
                    "submission_columns",
                    "submission_file_path"
                ],
                "properties": {
                    "train_dataset_path": {
                        "type": "string",
                        "description": "Path to the training dataset file."
                    },
                    "test_dataset_path": {
                        "type": "string",
                        "description": "Path to the testing dataset file."
                    },
                    "X_columns": {
                        "type": "list",
                        "description": "List of feature column names to be used for training (features must be numerical).",
                        "items": {
                            "type": "string"
                        }
                    },
                    "Y_column": {
                        "type": "string",
                        "description": "The name of the target label column."
                    },
                    "submission_columns": {
                        "type": "list",
                        "description": "List of columns from the test dataset to include in the submission file.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "submission_file_path": {
                        "type": "string",
                        "description": "File path where the prediction results (submission file) will be saved."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "xgboost_classification",
            "description": "Execute a xgboost classification model. This includes loading data (supports csv, xlsx, json), training the model with cross-validation, generating predictions on the test set, and saving the submission file.",
            "parameters": {
                "type": "object",
                "required": [
                    "train_dataset_path",
                    "test_dataset_path",
                    "X_columns",
                    "Y_column",
                    "submission_columns",
                    "submission_file_path"
                ],
                "properties": {
                    "train_dataset_path": {
                        "type": "string",
                        "description": "Path to the training dataset file."
                    },
                    "test_dataset_path": {
                        "type": "string",
                        "description": "Path to the testing dataset file."
                    },
                    "X_columns": {
                        "type": "list",
                        "description": "List of feature column names to be used for training (features must be numerical).",
                        "items": {
                            "type": "string"
                        }
                    },
                    "Y_column": {
                        "type": "string",
                        "description": "The name of the target label column."
                    },
                    "submission_columns": {
                        "type": "list",
                        "description": "List of columns from the test dataset to include in the submission file.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "submission_file_path": {
                        "type": "string",
                        "description": "File path where the prediction results (submission file) will be saved."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lightgbm_regression",
            "description": "Execute a lightgbm regression model. This includes loading data (supports csv, xlsx, json), training the model with cross-validation, generating predictions on the test set, and saving the submission file.",
            "parameters": {
                "type": "object",
                "required": [
                    "train_dataset_path",
                    "test_dataset_path",
                    "X_columns",
                    "Y_column",
                    "submission_columns",
                    "submission_file_path"
                ],
                "properties": {
                    "train_dataset_path": {
                        "type": "string",
                        "description": "Path to the training dataset file."
                    },
                    "test_dataset_path": {
                        "type": "string",
                        "description": "Path to the testing dataset file."
                    },
                    "X_columns": {
                        "type": "list",
                        "description": "List of feature column names to be used for training (features must be numerical).",
                        "items": {
                            "type": "string"
                        }
                    },
                    "Y_column": {
                        "type": "string",
                        "description": "The name of the target label column."
                    },
                    "submission_columns": {
                        "type": "list",
                        "description": "List of columns from the test dataset to include in the submission file.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "submission_file_path": {
                        "type": "string",
                        "description": "File path where the prediction results (submission file) will be saved."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lightgbm_classification",
            "description": "Execute a lightgbm classification model. This includes loading data (supports csv, xlsx, json), training the model with cross-validation, generating predictions on the test set, and saving the submission file.",
            "parameters": {
                "type": "object",
                "required": [
                    "train_dataset_path",
                    "test_dataset_path",
                    "X_columns",
                    "Y_column",
                    "submission_columns",
                    "submission_file_path"
                ],
                "properties": {
                    "train_dataset_path": {
                        "type": "string",
                        "description": "Path to the training dataset file."
                    },
                    "test_dataset_path": {
                        "type": "string",
                        "description": "Path to the testing dataset file."
                    },
                    "X_columns": {
                        "type": "list",
                        "description": "List of feature column names to be used for training (features must be numerical).",
                        "items": {
                            "type": "string"
                        }
                    },
                    "Y_column": {
                        "type": "string",
                        "description": "The name of the target label column."
                    },
                    "submission_columns": {
                        "type": "list",
                        "description": "List of columns from the test dataset to include in the submission file.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "submission_file_path": {
                        "type": "string",
                        "description": "File path where the prediction results (submission file) will be saved."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "catboost_regression",
            "description": "Execute a catboost regression model. This includes loading data (supports csv, xlsx, json), training the model with cross-validation, generating predictions on the test set, and saving the submission file.",
            "parameters": {
                "type": "object",
                "required": [
                    "train_dataset_path",
                    "test_dataset_path",
                    "X_columns",
                    "Y_column",
                    "submission_columns",
                    "submission_file_path"
                ],
                "properties": {
                    "train_dataset_path": {
                        "type": "string",
                        "description": "Path to the training dataset file."
                    },
                    "test_dataset_path": {
                        "type": "string",
                        "description": "Path to the testing dataset file."
                    },
                    "X_columns": {
                        "type": "list",
                        "description": "List of feature column names to be used for training (features must be numerical).",
                        "items": {
                            "type": "string"
                        }
                    },
                    "Y_column": {
                        "type": "string",
                        "description": "The name of the target label column."
                    },
                    "submission_columns": {
                        "type": "list",
                        "description": "List of columns from the test dataset to include in the submission file.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "submission_file_path": {
                        "type": "string",
                        "description": "File path where the prediction results (submission file) will be saved."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "catboost_classification",
            "description": "Execute a catboost classification model. This includes loading data (supports csv, xlsx, json), training the model with cross-validation, generating predictions on the test set, and saving the submission file.",
            "parameters": {
                "type": "object",
                "required": [
                    "train_dataset_path",
                    "test_dataset_path",
                    "X_columns",
                    "Y_column",
                    "submission_columns",
                    "submission_file_path"
                ],
                "properties": {
                    "train_dataset_path": {
                        "type": "string",
                        "description": "Path to the training dataset file."
                    },
                    "test_dataset_path": {
                        "type": "string",
                        "description": "Path to the testing dataset file."
                    },
                    "X_columns": {
                        "type": "list",
                        "description": "List of feature column names to be used for training (features must be numerical).",
                        "items": {
                            "type": "string"
                        }
                    },
                    "Y_column": {
                        "type": "string",
                        "description": "The name of the target label column."
                    },
                    "submission_columns": {
                        "type": "list",
                        "description": "List of columns from the test dataset to include in the submission file.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "submission_file_path": {
                        "type": "string",
                        "description": "File path where the prediction results (submission file) will be saved."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "machine_learning_tool_creation",
            "description": "Create a **general-purpose, reusable machine learning tool** when necessary to solve the current task. The generated tool should be **dataset-agnostic** and applicable to a wide range of machine learning scenarios rather than being tailored to a specific dataset or variable. The tool will be saved as a Python script, executed immediately, and return the execution results, including any printed outputs or errors. The created tool is expected to be reusable in future machine learning tasks within an autonomous data science pipeline.",
            "parameters": {
                "type": "object",
                "required": ["code", "script_file_name"],
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code of the newly created machine learning tool. The tool must follow the strict format requirement and be general enough for reuse as: ```python\ndef tool_name(parameters):\n    # detail of the code\n\n# execute the tool\nif __name__ == '__main__':\n    parameters = {...}\n    tool_name(parameters)\n```"
                    },
                    "script_file_name": {
                        "type": "string",
                        "description": "The name of the Python script of the newly created tool will be saved and executed."
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
# import re
# async def run_loop(messages, tools):
#     choice = await llm.generate(messages, tools=tools)
#     return choice
# 
# llm_config = models_config.get('deepseek-ai/DeepSeek-V3.1-Terminus_v32')
# llm = create_llm_instance(llm_config)
# tool_map = {
#     "logistic_regression": logistic_regression,
#     "linear_regression": linear_regression,
#     "random_forest_regression": random_forest_regression,
#     "random_forest_classification": random_forest_classification,
#     "xgboost_regression": xgboost_regression,
#     "xgboost_classification": xgboost_classification,
#     "lightgbm_regression": lightgbm_regression,
#     "lightgbm_classification": lightgbm_classification,
#     "catboost_regression": catboost_regression,
#     "catboost_classification": catboost_classification,
#     "machine_learning_tool_creation": machine_learning_tool_creation
# }
# 
# # from transformers import AutoTokenizer
# # tokenizer = AutoTokenizer.from_pretrained("verl/Qwen3-8B")
# 
# work_dir = './test'
# count = 1
# task = 'use the catboost_regression tool to predict the score column using age and years_of_experience columns, and use the same file to test.'
# prompt = MODEL_DEVELOPMENT_PROMPT.format(dataset_file='synthetic_dataset.csv', task=task)
# messages = [{"role": "system", "content": MODEL_AGENT_SYS_PROMPT},
#             {"role": "user", "content": prompt}]
# choice = asyncio.run(run_loop(messages, machine_learning_tools))
# finish_reason = choice.finish_reason
# if finish_reason == "tool_calls":
#     messages.append(dict(choice.message))
#     for tool_call in choice.message.tool_calls:
#         tool_call_name = tool_call['function']['name']
#         try:
#             tool_call_arguments = json.loads(tool_call['function']['arguments'])
#             tool_function = tool_map[tool_call_name]
#             tool_result = tool_function(**tool_call_arguments, work_dir=work_dir)
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
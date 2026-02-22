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
from typing import Any, Optional
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
import re
import tiktoken
import asyncio
import pandas as pd
import warnings
import numpy as np
import torch
import os
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


async def logistic_regression(train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir: str = None):
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


async def linear_regression(train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir: str = None):
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
    

async def random_forest_regression(train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir: str = None):
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
    

async def random_forest_classification(train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir: str = None):
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


async def xgboost_regression(train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir: str = None):
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


async def xgboost_classification(train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir: str = None):
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

    
async def lightgbm_regression(train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir: str = None):
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
    

async def lightgbm_classification(train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir: str = None):
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
    

async def catboost_regression(train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir: str = None):
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

    
async def catboost_classification(train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir: str = None):
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


async def machine_learning_tool_creation(code, script_file_name, work_dir):
    script_file_name = os.path.normpath(script_file_name)
    try:
        await write_program(code, os.path.join(work_dir, script_file_name))
        success, output = await execute_script(script_file_name, work_dir)
    except Exception as e:
        return f"Machine learning task failed. The reason is: Error occurred while executing the script: {str(e)}"
    if success:
        if output is None or output == '':
            return f"Machine learning task completed successfully. The relevant code is as follows: \n{code}."
        return f"Machine learning task completed successfully. The relevant code is as follows: \n{code}\nThe result is as follows: \n{output}."
    else:
        return f"Machine learning task failed. The relevant code is as follows: \n{code}\nThe reason is: {output}"


class LogisticRegressionTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["train_dataset_path", "test_dataset_path", "X_columns", "Y_column", "submission_columns", "submission_file_path", "work_dir"]

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
            return ToolResponse(text=f"Machine learning task failed. The following parameters are missing: {', '.join(missing_list)}")

        train_dataset_path = parameters.get("train_dataset_path")
        test_dataset_path = parameters.get("test_dataset_path")
        X_columns = parameters.get("X_columns")
        Y_column = parameters.get("Y_column")
        submission_columns = parameters.get("submission_columns")
        submission_file_path = parameters.get("submission_file_path")
        work_dir = parameters.get("work_dir")

        result_message = await logistic_regression(
            train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir
        )
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class LinearRegressionTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["train_dataset_path", "test_dataset_path", "X_columns", "Y_column", "submission_columns", "submission_file_path", "work_dir"]

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
            return ToolResponse(text=f"Machine learning task failed. The following parameters are missing: {', '.join(missing_list)}")

        train_dataset_path = parameters.get("train_dataset_path")
        test_dataset_path = parameters.get("test_dataset_path")
        X_columns = parameters.get("X_columns")
        Y_column = parameters.get("Y_column")
        submission_columns = parameters.get("submission_columns")
        submission_file_path = parameters.get("submission_file_path")
        work_dir = parameters.get("work_dir")

        result_message = await linear_regression(
            train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir
        )
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class RandomForestRegressionTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["train_dataset_path", "test_dataset_path", "X_columns", "Y_column", "submission_columns", "submission_file_path", "work_dir"]

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
            return ToolResponse(text=f"Machine learning task failed. The following parameters are missing: {', '.join(missing_list)}")

        train_dataset_path = parameters.get("train_dataset_path")
        test_dataset_path = parameters.get("test_dataset_path")
        X_columns = parameters.get("X_columns")
        Y_column = parameters.get("Y_column")
        submission_columns = parameters.get("submission_columns")
        submission_file_path = parameters.get("submission_file_path")
        work_dir = parameters.get("work_dir")

        result_message = await random_forest_regression(
            train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir
        )
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

class RandomForestClassificationTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["train_dataset_path", "test_dataset_path", "X_columns", "Y_column", "submission_columns", "submission_file_path", "work_dir"]

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
            return ToolResponse(text=f"Machine learning task failed. The following parameters are missing: {', '.join(missing_list)}")

        train_dataset_path = parameters.get("train_dataset_path")
        test_dataset_path = parameters.get("test_dataset_path")
        X_columns = parameters.get("X_columns")
        Y_column = parameters.get("Y_column")
        submission_columns = parameters.get("submission_columns")
        submission_file_path = parameters.get("submission_file_path")
        work_dir = parameters.get("work_dir")

        result_message = await random_forest_classification(
            train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir
        )
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

class XGBoostRegressionTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["train_dataset_path", "test_dataset_path", "X_columns", "Y_column", "submission_columns", "submission_file_path", "work_dir"]

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
            return ToolResponse(text=f"Machine learning task failed. The following parameters are missing: {', '.join(missing_list)}")

        train_dataset_path = parameters.get("train_dataset_path")
        test_dataset_path = parameters.get("test_dataset_path")
        X_columns = parameters.get("X_columns")
        Y_column = parameters.get("Y_column")
        submission_columns = parameters.get("submission_columns")
        submission_file_path = parameters.get("submission_file_path")
        work_dir = parameters.get("work_dir")

        result_message = await xgboost_regression(
            train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir
        )
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

class XGBoostClassificationTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["train_dataset_path", "test_dataset_path", "X_columns", "Y_column", "submission_columns", "submission_file_path", "work_dir"]

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
            return ToolResponse(text=f"Machine learning task failed. The following parameters are missing: {', '.join(missing_list)}")

        train_dataset_path = parameters.get("train_dataset_path")
        test_dataset_path = parameters.get("test_dataset_path")
        X_columns = parameters.get("X_columns")
        Y_column = parameters.get("Y_column")
        submission_columns = parameters.get("submission_columns")
        submission_file_path = parameters.get("submission_file_path")
        work_dir = parameters.get("work_dir")

        result_message = await xgboost_classification(
            train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir
        )
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

class LightGBMRegressionTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["train_dataset_path", "test_dataset_path", "X_columns", "Y_column", "submission_columns", "submission_file_path", "work_dir"]

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
            return ToolResponse(text=f"Machine learning task failed. The following parameters are missing: {', '.join(missing_list)}")

        train_dataset_path = parameters.get("train_dataset_path")
        test_dataset_path = parameters.get("test_dataset_path")
        X_columns = parameters.get("X_columns")
        Y_column = parameters.get("Y_column")
        submission_columns = parameters.get("submission_columns")
        submission_file_path = parameters.get("submission_file_path")
        work_dir = parameters.get("work_dir")

        result_message = await lightgbm_regression(
            train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir
        )
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

class LightGBMClassificationTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["train_dataset_path", "test_dataset_path", "X_columns", "Y_column", "submission_columns", "submission_file_path", "work_dir"]

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
            return ToolResponse(text=f"Machine learning task failed. The following parameters are missing: {', '.join(missing_list)}")

        train_dataset_path = parameters.get("train_dataset_path")
        test_dataset_path = parameters.get("test_dataset_path")
        X_columns = parameters.get("X_columns")
        Y_column = parameters.get("Y_column")
        submission_columns = parameters.get("submission_columns")
        submission_file_path = parameters.get("submission_file_path")
        work_dir = parameters.get("work_dir")

        result_message = await lightgbm_classification(
            train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir
        )
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

class CatBoostRegressionTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["train_dataset_path", "test_dataset_path", "X_columns", "Y_column", "submission_columns", "submission_file_path", "work_dir"]

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
            return ToolResponse(text=f"Machine learning task failed. The following parameters are missing: {', '.join(missing_list)}")

        train_dataset_path = parameters.get("train_dataset_path")
        test_dataset_path = parameters.get("test_dataset_path")
        X_columns = parameters.get("X_columns")
        Y_column = parameters.get("Y_column")
        submission_columns = parameters.get("submission_columns")
        submission_file_path = parameters.get("submission_file_path")
        work_dir = parameters.get("work_dir")

        result_message = await catboost_regression(
            train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir
        )
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

class CatBoostClassificationTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["train_dataset_path", "test_dataset_path", "X_columns", "Y_column", "submission_columns", "submission_file_path", "work_dir"]

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
            return ToolResponse(text=f"Machine learning task failed. The following parameters are missing: {', '.join(missing_list)}")

        train_dataset_path = parameters.get("train_dataset_path")
        test_dataset_path = parameters.get("test_dataset_path")
        X_columns = parameters.get("X_columns")
        Y_column = parameters.get("Y_column")
        submission_columns = parameters.get("submission_columns")
        submission_file_path = parameters.get("submission_file_path")
        work_dir = parameters.get("work_dir")

        result_message = await catboost_classification(
            train_dataset_path, test_dataset_path, X_columns, Y_column, submission_columns, submission_file_path, work_dir
        )
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

class MachineLearningToolCreationTool(BaseTool):
    """Tool for creating and executing a machine learning tool script."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["code", "script_file_name", "work_dir"]

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
            return ToolResponse(text=f"Machine learning task failed. The following parameters are missing: {', '.join(missing_list)}")

        code = parameters.get("code")
        script_file_name = parameters.get("script_file_name")
        work_dir = parameters.get("work_dir")

        result_message = await machine_learning_tool_creation(code, script_file_name, work_dir)
        return ToolResponse(text=result_message)

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

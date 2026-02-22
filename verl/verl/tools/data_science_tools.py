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
import json
import shlex
import numpy as np
import torch
import sys
from heapq import nlargest
from verl.tools.util import *
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
logger = logging.getLogger(__name__)


DATA_CLEARNER_SYS_PROMPT = """You are a data science expert specializing in data cleaning tasks. You have access to a set of tools that can help solve these tasks. When given a dataset path and a data cleaning task description, your first step is to check if the provided tools can directly solve the task. If they can, use the appropriate tool to perform the cleaning. The tool will automatically save the cleaned dataset.

If the tools cannot directly solve the task, you should use the `data_cleaning_tool_creation` tool to create a new tool to address the task based on the description provided.

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

DATA_CLEANING_PROMPT = """
You are given a dataset located at {dataset_file}. Your task is to clean the dataset according to the following requirements:

# DATA CLEANING TASK #
{task}

After cleaning, save the cleaned dataset to {saved_dataset_file}.
"""

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

MODEL_AGENT_SYS_PROMPT = """You are a data science expert specializing in machine learning or deep learning tasks. You have access to a set of tools that can help solve these tasks. When given a dataset path and a machine learning task description, your first step is to check if the provided tools can directly solve the task. If they can, use the appropriate tool to solve the machine learning task. The tool will automatically save the submission file.

If the tools cannot directly solve the task, you should use the `machine_learning_tool_creation` tool to create a new tool to address the task based on the description provided.

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

MODEL_DEVELOPMENT_PROMPT = """
You are given a training dataset located at {train_dataset_path}, and a testing dataset located at {test_dataset_path}. Your task is to solve the machine learning task below:

# MODELING TASK #
{task}

If you create a new tool to solve the task, you should strictly follow the instruction below:
### Instructions:
1. **Load the Dataset**: Start by loading the dataset.
2. **Design the Model**: Based on the specified task, choose an appropriate machine learning or deep learning model.
3. **Train the Model**: Train the chosen model using the provided data. Ensure that the model is optimized and tuned for better performance.
4. **Validate the Model**: Evaluate the model's performance using suitable metrics (e.g., accuracy, F1 score, RMSE, etc.) on a validation set.
5. **Print Results**: Print the model's performance metrics (e.g., accuracy, loss, etc.) for inspection. This will allow further iteration on model design or training if necessary.
6. **Make Predictions**: Use the trained model to make predictions on the specified test data.
7. **Save the Results**: After prediction, save the prediction results as specified.

Please ensure that:
- The model is appropriately designed and trained according to the task.
- The predictions are saved in the correct format.
- The printed results are clear and can be used for further iteration of the model.
"""

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


DEBUG_AGENT_SYS_PROMPT = """You are a data science expert specializing in debugging code and troubleshooting issues in data science tasks. Your responsibility is to identify errors, inefficiencies, or bugs in the given code or process and provide solutions to fix them. You should analyze the task, locate potential issues, and suggest or implement fixes while ensuring the solution follows best practices."""

CODE_DEBUG_PROMPT = """However, there are some bugs in the code. Here is the execution result:
# Execution Result:
{observation}

---

Based on the provided execution result, please revise the script to fix these bugs. Your task is to address the error indicated in the result, and refine or modify the code as needed to ensure it works correctly.

The Python code should strictly follow the format below:
```python
# Provide the corrected python code here.
```
"""


CONTEXT_SUMMARIZE_PROMPT = """Your task is to summarize and distill all useful information obtained from the entire interaction history above, in order to support long-horizon decision making, planning, and tool usage in subsequent steps.

IMPORTANT:
- Do NOT summarize, restate, or infer the original problem statement.
- The original problem will be provided separately in a later step.
- Focus exclusively on summarizing information derived from the interaction history (e.g., decisions, constraints, intermediate results, environment or tool-related details).
- You must perform the summarization directly using your internal reasoning. Do NOT call, invoke, or rely on any external tools or functions during this process.

**Objectives**
Extract high-value, task-relevant information from the interaction history.
Remove redundancy, verbosity, and irrelevant conversational content.
Preserve actionable knowledge, constraints, decisions, and intermediate results.
Produce a compact yet information-complete summary that can be used as the agent’s persistent context or state.

**What to Include**
Summarize information including, but not limited to:
Environment-related information.
Data-related information.
Important decisions, conclusions, or resolved questions.
Derived insights, intermediate results, or partial solutions.
Tool usage rules, formats, or protocols established during the interaction.

**What to Exclude**
Polite language, greetings, or conversational fillers.
Repeated or superseded information.
Failed attempts unless they reveal an important constraint or insight.
Speculative or uncertain content not supported by the interaction.

**Output Requirements (Strict)**
The output must be a numbered list, strictly following the format:

1. …
2. …
3. …
   …

Each item must be:
Self-contained.
Concise yet precise.
Focused on a single piece of useful information.

Do NOT add headings, subheadings, or bullet points.
Do NOT include explanations outside the numbered list.
Do NOT explicitly reference the original conversation (e.g., “the user said above”).

**Role Awareness**
You are not solving the task itself.
You are producing a compressed, structured memory to be consumed in subsequent steps.
Ensure the summary is faithful, loss-minimized, and operationally useful.

Begin the summary now."""


SQL_TEMPLATE = """
import sqlite3
import pandas as pd
import os

def execute_sql(file_path, command, output_path):
    # make sure the file path is correct
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {{file_path}}")
        return

    # Connect to the SQLite database
    conn = sqlite3.connect(file_path)
    
    try:
        # Execute the SQL command and fetch the results
        df = pd.read_sql_query(command, conn)
        
        # Check if the output should be saved to a CSV file or printed directly
        if output_path.lower().endswith(".csv"):
            df.to_csv(output_path, index=False)
            print(f"Output saved to: {{output_path}}")
        else:
            print(df)
    except Exception as e:
        print(f"ERROR: {{e}}")
    finally:
        # Close the connection to the database
        conn.close()

# Example usage
file_path = "{file_path}"  # Path to your SQLite database file
command = "{code}"             # SQL command to be executed
output_path = "{output}" # Path to save the output as a CSV or "directly"

execute_sql(file_path, command, output_path)

"""

code_instructions = (
    "\n\n"
    "Please write your code solution in Python. "
    "Return ONLY the complete, runnable code without explanations. "
    "Use proper Python syntax and formatting. "
)


def _extract_code_from_markdown(text: str) -> str:
    """
    Extract code from markdown code blocks in the response.
    
    Args:
        text: The text containing possible markdown code blocks
        
    Returns:
        The extracted code as a string, or empty string if no code blocks found
    """
    # Look for Python code blocks (```python ... ```)
    python_pattern = r"```python\s*([\s\S]*?)\s*```"
    python_matches = re.findall(python_pattern, text)

    if python_matches:
        # Join all Python code blocks
        return "\n\n".join(python_matches)
    
    # If no Python blocks found, look for generic code blocks (``` ... ```)
    generic_pattern = r"```\s*([\s\S]*?)\s*```"
    generic_matches = re.findall(generic_pattern, text)
    
    if generic_matches:
        # Join all generic code blocks
        return "\n\n".join(generic_matches)
    
    # No code blocks found
    return text

def _extract_xml_from_response(text: str, field_dict: dict) -> str:
    """
    Extract XML fields from the response.
    
    Args:
        text: The text containing possible XML fields
        field_dict: A dictionary mapping field names to XML tags
        
    Returns:
        The extracted XML fields as a string, or empty string if no fields found
    """
    pattern = r"<(\w+)>(.*?)</\1>"
    matches = re.findall(pattern, text, re.DOTALL)
    
    found_fields = {match[0]: match[1].strip() for match in matches}

    for field_name in field_dict.keys():
        is_required = field_dict[field_name]['default'] is None
        if is_required and field_name not in found_fields:
            raise print(f"Field '{field_name}' is missing or empty.")
        
    return found_fields


class EnvException(Exception):
    def __init__(self, message):
        self.message = message 
    def __str__(self):
        return self.message
        

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

    
async def exec_run(cmd: str, workdir: str):
    try:
        cmd_list = shlex.split(cmd)
        process = await asyncio.create_subprocess_exec(
            *cmd_list,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=900)
        if process.returncode != 0:
            return False, stderr.decode(errors='replace')
        return True, stdout.decode(errors='replace')
    except asyncio.TimeoutError:
        # 超时处理：必须手动杀死子进程，否则它会在后台继续运行
        process.kill()
        # 等待进程完全退出以清理僵尸进程
        await process.wait()
        return False, "Timeout"
    except UnicodeDecodeError as e:
        return False, "UnicodeDecodeError"
    except Exception as e:
        return False, f"Unknown error: {e}"

    
async def _execute_sql_code(file_path, code, output: str, work_dir: str) -> str:
    script_content = SQL_TEMPLATE.format(file_path=file_path, code=code, output=output)
    temp_file_path = "temp_sql_script.py"
    script_content = script_content.replace('"', '\\"').replace('`', '\\`').replace('$', '\\$')

    with open(os.path.join(work_dir, temp_file_path), "w+", encoding="utf-8") as f:
        f.write(script_content)
    write_program(script_content, os.path.join(work_dir, temp_file_path))
    _, observation = await execute_script(temp_file_path, work_dir)

    return observation

def update_working_directory(current: str, changed: Optional[str] = None) -> str:
    """ Resolves absolute path from the current working directory path and the argument of the `cd` command
    @args:
        current (str): the current working directory
        changed (Optional[str]): the changed working directory, argument of shell `cd` command
    @return:
        new_path (str): absolute path of the new working directory in the container
    """
    if not changed:
        return current
    if changed[0] == "/":
        current = ""

    path = []
    for segment in (current + "/" + changed).split("/"):
        if segment == "..":
            if path:
                path.pop()
        elif segment and segment != ".":
            path.append(segment)
    new_path = "/" + "/".join(path)
    return new_path

async def execute_command(command: str, work_dir: str):
    cmd = command
    exit_code, output = await exec_run(cmd, workdir=work_dir)
    if "venv" in command:
        return False, "Creating a new python environment is not allowed in the container. You can use 'pip install' to install the required packages."
    is_cd_flag = command.strip().startswith("cd ")
    if is_cd_flag:
        changed = command[command.index("cd ") + 3:].strip()
        if "&&" in changed:
            changed = changed[:changed.index("&&")].strip()
        work_dir = update_working_directory(work_dir, changed)
        return True, f"The command to change directory to {work_dir} is executed successfully."
        
    return exit_code, output


def write_program(assistant_output, out_fname):
    match = re.search(r"```python(.*?)```", assistant_output, re.DOTALL)
    if match:
        result = match.group(1).strip()
    else:
        result = assistant_output.strip()

    with open(out_fname, "w+", encoding="utf-8") as f:
        f.write(result)


def process_with_code_fill(prompt: str) -> str:
    instructions = prompt + code_instructions
    return instructions

def process_with_xml_fill(prompt: str, field_dict: dict) -> str:
    examples = []
    for field_name in field_dict.keys():
        description = field_dict[field_name]['description']
        examples.append(f"<{field_name}>{description}</{field_name}>")
    example_str = "\n".join(examples)
    
    instructions = prompt + f"\n# Response format (must be strictly followed) (do not include any other formats except for the given XML format):\n{example_str}"
    return instructions

async def call_with_mode(kwargs, system_prompt: str, prompt: str, mode: str, field_dict: dict = {}, memory: list = None) -> str:
    if mode == "code_fill":
        prompt = process_with_code_fill(prompt)
    elif mode == 'xml_fill':
        prompt = process_with_xml_fill(prompt, field_dict)

    tokenizer = kwargs.get("tokenizer")
    server_manager = kwargs.get("server_manager")
    request_id = kwargs.get("request_id")
    sampling_params = kwargs.get("sampling_params")

    messages = [{"role": "system", "content": system_prompt}]
    if memory is not None:
        for m in memory:
            messages.append(m)
    messages.append({"role": "user", "content": prompt})

    output = await server_manager.generate(
        request_id=request_id,
        prompt_ids=tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True
        ),
        sampling_params=sampling_params
    )

    response = tokenizer.decode(output.token_ids[:-1])
    if mode == 'code_fill':
        code = _extract_code_from_markdown(response)
        return code
    elif mode == 'xml_fill':
        found_fields = _extract_xml_from_response(response, field_dict)
        return found_fields


async def call_tool(tool_call: FunctionCall, tools: dict, domain: str, work_dir: str) -> ToolResponse:
    """Call tool and return tool response."""
    tool, instance_id = None, None
    use_created_tool = False
    try:
        tool_name = tool_call.name
        tool_args = json.loads(tool_call.arguments)
        if tool_name not in tools:
            with open(f"examples/sglang_multiturn/config/tool_config/created_tools/{domain}_tools.json", "r", encoding='utf-8') as f:
                created_tools = json.load(f)
            if tool_name in created_tools:
                tool_code = created_tools[tool_name]['code']
                tool_result = execute_tool(tool_code, tool_name, tool_args, domain, work_dir=work_dir)
                update_tool_num(domain, tool_name)
                return ToolResponse(text=tool_result)
            else:
                return ToolResponse(text=f"Error when executing tool {tool_name}: Tool {tool_name} not found.")
        else:
            tool_args['work_dir'] = work_dir
            tool = tools[tool_name]
            instance_id, _ = await tool.create()
            tool_execution_response = await tool.execute(instance_id, tool_args)

    except Exception as e:
        logger.warning(f"Error when executing tool {tool_call.name}: {e}")
        return ToolResponse(
            text=f"Error when executing tool {tool_call.name}: {e}",
        )
    finally:
        if not use_created_tool:
            if tool and instance_id:
                await tool.release(instance_id)

    return tool_execution_response


class DataCleaningTool(BaseTool):
    """Tool for data cleaning (removing duplicates, handling missing values, etc.)."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.system_prompt = DATA_CLEARNER_SYS_PROMPT
        self.prompt_template = DATA_CLEANING_PROMPT
        self.count = 0
        self.max_step = 3
        self.tool_parser = None
        self.required_keys = ["dataset_path", "task", "work_dir"]

        self.response_length = 24576
        tool_config_path = 'examples/sglang_multiturn/config/tool_config/data_cleaning_tool_config.yaml'
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_config = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

        if os.path.exists(f"examples/sglang_multiturn/config/tool_config/created_tools/data_cleaning_tools.json"):    
            with open(f"examples/sglang_multiturn/config/tool_config/created_tools/data_cleaning_tools.json", "r", encoding='utf-8') as f:
                created_tools = json.load(f)
        else:
            created_tools = {}
        if os.path.exists(f"examples/sglang_multiturn/config/tool_config/created_tools/data_cleaning_tool_count.json"):    
            with open(f"examples/sglang_multiturn/config/tool_config/created_tools/data_cleaning_tool_count.json", "r", encoding='utf-8') as f:
                tool_count = json.load(f)
        else:
            tool_count = {}
        top_k = [k for k, _ in nlargest(10, tool_count.items(), key=lambda x: x[1]) if tool_count[k] >= 3]
        add_tools = [created_tools[tool_name]['tool_config'] for tool_name in top_k]
        self.all_tools = self.tool_config + add_tools
        

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
            return ToolResponse(text=f"Data cleaning task failed. The following parameters are missing: {', '.join(missing_list)}"), None
        
        dataset_path = parameters.get("dataset_path")
        task = parameters.get("task")
        work_dir = parameters.get("work_dir")
        server_manager = kwargs.get("server_manager")
        tokenizer = kwargs.get("tokenizer")
        sampling_params = kwargs.get("sampling_params")

        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + "_cleaned." + dataset_path.split(".")[-1]
        prompt = self.prompt_template.format(dataset_file=dataset_path, task=task, saved_dataset_file=saved_dataset_path)

        agent_data = AgentData(
            messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
            image_data=None,
            metrics={},
            request_id=instance_id,
            tools_kwargs={}
        )
        
        agent_data.prompt_ids = tokenizer.apply_chat_template(
            agent_data.messages,
            tools=self.tool_config,
            add_generation_prompt=True,
            tokenize=True,
        )
        
        for i in range(self.max_step):
            agent_data.user_turns += 1
            output = await server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params
            )
            agent_data.response_ids = output.token_ids
            agent_data.prompt_ids += agent_data.response_ids
            agent_data.response_mask += [1] * len(agent_data.response_ids)
            agent_data.assistant_turns += 1
            if output.log_probs:
                agent_data.response_logprobs += output.log_probs

            if self.tool_parser is None:
                self.tool_parser = ToolParser.get_tool_parser('hermes', tokenizer)

            # Extract tool calls
            _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
            
            if agent_data.tool_calls:
                tool_call = agent_data.tool_calls[0]
                response = await call_tool(tool_call, self.tools, 'data_cleaning', work_dir)
                if tool_call.name not in self.tools:
                    add_tools = get_add_tools('data_cleaning')
                    self.all_tools = self.tool_config + add_tools
                response_text = response.text
            else:
                response_text = tokenizer.decode(output.token_ids[:-1])
                return ToolResponse(text=response_text), None
                
            if response_text.startswith("Data cleaning task completed successfully."):
                # add created tools
                if tool_call.name == "data_cleaning_tool_creation":
                    try:
                        tool_args = json.loads(tool_call.arguments)
                        success, add_tools = await update_tools(tool_args['code'], 'data_cleaning', instance_id, server_manager, tokenizer, sampling_params)
                        if success:
                            self.all_tools = self.tool_config + add_tools
                    except Exception as e:
                        print("Update tool failed due to: ", e)

                prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
                response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
                trajectory_output = AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids[: self.response_length],
                    response_mask=agent_data.response_mask[: self.response_length],
                    multi_modal_data={},
                    response_logprobs=agent_data.response_logprobs[: self.response_length],
                    num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
                    metrics=agent_data.metrics,
                    extra_fields={},
                )
                # compute turn_scores
                turn_scores = 0.1
                trajectory_output.extra_fields.update({"turn_scores": [turn_scores], "work_dir": work_dir, "init_files_hash": None, "sub_agent": True, "uid": uuid4().hex, "id": agent_data.request_id, "final": False})

                return ToolResponse(text=response_text), trajectory_output

            if i == self.max_step - 1:
                prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
                response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
                trajectory_output = AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids[: self.response_length],
                    response_mask=agent_data.response_mask[: self.response_length],
                    multi_modal_data={},
                    response_logprobs=agent_data.response_logprobs[: self.response_length],
                    num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
                    metrics=agent_data.metrics,
                    extra_fields={},
                )
                # compute turn_scores
                turn_scores = -0.1
                trajectory_output.extra_fields.update({"turn_scores": [turn_scores], "work_dir": work_dir, "init_files_hash": None, "sub_agent": True, "uid": uuid4().hex, "id": agent_data.request_id, "final": False})

                return ToolResponse(text=f"Data cleaning task failed. Please provide more detailed information about the data cleaning task or use an alternative tool to proceed with the process."), trajectory_output

            response_ids = tokenizer.apply_chat_template([{"role": "tool", "content": response_text or ""}], add_generation_prompt=True, tokenize=True)
            agent_data.prompt_ids += response_ids
            agent_data.response_mask += [0] * len(response_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(response_ids)

            if len(agent_data.response_mask) >= self.response_length:
                return ToolResponse(text=f"Data cleaning task failed. The reason is the messages length exceeds the maximum length. Please provide more detailed information about the data cleaning task or use an alternative tool to proceed with the process."), None
            
        # return ToolResponse(text=f"Data cleaning task failed. Please provide more detailed information about the data cleaning task or use an alternative tool to proceed with the process."), None

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class FeatureEngineeringTool(BaseTool):
    """Tool for feature engineering (scaling, encoding, etc.)."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.system_prompt = FEATURE_ENGINEER_SYS_PROMPT
        self.prompt_template = FEATURE_ENGINEERING_PROMPT
        self.count = 0
        self.max_step = 3
        self.tool_parser = None
        self.required_keys = ["dataset_path", "task", "work_dir"]

        self.response_length = 24576
        tool_config_path = 'examples/sglang_multiturn/config/tool_config/feature_engineering_tool_config.yaml'
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_config = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

        if os.path.exists(f"examples/sglang_multiturn/config/tool_config/created_tools/feature_engineering_tools.json"):    
            with open(f"examples/sglang_multiturn/config/tool_config/created_tools/feature_engineering_tools.json", "r", encoding='utf-8') as f:
                created_tools = json.load(f)
        else:
            created_tools = {}
        if os.path.exists(f"examples/sglang_multiturn/config/tool_config/created_tools/feature_engineering_tool_count.json"):    
            with open(f"examples/sglang_multiturn/config/tool_config/created_tools/feature_engineering_tool_count.json", "r", encoding='utf-8') as f:
                tool_count = json.load(f)
        else:
            tool_count = {}
        top_k = [k for k, _ in nlargest(10, tool_count.items(), key=lambda x: x[1]) if tool_count[k] >= 3]
        add_tools = [created_tools[tool_name]['tool_config'] for tool_name in top_k]
        self.all_tools = self.tool_config + add_tools

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
            return ToolResponse(text=f"Feature engineering task failed. The following parameters are missing: {', '.join(missing_list)}"), None
        
        dataset_path = parameters.get("dataset_path")
        task = parameters.get("task")
        work_dir = parameters.get("work_dir")
        server_manager = kwargs.get("server_manager")
        tokenizer = kwargs.get("tokenizer")
        sampling_params = kwargs.get("sampling_params")

        saved_dataset_path = ".".join(dataset_path.split(".")[:-1]) + "_feature_engineered." + dataset_path.split(".")[-1]
        prompt = self.prompt_template.format(dataset_file=dataset_path, task=task, saved_dataset_file=saved_dataset_path)
        
        agent_data = AgentData(
            messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
            image_data=None,
            metrics={},
            request_id=instance_id,
            tools_kwargs={}
        )
    
        agent_data.prompt_ids = tokenizer.apply_chat_template(
            agent_data.messages,
            tools=self.tool_config,
            add_generation_prompt=True,
            tokenize=True,
        )
        
        for i in range(self.max_step):
            agent_data.user_turns += 1
            output = await server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params
            )
            agent_data.response_ids = output.token_ids
            agent_data.prompt_ids += agent_data.response_ids
            agent_data.response_mask += [1] * len(agent_data.response_ids)
            agent_data.assistant_turns += 1
            if output.log_probs:
                agent_data.response_logprobs += output.log_probs

            if self.tool_parser is None:
                self.tool_parser = ToolParser.get_tool_parser('hermes', tokenizer)

            # Extract tool calls
            _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
            
            if agent_data.tool_calls:
                tool_call = agent_data.tool_calls[0]
                response = await call_tool(tool_call, self.tools, 'feature_engineering', work_dir)
                if tool_call.name not in self.tools:
                    add_tools = get_add_tools('feature_engineering')
                    self.all_tools = self.tool_config + add_tools
                response_text = response.text
            else:
                response_text = tokenizer.decode(output.token_ids[:-1])
                return ToolResponse(text=response_text), None
                
            if response_text.startswith("Feature engineering task completed successfully."):
                # add created tools
                if tool_call.name == "feature_engineering_tool_creation":
                    try:
                        tool_args = json.loads(tool_call.arguments)
                        success, add_tools = await update_tools(tool_args['code'], 'feature_engineering', instance_id, server_manager, tokenizer, sampling_params)
                        if success:
                            self.all_tools = self.tool_config + add_tools
                    except Exception as e:
                        print("Update tool failed due to: ", e)

                prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
                response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
                trajectory_output = AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids[: self.response_length],
                    response_mask=agent_data.response_mask[: self.response_length],
                    multi_modal_data={},
                    response_logprobs=agent_data.response_logprobs[: self.response_length],
                    num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
                    metrics=agent_data.metrics,
                    extra_fields={},
                )
                # compute turn_scores
                turn_scores = 0.1
                trajectory_output.extra_fields.update({"turn_scores": [turn_scores], "work_dir": work_dir, "init_files_hash": None, "sub_agent": True, "uid": uuid4().hex, "id": agent_data.request_id, "final": False})

                return ToolResponse(text=response_text), trajectory_output
            
            if i == self.max_step - 1:
                prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
                response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
                trajectory_output = AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids[: self.response_length],
                    response_mask=agent_data.response_mask[: self.response_length],
                    multi_modal_data={},
                    response_logprobs=agent_data.response_logprobs[: self.response_length],
                    num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
                    metrics=agent_data.metrics,
                    extra_fields={},
                )
                # compute turn_scores
                turn_scores = -0.1
                trajectory_output.extra_fields.update({"turn_scores": [turn_scores], "work_dir": work_dir, "init_files_hash": None, "sub_agent": True, "uid": uuid4().hex, "id": agent_data.request_id, "final": False})

                return ToolResponse(text=f"Feature engineering task failed. Please provide more detailed information about the feature engineering task or use an alternative tool to proceed with the process."), trajectory_output

            response_ids = tokenizer.apply_chat_template([{"role": "tool", "content": response_text or ""}], add_generation_prompt=True, tokenize=True)
            agent_data.prompt_ids += response_ids
            agent_data.response_mask += [0] * len(response_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(response_ids)

            if len(agent_data.response_mask) >= self.response_length:
                return ToolResponse(text=f"Feature engineering task failed. The reason is the messages length exceeds the maximum length. Please provide more detailed information about the feature engineering task or use an alternative tool to proceed with the process."), None
            
        # return ToolResponse(text=f"Feature engineering task failed. Please provide more detailed information about the feature engineering task or use an alternative tool to proceed with the process."), None

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class ModelDevelopmentTool(BaseTool):
    """Tool for model development (training, validation)."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.system_prompt = MODEL_AGENT_SYS_PROMPT
        self.prompt_template = MODEL_DEVELOPMENT_PROMPT
        self.count = 0
        self.max_step = 3
        self.tool_parser = None
        self.required_keys = ["train_dataset_path", "test_dataset_path", "task", "work_dir"]

        self.response_length = 24576
        tool_config_path = 'examples/sglang_multiturn/config/tool_config/machine_learning_tool_config.yaml'
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_config = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

        if os.path.exists(f"examples/sglang_multiturn/config/tool_config/created_tools/machine_learning_tools.json"):    
            with open(f"examples/sglang_multiturn/config/tool_config/created_tools/machine_learning_tools.json", "r", encoding='utf-8') as f:
                created_tools = json.load(f)
        else:
            created_tools = {}
        if os.path.exists(f"examples/sglang_multiturn/config/tool_config/created_tools/machine_learning_tool_count.json"):    
            with open(f"examples/sglang_multiturn/config/tool_config/created_tools/machine_learning_tool_count.json", "r", encoding='utf-8') as f:
                tool_count = json.load(f)
        else:
            tool_count = {}
        top_k = [k for k, _ in nlargest(10, tool_count.items(), key=lambda x: x[1]) if tool_count[k] >= 3]
        add_tools = [created_tools[tool_name]['tool_config'] for tool_name in top_k]
        self.all_tools = self.tool_config + add_tools

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
            return ToolResponse(text=f"Machine learning task failed. The following parameters are missing: {', '.join(missing_list)}"), None
        
        train_dataset_path = parameters.get("train_dataset_path")
        test_dataset_path = parameters.get("test_dataset_path")
        task = parameters.get("task")
        work_dir = parameters.get("work_dir")
        server_manager = kwargs.get("server_manager")
        tokenizer = kwargs.get("tokenizer")
        sampling_params = kwargs.get("sampling_params")

        prompt = self.prompt_template.format(train_dataset_path=train_dataset_path, test_dataset_path=test_dataset_path, task=task)
        
        agent_data = AgentData(
            messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
            image_data=None,
            metrics={},
            request_id=instance_id,
            tools_kwargs={}
        )
    
        agent_data.prompt_ids = tokenizer.apply_chat_template(
            agent_data.messages,
            tools=self.tool_config,
            add_generation_prompt=True,
            tokenize=True,
        )
        
        for i in range(self.max_step):
            agent_data.user_turns += 1
            output = await server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params
            )
            agent_data.response_ids = output.token_ids
            agent_data.prompt_ids += agent_data.response_ids
            agent_data.response_mask += [1] * len(agent_data.response_ids)
            agent_data.assistant_turns += 1
            if output.log_probs:
                agent_data.response_logprobs += output.log_probs

            if self.tool_parser is None:
                self.tool_parser = ToolParser.get_tool_parser('hermes', tokenizer)

            # Extract tool calls
            _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
            
            if agent_data.tool_calls:
                tool_call = agent_data.tool_calls[0]
                response = await call_tool(tool_call, self.tools, 'machine_learning', work_dir)
                if tool_call.name not in self.tools:
                    add_tools = get_add_tools('machine_learning')
                    self.all_tools = self.tool_config + add_tools
                response_text = response.text
            else:
                response_text = tokenizer.decode(output.token_ids[:-1])
                return ToolResponse(text=response_text), None
            
            if response_text.startswith("Machine learning task completed successfully."):
                # add created tools
                if tool_call.name == "machine_learning_tool_creation":
                    try:
                        tool_args = json.loads(tool_call.arguments)
                        success, add_tools = await update_tools(tool_args['code'], 'machine_learning', instance_id, server_manager, tokenizer, sampling_params)
                        if success:
                            self.all_tools = self.tool_config + add_tools
                    except Exception as e:
                        print("Update tool failed due to: ", e)

                prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
                response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
                trajectory_output = AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids[: self.response_length],
                    response_mask=agent_data.response_mask[: self.response_length],
                    multi_modal_data={},
                    response_logprobs=agent_data.response_logprobs[: self.response_length],
                    num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
                    metrics=agent_data.metrics,
                    extra_fields={},
                )
                # compute turn_scores
                turn_scores = 0.1
                trajectory_output.extra_fields.update({"turn_scores": [turn_scores], "work_dir": work_dir, "init_files_hash": None, "sub_agent": True, "uid": uuid4().hex, "id": agent_data.request_id, "final": False})

                return ToolResponse(text=response_text), trajectory_output
            
            if i == self.max_step - 1:
                prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
                response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
                trajectory_output = AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids[: self.response_length],
                    response_mask=agent_data.response_mask[: self.response_length],
                    multi_modal_data={},
                    response_logprobs=agent_data.response_logprobs[: self.response_length],
                    num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
                    metrics=agent_data.metrics,
                    extra_fields={},
                )
                # compute turn_scores
                turn_scores = -0.1
                trajectory_output.extra_fields.update({"turn_scores": [turn_scores], "work_dir": work_dir, "init_files_hash": None, "sub_agent": True, "uid": uuid4().hex, "id": agent_data.request_id, "final": False})

                return ToolResponse(text=f"Machine learning task failed. Please provide more detailed information about the machine learning task or use an alternative tool to proceed with the process."), trajectory_output

            response_ids = tokenizer.apply_chat_template([{"role": "tool", "content": response_text or ""}], add_generation_prompt=True, tokenize=True)
            agent_data.prompt_ids += response_ids
            agent_data.response_mask += [0] * len(response_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(response_ids)

            if len(agent_data.response_mask) >= self.response_length:
                return ToolResponse(text=f"Machine learning task failed. The reason is the messages length exceeds the maximum length. Please provide more detailed information about the machine learning task or use an alternative tool to proceed with the process."), None

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class VisualizationTool(BaseTool):
    """Tool for visualizing results (charts, graphs, etc.)."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.system_prompt = VISUALIZER_SYS_PROMPT
        self.prompt_template = VISUALIZATION_PROMPT
        self.count = 0
        self.field_dict = {
            "response": {"default": "", "description": "Your visualization code for this problem"},
            "file_name": {"default": "", "description": "The image name of the visualization result"}
        }
        self.required_keys = ["dataset_path", "task", "saved_plot_path", "work_dir"]
        self.max_step = 3
        self.tool_parser = None
        self.response_length = 24576
        tool_config_path = 'examples/sglang_multiturn/config/tool_config/visualization_tool_config.yaml'
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_config = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

        if os.path.exists(f"examples/sglang_multiturn/config/tool_config/created_tools/visualization_tools.json"):    
            with open(f"examples/sglang_multiturn/config/tool_config/created_tools/visualization_tools.json", "r", encoding='utf-8') as f:
                created_tools = json.load(f)
        else:
            created_tools = {}
        if os.path.exists(f"examples/sglang_multiturn/config/tool_config/created_tools/visualization_tool_count.json"):    
            with open(f"examples/sglang_multiturn/config/tool_config/created_tools/visualization_tool_count.json", "r", encoding='utf-8') as f:
                tool_count = json.load(f)
        else:
            tool_count = {}
        top_k = [k for k, _ in nlargest(10, tool_count.items(), key=lambda x: x[1]) if tool_count[k] >= 3]
        add_tools = [created_tools[tool_name]['tool_config'] for tool_name in top_k]
        self.all_tools = self.tool_config + add_tools

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
            return ToolResponse(text=f"Visualization task failed. The following parameters are missing: {', '.join(missing_list)}"), None
        
        dataset_path = parameters.get("dataset_path")
        task = parameters.get("task")
        saved_plot_path = parameters.get("saved_plot_path")
        work_dir = parameters.get("work_dir")
        server_manager = kwargs.get("server_manager")
        tokenizer = kwargs.get("tokenizer")
        sampling_params = kwargs.get("sampling_params")

        prompt = self.prompt_template.format(dataset_file=dataset_path, task=task, saved_plot_file=saved_plot_path)

        agent_data = AgentData(
            messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
            image_data=None,
            metrics={},
            request_id=instance_id,
            tools_kwargs={}
        )
        agent_data.prompt_ids = tokenizer.apply_chat_template(
            agent_data.messages,
            tools=self.all_tools,
            add_generation_prompt=True,
            tokenize=True,
        )
        for i in range(self.max_step):
            agent_data.user_turns += 1
            output = await server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params
            )
            agent_data.response_ids = output.token_ids
            agent_data.prompt_ids += agent_data.response_ids
            agent_data.response_mask += [1] * len(agent_data.response_ids)
            agent_data.assistant_turns += 1
            if output.log_probs:
                agent_data.response_logprobs += output.log_probs

            if self.tool_parser is None:
                self.tool_parser = ToolParser.get_tool_parser('hermes', tokenizer)

            # Extract tool calls
            _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
            
            if agent_data.tool_calls:
                tool_call = agent_data.tool_calls[0]
                response = await call_tool(tool_call, self.tools, 'visualization', work_dir)
                if tool_call.name not in self.tools:
                    add_tools = get_add_tools('visualization')
                    self.all_tools = self.tool_config + add_tools
                response_text = response.text
            else:
                response_text = tokenizer.decode(output.token_ids[:-1])
                return ToolResponse(text=response_text), None
                
            if response_text.startswith("Visualization task completed successfully."):
                # add created tools
                if tool_call.name == "visualization_tool_creation":
                    try:
                        tool_args = json.loads(tool_call.arguments)
                        success, add_tools = await update_tools(tool_args['code'], 'visualization', instance_id, server_manager, tokenizer, sampling_params)
                        if success:
                            self.all_tools = self.tool_config + add_tools
                    except Exception as e:
                        print("Update tool failed due to: ", e)

                prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
                response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
                trajectory_output = AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids[: self.response_length],
                    response_mask=agent_data.response_mask[: self.response_length],
                    multi_modal_data={},
                    response_logprobs=agent_data.response_logprobs[: self.response_length],
                    num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
                    metrics=agent_data.metrics,
                    extra_fields={},
                )
                # compute turn_scores
                turn_scores = 0.1
                trajectory_output.extra_fields.update({"turn_scores": [turn_scores], "work_dir": work_dir, "init_files_hash": None, "sub_agent": True, "uid": uuid4().hex, "id": agent_data.request_id, "final": False})

                return ToolResponse(text=response_text), trajectory_output

            if i == self.max_step - 1:
                prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
                response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
                trajectory_output = AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids[: self.response_length],
                    response_mask=agent_data.response_mask[: self.response_length],
                    multi_modal_data={},
                    response_logprobs=agent_data.response_logprobs[: self.response_length],
                    num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
                    metrics=agent_data.metrics,
                    extra_fields={},
                )
                # compute turn_scores
                turn_scores = -0.1
                trajectory_output.extra_fields.update({"turn_scores": [turn_scores], "work_dir": work_dir, "init_files_hash": None, "sub_agent": True, "uid": uuid4().hex, "id": agent_data.request_id, "final": False})

                return ToolResponse(text=f"Visualization task failed. Please provide more detailed information about the visualization task or use an alternative tool to proceed with the process."), trajectory_output

            response_ids = tokenizer.apply_chat_template([{"role": "tool", "content": response_text or ""}], add_generation_prompt=True, tokenize=True)
            agent_data.prompt_ids += response_ids
            agent_data.response_mask += [0] * len(response_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(response_ids)

            if len(agent_data.response_mask) >= self.response_length:
                return ToolResponse(text=f"Visualization task failed. The reason is the messages length exceeds the maximum length. Please provide more detailed information about the visualization task or use an alternative tool to proceed with the process."), None
            
        # return ToolResponse(text=f"Visualization task failed. Please provide more detailed information about the visualization task or use an alternative tool to proceed with the process."), None

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class BashTool(BaseTool):
    """Tool for executing bash commands."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["code", "work_dir"]

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
            return ToolResponse(text=f"Bash command execution failed. The following parameters are missing: {', '.join(missing_list)}"), None
        
        code = parameters.get("code")
        work_dir = parameters.get("work_dir")
        
        success, output = await execute_command(code, work_dir)

        if success and (output is None or output == ''):
            return ToolResponse(text="Bash command executed successfully. No output."), None
        
        ## If observation is too long, we only keep the last ~1k tokens.
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(output)
        if len(tokens) >= 1000:
            head_tokens = tokens[:500]
            tail_tokens = tokens[-500:]
            output = enc.decode(head_tokens) + "\n...\n" + enc.decode(tail_tokens)
        if success:
            return ToolResponse(text=output), None
        else:
            return ToolResponse(text=f"Bash command execution failed. The reason is: {output}."), None

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class PythonTool(BaseTool):
    """Tool for executing Python code."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["code", "file_path", "work_dir"]
        

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
            return ToolResponse(text=f"Python script execution failed. The following parameters are missing: {', '.join(missing_list)}"), None
        
        code = parameters.get("code")
        file_path = parameters.get("file_path")
        work_dir = parameters.get("work_dir")

        try:
            write_program(code, os.path.join(work_dir, file_path))
            success, output = await execute_script(file_path, work_dir)
        except Exception as e:
            return ToolResponse(text=f"Python script execution failed. The reason is: {str(e)}."), None

        if success:
            if output is None or output == '':
                return ToolResponse(text="Python script executed successfully. No output."), None
            return ToolResponse(text=output), None
        else:
            return ToolResponse(text=f"Python script execution failed. The reason is: {output}."), None

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class SqlTool(BaseTool):
    """Tool for executing SQL commands."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.required_keys = ["code", "dataset_path", "output_file", "work_dir"]

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
            return ToolResponse(text=f"SQL command execution failed. The following parameters are missing: {', '.join(missing_list)}"), None
        
        code = parameters.get("code")
        dataset_path = parameters.get("dataset_path")
        output_file = parameters.get("output_file")
        work_dir = parameters.get("work_dir")

        try:
            obs = await _execute_sql_code(dataset_path, code, output_file, work_dir)
        except Exception as e:
            return ToolResponse(text=f"SQL command execution failed. The reason is: {str(e)}."), None

        if obs is None or obs == "":
            return ToolResponse(text="SQL command executed successfully. No output."), None
        else:
            ## If observation is too long, we only keep the last ~1k tokens.
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(obs)
            if len(tokens) >= 1000:
                head_tokens = tokens[:500]
                tail_tokens = tokens[-500:]
                obs = enc.decode(head_tokens) + "\n...\n" + enc.decode(tail_tokens)
            return ToolResponse(text=obs), None

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class DebuggingTool(BaseTool):
    """Tool for debugging code."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.system_prompt = DEBUG_AGENT_SYS_PROMPT
        self.count = 0
        self.max_step = 3
        self.response_length = 24576
        self.required_keys = ["task", "code", "work_dir"]

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
            return ToolResponse(text=f"Debugging task failed. The following parameters are missing: {', '.join(missing_list)}"), None
        
        task = parameters.get("task")
        code = parameters.get("code")
        work_dir = parameters.get("work_dir")
        server_manager = kwargs.get("server_manager")
        tokenizer = kwargs.get("tokenizer")
        sampling_params = kwargs.get("sampling_params")

        if code.endswith('.py'):
            file_path = code
            success, observation = await execute_script(file_path, work_dir)
            with open(os.path.join(work_dir, file_path), 'r') as f:
                code = f.read()
        else:
            self.count += 1
            file_path = f'debug_{self.count}.py'
            write_program(code, os.path.join(work_dir, file_path))
            success, observation = await execute_script(file_path, work_dir)
            
        prompt = f"Task: {task}\nCode:\n```python\n{code}\n```\n\n" + CODE_DEBUG_PROMPT.format(observation=observation)
        
        agent_data = AgentData(
            messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
            image_data=None,
            metrics={},
            request_id=instance_id,
            tools_kwargs={}
        )
    
        agent_data.prompt_ids = tokenizer.apply_chat_template(
            agent_data.messages,
            tools=None,
            add_generation_prompt=True,
            tokenize=True,
        )
        
        for i in range(self.max_step):
            agent_data.user_turns += 1
            output = await server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params
            )
            agent_data.response_ids = output.token_ids
            agent_data.prompt_ids += agent_data.response_ids
            agent_data.response_mask += [1] * len(agent_data.response_ids)
            agent_data.assistant_turns += 1
            if output.log_probs:
                agent_data.response_logprobs += output.log_probs

            response_text = tokenizer.decode(output.token_ids)
            self.count += 1
            write_program(response_text, os.path.join(work_dir, f"debug_{self.count}.py"))
            success, output = await execute_script(f"debug_{self.count}.py", work_dir)
            try:
                code = re.search(r"```python(.*?)```", response_text, re.DOTALL).group(1).strip()
            except:
                code = ""
            if success:
                self.count = 0
                prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
                response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
                trajectory_output = AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids[: self.response_length],
                    response_mask=agent_data.response_mask[: self.response_length],
                    multi_modal_data={},
                    response_logprobs=agent_data.response_logprobs[: self.response_length],
                    num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
                    metrics=agent_data.metrics,
                    extra_fields={},
                )
                # compute turn_scores
                turn_scores = 0.1
                trajectory_output.extra_fields.update({"turn_scores": [turn_scores], "work_dir": work_dir, "init_files_hash": None, "sub_agent": True, "uid": uuid4().hex, "id": agent_data.request_id, "final": False})
                
                return ToolResponse(text=f"The refined code is as follows: \n\n```python\n{code}\n```\n\nThe code has been successfully executed. The execution result is as follows: \n{output}"), trajectory_output

            if i == self.max_step - 1:
                self.count = 0
                prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
                response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
                trajectory_output = AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids[: self.response_length],
                    response_mask=agent_data.response_mask[: self.response_length],
                    multi_modal_data={},
                    response_logprobs=agent_data.response_logprobs[: self.response_length],
                    num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
                    metrics=agent_data.metrics,
                    extra_fields={},
                )
                # compute turn_scores
                turn_scores = -0.1
                trajectory_output.extra_fields.update({"turn_scores": [turn_scores], "work_dir": work_dir, "init_files_hash": None, "sub_agent": True, "uid": uuid4().hex, "id": agent_data.request_id, "final": False})
                
                return ToolResponse(text=f"The refined code is as follows: \n\n```python\n{code}\n```\n\nThe code has failed to execute. The error message is as follows: \n{output}"), trajectory_output

            response_ids = tokenizer.apply_chat_template([{"role": "user", "content": CODE_DEBUG_PROMPT.format(observation=output)}], add_generation_prompt=True, tokenize=True)
            agent_data.prompt_ids += response_ids
            agent_data.response_mask += [0] * len(response_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(response_ids)

            if len(agent_data.response_mask) >= self.response_length:
                return ToolResponse(text=f"Debugging task failed. The reason is the messages length exceeds the maximum length. Please provide more detailed information about the debugging task or use an alternative tool to proceed with the process."), None
        # return ToolResponse(text=f"The refined code is as follows: \n\n```python\n{code}\n```\n\nThe code has failed to execute. The error message is as follows: \n{output}"), None

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class ContextSummarizeTool(BaseTool):
    """Tool for context summarize."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"status": "initialized"}
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, AgentLoopOutput]:
        return ToolResponse(text=CONTEXT_SUMMARIZE_PROMPT), None

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
from typing import List, Optional
import subprocess
import tiktoken
import re
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import base64
import sys
import json
import time
import shlex
import numpy as np
import torch
import ast
from pathlib import Path
from heapq import nlargest
import json5
from llm.async_llm import LLMsConfig, create_llm_instance
from datasets import load_dataset
import aiofiles
import pandas as pd
from utils.prompt import Tool_Configuration_Extraction_PROMPT

# os.environ["HTTP_PROXY"] = "http://127.0.0.1:20171"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:20171"


SQL_TEMPLATE = """
import sqlite3
import pandas as pd
import os

def execute_sql(dataset_path, command, output_path):
    # make sure the file path is correct
    if not os.path.exists(dataset_path):
        print(f"ERROR: File not found: {{dataset_path}}")
        return

    # Connect to the SQLite database
    conn = sqlite3.connect(dataset_path)
    
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
dataset_path = "{dataset_path}"  # Path to your SQLite database file
command = "{code}"             # SQL command to be executed
output_path = "{output}" # Path to save the output as a CSV or "directly"

execute_sql(dataset_path, command, output_path)

"""


class EnvException(Exception):
    def __init__(self, message):
        self.message = message 
    def __str__(self):
        return self.message
    

def execute_script(script_path, work_dir):
    try:
        python = sys.executable
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES") or str(np.arange(torch.cuda.device_count()).tolist())[1:-1]
        cmd_list = [python, "-u", script_path]
        try:
            exec_res = subprocess.run(cmd_list, shell=False, capture_output=True, timeout=1800, text=True, cwd=work_dir, env=my_env)
        except subprocess.TimeoutExpired:
            return False, "Timeout"

        if exec_res.returncode != 0:
            raw = exec_res.stderr
        else:
            raw = exec_res.stdout
        
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(raw)
        if len(tokens) >= 1000:
            head = enc.decode(tokens[:500])
            tail = enc.decode(tokens[-500:])
            output = f"{head}\n...\n{tail}"
        else:
            output = raw
        
        return exec_res.returncode == 0, output

    except Exception as e:
        raise EnvException(f"Something went wrong in executing {script_path}: {e}. Please check if it is ready to be executed.")
    

def exec_run(cmd, workdir: str):
    try:
        cmd_list = shlex.split(cmd)
        exec_res = subprocess.run(cmd_list, shell=False, capture_output=True, timeout=1800, text=True, cwd=workdir)
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    
    if exec_res.returncode != 0:
        return False, exec_res.stderr
    else:
        return True, exec_res.stdout
    
def execute_sql_code(dataset_path, code, output: str, work_dir: str) -> str:
    script_content = SQL_TEMPLATE.format(dataset_path=dataset_path, code=code, output=output)
    temp_file_path = f"temp_sql_script.py"
    script_content = script_content.replace('"', '\\"').replace('`', '\\`').replace('$', '\\$')
    with open(os.path.join(work_dir, temp_file_path), "w+", encoding="utf-8") as f:
        f.write(script_content)
    write_program(script_content, os.path.join(work_dir, temp_file_path))
    _, observation = execute_script(temp_file_path, work_dir)

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

def execute_command(command: str, work_dir: str):
    cmd = command
    exit_code, output = exec_run(cmd, workdir=work_dir)
    if "venv" in command or "conda create" in command:
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


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


async def image_evaluate(image_path, ground_truth_path, max_try=5):
    score = 0
    success = False
    delays = [30, 90, 180, 180, 180]
    models_config = LLMsConfig.default()
    llm_config = models_config.get('gpt-4o')
    llm_gpt = create_llm_instance(llm_config)
    
    if not os.path.exists(f'{image_path}'):
        return score
    else:
        base64_image1 = encode_image(image_path)
        base64_image2 = encode_image(ground_truth_path)

        for i in range(max_try):
            messages=[
                {
                  "role": "user",
                  "content": [
                    {
                      "type": "text",
                      "text": f'''You are an excellent judge at evaluating visualization plots between a model generated plot and the ground truth. You will be giving scores on how well it matches the ground truth plot.

                       The generated plot will be given to you as the first figure.
                       Another plot will be given to you as the second figure, which is the desired outcome of the user query, meaning it is the ground truth for you to reference.
                       Please compare the two figures head to head and rate them.
                       Suppose the second figure has a score of 100, rate the first figure on a scale from 0 to 100.
                       Scoring should be carried out in the following aspect:
                       1. Plot correctness: 
                       Compare closely between the generated plot and the ground truth, the more resemblance the generated plot has compared to the ground truth, the higher the score. The score should be proportionate to the resemblance between the two plots.
                       In some rare occurrence, see if the data points are generated randomly according to the query, if so, the generated plot may not perfectly match the ground truth, but it is correct nonetheless.
                       Only rate the first figure, the second figure is only for reference.
                       If the first figure is blank, that means the code failed to generate a figure. Give a score of 0 on the Plot correctness.
                        After scoring from the above aspect, please give a final score. The final score is preceded by the [FINAL SCORE] token.
                       For example [FINAL SCORE]: 40.''',
                    },
                    {
                      "type": "image_url",
                      "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image1}",
                      },
                    },
                    {
                      "type": "image_url",
                      "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image2}",
                      },
                    },
                  ],
                }
              ]
            try:
                choice = await llm_gpt.generate(messages)
                result = choice.message.content
                match = re.search(r"\[FINAL SCORE\]: (\d+)", result)
                if match:
                    success = True
                    score = int(match.group(1))
                    break
            except:
                time.sleep(delays[i])
                continue
        if not success:
            raise Exception("Failed to evaluate image after {} tries".format(max_try))
        time.sleep(10)
    return score / 100
    

def extract_tool_details(source_code):
    """
    从代码中提取 import 语句和函数定义，并将它们组合在一起。
    """
    try:
        tree = ast.parse(source_code)
        lines = source_code.splitlines()
        
        import_statements = []
        function_nodes = []

        # 第一次遍历：识别所有的 import 和 函数定义
        for node in tree.body:
            # 处理 import os 或 from math import ...
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_code = "\n".join(lines[node.lineno - 1 : node.end_lineno])
                import_statements.append(import_code)
            
            # 处理函数定义
            elif isinstance(node, ast.FunctionDef):
                function_nodes.append(node)

        # 合并所有的 import 语句为一个字符串
        full_imports = "\n".join(import_statements)
        
        results = []
        # 第二次遍历：为每个函数生成包含 import 的完整代码块
        for node in function_nodes:
            func_name = node.name
            func_body_code = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            
            # 组合：Import 部分 + 函数部分
            # 这样提取出的 tool_function 可以直接独立运行
            combined_code = f"{full_imports}\n\n{func_body_code}" if full_imports else func_body_code
            
            results.append({
                "tool_name": func_name,
                "tool_function": combined_code
            })
        
        return results
    except Exception as e:
        return {"error": str(e)}


def validate_tool_config(tool_config: dict) -> None:
    """
    Validate whether the given tool_config is a valid function-style tool configuration.

    Raises:
        ValueError: if the tool_config is invalid.
    """

    # ---------- Top-level checks ----------
    if not isinstance(tool_config, dict):
        raise ValueError("Tool config must be a dictionary.")

    if tool_config.get("type") != "function":
        raise ValueError("Tool config must have type == 'function'.")

    if "function" not in tool_config:
        raise ValueError("Tool config must contain a 'function' field.")

    function_block = tool_config["function"]
    if not isinstance(function_block, dict):
        raise ValueError("'function' must be a dictionary.")

    # ---------- Function metadata ----------
    name = function_block.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Function name must be a non-empty string.")

    description = function_block.get("description")
    if not isinstance(description, str) or not description:
        raise ValueError("Function description must be a non-empty string.")

    # ---------- Parameters block ----------
    parameters = function_block.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("'parameters' must be a dictionary.")

    if parameters.get("type") != "object":
        raise ValueError("'parameters.type' must be 'object'.")

    required = parameters.get("required")
    properties = parameters.get("properties")

    if not isinstance(required, list):
        raise ValueError("'parameters.required' must be a list.")

    if not isinstance(properties, dict):
        raise ValueError("'parameters.properties' must be a dictionary.")

    # ---------- Properties validation ----------
    for param_name, param_spec in properties.items():
        if not isinstance(param_name, str) or not param_name:
            raise ValueError("Parameter names must be non-empty strings.")

        if not isinstance(param_spec, dict):
            raise ValueError(f"Specification for parameter '{param_name}' must be a dictionary.")

        param_type = param_spec.get("type")
        if not isinstance(param_type, str) or not param_type:
            raise ValueError(f"Parameter '{param_name}' must have a valid 'type' field.")

        param_desc = param_spec.get("description")
        if not isinstance(param_desc, str) or not param_desc:
            raise ValueError(f"Parameter '{param_name}' must have a non-empty 'description'.")

    # ---------- Required parameters consistency ----------
    for req_param in required:
        if req_param not in properties:
            raise ValueError(
                f"Required parameter '{req_param}' is not defined in properties."
            )

    # ---------- Passed all checks ----------
    return None


def execute_tool(code, tool_name, arguments, domain, work_dir):
    current_dir = Path.cwd()
    if domain == 'data_clean':
        domain = 'Data cleaning'
    elif domain == 'feature_engineering':
        domain = 'Feature engineering'
    elif domain == 'modeling':
        domain = 'Machine learning'
    elif domain == 'visualization':
        domain = 'Visualization'
    try:
        workdir_path = Path(work_dir).resolve()
        os.chdir(workdir_path)

        exec_context = {}
        exec(code, exec_context, exec_context)
        my_tool_func = exec_context[tool_name]
        try:
            raw = my_tool_func(**arguments)
        except:
            raw = my_tool_func(arguments)

        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(raw)
        if len(tokens) >= 1000:
            head = enc.decode(tokens[:500])
            tail = enc.decode(tokens[-500:])
            output = f"{head}\n...\n{tail}"
        else:
            output = raw
        
        return f"{domain} task completed successfully. The output is as follows: \n{output}"
    except:
        return f"{domain} task failed. Try again."
    finally:
        os.chdir(current_dir)


async def update_tools(code, dataset, domain, llm_config, num=10):
    llm = create_llm_instance(llm_config)
    # add tools
    if os.path.exists(f"utils/{dataset}/created_tools/{domain}_tools.json"):    
        with open(f"utils/{dataset}/created_tools/{domain}_tools.json", "r", encoding='utf-8') as f:
            tools = json.load(f)
    else:
        tools = {}
    if os.path.exists(f"utils/{dataset}/created_tools/{domain}_tool_count.json"):    
        with open(f"utils/{dataset}/created_tools/{domain}_tool_count.json", "r", encoding='utf-8') as f:
            tool_count = json.load(f)
    else:
        tool_count = {}
    results = extract_tool_details(code)
    if type(results) == dict:
        return
    else:
        messages_list = []
        for result in results:
            messages = [{"role": "user", "content": Tool_Configuration_Extraction_PROMPT.format(code=result["tool_function"])}]
            choice = await llm.generate(messages)
            content = choice.message.content
            if '</think>' in content:
                content = content.split('</think>')[-1]
            match = re.search(r"```json(.*?)```", content, re.DOTALL)
            if match:
                tool_config = match.group(1).strip()
            else:
                tool_config = content.strip()
            tool_config = json5.loads(tool_config)
            try:
                validate_tool_config(tool_config)
            except ValueError as e:
                print(f"Invalid tool config: {e}")
                return False, None, None

            messages.append({"role": "assistant", "content": content})
            messages_list.append(messages)
            tool_name = result["tool_name"]
            if tool_name in tools:
                tool_count[tool_name] += 1
            else:
                tool_count[tool_name] = 1
            # tool may not the same, update in the future
            tools[tool_name] = {"code": result["tool_function"], "tool_config": tool_config}

        top_k = [k for k, _ in nlargest(num, tool_count.items(), key=lambda x: x[1]) if tool_count[k] >= 3]
        add_tools = [tools[tool_name]['tool_config'] for tool_name in top_k]
        with open(f"utils/{dataset}/created_tools/{domain}_tools.json", "w", encoding='utf-8') as f:
            json.dump(tools, f, indent=4, ensure_ascii=False)
        with open(f"utils/{dataset}/created_tools/{domain}_tool_count.json", "w", encoding='utf-8') as f:
            json.dump(tool_count, f, indent=4, ensure_ascii=False)
    
    return True, add_tools, messages_list


def update_tool_num(dataset, domain, tool_name, num=10):
    with open(f"utils/{dataset}/created_tools/{domain}_tool_count.json", "r", encoding='utf-8') as f:
        tool_count = json.load(f)
    tool_count[tool_name] += 1
    with open(f"utils/{dataset}/created_tools/{domain}_tool_count.json", "w", encoding='utf-8') as f:
        json.dump(tool_count, f, indent=4, ensure_ascii=False)
    with open(f"utils/{dataset}/created_tools/{domain}_tools.json", "r", encoding='utf-8') as f:
        tools = json.load(f)
    top_k = [k for k, _ in nlargest(num, tool_count.items(), key=lambda x: x[1]) if tool_count[k] >= 3]
    add_tools = [tools[tool_name]['tool_config'] for tool_name in top_k]
    
    return add_tools


async def load_data(dataset) -> List[dict]:
    if dataset == 'SAB':
        dataset = load_dataset("osunlp/ScienceAgentBench", split="validation")
        data = [item for item in dataset]
        return data
    elif dataset == 'DSBench':
        data = []
        with open("datasets/dsbench.jsonl", "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    elif dataset.startswith("datamind"):
        data = []
        with open(f"datasets/datamind.jsonl", "r") as f:
            for line in f:
                data_item = json.loads(line.strip())
                if dataset == "datamind_sql" and data_item["source"] == "darl/sql":
                    id_data = pd.read_csv('selected_ids/datamind_sql.txt', header=None, names=['task_id'])
                    id_list = list(id_data['task_id'])
                    if data_item['task_id'] in id_list:
                        data.append(data_item)
                elif dataset == "datamind_python" and data_item["source"] == "darl/python":
                    id_data = pd.read_csv('selected_ids/datamind_python.txt', header=None, names=['task_id'])
                    id_list = list(id_data['task_id'])
                    if data_item['task_id'] in id_list:
                        data.append(data_item)
        return data
    elif dataset == 'datascience_instruct':
        data = []
        id_data = pd.read_csv('selected_ids/datascience_instruct.txt', header=None, names=['task_id'])
        id_list = list(id_data['task_id'])
        with open(f"datasets/datascience_instruct.jsonl", "r") as f:
            for line in f:
                data_item = json.loads(line)
                if data_item['task_id'] in id_list:
                    data.append(data_item)
        return data
    elif dataset == 'matplotbench':
        data = []
        with open('verl/MatPlotBench/benchmark_instructions.json', 'r') as f:
            data = json.load(f)
        return data
    file_path = f"datasets/{dataset.lower()}.jsonl"
    data = []
    async with aiofiles.open(file_path, mode="r", encoding="utf-8") as file:
        async for line in file:
            data.append(json.loads(line))

    return data

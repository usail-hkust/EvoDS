import json
import json5
import os
import ast
from pathlib import Path
import re
from heapq import nlargest
from verl.experimental.agent_loop.data_science_agent_loop import AgentData


Tool_Configuration_Extraction_PROMPT = """You are a **Tool Configuration Extraction Assistant**.

You are provided with the source code of a single function as:
{code}

Your task is to analyze the function definition and implementation, and extract a corresponding tool configuration strictly following the format below:

```json
{{
  "type": "function",
  "function": {{
    "name": "tool name",
    "description": "tool description.",
    "parameters": {{
      "type": "object",
      "required": ["required parameters"],
      "properties": {{
        "parameter name": {{
          "type": "parameter type",
          "description": "parameter description"
        }}
      }}
    }}
  }}
}}
```

**Extraction Rules**
The tool name must exactly match the function name.
The tool description must concisely describe the purpose and behavior of the function.
Parameters must be inferred from the function signature.
Required parameters are those without default values.
Optional parameters are those with default values.

**Output Requirements**
Output **only** the tool configuration in valid JSON as:
```json
content.
```
Do NOT include explanations, comments, or markdown.
Do NOT include any content outside the JSON object.
Follow the exact schema and field names shown above.

**Additional Constraints**
Do NOT invent parameters that are not present in the function.
Do NOT omit parameters defined in the function signature.
If the function takes no parameters, use:

```
"required": [],
"properties": {{}}
```

Generate the tool configuration now."""


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
            output = my_tool_func(**arguments)
        except:
            output = my_tool_func(arguments)
        return f"{domain} task completed successfully. The output is as follows: \n{output}"
    except:
        return f"{domain} task failed. Try again."
    finally:
        os.chdir(current_dir)


async def update_tools(code, domain, instance_id: str, server_manager, tokenizer, sampling_params, num=10):
    # add tools
    if os.path.exists(f"examples/sglang_multiturn/config/tool_config/created_tools/{domain}_tools.json"):    
        with open(f"examples/sglang_multiturn/config/tool_config/created_tools/{domain}_tools.json", "r", encoding='utf-8') as f:
            tools = json.load(f)
    else:
        tools = {}
    if os.path.exists(f"examples/sglang_multiturn/config/tool_config/created_tools/{domain}_tool_count.json"):    
        with open(f"examples/sglang_multiturn/config/tool_config/created_tools/{domain}_tool_count.json", "r", encoding='utf-8') as f:
            tool_count = json.load(f)
    else:
        tool_count = {}
    results = extract_tool_details(code)
    if type(results) == dict:
        return
    else:
        for result in results:
            messages = [{"role": "user", "content": Tool_Configuration_Extraction_PROMPT.format(code=result["tool_function"])}]
            agent_data = AgentData(
                messages=messages,
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
            output = await server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params
            )
            content = tokenizer.decode(output.token_ids[:-1])
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
                return False, None

            tool_name = result["tool_name"]
            if tool_name in tools:
                tool_count[tool_name] += 1
            else:
                tool_count[tool_name] = 1
            # tool may not the same, update in the future
            tools[tool_name] = {"code": result["tool_function"], "tool_config": tool_config}

        top_k = [k for k, _ in nlargest(num, tool_count.items(), key=lambda x: x[1]) if tool_count[k] >= 3]
        add_tools = [tools[tool_name]['tool_config'] for tool_name in top_k]
        with open(f"examples/sglang_multiturn/config/tool_config/created_tools/{domain}_tools.json", "w", encoding='utf-8') as f:
            json.dump(tools, f, indent=4, ensure_ascii=False)
        with open(f"examples/sglang_multiturn/config/tool_config/created_tools/{domain}_tool_count.json", "w", encoding='utf-8') as f:
            json.dump(tool_count, f, indent=4, ensure_ascii=False)
    
    return True, add_tools


def update_tool_num(domain, tool_name):
    with open(f"examples/sglang_multiturn/config/tool_config/created_tools/{domain}_tool_count.json", "r", encoding='utf-8') as f:
        tool_count = json.load(f)
    tool_count[tool_name] += 1
    with open(f"examples/sglang_multiturn/config/tool_config/created_tools/{domain}_tool_count.json", "w", encoding='utf-8') as f:
        json.dump(tool_count, f, indent=4, ensure_ascii=False)


def get_add_tools(domain, num=10):
    with open(f"examples/sglang_multiturn/config/tool_config/created_tools/{domain}_tool_count.json", "r", encoding='utf-8') as f:
        tool_count = json.load(f)
    with open(f"examples/sglang_multiturn/config/tool_config/created_tools/{domain}_tools.json", "r", encoding='utf-8') as f:
        tools = json.load(f)
    top_k = [k for k, _ in nlargest(num, tool_count.items(), key=lambda x: x[1]) if tool_count[k] >= 3]
    add_tools = [tools[tool_name]['tool_config'] for tool_name in top_k]
    
    return add_tools
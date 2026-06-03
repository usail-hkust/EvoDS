from agents.manager import Manager
from agents.data_cleaner import DataCleaner
from agents.feature_enginner import FeatureEnginner
from agents.model_developer import ModelAgent
from agents.visualizer import Visualizer
from agents.bash import BashAgent
from agents.sql import SqlAgent
from agents.python import PythonAgent
from agents.debugger import DebugAgent
from agents.context_summarizer import ContextSummarize
from utils.data_cleaning import data_cleaning_tools
from utils.feature_engineering import feature_engineering_tools
from utils.model_development import machine_learning_tools
from utils.visualization import visualization_tools
from utils.prompt import CONTINUE_PROMPT
import os
import aiofiles, asyncio
import json
from uuid import uuid4
from transformers import AutoTokenizer



class EvoDS():
    def __init__(self, llm_config, tools, max_steps, dataset, add_tool_nums, context_tokens):
        self.tools = tools
        self.max_steps = max_steps
        self.context_tokens = context_tokens
        self.manager = Manager(llm_config, tools, max_steps)
        self.data_cleaner = DataCleaner(llm_config, data_cleaning_tools, dataset, add_tool_nums)
        self.feature_enginner = FeatureEnginner(llm_config, feature_engineering_tools, dataset, add_tool_nums)
        self.model_agent = ModelAgent(llm_config, machine_learning_tools, dataset, add_tool_nums)
        self.visualizer = Visualizer(llm_config, visualization_tools, dataset, add_tool_nums)
        self.bash_agent = BashAgent()
        self.sql_agent = SqlAgent()
        self.python_agent = PythonAgent()
        self.debug_agent = DebugAgent(llm_config)
        self.context_summarize_tool = ContextSummarize()
        self.tool_map = {
            "data_cleaning": self.data_cleaning,
            "feature_engineering": self.feature_engineering,
            "model_development": self.model_development,
            "visualization": self.visualization,
            "bash": self.bash,
            "sql": self.sql,
            "python": self.python,
            "debugging": self.debugging,
            "context_summarize": self.context_summarize,
        }
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.logs = []
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_config.model)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen3-8B")

    async def data_cleaning(self, dataset_path: str, task: str, global_task: str, work_dir: str):
        dataset_path = os.path.normpath(dataset_path)
        if "data_cleaning" not in self.function_call_count:
            self.function_call_count["data_cleaning"] = 0
        self.function_call_count["data_cleaning"] += 1
        return await self.data_cleaner(dataset_path, task, global_task, work_dir)

    async def feature_engineering(self, dataset_path: str, task: str, global_task: str, work_dir: str):
        dataset_path = os.path.normpath(dataset_path)
        if "feature_engineering" not in self.function_call_count:
            self.function_call_count["feature_engineering"] = 0
        self.function_call_count["feature_engineering"] += 1
        return await self.feature_enginner(dataset_path, task, global_task, work_dir)

    async def model_development(self, train_dataset_path: str, test_dataset_path, task: str, global_task: str, work_dir: str):
        train_dataset_path = os.path.normpath(train_dataset_path)
        test_dataset_path = os.path.normpath(test_dataset_path)
        if "model_development" not in self.function_call_count:
            self.function_call_count["model_development"] = 0
        self.function_call_count["model_development"] += 1
        return await self.model_agent(train_dataset_path, test_dataset_path, task, global_task, work_dir)

    async def visualization(self, dataset_path: str, task: str, saved_plot_path: str, global_task: str, work_dir: str):
        dataset_path = os.path.normpath(dataset_path)
        saved_plot_path = os.path.normpath(saved_plot_path)
        if "visualization" not in self.function_call_count:
            self.function_call_count["visualization"] = 0
        self.function_call_count["visualization"] += 1
        return await self.visualizer(dataset_path, task, saved_plot_path, global_task, work_dir)

    def bash(self, code: str, global_task: str, work_dir: str):
        if "bash" not in self.function_call_count:
            self.function_call_count["bash"] = 0
        self.function_call_count["bash"] += 1
        return self.bash_agent(code, work_dir)

    def sql(self, code: str, dataset_path: str, output_file: str, global_task: str, work_dir: str):
        if "sql" not in self.function_call_count:
            self.function_call_count["sql"] = 0
        self.function_call_count["sql"] += 1
        return self.sql_agent(code, dataset_path, output_file, work_dir)

    def python(self, code: str, file_path: str, global_task: str, work_dir: str):
        if "python" not in self.function_call_count:
            self.function_call_count["python"] = 0
        self.function_call_count["python"] += 1
        return self.python_agent(code, file_path, work_dir)

    async def debugging(self, task: str, code: str, global_task: str, work_dir: str):
        if "debugging" not in self.function_call_count:
            self.function_call_count["debugging"] = 0
        self.function_call_count["debugging"] += 1
        return await self.debug_agent(task, code, work_dir)

    async def context_summarize(self, global_task: str, work_dir: str):
        return await self.context_summarize_tool()


    async def save_messages_async(self, output_dir: str):
        dir_path = os.path.join(output_dir, 'messages')
        # os.makedirs(dir_path, exist_ok=True)

        # 统一写文件协程
        async def _write_one(file_name, history):
            async with aiofiles.open(os.path.join(dir_path, file_name), 'w', encoding='utf-8') as f:
                await f.write(json.dumps(history, indent=4, ensure_ascii=False))

        # 批量并发写
        tasks = []
        if 'data_cleaning' in self.function_call_count:
            tasks.append(_write_one("data_cleaning.json", self.data_cleaner.history))
        if 'feature_engineering' in self.function_call_count:
            tasks.append(_write_one("feature_engineering.json", self.feature_enginner.history))
        if 'model_development' in self.function_call_count:
            tasks.append(_write_one("model_development.json", self.model_agent.history))
        if 'visualization' in self.function_call_count:
            tasks.append(_write_one("visualization.json", self.visualizer.history))
        if 'debugging' in self.function_call_count:
            tasks.append(_write_one("debugging.json", self.debug_agent.history))

        await asyncio.gather(*tasks)


    def get_usage_summary(self):
        """Get a summary of token usage and costs"""
        for agent in [self.manager, self.data_cleaner, self.feature_enginner, self.model_agent, self.visualizer, self.debug_agent]:
            self.total_input_tokens += agent.llm.usage_tracker.total_input_tokens
            self.total_output_tokens += agent.llm.usage_tracker.total_output_tokens
        
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
        }
    
    def reset(self):
        """Reset token usage and cost counters"""
        for agent in [self.manager, self.data_cleaner, self.feature_enginner, self.model_agent, self.visualizer, self.debug_agent]:
            agent.llm.reset_usage()
            agent.history = []
            agent.count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.data_cleaner.count = 0
        self.feature_enginner.count = 0
        self.model_agent.count = 0
        self.visualizer.count = 0
        self.function_call_count = {}
        self.logs = []

    def get_tokens(self, messages, system_prompt, model):
        token_num = len(self.tokenizer.apply_chat_template([{'role': 'system', 'content': system_prompt}], tokenize=True, add_generation_prompt=True))
        token_num += len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
        return token_num

    async def generate_output(self, problem, work_dir):
        input_text = f"# GLOBAL TASK CONTEXT #\n{problem}"
        logs = []
        messages = [{"role": "user", "content": input_text}]
        logs.append([{"role": "system", "content": self.manager.system_prompt}])
        finish_reason = None
        done = False
        result = ''
        final_result = ''
        repeat_num = 0
        last_tool_call = ''
        count = 0
        os.makedirs(os.path.join(work_dir, 'messages'), exist_ok=True)
        try:
            is_summarize = False
            for i in range(self.max_steps):
                logs.append(messages.copy())
                token_num = self.get_tokens(messages, self.manager.system_prompt, self.manager.llm.config.model)
                if token_num > self.context_tokens and not is_summarize:
                    tool_call_id = uuid4().hex
                    sum_mes = {'content': "The message is too long, let's summarize it first before continuing.", 'role': 'assistant', 'tool_calls': [{'id': tool_call_id, 'function': {'arguments': '{}', 'name': 'context_summarize'}, 'type': 'function'}]}
                    logs[-1].append(sum_mes)
                    messages.append(sum_mes)
                    tool_function = self.tool_map['context_summarize']
                    tool_result = await tool_function(global_task=input_text, work_dir=work_dir)
                    tool_message = {"role": 'user', "content": json.dumps({'result': tool_result})}
                    is_summarize = True
                    messages.append(tool_message)
                    logs[-1].append(tool_message)
                    continue
                choice = await self.manager.action(messages)
                logs[-1].append(dict(choice.message))
                if is_summarize:
                    finish_reason = 'stop'
                else:
                    finish_reason = choice.finish_reason
                if finish_reason == "tool_calls":
                    if repeat_num >= 2:
                        messages = [{"role": "user", "content": input_text}]
                        self.manager.update_system_prompt(i)
                    else:
                        messages.append(dict(choice.message))
                    for tool_call in choice.message.tool_calls:
                        tool_call_name = tool_call['function']['name']
                        try:
                            tool_call_arguments = json.loads(tool_call['function']['arguments'])
                            tool_function = self.tool_map[tool_call_name]
                            if last_tool_call == tool_call_name + '\n'+ str(tool_call_arguments):
                                repeat_num += 1
                            else:
                                repeat_num = 0
                            last_tool_call = tool_call_name + '\n'+ str(tool_call_arguments)
                            tool_result = await tool_function(**tool_call_arguments, global_task=input_text, work_dir=work_dir)
                        except Exception as e:
                            tool_result = f"Tool call failed due to {str(e)}"
                        print("tool_result:", tool_result)
                        if tool_call_name == 'context_summarize':
                            tool_message = {
                                "role": 'user',
                                "content": json.dumps({'result': tool_result}),
                            }
                            is_summarize = True
                        else:
                            tool_message = {
                                "role": 'tool',
                                "tool_call_id": tool_call['id'],
                                "name": tool_call_name,
                                "content": json.dumps({'result': tool_result}),
                            }
                        messages.append(tool_message)
                        logs[-1].append(tool_message)
                else:
                    messages.append(dict(choice.message))
                    result = choice.message.content
                    if is_summarize:
                        count += 1
                        async with aiofiles.open(os.path.join(work_dir, 'messages', f"messages_summarize_{count}.json"), "w", encoding="utf-8") as f:
                            await f.write(json.dumps(messages, ensure_ascii=False, indent=4))
                        is_summarize = False
                        self.manager.update_system_prompt(i)
                        messages = [{"role": "user", "content": CONTINUE_PROMPT.format(task=input_text, summary=result)}]
                        continue
                    else:
                        done = True
                        break
        except Exception as e:
            print(f"Error in generate_output: {str(e)}")
            result = f"Error: {str(e)}"
            done = False

        async with aiofiles.open(os.path.join(work_dir, "messages.json"), "w", encoding="utf-8") as f:
            await f.write(json.dumps(messages, ensure_ascii=False, indent=4))
        usage = self.get_usage_summary()
        async with aiofiles.open(os.path.join(work_dir, "usage.json"), "w", encoding="utf-8") as f:
            await f.write(json.dumps(usage))
        async with aiofiles.open(os.path.join(work_dir, "logs.json"), "w", encoding="utf-8") as f:
            await f.write(json.dumps(logs, indent=4, ensure_ascii=False))
        await self.save_messages_async(work_dir)

        return done, result
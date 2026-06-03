from agents.base_agent import BaseAgent
from utils.model_development import *
import json
from heapq import nlargest
from utils.util import *



class ModelAgent(BaseAgent):
    def __init__(self, llm_config, tools, dataset, add_tool_nums, max_steps=3):
        self.tools = tools
        self.dataset = dataset
        self.add_tool_nums = add_tool_nums
        if os.path.exists(f"utils/{self.dataset}/created_tools/modeling_tools.json"):    
            with open(f"utils/{self.dataset}/created_tools/modeling_tools.json", "r", encoding='utf-8') as f:
                created_tools = json.load(f)
        else:
            created_tools = {}
        if os.path.exists(f"utils/{self.dataset}/created_tools/modeling_tool_count.json"):    
            with open(f"utils/{self.dataset}/created_tools/modeling_tool_count.json", "r", encoding='utf-8') as f:
                tool_count = json.load(f)
        else:
            tool_count = {}
        top_k = [k for k, _ in nlargest(self.add_tool_nums, tool_count.items(), key=lambda x: x[1]) if tool_count[k] >= 3]
        add_tools = [created_tools[tool_name]['tool_config'] for tool_name in top_k]
        self.all_tools = self.tools + add_tools

        self.max_steps = max_steps
        self.tool_map = {
            "logistic_regression": logistic_regression,
            "linear_regression": linear_regression,
            "random_forest_regression": random_forest_regression,
            "random_forest_classification": random_forest_classification,
            "xgboost_regression": xgboost_regression,
            "xgboost_classification": xgboost_classification,
            "lightgbm_regression": lightgbm_regression,
            "lightgbm_classification": lightgbm_classification,
            "catboost_regression": catboost_regression,
            "catboost_classification": catboost_classification,
            "machine_learning_tool_creation": machine_learning_tool_creation
        }
        self.system_prompt = MODEL_AGENT_SYS_PROMPT
        super().__init__("model_agent", self.system_prompt, llm_config)
        self.count = 0
        self.history = []

    async def __call__(self, train_dataset_path, test_dataset_path, task, global_task, work_dir):
        self.count += 1
        prompt = MODEL_DEVELOPMENT_PROMPT.format(train_dataset_path=train_dataset_path, test_dataset_path=test_dataset_path, global_task=global_task, task=task)
        messages = [{"role": "user", "content": prompt}]
        for i in range(self.max_steps):
            choice = await self.llm.generate(messages, tools=self.all_tools)
            finish_reason = choice.finish_reason
            if finish_reason == "tool_calls":
                messages.append(dict(choice.message))
                for tool_call in choice.message.tool_calls:
                    tool_call_name = tool_call['function']['name']
                    try:
                        tool_call_arguments = json.loads(tool_call['function']['arguments'])
                        if tool_call_name in self.tool_map:
                            tool_function = self.tool_map[tool_call_name]
                            tool_result = tool_function(**tool_call_arguments, work_dir=work_dir)
                        else:
                            with open(f"utils/{self.dataset}/created_tools/modeling_tools.json", "r", encoding='utf-8') as f:
                                created_tools = json.load(f)
                            tool_code = created_tools[tool_call_name]['code']
                            tool_result = execute_tool(tool_code, tool_call_name, tool_call_arguments, 'modeling', work_dir=work_dir)
                            add_tools = update_tool_num(self.dataset, 'modeling', tool_call_name, self.add_tool_nums)
                            self.all_tools = self.tools + add_tools
                    except Exception as e:
                        tool_result = f"Tool call failed due to {str(e)}"
                    if tool_call_name == "machine_learning_tool_creation" and tool_result.startswith('Machine learning task completed successfully.'):
                        try:
                            success, add_tools, tool_extraction_messages = await update_tools(tool_call_arguments['code'], self.dataset, 'modeling', self.llm_config, self.add_tool_nums)
                            if success:
                                self.all_tools = self.tools + add_tools
                                if os.path.exists(os.path.join(work_dir, 'messages', f"tool_extraction_messages.json")):    
                                    with open(os.path.join(work_dir, 'messages', f"tool_extraction_messages.json"), "r", encoding='utf-8') as f:
                                        all_tool_extraction_messages = json.load(f)
                                else:
                                    all_tool_extraction_messages = []
                                update_tool_extraction_messages = all_tool_extraction_messages + tool_extraction_messages
                                with open(os.path.join(work_dir, 'messages', f"tool_extraction_messages.json"), "w", encoding='utf-8') as f:
                                    json.dump(update_tool_extraction_messages, f, indent=4, ensure_ascii=False)
                        except Exception as e:
                            print("Update tool failed due to: ", e)
                    print("tool_result:", tool_result)
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call['id'],
                        "name": tool_call_name,
                        "content": json.dumps({'result': tool_result}),
                    }
                    messages.append(tool_message)
                    if tool_result.startswith("Machine learning task completed successfully."):
                        messages.insert(0, {"role": "system", "content": self.system_prompt})
                        self.history.append(messages)
                        return tool_result
            else:
                messages.append({"role": "assistant", "content": choice.message.content})
                messages.insert(0, {"role": "system", "content": self.system_prompt})
                self.history.append(messages)
                return choice.message.content
        messages.insert(0, {"role": "system", "content": self.system_prompt})
        self.history.append(messages)
        return f"Machine learning task failed. Please provide more detailed information about the machine learning task or use an alternative tool to proceed with the process."
    
from agents.base_agent import BaseAgent
from utils.visualization import *
import json
from heapq import nlargest
from utils.util import *



class Visualizer(BaseAgent):
    def __init__(self, llm_config, tools, dataset, add_tool_nums, max_steps=3):
        self.tools = tools
        self.dataset = dataset
        self.add_tool_nums = add_tool_nums
        if os.path.exists(f"utils/{self.dataset}/created_tools/visualization_tools.json"):    
            with open(f"utils/{self.dataset}/created_tools/visualization_tools.json", "r", encoding='utf-8') as f:
                created_tools = json.load(f)
        else:
            created_tools = {}
        if os.path.exists(f"utils/{self.dataset}/created_tools/visualization_tool_count.json"):    
            with open(f"utils/{self.dataset}/created_tools/visualization_tool_count.json", "r", encoding='utf-8') as f:
                tool_count = json.load(f)
        else:
            tool_count = {}
        top_k = [k for k, _ in nlargest(self.add_tool_nums, tool_count.items(), key=lambda x: x[1]) if tool_count[k] >= 3]
        add_tools = [created_tools[tool_name]['tool_config'] for tool_name in top_k]
        self.all_tools = self.tools + add_tools

        self.max_steps = max_steps
        self.tool_map = {
            "plot_line": plot_line,
            "plot_bar": plot_bar,
            "plot_histogram": plot_histogram,
            "plot_boxplot": plot_boxplot,
            "plot_scatter": plot_scatter,
            "plot_heatmap": plot_heatmap,
            "plot_pie": plot_pie,
            "plot_pairplot": plot_pairplot,
            "visualization_tool_creation": visualization_tool_creation
        }
        self.system_prompt = VISUALIZER_SYS_PROMPT
        super().__init__("visualizer", self.system_prompt, llm_config)
        self.count = 0
        self.history = []

    async def __call__(self, dataset_file, task, saved_plot_file, work_dir):
        self.count += 1
        prompt = VISUALIZATION_PROMPT.format(dataset_file=dataset_file, task=task, saved_plot_file=saved_plot_file)
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
                            with open(f"utils/{self.dataset}/created_tools/visualization_tools.json", "r", encoding='utf-8') as f:
                                created_tools = json.load(f)
                            tool_code = created_tools[tool_call_name]['code']
                            tool_result = execute_tool(tool_code, tool_call_name, tool_call_arguments, 'visualization', work_dir=work_dir)
                            add_tools = update_tool_num(self.dataset, 'visualization', tool_call_name, self.add_tool_nums)
                            self.all_tools = self.tools + add_tools
                    except Exception as e:
                        tool_result = f"Tool call failed due to {str(e)}"
                    if tool_call_name == "visualization_tool_creation" and tool_result.startswith('Visualization task completed successfully.'):
                        try:
                            success, add_tools, tool_extraction_messages = await update_tools(tool_call_arguments['code'], self.dataset, 'visualization', self.llm_config, self.add_tool_nums)
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
                    if tool_result.startswith("Visualization task completed successfully."):
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
        return f"Visualization task failed. Please provide more detailed information about the visualization task or use an alternative tool to proceed with the process."


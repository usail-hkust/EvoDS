from agents.base_agent import BaseAgent
from utils.prompt import MANAGER_SYS_PROMPT


class Manager(BaseAgent):
    def __init__(self, llm_config, tools, max_steps):
        self.tools = tools
        self.max_steps = max_steps
        self.history = []
        self.count = 0
        self.tool_space = ["bash", "python", "sql", "data_cleaning", "feature_engineering", "model_development", "visualization", "debugging", "context_summarize"]
        self.system_prompt = MANAGER_SYS_PROMPT.format(max_steps=max_steps)
        super().__init__("manager", self.system_prompt, llm_config)
    
    async def action(self, messages):
        return await self.llm.generate(messages, tools=self.tools)
    
    def update_system_prompt(self, step):
        step_left = self.max_steps - step
        self.system_prompt = MANAGER_SYS_PROMPT.format(max_steps=step_left)
        self.llm.sys_msg = self.system_prompt
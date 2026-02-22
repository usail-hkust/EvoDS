from utils.util import *
from agents.base_agent import BaseAgent
from utils.debug import *


class DebugAgent(BaseAgent):
    def __init__(self, llm_config, max_steps=3):
        self.system_prompt = DEBUG_AGENT_SYS_PROMPT
        self.max_steps = max_steps
        self.count = 0
        self.history = []
        super().__init__("debug_agent", self.system_prompt, llm_config)

    async def __call__(self, task, code, work_dir):
        if code.endswith('.py'):
            file_path = code.copy()
            success, observation = execute_script(file_path, work_dir)
            with open(os.path.join(work_dir, file_path), 'r') as f:
                code = f.read()
        else:
            file_path = f'debug_{self.count}.py'
            write_program(code, os.path.join(work_dir, file_path))
            success, observation = execute_script(file_path, work_dir)
            self.count += 1
        
        memory = None
        self.count += 1
        prompt = f"Task: {task}\nCode: {code}\n" + CODE_DEBUG_PROMPT.format(observation=observation)
        messages = [{"role": "user", "content": prompt}]
        for i in range(self.max_steps):
            # debug
            choice = await self.llm.generate(messages)
            code = choice.message.content
            messages.append({"role": "assistant", "content": code})

            # execute
            write_program(code, os.path.join(work_dir, file_path))
            success, observation = execute_script(file_path, work_dir)

            ## If observation is too long, we only keep the last ~2k tokens.
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(observation)
            if len(tokens) >= 1000:
                head_tokens = tokens[:500]
                tail_tokens = tokens[-500:]
                observation = enc.decode(head_tokens) + "\n...\n" + enc.decode(tail_tokens)

            # If the script has been successfully executed: Exit.
            if success:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
                self.history.append(messages)
                return f"The refined code is as follows: \n{code}\nThe code has been successfully executed. The execution result is as follows: \n{observation}"
            else:
                messages.append({"role": "user", "content": CODE_DEBUG_PROMPT.format(observation=observation)})
        messages.insert(0, {"role": "system", "content": self.system_prompt})
        self.history.append(messages)
        return f"The refined code is as follows: \n{code}\nThe code has not been successfully executed. The execution result is as follows: \n{observation}"


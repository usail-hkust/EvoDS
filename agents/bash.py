from utils.util import *


class BashAgent():
    def __init__(self):
        self.role = "bash_agent"

    async def __call__(self, code, work_dir):
        success, output = execute_command(code, work_dir)

        if success and (output is None or output == ''):
            return "Bash command executed successfully. No output."

        ## If observation is too long, we only keep the last ~1k tokens.
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(output)
        if len(tokens) >= 1000:
            head_tokens = tokens[:500]
            tail_tokens = tokens[-500:]
            output = enc.decode(head_tokens) + "\n...\n" + enc.decode(tail_tokens)
        
        if success:
            return output
        else:
            return f"Bash command execution failed. The reason is: {output}."
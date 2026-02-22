from utils.util import *


class SqlAgent():
    def __init__(self):
        self.role = "sql_agent"
    
    async def __call__(self, code, dataset_path, output_file, work_dir):
        obs = execute_sql_code(dataset_path, code, output_file, work_dir)
        if obs is None or obs == '':
            obs = f"SQL command executed successfully. No output."

        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(obs)
        if len(tokens) >= 1000:
            head_tokens = tokens[:500]
            tail_tokens = tokens[-500:]
            obs = enc.decode(head_tokens) + "\n...\n" + enc.decode(tail_tokens)
        
        return obs
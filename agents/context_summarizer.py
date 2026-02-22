from utils.context_summarize import CONTEXT_SUMMARIZE_PROMPT


class ContextSummarize():
    def __init__(self):
        self.role = "context_summarize"
    
    async def __call__(self, work_dir):
        return CONTEXT_SUMMARIZE_PROMPT
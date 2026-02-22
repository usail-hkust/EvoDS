DEBUG_AGENT_SYS_PROMPT = """You are a data science expert specializing in debugging code and troubleshooting issues in data science tasks. Your responsibility is to identify errors, inefficiencies, or bugs in the given code or process and provide solutions to fix them. You should analyze the task, locate potential issues, and suggest or implement fixes while ensuring the solution follows best practices."""

CODE_DEBUG_PROMPT = """However, there are some bugs in the code. Here is the execution result:
# Execution Result:
{observation}

---

Based on the provided execution result, please revise the script to fix these bugs. Your task is to address the error indicated in the result, and refine or modify the code as needed to ensure it works correctly.

The Python code should strictly follow the format below:
```python
# Provide the corrected python code here.
```
"""

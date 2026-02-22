from utils.util import *


class PythonAgent():
    def __init__(self):
        self.role = "python_agent"

    async def __call__(self, code, file_path, work_dir):
        write_program(code, os.path.join(work_dir, file_path))
        success, output = execute_script(file_path, work_dir)
        if success:
            if output is None or output == '':
                return "Python script executed successfully. No output."
            return output
        else:
            return f"Python script execution failed. The reason is: {output}."
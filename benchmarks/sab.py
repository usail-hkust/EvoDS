import os
import asyncio
import asyncio
import json
import shutil
import tiktoken
from utils.prompt import *
from pathlib import Path
import importlib


def get_sys_msg(task, data_id):
    sys_msg = "Task: " + task["task_inst"] + "\n" + str(task["domain_knowledge"]) + "\n" + "Format: Please strictly adhere to the task requirements when saving the results. Specifically, for CSV files, if the required column names are not explicitly specified, you must use exactly the column names from the test data and must not modify or replace them."

    dataset_preview = task["dataset_preview"]
    if dataset_preview:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(dataset_preview)
        if len(tokens) >= 1000:
            head_tokens = tokens[:500]
            tail_tokens = tokens[-500:]
            dataset_preview = enc.decode(head_tokens) + "\n...\n" + enc.decode(tail_tokens)

    sys_msg += "\n" + DATA_INFO_PROMPT.format(dataset_path = data_id, dataset_folder_tree = task['dataset_folder_tree'], dataset_preview = dataset_preview)

    return sys_msg

def calculate_sab_score(data_item, data_id, work_dir):
    output_path = data_item["output_fname"]
    eval_program_path = 'benchmark/eval_programs'

    if not Path(output_path).exists():
        return 0, "The program does not save its output correctly."

    # Run evaluation script
    eval_fname = str(Path(eval_program_path, data_item["eval_script_name"]))
    eval_fname = eval_fname.replace("/", '.')[:-3] # remove ".py" suffix
    
    try:
        module = importlib.import_module(eval_fname)  # 动态导入模块
        result = module.eval() 

        success, log_info = result
        return success, log_info
    except Exception as e:
        return 0, repr(e)

async def sab(data_item, agent, model):
    agent.reset()
    model = model.split('/')[-1]
    data_id = data_item['instance_id']
    work_dir = os.path.join(f"output/sab_{model}/", str(data_id))
    if os.path.exists(os.path.join(work_dir, "result.json")):
        return
    if os.path.exists(work_dir):
        proc = await asyncio.create_subprocess_shell(f"rm -rf {work_dir}")
        await proc.wait()
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.join(work_dir, "pred_results"), exist_ok=True)

    data_id = data_item['dataset_folder_tree'].splitlines()[0].split('|-- ')[1].split('/')[0]
    src_code = os.path.join('benchmark/datasets/', data_id)

    shutil.copytree(src_code, os.path.join(work_dir, data_id), dirs_exist_ok=True)
    input_text = get_sys_msg(data_item, data_id)
        
    done, result = await agent.generate_output(input_text, work_dir)

    shutil.copytree('benchmark/eval_programs', os.path.join(work_dir, 'benchmark/eval_programs'), dirs_exist_ok=True)
    current_dir = os.getcwd()
    os.chdir(work_dir)
    score, eval_log = calculate_sab_score(data_item, data_id, work_dir)
    os.chdir(current_dir)
    shutil.rmtree(os.path.join(work_dir, 'benchmark'))

    result = {
        'id': data_item['instance_id'],
        'input_text': input_text,
        "result": result,
        'score': score,
        'eval_log': eval_log
    }

    with open(os.path.join(work_dir, "result.json"), 'w') as f:
        json.dump(result, f, indent=2)

    # for file in os.listdir(work_dir):
    #     if file in ['messages', 'messages.json', 'result.json', 'logs.json', 'usage.json', 'pred_results']:
    #         continue
    #     else:
    #         try:
    #             os.remove(os.path.join(work_dir, file))
    #         except:
    #             shutil.rmtree(os.path.join(work_dir, file))
import os
import asyncio
import aiofiles, aioshutil, asyncio
from typing import Dict
import hashlib
from evaluate import configs
from evaluate.da import Evaluator
import json
import shutil
import aiofiles.os
from benchmarks.matplotbench import image_evaluate


def calculate_sha256(file_path):
    with open(file_path, 'rb') as f:
        file_data = f.read()
        return hashlib.sha256(file_data).hexdigest()

def find_diff_files_init(init_file_dict, output_dir)-> Dict:
    init_file_paths = init_file_dict.keys()
    added_files_list = []
    changed_files_list = []
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            file_path = os.path.join(root, f)
            if file_path not in init_file_paths:
                added_files_list.append(file_path)
            else:
                if init_file_dict[file_path] != calculate_sha256(file_path):
                    changed_files_list.append(file_path)
    return {"added_files": added_files_list, "changed_files": changed_files_list}

def post_process(data, output_dir, init_files_hash):
    """
    Evaluate whether the task is successfully completed.
    """
    diff_files = find_diff_files_init(init_files_hash, output_dir)
    post_process_func = data["post_process"] if "post_process" in data else []
    post_process_files = []
    errors = []

    try:
        for post_process_f in post_process_func:
            process_function = getattr(configs, post_process_f, None)
            post_files, error = process_function(output_dir)
            post_files = post_files if isinstance(post_files, list) else list(post_files)
            post_process_files.extend(post_files)
            errors.append(error)
    except:
        post_process_files = []
        errors = []

    return {**diff_files, "post_process_files": post_process_files, "error": errors}

def get_env_files_hash(output_dir) -> Dict[str, str]:
    """
    Returns:
        Dict[str, str]: a dictionary of the hash of the files in the
          environment
    """
    files_hash = {}
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            file_path = os.path.join(root, f)
            files_hash[file_path] = calculate_sha256(file_path)
    return files_hash


async def evaluate_plot(data_item, work_dir, num=None):
    data_id = data_item['id']
    for target_file in os.listdir(f"da_code/gold/{data_id}"):
        if target_file in ['plot.json', 'result.npy']:
            continue
        break
    target_path = os.path.join(f'da_code/gold/{data_id}', target_file)
    if num:
        source_dir = os.path.join(work_dir, num)
    else:
        source_dir = work_dir
    source_path_1 = os.path.join(source_dir, target_file)
    source_path_2 = None
    for source_file in os.listdir(source_dir):
        if source_file.endswith('.png') or source_file.endswith('.jpg'):
            source_path_2 = os.path.join(source_dir, source_file)
            break
    if os.path.exists(source_path_1):
        score = await image_evaluate(source_path_1, target_path)
    elif source_path_2 is not None and os.path.exists(source_path_2):
        score = await image_evaluate(source_path_2, target_path)
    else:
        score = 0

    return score


async def da_code(data_item, agent, model):
    agent.reset()

    input_text = data_item['instruction']
    model = model.split('/')[-1]
    work_dir = os.path.join(f"output/da_code_{model}/", data_item["id"])
    if os.path.exists(os.path.join(work_dir, "dabench/result.json")):
        return

    if os.path.exists(work_dir):
        proc = await asyncio.create_subprocess_shell(f"rm -rf {work_dir}")
        await proc.wait()
    os.makedirs(work_dir, exist_ok=True)
    src_code = os.path.join('da_code/source/', data_item["id"])
    await aioshutil.copytree(src_code, work_dir, dirs_exist_ok=True)

    loop = asyncio.get_event_loop()
    init_files_hash = await loop.run_in_executor(None, get_env_files_hash, work_dir)

    done, result = await agent.generate_output(input_text, work_dir)

    await aiofiles.os.makedirs(os.path.join(work_dir, "dabench"), exist_ok=True)
    result_files = await asyncio.get_running_loop().run_in_executor(
        None, post_process, data_item, work_dir, init_files_hash
    )
    dabench_result = {"finished": done, "result": result, "result_files": result_files}
    async with aiofiles.open(os.path.join(work_dir, "dabench/result.json"), 'w', encoding='utf-8') as f:
        await f.write(json.dumps(dabench_result, indent=2, ensure_ascii=False))

    # compute score
    evaluator = Evaluator(data=data_item, eval_config_path='da_code/configs/eval/eval_all.jsonl', output_dir=work_dir, gold_dir='da_code/gold/', timeout_seconds=300)
    try:
        score = evaluator.evaluate()
    except Exception as e:
        print('Error:', e)
        score = 0
    dabench_result['score'] = score

    if data_item["id"].startswith('plot-'):
        score_new = await evaluate_plot(data_item, work_dir)
        dabench_result['score'] = score_new
        dabench_result['score_old'] = score

    async with aiofiles.open(os.path.join(work_dir, "dabench/result.json"), 'w', encoding='utf-8') as f:
        await f.write(json.dumps(dabench_result, indent=2, ensure_ascii=False))

    # for file in os.listdir(work_dir):
    #     if file in ['messages', 'messages.json', 'dabench', 'logs.json', 'usage.json'] or file.endswith('.png') or file.endswith('.jpg'):
    #         continue
    #     else:
    #         try:
    #             os.remove(os.path.join(work_dir, file))
    #         except:
    #             shutil.rmtree(os.path.join(work_dir, file))
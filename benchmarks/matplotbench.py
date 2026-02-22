import os
import asyncio
import aiofiles, aioshutil, asyncio
import json
import shutil
from utils.util import image_evaluate


async def matplotbench(data_item, agent, model):
    agent.reset()
    data_id = data_item['id']
    input_text = f"Visualization task: \n\n{data_item['expert_instruction']}\n\nFormat: Ensure that the visualization results are saved in the current directory, strictly adhering to the task requirements, and name the file 'plot.png'."
    model = model.split('/')[-1]
    work_dir = os.path.join(f"output/matplotbench_{model}/", str(data_id))
    if os.path.exists(os.path.join(work_dir, "result.json")):
        return

    if os.path.exists(work_dir):
        proc = await asyncio.create_subprocess_shell(f"rm -rf {work_dir}")
        await proc.wait()
    os.makedirs(work_dir, exist_ok=True)
    src_table = os.path.join('verl/MatPlotBench/data', str(data_id))
    if os.path.exists(src_table):
        await aioshutil.copytree(src_table, work_dir, dirs_exist_ok=True)
    done, result = await agent.generate_output(input_text, work_dir)

    pred_path = os.path.join(work_dir, "plot.png")
    gt_path = os.path.join(f"verl/MatPlotBench/ground_truth/example_{data_id}.png")
    score = await image_evaluate(pred_path, gt_path)
    eval_logging = {
        "id": data_id, 
        "result": result,
        "score": score
    }
    async with aiofiles.open(os.path.join(work_dir, "result.json"), 'w', encoding='utf-8') as f:
        await f.write(json.dumps(eval_logging, indent=2))

    # for file in os.listdir(work_dir):
    #     if file in ['messages', 'messages.json', 'result.json', 'logs.json', 'usage.json', 'plot.png']:
    #         continue
    #     else:
    #         try:
    #             os.remove(os.path.join(work_dir, file))
    #         except:
    #             shutil.rmtree(os.path.join(work_dir, file))
import os
import asyncio
import aiofiles, aioshutil, asyncio
import json
import shutil



def compute_score_dsbench(data_id, work_dir):
    gt_path = "verl/dsbench/answers/"
    python_path = "verl/dsbench/evaluation/"
    save_path = f"verl/dsbench/save_performance/"

    answer_file = os.path.join(gt_path, data_id, 'test_answer.csv')
    pred_file = os.path.join(work_dir, 'prediction.csv')

    if os.path.exists(pred_file):
        os.makedirs(os.path.join(work_dir, data_id), exist_ok=True)
        os.system(f"python {python_path}{data_id}_eval.py --answer_file {answer_file} --predict_file {pred_file} --path {work_dir} --name {data_id}")
        
        flag = False ## whetehr bigger is better
        with open(os.path.join(os.path.join(save_path, 'GT'), data_id, "result.txt"), "r") as f:
            gt = eval(f.read().strip())
        with open(os.path.join(os.path.join(save_path, 'baseline'), data_id, "result.txt"), "r") as f:
            bl = eval(f.read().strip())
        if gt > bl:
            flag = True
    
        if not os.path.exists(os.path.join(work_dir, data_id, "result.txt")):
            score = 0
        else:
            with open(os.path.join(work_dir, data_id, "result.txt"), "r") as f:
                pre = f.read().strip()
            if pre == "nan":
                score = 0
            else:
                pre = eval(pre)
                score = max(0, (pre-bl)/(gt-bl))
    else:
        score = 0

    return score


async def dsbench(data_item, agent, model):
    agent.reset()
    data_id = data_item['name']
    with open(os.path.join('verl/dsbench/task', data_id + ".txt"), "r") as f:
        task = f.read()

    input_text = f"Machine Learning task: \n\n{task}\n\nFormat: Ensure that the prediction results are saved in the current directory, strictly adhering to the task requirements, and name the file 'prediction.csv'."

    model = model.split('/')[-1]
    work_dir = os.path.join(f"output/dsbench_{model}/", data_id)
    if os.path.exists(os.path.join(work_dir, "result.json")):
        return

    if os.path.exists(work_dir):
        proc = await asyncio.create_subprocess_shell(f"rm -rf {work_dir}")
        await proc.wait()
    os.makedirs(work_dir, exist_ok=True)
    src_code = os.path.join('verl/dsbench/data_resplit', data_id)
    await aioshutil.copytree(src_code, work_dir, dirs_exist_ok=True)

    done, result = await agent.generate_output(input_text, work_dir)

    score = compute_score_dsbench(data_id, work_dir)

    dsbench_result = {
        "id": data_id, 
        "result": result,
        "score": score
    }

    async with aiofiles.open(os.path.join(work_dir, "result.json"), 'w', encoding='utf-8') as f:
        await f.write(json.dumps(dsbench_result, indent=2, ensure_ascii=False))

    # for file in os.listdir(work_dir):
    #     if file in ['messages', 'prediction.csv', 'messages.json', 'result.json', 'logs.json', 'usage.json', data_id]:
    #         continue
    #     else:
    #         try:
    #             os.remove(os.path.join(work_dir, file))
    #         except:
    #             shutil.rmtree(os.path.join(work_dir, file))


from verl.mledojo.gym.competition import CompetitionRegistry
from verl.mledojo.competitions import get_metric
import os
import asyncio
import aiofiles, aioshutil, asyncio
import json
import shutil
import pandas as pd


def mledojo_reward_function(work_dir, data_id: str) -> float:
    score = 0

    # register the competition
    registry = CompetitionRegistry()
    registry.register(
        name=data_id,
        data_dir=os.path.join('verl/mledojo/data/prepared', data_id, 'data'),
        metric_class=get_metric(data_id)
    )
    metric = registry._competitions[data_id].metric_class()

    if not os.path.exists(os.path.join(work_dir, "submission.csv")):
        return 0
    else:
        y_pred = pd.read_csv(os.path.join(work_dir, "submission.csv"))
        y_true = pd.read_csv(os.path.join('verl/mledojo/data/prepared', data_id, 'data/private/test_answer.csv'))
        try:
            metric.validate_submission(y_pred, y_true)
        except Exception as e:
            return 0
        score = metric.evaluate(y_true, y_pred)

        leaderboard = pd.read_csv(f'verl/mledojo/competitions/{data_id}/info/private_leaderboard.csv')
        cmp_op = (lambda a, b: a <= b) if metric.higher_is_better else (lambda a, b: a >= b)
        idx = leaderboard['score'].shape[0]
        for i, s in enumerate(leaderboard['score']):
            if cmp_op(s, score):
                idx = i
                break
        rank = idx + 1
        reward = 1 - rank / leaderboard['score'].shape[0]

        return reward

async def mledojo(data_item, agent, model):
    agent.reset()
    data_id = data_item['task_id']
    with open(os.path.join('verl/mledojo/competitions', data_id, 'info/description' + ".txt"), "r") as f:
        task = f.read()

    input_text = f"Machine Learning task: \n\n{task}\n\nFormat: Ensure that the prediction results are saved in the current directory, strictly adhering to the task requirements, and name the file 'submission.csv'."
    model = model.split('/')[-1]
    work_dir = os.path.join(f"output/mledojo_{model}/", str(data_item["task_id"]))
    if os.path.exists(os.path.join(work_dir, "result.json")):
        return

    if os.path.exists(work_dir):
        proc = await asyncio.create_subprocess_shell(f"rm -rf {work_dir}")
        await proc.wait()
    os.makedirs(work_dir, exist_ok=True)
    src_table = os.path.join('verl/mledojo/data/prepared', data_id, 'data/public')
    await aioshutil.copytree(src_table, work_dir, dirs_exist_ok=True)
    done, result = await agent.generate_output(input_text, work_dir)

    score = mledojo_reward_function(work_dir, data_id)
    eval_logging = {
        "id": data_item['task_id'], 
        "result": result,
        "score": score
    }
    async with aiofiles.open(os.path.join(work_dir, "result.json"), 'w', encoding='utf-8') as f:
        await f.write(json.dumps(eval_logging, indent=2))

    # for file in os.listdir(work_dir):
    #     if file in ['messages', 'messages.json', 'result.json', 'logs.json', 'usage.json', 'submission.csv']:
    #         continue
    #     else:
    #         try:
    #             os.remove(os.path.join(work_dir, file))
    #         except:
    #             shutil.rmtree(os.path.join(work_dir, file))
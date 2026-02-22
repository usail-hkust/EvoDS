import os
import asyncio
import aiofiles, aioshutil, asyncio
import json
import shutil
import re
import math
import pandas as pd


def is_equal_relative(response, label):
    if response == label:
        return True
    else:
        try:
            if ',' in label:
                label = label.replace(',', '')
            if ',' in response:
                response = response.replace(',', '')
            return abs((float(response) - float(label))/float(label)) < 0.001
        except:
            return False
        
def extract_format(input_string):
    pattern = r"@(\w+)\[(.*?)\]"
    matches = re.findall(pattern, input_string)
    answer_names = [match[0] for match in matches]
    answers = [match[1] for match in matches]
    return answer_names, answers


async def datamind_python(data_item, agent, model):
    if data_item['ground_truth'] == []:
        return
    agent.reset()
    input_text = f"Question: {data_item['problem']}\n\nOutput Format: {data_item['format']}\n\nData File: {data_item['file_name']}"

    model = model.split('/')[-1]
    work_dir = os.path.join(f"output/datamind_python_{model}/", str(data_item["task_id"]))
    if os.path.exists(os.path.join(work_dir, "result.json")):
        return

    if os.path.exists(work_dir):
        proc = await asyncio.create_subprocess_shell(f"rm -rf {work_dir}")
        await proc.wait()
    os.makedirs(work_dir, exist_ok=True)
    src_table = os.path.join('verl/datamind/train_files', data_item["file_name"])
    await aioshutil.copy(src_table, work_dir)
    done, result = await agent.generate_output(input_text, work_dir)

    try:
        label_answers = {ans[0]: ans[1] for ans in data_item['ground_truth']}
        answer_names, answers = extract_format(result)
        extracted_answers = dict(zip(answer_names, answers))
        correct_answers = {ans_name: is_equal_relative(extracted_answers.get(ans_name), label_answers[ans_name]) for ans_name
                               in label_answers.keys()}
        score = sum(correct_answers.values()) / len(correct_answers)
    except:
        score = 0
    eval_logging = {
        "id": data_item['task_id'], 
        "result": result,
        "score": score,
        "correct_answers": correct_answers,
        "label_answers": label_answers
    }
    async with aiofiles.open(os.path.join(work_dir, "result.json"), 'w', encoding='utf-8') as f:
        await f.write(json.dumps(eval_logging, indent=2))
    
    # for file in os.listdir(work_dir):
    #     if file in ['messages', 'messages.json', 'result.json', 'logs.json', 'usage.json']:
    #         continue
    #     else:
    #         try:
    #             os.remove(os.path.join(work_dir, file))
    #         except:
    #             shutil.rmtree(os.path.join(work_dir, file))



def datamind_sql_reward_function(id: str, gold_csv_results_dir: str, pred_csv_results_dir: str) -> float:
    def compare_pandas_table(pred, gold, condition_cols=[], ignore_order=False):
        """_summary_

        Args:
            pred (Dataframe): _description_
            gold (Dataframe): _description_
            condition_cols (list, optional): _description_. Defaults to [].
            ignore_order (bool, optional): _description_. Defaults to False.

        """
        # print('condition_cols', condition_cols)
        
        tolerance = 1e-2

        def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
            if ignore_order_:
                v1, v2 = (sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))),
                        sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float)))))
            if len(v1) != len(v2):
                return False
            for a, b in zip(v1, v2):
                if pd.isna(a) and pd.isna(b):
                    continue
                elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if not math.isclose(float(a), float(b), abs_tol=tol):
                        return False
                elif a != b:
                    return False
            return True
        
        if condition_cols != []:
            gold_cols = gold.iloc[:, condition_cols]
        else:
            gold_cols = gold
        pred_cols = pred

        t_gold_list = gold_cols.transpose().values.tolist()
        t_pred_list = pred_cols.transpose().values.tolist()
        score = 1
        for _, gold in enumerate(t_gold_list):
            if not any(vectors_match(gold, pred, ignore_order_=ignore_order) for pred in t_pred_list):
                score = 0
            else:
                for j, pred in enumerate(t_pred_list):
                    if vectors_match(gold, pred, ignore_order_=ignore_order):
                        break

        return score

    try:
        pred_csv_path = os.path.join(pred_csv_results_dir, 'result.csv')
        if os.path.exists(pred_csv_path) == False:
            return 0

        gold_csv_path = os.path.join(gold_csv_results_dir, f"{id}.csv")
        score = compare_pandas_table(pd.read_csv(pred_csv_path), pd.read_csv(gold_csv_path), ignore_order=True)
    except Exception as e:
        score = 0

    return score


async def datamind_sql(data_item, agent, model):
    agent.reset()
    input_text = f"Question: {data_item['problem']}\n\nOutput Format: {data_item['format']}\n\nData File: {data_item['file_name']}"

    model = model.split('/')[-1]
    work_dir = os.path.join(f"output/datamind_sql_{model}/", str(data_item["task_id"]))
    if os.path.exists(os.path.join(work_dir, "result.json")):
        return

    if os.path.exists(work_dir):
        proc = await asyncio.create_subprocess_shell(f"rm -rf {work_dir}")
        await proc.wait()
    os.makedirs(work_dir, exist_ok=True)
    src_table = os.path.join('verl/datamind/train_files', data_item["file_name"])
    await aioshutil.copy(src_table, work_dir)
    done, result = await agent.generate_output(input_text, work_dir)

    score = datamind_sql_reward_function(data_item['task_id'], "verl/datamind/gold_csv_results", work_dir)
    eval_logging = {
        "id": data_item['task_id'], 
        "result": result,
        "score": score
    }
    async with aiofiles.open(os.path.join(work_dir, "result.json"), 'w', encoding='utf-8') as f:
        await f.write(json.dumps(eval_logging, indent=2))

    # for file in os.listdir(work_dir):
    #     if file in ['messages', 'messages.json', 'result.json', 'logs.json', 'usage.json', 'result.csv']:
    #         continue
    #     else:
    #         try:
    #             os.remove(os.path.join(work_dir, file))
    #         except:
    #             shutil.rmtree(os.path.join(work_dir, file))
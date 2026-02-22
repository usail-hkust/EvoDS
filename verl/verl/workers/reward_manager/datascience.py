from verl import DataProto
from verl.workers.reward_manager import register
from verl.utils.reward_score import default_compute_score
import torch
import json
from collections import defaultdict
import re
import os
import shutil
import pandas as pd
import math
from mledojo.gym.competition import CompetitionRegistry
from mledojo.competitions import get_metric
import base64
import requests
import time


@register("datascience")
class DSRewardManager:
    """The custom reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    def extract_block(self, text: str):
        full_block = text.split('\nassistant\n')[-1]
        if not full_block:
            return None, None
        result = re.search(r'(?s)</think>\n\n(.*)$', full_block)
        if result:
            result = result.group(1)
        else:
            result = full_block.strip()
        return full_block, result
    
    def extract_format(self, input_string):
        pattern = r"@(\w+)\[(.*?)\]"
        matches = re.findall(pattern, input_string)
        answer_names = [match[0] for match in matches]
        answers = [match[1] for match in matches]
        return answer_names, answers

    def __call__(self, data: DataProto, return_dict: bool = False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # if "rm_scores" in data.batch.keys():
        #     if return_dict:
        #         reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
        #         reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
        #         return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
        #     else:
        #         return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            if data_item.non_tensor_batch["sub_agent"] == 1:
                reward = data_item.non_tensor_batch["turn_scores"][0]
                reward_extra_info["acc"].append(reward)
                reward_tensor[i, valid_response_length - 1] = reward
            else:
                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                eos_token = self.tokenizer.eos_token
                if response_str.endswith(eos_token):
                    response_str = response_str[: -len(eos_token)]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]

                if data_source == 'matplotbench':
                    ground_truth = data_item.non_tensor_batch["reward_model"]['ground_truth']
                elif data_source == 'mledojo':
                    ground_truth = data_item.non_tensor_batch["reward_model"]['ground_truth']
                elif data_source == 'darl/sql':
                    ground_truth = data_item.non_tensor_batch["reward_model"]['ground_truth']
                elif data_source == 'darl/python':
                    ground_truth = eval(data_item.non_tensor_batch["reward_model"]['ground_truth'])
                elif data_source == 'datascience_instruct':
                    ground_truth = eval(data_item.non_tensor_batch["reward_model"]['ground_truth'])
                elif data_source == 'dsbench':
                    ground_truth = data_item.non_tensor_batch["reward_model"]['ground_truth']
                else:
                    raise ValueError(f"Unknown data source: {data_item.non_tensor_batch[self.reward_fn_key]}")

                extra_info = data_item.non_tensor_batch.get("extra_info", None)

                result = self.reward_func(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    data_item=data_item,
                )
            
                score: float
                if isinstance(result, dict):
                    score = result["score"]
                    # Store the information including original reward
                    for key, value in result.items():
                        reward_extra_info[key].append(value)
                else:
                    score = result
                    reward_extra_info["acc"].append(score)

                reward = score

                reward_tensor[i, valid_response_length - 1] = reward

                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    print("[ground_truth]", ground_truth)
                    if isinstance(result, dict):
                        for key, value in result.items():
                            print(f"[{key}]", value)
                    else:
                        print("[score]", score)

                # remove work_dir
                work_dir = str(data_item.non_tensor_batch['work_dir'])
                if os.path.exists(work_dir):
                    shutil.rmtree(work_dir)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def reward_func(self, data_source, solution_str, ground_truth, extra_info, data_item):
        if "acc" in data_item.non_tensor_batch:
            score = data_item.non_tensor_batch.get("acc")
            turn_scores = data_item.non_tensor_batch.get("turn_scores")
            if score > 0:
                reward = score + turn_scores[0]
            else:
                reward = score
            return reward
        """Reward function that gives higher scores to longer completions."""
        if data_source == 'matplotbench':
            return self.reward_func_matplotbench(data_item)
        elif data_source == 'mledojo':
            return self.reward_func_mledojo(data_item)
        elif data_source == 'darl/sql':
            return self.reward_func_darl_sql(data_item)
        elif data_source == 'darl/python' or data_source == 'datascience_instruct':
            return self.reward_func_darl_python_datascience_instruct(solution_str, ground_truth)
        elif data_source == 'dsbench':
            return self.reward_func_dsbench(data_item)
        else:
            raise ValueError(f"Unknown data source: {data_source}")

    def reward_func_darl_python_datascience_instruct(self, solution_str, ground_truth):
        label_answers = {ans[0]: ans[1] for ans in ground_truth}
        full_content, result = self.extract_block(solution_str)
        if not result:
            return 0.0
        answer_names, answers = self.extract_format(result)
        extracted_answers = dict(zip(answer_names, answers))
        correct_answers = {ans_name: self.is_equal_relative(extracted_answers.get(ans_name), label_answers[ans_name]) for ans_name in label_answers.keys()}
        score = sum(correct_answers.values()) / len(correct_answers)

        return float(score)

    def reward_func_darl_sql(self, data_item):
        raw_data = json.loads(data_item.non_tensor_batch['extra_info']['raw_data'])
        work_dir = str(data_item.non_tensor_batch['work_dir'])
        data_id = raw_data['task_id']
        score = self.datamind_sql_reward_function(data_id, "datamind/gold_csv_results", work_dir)
        return float(score)

    def reward_func_dsbench(self, data_item):
        raw_data = json.loads(data_item.non_tensor_batch['extra_info']['raw_data'])
        work_dir = str(data_item.non_tensor_batch['work_dir'])
        data_id = raw_data['name']
        try:
            score = self.compute_score_dsbench(data_id, work_dir)
        except Exception as e:
            print("Error: ", e)
            score = 0

        return float(score)
    
    def reward_func_mledojo(self, data_item):
        raw_data = json.loads(data_item.non_tensor_batch['extra_info']['raw_data'])
        work_dir = str(data_item.non_tensor_batch['work_dir'])
        data_id = raw_data['task_id']
        try:
            score = self.mledojo_reward_function(data_id, work_dir)
        except Exception as e:
            print("Error: ", e)
            score = 0
        return float(score)

    def reward_func_matplotbench(self, data_item):
        raw_data = json.loads(data_item.non_tensor_batch['extra_info']['raw_data'])
        work_dir = str(data_item.non_tensor_batch['work_dir'])
        data_id = raw_data['id']
        pred_path = os.path.join(work_dir, "plot.png")
        gt_path = os.path.join(f"MatPlotBench/ground_truth/example_{data_id}.png")
        score = self.image_evaluate(pred_path, gt_path)

        return float(score)

    def is_equal_relative(self, response, label):
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

    def datamind_sql_reward_function(self, id: str, gold_csv_results_dir: str, pred_csv_results_dir: str) -> float:
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

    def compute_score_dsbench(self, data_id, work_dir):
        gt_path = "dsbench/answers/"
        python_path = "dsbench/evaluation/"
        save_path = f"dsbench/save_performance/"

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

    def mledojo_reward_function(self, data_id, work_dir) -> float:
        score = 0

        # register the competition
        registry = CompetitionRegistry()
        registry.register(
            name=data_id,
            data_dir=os.path.join('mledojo/data/prepared', data_id, 'data'),
            metric_class=get_metric(data_id)
        )
        metric = registry._competitions[data_id].metric_class()

        if not os.path.exists(os.path.join(work_dir, "submission.csv")):
            return 0
        else:
            y_pred = pd.read_csv(os.path.join(work_dir, "submission.csv"))
            y_true = pd.read_csv(os.path.join('mledojo/data/prepared', data_id, 'data/private/test_answer.csv'))
            try:
                metric.validate_submission(y_pred, y_true)
            except Exception as e:
                return 0
            score = metric.evaluate(y_true, y_pred)

            leaderboard = pd.read_csv(f'mledojo/competitions/{data_id}/info/private_leaderboard.csv')
            cmp_op = (lambda a, b: a <= b) if metric.higher_is_better else (lambda a, b: a >= b)
            idx = leaderboard['score'].shape[0]
            for i, s in enumerate(leaderboard['score']):
                if cmp_op(s, score):
                    idx = i
                    break
            rank = idx + 1
            reward = 1 - rank / leaderboard['score'].shape[0]

            return reward

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def image_evaluate(self, image_path, ground_truth_path, max_try=3):
        score = 0
        success = False
        delays = [30, 90, 180]
        if not os.path.exists(f'{image_path}'):
            return score
        else:
            base64_image1 = self.encode_image(image_path)
            base64_image2 = self.encode_image(ground_truth_path)
    
            for i in range(max_try):
                raw_request = {
                    "model": 'gpt-4o',
                    "temperature": 0.2,
                    "top_p": 1,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0
                    }

                HEADERS = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer {}".format("your_api_key")
                }
                messages=[
                    {
                      "role": "user",
                      "content": [
                        {
                          "type": "text",
                          "text": f'''You are an excellent judge at evaluating visualization plots between a model generated plot and the ground truth. You will be giving scores on how well it matches the ground truth plot.
    
                           The generated plot will be given to you as the first figure.
                           Another plot will be given to you as the second figure, which is the desired outcome of the user query, meaning it is the ground truth for you to reference.
                           Please compare the two figures head to head and rate them.
                           Suppose the second figure has a score of 100, rate the first figure on a scale from 0 to 100.
                           Scoring should be carried out in the following aspect:
                           1. Plot correctness: 
                           Compare closely between the generated plot and the ground truth, the more resemblance the generated plot has compared to the ground truth, the higher the score. The score should be proportionate to the resemblance between the two plots.
                           In some rare occurrence, see if the data points are generated randomly according to the query, if so, the generated plot may not perfectly match the ground truth, but it is correct nonetheless.
                           Only rate the first figure, the second figure is only for reference.
                           If the first figure is blank, that means the code failed to generate a figure. Give a score of 0 on the Plot correctness.
                            After scoring from the above aspect, please give a final score. The final score is preceded by the [FINAL SCORE] token.
                           For example [FINAL SCORE]: 40.''',
                        },
                        {
                          "type": "image_url",
                          "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image1}",
                          },
                        },
                        {
                          "type": "image_url",
                          "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image2}",
                          },
                        },
                      ],
                    }
                  ]
                try:
                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=HEADERS, data=json.dumps({"messages": messages, **raw_request}))
                    response.raise_for_status()
                    response = response.json()
    
                    result = response['choices'][0]['message']['content']
                    match = re.search(r"\[FINAL SCORE\]: (\d+)", result)
                    if match:
                        success = True
                        score = int(match.group(1))
                        break
                except:
                    time.sleep(delays[i])
                    continue
            if not success:
                raise Exception("Failed to evaluate image after {} tries".format(max_try))
            time.sleep(10)
        return score / 100

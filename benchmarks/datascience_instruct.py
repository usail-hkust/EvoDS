import os
import asyncio
import aiofiles, aioshutil, asyncio
import json
import shutil
import re


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


async def datascience_instruct(data_item, agent, model):
    if data_item['ground_truth'] == []:
        return
    agent.reset()
    input_text = f"Question: {data_item['problem']}\n\nOutput Format: {data_item['format']}\n\nData Files: {data_item['file_list']}"

    model = model.split('/')[-1]
    work_dir = os.path.join(f"output/datascience_instruct_{model}/", str(data_item["task_id"]))
    if os.path.exists(os.path.join(work_dir, "result.json")):
        return

    if os.path.exists(work_dir):
        proc = await asyncio.create_subprocess_shell(f"rm -rf {work_dir}")
        await proc.wait()
    os.makedirs(work_dir, exist_ok=True)
    for file in data_item['file_list']:
        src_table = os.path.join(f'verl/deepanalyze/files/{data_item["workspace"]}', file)
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
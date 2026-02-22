import re
import os
import asyncio
import aiofiles, aioshutil, asyncio
import json
import shutil
from utils.prompt import *
from llm.async_llm import LLMsConfig, create_llm_instance
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:20171"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:20171"


def extract_format(input_string):
    pattern = r"@(\w+)\[(.*?)\]"
    matches = re.findall(pattern, input_string)
    answer_names = [match[0] for match in matches]
    answers = [match[1] for match in matches]
    return answer_names, answers

def is_equal(response, label):
    if response == label:
        return True
    else:
        try:
            return abs(float(response) - float(label)) < 1e-6
        except:
            return False


async def load_label_async(target_id: str):
    async with aiofiles.open('dabench/da-dev-labels.jsonl', 'r') as f:
        async for line in f:
            cfg = json.loads(line)
            if cfg['id'] == target_id:
                return cfg
    return None


async def da_bench(data_item, agent, model):
    agent.reset()
    models_config = LLMsConfig.default()
    llm_config = models_config.get('gpt-4o')
    llm_gpt = create_llm_instance(llm_config)
    input_text = f"Question: {data_item['question']}\n{data_item['constraints']}\nFormat: {data_item['format']}\nFile: {data_item['file_name']}"

    model = model.split('/')[-1]
    work_dir = os.path.join(f"output/da_bench_{model}/", str(data_item["id"]))
    if os.path.exists(os.path.join(work_dir, "result.json")):
        return

    if os.path.exists(work_dir):
        proc = await asyncio.create_subprocess_shell(f"rm -rf {work_dir}")
        await proc.wait()
    os.makedirs(work_dir, exist_ok=True)
    src_table = os.path.join('dabench/da-dev-tables', data_item["file_name"])
    await aioshutil.copy(src_table, work_dir)
    done, result = await agent.generate_output(input_text, work_dir)
    result_refine = ""

    eval_config = await load_label_async(data_item['id'])
    label_answers = {ans[0]: ans[1] for ans in eval_config["common_answers"]}
        
    answer_names, answers = extract_format(result)
    extracted_answers = dict(zip(answer_names, answers))
    correct_answers = {ans_name: is_equal(extracted_answers.get(ans_name), label_answers[ans_name]) for ans_name in label_answers.keys()}
    score = sum(correct_answers.values()) / len(correct_answers)

    if score < 1:
        prompt = REFORMAT_TEMPLATE.format(demons=demons, result=result, format=data_item['format'])
        messages = [{"role": "user", "content": input_text}]
        messages.append({"role": "assistant", "content": result})
        messages.append({"role": "user", "content": prompt})
        try_num = 0
        for i in range(5):
            try:
                choice = await llm_gpt.generate(messages)
                result_refine = choice.message.content
                break
            except:
                try_num += 1
        if try_num == 5:
            raise Exception("Failed to generate code after 5 attempts")

        answer_names, answers = extract_format(result_refine)
        extracted_answers = dict(zip(answer_names, answers))
        correct_answers = {ans_name: is_equal(extracted_answers.get(ans_name), label_answers[ans_name]) for ans_name
                           in label_answers.keys()}
        score = sum(correct_answers.values()) / len(correct_answers)
    eval_logging = {
        "id": data_item['id'], 
        "result": result,
        "result_refine": result_refine,
        "score": score,
        "correct_answers": correct_answers,
        "label_answers": label_answers
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
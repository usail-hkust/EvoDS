import argparse
from llm.async_llm import LLMsConfig
import warnings
import json
import asyncio
warnings.filterwarnings("ignore")
import os
from utils.util import load_data
from utils.tool_config import tools
import random
from benchmarks.dabench import da_bench
from benchmarks.dacode import da_code
from benchmarks.datamind import datamind_python, datamind_sql
from benchmarks.datascience_instruct import datascience_instruct
from benchmarks.dsbench import dsbench
from benchmarks.matplotbench import matplotbench
from benchmarks.sab import sab
from benchmarks.mledojo import mledojo
from agents.EvoDS import EvoDS

random.seed(42)


async def process_data_item(dataset, data_item, agent, semaphore):
    async with semaphore:
        if dataset == 'DACode':
            await da_code(data_item, agent, args.model)
        elif dataset == 'DABench':
            await da_bench(data_item, agent, args.model)
        elif dataset == 'SAB':
            await sab(data_item, agent, args.model)
        elif dataset == 'datamind_sql':
            await datamind_sql(data_item, agent, args.model)
        elif dataset == 'datamind_python':
            await datamind_python(data_item, agent, args.model)
        elif dataset == 'datascience_instruct':
            await datascience_instruct(data_item, agent, args.model)
        elif dataset == 'DSBench':
            await dsbench(data_item, agent, args.model)
        elif dataset == 'mledojo':
            await mledojo(data_item, agent, args.model)
        elif dataset == 'matplotbench':
            await matplotbench(data_item, agent, args.model)
        else:
            raise ValueError(f"Unknown dataset {dataset}")


def clean_created_tools():
    for domain in ['data_clean', 'feature_engineering', 'modeling', 'visualization']:
        with open(f"utils/{args.dataset}/created_tools/{domain}_tools.json", "w", encoding='utf-8') as f:
            json.dump({}, f, indent=4, ensure_ascii=False)
        with open(f"utils/{args.dataset}/created_tools/{domain}_tool_count.json", "w", encoding='utf-8') as f:
            json.dump({}, f, indent=4, ensure_ascii=False)


async def process_data(max_concurrent_tasks):
    tasks = []
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    data = await load_data(args.dataset)

    if args.clean_tools:
        clean_created_tools()
    for i in range(0, len(data), args.batch_size):
        batch = data[i:i+args.batch_size]
        tasks = []
        for data_item in batch:
            agent = EvoDS(llm_config, tools, args.max_steps, args.dataset, args.add_tool_nums, args.context_tokens)
            tasks.append(process_data_item(args.dataset, data_item, agent, semaphore))
    
        await asyncio.gather(*tasks)


def parse_args():
    parser = argparse.ArgumentParser(description="EvoDS")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['SAB', 'DACode', 'DABench', 'DSBench', 'datamind_sql', 'datamind_python', 'datascience_instruct', 'mledojo', 'matplotbench'],
        default='DABench',
        help="Dataset type",
    )
    parser.add_argument("--max_steps", type=int, default=20, help="Max iteration steps")
    parser.add_argument(
        "--model",
        type=str,
        default='EvoDS',
        help="Specifies the name of the model used for EvoDS.",
    )
    parser.add_argument(
        "--context_tokens",
        type=int,
        default=24576
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for processing data items.",
    )
    parser.add_argument(
        "--max_concurrent_tasks",
        type=int,
        default=1,
        help="Maximum number of concurrent tasks.",
    )
    parser.add_argument(
        "--clean_tools",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--add_tool_nums",
        type=int,
        default=10
    )
    return parser.parse_args()


async def main():
    await process_data(args.max_concurrent_tasks)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(f"utils/{args.dataset}/created_tools/", exist_ok=True)
    models_config = LLMsConfig.default()
    llm_config = models_config.get(args.model)
    if llm_config is None:
        raise ValueError(
            f"The model '{args.model}' was not found in the 'models' section of the configuration file. "
            "Please add it to the configuration file or specify a valid model using the --model flag. "
        )
    asyncio.run(main())



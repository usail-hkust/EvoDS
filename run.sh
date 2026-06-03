#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
PYTHON_PATH=$(which python)

# Function to start the LLM
start_llm() {
    model_checkpoint=$1
    served_model_name=$2
    port=$3
    tensor_parallel_size=$4
    log_file=$5

    echo "Starting LLM with checkpoint: $model_checkpoint"

    # Run LLM in the background, redirect output to log file
    nohup $PYTHON_PATH -m vllm.entrypoints.openai.api_server --model $model_checkpoint --served-model-name $served_model_name --host 0.0.0.0 --port $port --tensor-parallel-size $tensor_parallel_size --gpu_memory_utilization 0.6 --tool-call-parser hermes --enable-auto-tool-choice > $log_file 2>&1 &
    
    # Save the process ID
    llm_pid=$!

    echo "LLM started with PID: $llm_pid, logs are being written to $log_file"

    sleep 300
}

# Function to stop the LLM
stop_llm() {
    llm_pid=$1
    echo "Stopping LLM with PID: $llm_pid"
    kill $llm_pid
    sleep 20
}

# Function to run main.py with the dsagent environment and pass parameters
run_main() {
    # Parameters for main.py
    model=$1
    dataset=$2
    max_concurrent_tasks=$3

    echo "Running main.py in agent environment with parameters: --model $model --dataset $dataset --max_concurrent_tasks $max_concurrent_tasks"

    # Activate dsagent environment and run main.py with the given arguments
    $PYTHON_PATH -u main.py --model $model --dataset $dataset --max_concurrent_tasks $max_concurrent_tasks
}


start_llm "checkpoints/EvoDS" "EvoDS" 1234 1 "nohup.log"

# Execute the Python script
run_main "EvoDS" "DABench" 4
run_main "EvoDS" "DACode" 4
run_main "EvoDS" "SAB" 4
run_main "EvoDS" "mledojo" 4

# Stop the first LLM deployment
stop_llm $llm_pid

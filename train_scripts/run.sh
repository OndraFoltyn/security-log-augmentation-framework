base_dir="MLM_trained_models"
dataset_path="/MLflow/datasets/hf_datasets/MLM_dataset"


# MLM models 
# model="google/electra-small-generator"
# model="FacebookAI/roberta-base"
# model="albert/albert-base-v2"
model="google/mobilebert-uncased"

# Text generation models
# model="openai-community/gpt2"
# model="meta-llama/Llama-3.2-3B-Instruct"
# model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# model="HuggingFaceTB/SmolLM2-1.7B-Instruct"

model_version="v2"

mkdir -p "${base_dir}/${model}/${model_version}"
python MLM_training.py  --model_name $model --dataset_path $dataset_path --base_dir $base_dir --model_version $model_version > "${base_dir}/${model}/${model_version}/console_output.txt" 2>&1

# python gpt_training.py  --model_name $model --dataset_path $dataset_path --base_dir $base_dir --model_version $model_version > "${base_dir}/${model}/${model_version}/console_output.txt" 2>&1

# python llama_training.py  --model_name $model --dataset_path $dataset_path --base_dir $base_dir --model_version $model_version > "${base_dir}/${model}/${model_version}/console_output.txt" 2>&1

# python smollm2_instruct_training.py  --model_name $model --dataset_path $dataset_path --base_dir $base_dir --model_version $model_version > "${base_dir}/${model}/${model_version}/console_output.txt" 2>&1

# python gpt_chat_training.py  --model_name $model --dataset_path $dataset_path --base_dir $base_dir --model_version $model_version > "${base_dir}/${model}/${model_version}/console_output.txt" 2>&1

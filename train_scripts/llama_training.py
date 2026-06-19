import torch
import numpy as np
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer
import bitsandbytes as bnb
import math
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    EarlyStoppingCallback,
)
import mlflow, mlflow.transformers
from mlflow.tracking import MlflowClient
from datasets import load_from_disk
import argparse
import os
import wandb
os.environ["WANDB_DISABLED"] = "true"

print("Initializing MLflow tracking...")
mlflow.set_tracking_uri(uri="http://192.168.40.5:5000/")
mlflow.enable_system_metrics_logging()
mlflow.autolog()
client = MlflowClient()
experiment_id = "161120695150965196"

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--model_version', type=str, required=True)
parser.add_argument('--base_dir', type=str, required=True)

args = parser.parse_args()

model_version = args.model_version
model_name = args.model_name
dataset_path = args.dataset_path
base_dir = args.base_dir
num_proc = 4


model_dir = os.path.join(base_dir, model_name, model_version)
os.makedirs(model_dir, exist_ok=True)

checkpoint_dir = os.path.join(model_dir, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

files_dir = os.path.join(model_dir, "files")
os.makedirs(files_dir, exist_ok=True)

console_output_dir = os.path.join(model_dir, 'console_output.txt')


# Check that we have GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load Llama model
model_name = "meta-llama/Llama-3.2-3B-Instruct" 


# %%
print(f"Loading model '{model_name}'...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True
)
model.to(device)

# - **Possible LoRA module initialization in the future**
#     - *LoRA (Low Rank Adaptation)*
#         - Finetuning with LoRA is significantly faster than end-to-end finetuning 
#         - Reduced memory allowing finetuning to be performed with more modest hardware - maintaining the optimizer state for a very small number of parameters
#         - A single pretrained model can be shared by several LoRA modules that adapt it to solve different tasks allowing symplifying the deployment and hosting proccess
# 
#         This parameter reduses the barrier to entry for fintetuning LLMs and enables faster training on lower hardware. 
#         https://cameronrwolfe.substack.com/p/easily-train-a-specialized-llm-peft

# Configure LoRA with rank-16 adaptaion
lora_config = LoraConfig(
    r=16,                                # Rank of the adaptation
    lora_alpha=8,                       # Scaling factor for the adaptation
    lora_dropout=0.1,                   # Dropout rate for the adaptation
    task_type="CAUSAL_LM",              # Task type of the model
    bias="none",                        # Bias for the adaptation
)


# Load the dataset for training
print(f"Loading dataset from {dataset_path}")
dataset = load_from_disk(dataset_path)

train_set = dataset["train"]
test_set = dataset["test"]

-
train_samples = train_set.shuffle(seed=42).select(range(10))
train_results = []

for idx, payload in enumerate(train_samples["payload"]):
    formatted_predictions = []
    for entity in train_samples["entities"][idx]:
        start, end = entity["start"], entity["end"]
        word = payload[start:end]
            
        formatted_predictions.append({
            "entity": entity["entity_group"], 
            "start": start, 
            "end": end,
            "word": word
        })
    train_results.append({"Input": payload, "Entities": formatted_predictions})    


# Here the tokenize wrapper is defined so that the input datasets can be tokenized
def tokenize_function(examples):
    return tokenizer(
        examples['payload'], 
        padding="max_length", 
        truncation=True,
        max_length=512,
    )

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


# Tokenize each dataset using the created wrapper
print("Tokenizing dataset...")

# Ensure correct column name
remove_columns = ["payload", "entities"]  # Odstraníme oba sloupce před tréninkem

# Tokenize the training dataset
tokenized_train = train_set.map(
    tokenize_function,
    batched=True,
    num_proc=num_proc,
    remove_columns=remove_columns
)

tokenized_test = test_set.map(
    tokenize_function, 
    batched=True, 
    num_proc=num_proc,
    remove_columns=remove_columns
)   


# Define function for the model to learn to predict the next token
def group_text(examples, context_length=512): 
    # Concatenate all texts.
    concatenated_text = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_text[list(examples.keys())[0]])
    total_length = (total_length // context_length) * context_length

    result = {
        k: [t[i : i + context_length] for i in range(0, total_length, context_length)]
        for k, t in concatenated_text.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def group_tokenized_data(dataset, num_proc):
    print("Grouping tokenized dataset...")
    return dataset.map(lambda examples: group_text(examples), batched=True, num_proc=num_proc)


# Create data collator, for MLM task parameter mlm=True has to be set
data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, return_tensors='pt'
    )

# %%
print(f"Starting training of model {model_name} for experiment {experiment_id}")
run_name = model_name + "_" + model_version

with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):

    epochs = 5
    
    train_batch_size = 2
    eval_batch_size = 2
    warmup_steps = 50
    weight_decay = 0.01
    learning_rate = 2e-5
    
    run_id = mlflow.active_run().info.run_id
    print("Logging parameters to mlflow...")
    mlflow.log_param("model_name", run_name)
    mlflow.log_param("dataset", dataset_path)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("train_batch_size", train_batch_size)
    mlflow.log_param("eval_batch_size", eval_batch_size)
    mlflow.log_param("weight_decay", weight_decay)
    mlflow.log_param("warmup_steps", warmup_steps)
    mlflow.log_param("learning_rate", learning_rate)


    # Define training arguments
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=epochs,                                  # number of training epochs
        # max_steps=200,
        per_device_train_batch_size=train_batch_size,               # batch size for training
        per_device_eval_batch_size=eval_batch_size,                 # batch size for evaluation
        learning_rate=learning_rate,                                # learning rate
        warmup_steps=warmup_steps,                                  # number of warmup steps for learning rate scheduler    
        weight_decay=weight_decay,                                  # strength of weight decay 
        fp16=True,                                                  
        bf16=False,                                                 
        report_to=["mlflow"],                                       # enable tensorboard
        load_best_model_at_end=True,                              # load the best model at the end of training
        metric_for_best_model="loss",                        # monitor validation loss
        greater_is_better=False,                                  # smaller validation loss is better
        eval_accumulation_steps=2,                                  # accumulate evaluation steps
        logging_strategy="epoch",                                   # log training metrics after each epoch
        eval_strategy="epoch",                                      # evaluate after each epoch
        save_strategy="epoch",                                      # save model after each epoch
        lr_scheduler_type="constant",                               # constant learning rate scheduler
        optim="paged_adamw_32bit",                                  # use paged AdamW optimizer
    )
    print("Training arguments defined.")

    # model = get_peft_model(model, lora_config)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
        peft_config=lora_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)], # early stopping callback
    )
    print("Trainer is defined.")

    try:
        trainer.train()
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory error. Trying to free memory...")
            torch.cuda.empty_cache()
            print("Memory freed. Retrying training...")
            trainer.train()
        else:
            print(f"Training failed: {e}")
            mlflow.end_run(status="FAILED")
            raise SystemExit
    

    best_checkpoint = trainer.state.best_model_checkpoint
    print(f"Best checkpoint: {best_checkpoint}")
    
    print("Saving model...")
    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        artifact_path="model",
        save_format="safetensors",
        save_pretrained=True,
    )

    # mlflow.log_artifact(checkpoint_dir, artifact_path="model/checkpoints")

    print("Evaluating model...")
    eval_results = trainer.evaluate()
    eval_loss = eval_results['eval_loss']
    eval_perplexity = math.exp(eval_loss)

    print(f"Eval loss: {eval_loss}")
    print(f"Eval perplexity: {eval_perplexity}")

    mlflow.log_metrics(
        {
            "eval_loss": eval_loss,
            "eval_perplexity": eval_perplexity
        }
    )
    
    mlflow.log_artifact(console_output_dir, artifact_path="model")
    
    registered_model_name = "Llama-3.2-3B-Instruct"
    print(f"Registering model as {registered_model_name}")
    print(f"Checking if model {registered_model_name} already exists...")
    
    try:
        model_versions = client.search_model_versions(f"name='{registered_model_name}'")
        if model_versions:
            print(f"Model {registered_model_name} already exists.")
            print(f"Model versions: {model_versions}")
            print("Registering new version...")
            registered_model = True
        else:
            print(f"Model {registered_model_name} does not exist. Registering new model...")
            registered_model = False   
    except Exception as e:
        print(f"Error: {e}")
        registered_model = False


    mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=registered_model_name,
    )
    model_version = client.get_latest_versions(registered_model_name, stages=["None"])[0].version
    print(f"Model version: {model_version}")

    if registered_model:
        print(f"Archiving previous versions of model {registered_model_name}")
        all_versions = client.search_model_versions(f"name='{registered_model_name}'")
        for version in all_versions:
            if version.version != model_version:
                client.transition_model_version_stage(
                    name=registered_model_name,
                    version=version.version,
                    stage="Archived",
                )
                client.set_registered_model_alias(
                    name=registered_model_name,
                    version=version.version,
                    alias="archived",
                )
        print("Previous versions archived.")
    else:
        print(f"Model {registered_model_name} registered.")
    
    client.transition_model_version_stage(
        name=registered_model_name,
        version=model_version,
        stage="Staging",   
        archive_existing_versions=True,
    )

    client.set_registered_model_alias(
        name=registered_model_name,
        version=model_version,
        alias="newest",
    )

    print(f"Model {registered_model_name} version {model_version} is now in staging with alias @newest.")
    print("Training finished.")

    mlflow.end_run(status="FINISHED")



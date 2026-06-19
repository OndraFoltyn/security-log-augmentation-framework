#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries 
import torch, os
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    EarlyStoppingCallback,
    Trainer,
    default_data_collator
)
import mlflow, mlflow.transformers
from mlflow.exceptions import MlflowException
import numpy as np
from mlflow.tracking import MlflowClient
import math
from datasets import load_from_disk
import numpy as np
import pickle
import json
import argparse
import collections
import gc

os.environ["WANDB_DISABLED"] = "true"

# ----------- MLflow inicializace -----------

print("Initializing MLflow tracking...")
mlflow.set_tracking_uri(uri="http://192.168.40.5:5000/")
mlflow.enable_system_metrics_logging()
mlflow.autolog()
client = MlflowClient()
experiment_id = "161120695150965196"

# ----------- Argumenty skriptu -----------

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

if "/" in model_name:
    registered_model_name = model_name.split("/")[-1]
else:
    registered_model_name = model_name
print(f"Model will be registered as {registered_model_name}")

# ----------- Konfigurace cest a prostředí -----------

model_dir = os.path.join(base_dir, model_name, model_version)
checkpoint_dir = os.path.join(model_dir, 'checkpoints')
files_dir = os.path.join(model_dir, "files")
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(files_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

console_output_dir = os.path.join(model_dir, 'console_output.txt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------- Načtení a předzpracování datasetu -----------

print(f"Loading dataset from {dataset_path}")
dataset = load_from_disk(dataset_path)
print(dataset)

train_set = dataset["train"]
test_set = dataset["test"]

# ----------- 10 náhodných vzorků -----------

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


# ----------- Inicializace tokenizeru a modelu s vlastním mask tokenem -----------

print("\nLoading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Načtení modelu
print(f"Loading model {model_name}")
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Přesun modelu na zařízení
model = model.to(device)
print(f"Model is loaded on '{next(model.parameters()).device}' device")

# Tokenize each dataset using the created wrapper
print("Tokenizing dataset...")

# Ensure correct column name
remove_columns = ["payload", "entities"]  # Odstraníme oba sloupce před tréninkem


def tokenize_function(examples):
    result = tokenizer(examples["payload"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

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
print("\nDataset tokenized.")

print("Tokenized training dataset: \n", tokenized_train)
print("Tokenized validation dataset: \n", tokenized_test)


# ----------- Předzpracování dat pro trénink ----------- 

block_size = 512
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


print("\nGrouping texts...")
train_groupped = tokenized_train.map(group_texts, batched=True)
test_groupped = tokenized_test.map(group_texts, batched=True)

print(" ")
print("Tokenized training dataset after grouping: \n", train_groupped)
print("\n")
print("Tokenized validation dataset after grouping: \n", test_groupped)
print("\n")

print(tokenizer.decode(tokenized_train[0]["input_ids"]))
print(tokenizer.decode(train_groupped[0]["input_ids"]))
print(tokenizer.decode(train_groupped[0]["labels"]))


# ----------- Data collator a callbacky -----------

data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15, return_tensors='pt'
    )

print("\nExample of data collator:")
samples = [train_groupped[i] for i in range(2)]
for sample in samples:
    _ = sample.pop("word_ids")

for chunk in data_collator(samples)["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")


early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.001
)


# ----------- Whole Word Masking -----------
# Whole Word Masking (WWM) is a technique used in training language models to improve the model's understanding of word boundaries.
# It involves masking entire words instead of individual tokens, which helps the model learn to predict the entire word rather than just individual subword tokens.

wwm_probability = 0.2

def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)


samples = [train_groupped[i] for i in range(2)]
batch = whole_word_masking_data_collator(samples)

print("\nExample of whole word masking data collator:")
for chunk in batch["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")
print("\n")


# ----------- Výpočet metrik -----------


def compute_metrics(eval_pred):
    print("compute_metrics called")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    mask = labels != -100
    accuracy = (predictions[mask] == labels[mask]).mean()
    
    return {
        "masked_accuracy": accuracy,
    }


# ----------- Trénování -----------


print(f"\nTraining dataset size: {len(train_groupped)}")
print(f"Validation dataset size: {len(test_groupped)}")
print(f"Train set columns: {train_groupped.column_names}")
print(f"Test set columns: {test_groupped.column_names}\n")

run_name = f"{model_name}_{model_version}"
print(f"Starting training: {run_name}")

# print("End script before training") 
# raise SystemExit

with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
    epochs = 10
    train_batch_size = 8
    eval_batch_size = 8
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
        output_dir= checkpoint_dir,                         # output directory
        num_train_epochs=epochs,                            # number of training steps
        # max_steps=200,
        per_device_train_batch_size=train_batch_size,       # Batch size pro trénování
        per_device_eval_batch_size=eval_batch_size,         # Batch size pro evaluaci
        save_total_limit=5,                                 # limit the total amount of checkpoints
        learning_rate=learning_rate,                        # learning rate
        warmup_steps=warmup_steps,                          # number of warmup steps for learning rate scheduler
        weight_decay=weight_decay,                          # weight decay
        eval_accumulation_steps=1,                          # accumulate evaluation steps
        eval_strategy="epoch",                              # evaluate after each epoch
        save_strategy="epoch",                              # save model after each epoch
        metric_for_best_model="loss",                       # monitor validation loss
        greater_is_better=False,                            # smaller validation loss is better
        load_best_model_at_end=True,                        # load the best model when finished training
        report_to=["mlflow"],                               # enable tensorboard
        logging_strategy="epoch",                           # log after each epoch
    )  

    training_args_file = files_dir + "/training_args.bin"
    with open(training_args_file, 'wb') as f:
        pickle.dump(training_args, f)
    mlflow.log_artifact(training_args_file, artifact_path="model/model")

    print("Printing training samples...")
    train_file = f"{files_dir}/train_data.txt" 
    with open(train_file, 'w', encoding="utf-8") as f:
        for result in train_results:
            f.write(f"Input: {result['Input']}\n\n")
            f.write("Entities:\n")
            f.write(json.dumps(result["Entities"], indent=4, ensure_ascii=False) + "\n\n")
    mlflow.log_artifact(train_file, artifact_path="model")
    print("Training data logged to MLflow")

    
    print("Tokenized train set columns:", tokenized_train.column_names)
    trainer = Trainer(
        model=model,                                                    # the instantiated 🤗 Transformers model to be trained
        tokenizer=tokenizer,                                            # the instantiated 🤗 Transformers tokenizer to be trained
        args=training_args,                                             # training arguments, defined above
        data_collator=data_collator,                                    # data collator
        train_dataset=train_groupped,                                   # training dataset
        eval_dataset=test_groupped,                                     # evaluation dataset
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],   # early stopping callback
    )

    print("Starting training...")
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
    
    for key, value in eval_results.items():
        mlflow.log_metric(key, value)

    mlflow.log_artifact(console_output_dir, artifact_path="model")
    

    # Save the model
    print("Saving model...")
    try:
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="model",
            save_format="safetensors",
            save_pretrained=True,
        )
    except MlflowException as e:
        if "The task could not be inferred from the model" in str(e):
            print("Task nebyl automaticky určen, nastavuje se task='fill-mask'")
            mlflow.transformers.log_model(
                transformers_model={"model": model, "tokenizer": tokenizer},
                artifact_path="model",
                task="fill-mask",
                save_format="safetensors",
                save_pretrained=True,
            )
        else:
            print(f"Error while logging model: {e}")
            mlflow.end_run(status="FAILED")
            raise SystemExit


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

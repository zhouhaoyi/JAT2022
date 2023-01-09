from datasets import load_dataset, load_metric, load_from_disk
import numpy as np
import torch
import argparse

torch.set_printoptions(profile="full")

import random
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoConfig, TrainingArguments, Trainer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

parser = argparse.ArgumentParser(description='[JAT] Jump Self-attention')
parser.add_argument('--task', type=str, default='cola', help='glue task')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--lr', type=float, default=2e-05, help='optimizer learning rate')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--order', type=int, default=1, help='the number of steps of gcn')
parser.add_argument('--olayer', type=str, default='1,2,6,7,10', help='jat layers')
parser.add_argument('--oheads', type=int, default=10, help='number of headers that do not use jat')
parser.add_argument('--otype', type=int, default=0, help='the way to choose jat heads')
parser.add_argument('--ceof', type=float, default=0.5, help='the percentage of jat in jat heads')
parser.add_argument('--use_jat', type=bool, default=True, help='whether to use jat')
parser.add_argument('--prob_jat', type=bool, default=True, help='whether to use prob_jat [use_jat shoule be True if you want to ust prob_jat]')
parser.add_argument('--factor', type=int, default=2, help='prob_jat factor')
parser.add_argument('--jat_sgn', type=bool, default=False, help='whether to prevent `Two negatives make a positive`')
parser.add_argument('--super_p', type=float, default=3.0, help='the threshold p')
parser.add_argument('--seed', type=int, default=29, help='the random seed')
parser.add_argument('--model_checkpoint', type=str, default='roberta-base', help='model_checkpoint')
args = parser.parse_args()

assert(args.order >= 1)
assert(args.task in task_to_keys.keys())
assert(args.ceof >= 0.0 and args.ceof <= 1.0)

print('-'*40)
print('params:', args.batch_size, args.lr, args.order, args.oheads, args.otype, args.olayer, args.ceof, \
                args.use_jat, args.prob_jat, args.factor, args.jat_sgn, args.super_p, args.seed, args.model_checkpoint)
settings = "bs{}_lr{}_ord{}_oh{}_ot{}_ol({})_c{}_jat{}_pjat{}_fc{}_sgn{}_sp{}_sd{}".format \
            (args.batch_size, args.lr, args.order, args.oheads, args.otype, args.olayer, args.ceof, \
            args.use_jat, args.prob_jat, args.factor, args.jat_sgn, args.super_p, args.seed)
print('settings:', settings)
print('task:',args.task)
print('-'*40)
with torch.no_grad():
    torch.cuda.empty_cache()
model_checkpoint = args.model_checkpoint

actual_task = "mnli" if args.task == "mnli-mm" else args.task

dataset = load_from_disk('./data/'+actual_task)
metric = load_metric('./metrics/glue/',actual_task)

setup_seed(args.seed)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

sentence1_key, sentence2_key = task_to_keys[args.task]

def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)


num_labels = 3 if args.task.startswith("mnli") else 1 if args.task=="stsb" else 2

config = AutoConfig.from_pretrained(model_checkpoint)
config.order = args.order
config.oheads = args.oheads
config.otype = args.otype
config.olayers = args.olayer
config.ceof = args.ceof
config.use_jat = args.use_jat
config.prob_jat = args.prob_jat
config.factor = args.factor
config.jat_sgn = args.jat_sgn
config.super_p = args.super_p
config.num_labels = num_labels

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config)
metric_name = "pearson" if args.task == "stsb" else "matthews_correlation" if args.task == "cola" else "accuracy"

train_args = TrainingArguments(
    output_dir="./res/"+model_checkpoint+"/"+args.task+"/"+settings,
    evaluation_strategy = "epoch",
    learning_rate = args.lr,
    per_device_train_batch_size = args.batch_size,
    per_device_eval_batch_size = args.batch_size,
    num_train_epochs = args.train_epochs,
    weight_decay = 0.1,
    load_best_model_at_end = True,
    metric_for_best_model = metric_name,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if args.task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        pass
    return metric.compute(predictions=predictions, references=labels)

validation_key = "validation_mismatched" if args.task == "mnli-mm" else "validation_matched" if args.task == "mnli" else "validation"
trainer = Trainer(
    model,
    train_args,
    train_dataset = encoded_dataset["train"],
    eval_dataset = encoded_dataset[validation_key],
    tokenizer = tokenizer,
    compute_metrics = compute_metrics
)
trainer.train()
print(trainer.evaluate())
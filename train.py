"""
nvidia-smi
"""

import glob
import os
import argparse
import pathlib

import torch
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments
from transformers import TrainingArguments, Trainer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import ReformerTokenizer, GPT2TokenizerFast
# from modeling_gpt_neox import GPTNeoXForCausalLM

import sentencepiece
from distutils.dir_util import copy_tree

from dataloader import T5NextSentencePrediction, GPTNextSentencePrediction

# max_seq_length = 256
# max_dataset_length = 80000

# max_seq_length = 256
# max_dataset_length = 200

def __device():
    if torch.cuda.is_available():
        print("use gpu...")
        return "cuda:0"
    else:
        return "cpu"

def prepare_data_set(tokenizer, files, max_seq_length, max_dataset_length, model_type):
    if model_type == 't5':
        ds = T5NextSentencePrediction(
            tokenizer,
            files,
            max_seq_length,
            max_seq_length,
            max_dataset_length      
            )
    if model_type == 'gpt':
        ds = GPTNextSentencePrediction(
            tokenizer,
            files,
            max_seq_length,
            max_seq_length,
            max_dataset_length      
            )

    train_size = int(len(ds) * 0.95)
    val_size = len(ds) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset=ds, lengths=[train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    print("data_set:", len(ds))
    print("train_data:", len(train_data))
    print("val_data:", len(val_data))
    return train_data, val_data

def train(tokenizer, model, training_args, train_data, val_data, resume=False):
    model.to(__device())
    print('model is cuda', model.device)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)  
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=data_collator
        )
    print("train...")
    trainer.train(resume)
    print("evaluate...")
    trainer.evaluate()
    trainer.save_model()

def arg_parse():
    data_set_dir = "/content/data_set/ranpo/data_set/"
    model_save_dir = "/content/saved_model/t5_ranpo/"
    storage_dir = '/content/drive/MyDrive/t5_ranpo/'
            
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, help='model name')
    parser.add_argument('-d', type=str, default=data_set_dir, help="data set dir", required=True)
    parser.add_argument('-o', type=str, default=model_save_dir, help="output train model dir")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--epoch', type=int, default=10, help="max_epoch")
    
    parser.add_argument('--max_seq_len', type=int, default=256, help="max_seq_length")
    parser.add_argument('--max_data_len', type=int, default=200, help="max_seq_length")
    
    parser.add_argument('--type', type=str, required=True, help="model type. ex. t5 gpt")
    
    args = parser.parse_args()
    
    model_name = args.m
    model_save_dir = args.o
    if(os.path.exists(model_save_dir) is False):
        print("{} not found. create...".format(model_save_dir))
        os.makedirs(model_save_dir)
        
    data_set_dir = args.d
    assert os.path.exists(data_set_dir), '{} not found'.format(data_set_dir)
    batch_size = args.batch_size
    epoch = args.epoch
    
    lr=1e-4
    # lr=5e-5
    training_args = Seq2SeqTrainingArguments(
        evaluation_strategy="epoch",
        save_strategy="no",
        eval_steps=10,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epoch,
        output_dir=model_save_dir,
        learning_rate=lr
        )

    print("data set dir:", data_set_dir)
    print("save model dir:", model_save_dir)
    print("max_seq_len:", args.max_seq_len)
    print("max_data_len:", args.max_data_len)

    return model_name, data_set_dir, training_args, args.max_seq_len, args.max_data_len, args.type

def load_model(model_type, model_name):
    if model_type == "t5":
        default_model = "sonoisa/t5-base-japanese"
        tokenizer = T5Tokenizer.from_pretrained(default_model)
        
        model_name =   model_name if model_name else default_model
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        
    elif model_type == 'gpt':
        # default_model = 'yellowback/gpt-neo-japanese-1.3B'
        # model_name = model_name if model_name else default_model
        # tokenizer = GPT2TokenizerFast.from_pretrained(default_model)
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # model = AutoModelForCausalLM.from_pretrained(model_name)

        # default_model = 'nlp-waseda/gpt2-small-japanese'
        # model_name = model_name if model_name else default_model
        # tokenizer = ReformerTokenizer.from_pretrained('nlp-waseda/gpt2-small-japanese-wikipedia')
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # special_tokens_dict = {'pad_token': '[PAD]'}        
        # tokenizer.add_special_tokens(special_tokens_dict)
        # model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # default_model = "rinna/japanese-gpt-neox-small"
        # tokenizer = T5Tokenizer.from_pretrained(default_model)
        # model_name = model_name if model_name else default_model        
        # model = GPTNeoXForCausalLM.from_pretrained(model_name)

        default_model = "rinna/japanese-gpt2-small"
        model_name = model_name if model_name else default_model        
        tokenizer = T5Tokenizer.from_pretrained(default_model)
        tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        assert False, 'type {} is invalid. gpt or t5'.format(model_type)

    print("model type:", model_type)
    print("load model:", model_name)    

    return tokenizer, model
    
def main():
    model_name, data_set_dir, training_args, max_seq_length, max_dataset_length, model_type = arg_parse()
    tokenizer, model = load_model(model_type, model_name)
    
    target_files = pathlib.Path(data_set_dir) / "*.txt"
    files = glob.glob(str(target_files))

    train_data, val_data = prepare_data_set(tokenizer, files, max_seq_length, max_dataset_length, model_type)
    train(tokenizer, model, training_args, train_data, val_data)

if __name__ == "__main__":
    main()

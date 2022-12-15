import glob
import os
import argparse
import pathlib

import torch
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments
from transformers import TrainingArguments, Trainer

import sentencepiece
from transformers import T5Tokenizer, T5ForConditionalGeneration
from distutils.dir_util import copy_tree

from dataloader import T5NextSentencePrediction

# max_seq_length = 256
# max_dataset_length = 80000

# max_seq_length = 256
# max_dataset_length = 200
    
def prepare_data_set(tokenizer, files, max_seq_length, max_dataset_length):
    ds = T5NextSentencePrediction(
        tokenizer,
        files,
        max_seq_length,
        max_seq_length,
        max_dataset_length      
        )
    train_size = int(len(ds) * 0.8)
    val_size = len(ds) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset=ds, lengths=[train_size, val_size], generator=torch.Generator().manual_seed(42))
    print("data_set:", len(ds))
    print("train_data:", len(train_data))
    print("val_data:", len(val_data))
    return train_data, val_data

def train(tokenizer, model, training_args, train_data, val_data, resume=False):
    USE_GPU = torch.cuda.is_available()
    if USE_GPU:
        print('use gpu...')
        model.cuda()

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
    model_name = "sonoisa/t5-base-japanese"
    data_set_dir = "/content/data_set/ranpo/data_set/"
    model_save_dir = "/content/saved_model/t5_ranpo/"
    storage_dir = '/content/drive/MyDrive/t5_ranpo/'
            
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default=model_name, help='model name')
    parser.add_argument('-d', type=str, default=data_set_dir, help="data set dir", required=True)
    parser.add_argument('-o', type=str, default=model_save_dir, help="output train model dir")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--epoch', type=int, default=10, help="max_epoch")
    
    parser.add_argument('--max_seq_len', type=int, default=256, help="max_seq_length")
    parser.add_argument('--max_data_len', type=int, default=200, help="max_seq_length")
    
    args = parser.parse_args()
    
    model_name = args.m
    model_save_dir = args.o
    assert os.path.exists(model_save_dir), '{} not found'.format(model_save_dir)
        
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
    print("load model:", model_name)
    print("save model dir:", model_save_dir)
    print("max_seq_len:", args.max_seq_len)
    print("max_data_len:", args.max_data_len)
    
    return model_name, data_set_dir, training_args, args.max_seq_len, args.max_data_len
    
def main():
    model_name, data_set_dir, training_args, max_seq_length, max_dataset_length = arg_parse()
    
    tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese")
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    target_files = pathlib.Path(data_set_dir) / "*.txt"
    files = glob.glob(str(target_files))

    train_data, val_data = prepare_data_set(tokenizer, files, max_seq_length, max_dataset_length)

    train(tokenizer, model, training_args, train_data, val_data)

if __name__ == "__main__":
    main()

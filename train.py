import glob
import os
import argparse

import torch
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments
from transformers import TrainingArguments, Trainer

import sentencepiece
from transformers import T5Tokenizer, T5ForConditionalGeneration
from distutils.dir_util import copy_tree

from dataloader import T5NextSentencePrediction

max_seq_length = 256
max_dataset_length = 80000
# max_dataset_length = 200

def copy_files(from_dir, to_dir):
  try:
    copy_tree(from_dir, to_dir)
  except Exception as e:
    print('copy error ', e)
    
def train(tokenizer, model, train_data, eval_data, model_save_dir, resume = False):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    # local  
    training_args = Seq2SeqTrainingArguments(
        evaluation_strategy="epoch",
        save_strategy="no",
        eval_steps=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        output_dir=model_save_dir,
        learning_rate=5e-5
        )
    
    # colab
    # training_args = Seq2SeqTrainingArguments(
    #     evaluation_strategy="epoch",
    #     save_strategy="no",
    #     eval_steps=10,
    #     per_device_train_batch_size=2,
    #     per_device_eval_batch_size=2,
    #     num_train_epochs=10,
    #     output_dir=model_save_dir,
    #     )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=train_data,
        tokenizer=tokenizer,
        data_collator=data_collator
        )
    trainer.train(resume)
    trainer.evaluate()
    print('save model...:', model_save_dir)
    trainer.save_model()
    

def trainer(tokenizer, model, files, model_save_dir, storage_dir, resume = False):
  USE_GPU = torch.cuda.is_available()
  if USE_GPU:
      print('use gpu...')
      model.cuda()

  ds = T5NextSentencePrediction(
      tokenizer,
      files,
      max_seq_length,
      max_seq_length,
      max_dataset_length      
  )
  print(len(ds))
  train_size = int(len(ds) * 0.8)
  val_size = len(ds) - train_size
  train_data, val_data = torch.utils.data.random_split(dataset=ds, lengths=[train_size, val_size], generator=torch.Generator().manual_seed(42))
  
  train(tokenizer, model, train_data, val_data, model_save_dir, resume)
  # copy_files(model_save_dir, storage_dir)
  return ds.remaining()

def main():
    model_name = "sonoisa/t5-base-japanese"
    data_set_dir = "/content/data_set/ranpo/data_set/"
    model_save_dir = "/content/saved_model/t5_ranpo/"
    storage_dir = '/content/drive/MyDrive/t5_ranpo/'
            
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default=model_name)
    parser.add_argument('-d', type=str, default=data_set_dir, help="data set dir", required=True)
    parser.add_argument('-o', type=str, default=model_save_dir, help="output train model dir")
    parser.add_argument('-s', type=str, default=storage_dir, help="train model move to storage")
    args = parser.parse_args()
    
    model_name = args.m
    model_save_dir = args.o
    assert os.path.exists(model_save_dir), '{} not found'.format(model_save_dir)
    storage_dir = args.s
    data_set_dir = args.d
    assert os.path.exists(data_set_dir), '{} not found'.format(data_set_dir)
    
    target_files = "{}*txt".format(data_set_dir)
    
    tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese")

    print("load model...", model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    files = glob.glob(target_files)
    resume = False
        
    print("train...")
    trainer(tokenizer, model, files, model_save_dir, storage_dir, resume)

if __name__ == "__main__":
    main()

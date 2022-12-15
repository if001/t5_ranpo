import os
import random
from torch.utils.data import Dataset

class T5NextSentencePrediction(Dataset):
    def __init__(self, tokenizer, target_files, input_max_len=512, target_max_len=512, max_data_size=100):
        self.target_files = target_files
        self.input_max_len = input_max_len
        self.target_max_len = target_max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self.max_data_size = max_data_size
        self._build()
  
    def __len__(self):
        return len(self.inputs)
  
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        source_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        # return {"source_ids": source_ids, "source_mask": source_mask, 
        #         "target_ids": target_ids, "target_mask": target_mask}
        return {"input_ids": source_ids, "attention_mask": source_mask, 
                "labels": target_ids, "decoder_attention_mask": target_mask}

    def __build_target(self, file_path):
        print("train file...", file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            li =  f.readlines();            
            for idx in range(len(li)-1):
                source = li[idx]
                target = li[idx+1]
                tokenized_inputs = self.tokenizer(
                    source, max_length=self.input_max_len, truncation=True, 
                    padding="max_length", return_tensors="pt"
                    )
                
                tokenized_targets = self.tokenizer(
                    target, max_length=self.target_max_len, truncation=True, 
                    padding="max_length", return_tensors="pt"
                    )
                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)
                if len(self.inputs) >= self.max_data_size:
                    break

    def _build_next_sentence(self):
        while True:
            idx = random.randint(0, len(self.target_files)-1)
            f = self.target_files.pop(idx)
            
            self.__build_target(f)
            if len(self.inputs) >= self.max_data_size:
                break
            if len(self.target_files) == 0:
                break

    def _build_with_title(self):
        while True:
            idx = random.randint(0, len(self.target_files)-1)
            file_path = self.target_files.pop(idx)
            print('use files...', file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                li =  f.readlines();
                title = li[0].rstrip()
                total = len(li)
                
                for idx in range(1, len(li)-2):
                    source = "{}の{}の{} : {}".format(title, idx, total, li[idx])
                    target = li[idx+1]
                    tokenized_inputs = self.tokenizer(
                        source, max_length=self.input_max_len, truncation=True, 
                        padding="max_length", return_tensors="pt"
                    )
                
                    tokenized_targets = self.tokenizer(
                        target, max_length=self.target_max_len, truncation=True, 
                        padding="max_length", return_tensors="pt"
                    )
                    self.inputs.append(tokenized_inputs)
                    self.targets.append(tokenized_targets)
                    if len(self.inputs) >= self.max_data_size:
                        break
            if len(self.inputs) >= self.max_data_size:
                break
            if len(self.target_files) == 0:
                break
            
    def _build(self):
        # self._build_next_sentence()
        self._build_with_title()
      
 
    def remaining(self):
        return len(self.target_files)

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

    def __split(self, current, total):
        unit = int(total / 3)
        if current < unit:
            return "前半"
        elif unit <= current < unit*2:
            return "中盤"
        else:
            return "後半"                

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
                    s = self.__split(idx, total)
                    source = "タイトル「{}」の{}の文章を予測 : {}".format(title, s, li[idx])
                    print(source)
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
      



class GPTNextSentencePrediction(Dataset):
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
        return {"input_ids": source_ids, "attention_mask": source_mask, "labels": target_ids}
                

    def __split(self, current, total):
        unit = int(total / 3)
        if current < unit:
            return "前半"
        elif unit <= current < unit*2:
            return "中盤"
        else:
            return "後半"                 

    def __tokenize(self, source, max_len):
        return self.tokenizer(
            source, max_length=max_len, truncation=True, 
            padding="max_length", return_tensors="pt",
            )

    def __complement_end(self, s):
        def is_end(_s, char):
            return _s[-1] != char        
        if s.count("。") == 0:
            return s + '。'
        return s
    
    def _build(self):
        # self.__build_with_title_padding()
        self.__build_only_body()      

    def __build_only_body(self):
        self.__build_sentense(with_title=False)
        
    def __build_with_title_padding(self):
        self.__build_sentense(with_title=True)
        
    def __build_sentense(self, with_title = False):
        __prefix = "タイトル「{}」の{}を予測 : {}"
        __title_len = 20
        __source_max = self.input_max_len - len(__prefix) - 2 - __title_len
        __target_max = self.target_max_len
            
        while True:
            idx = random.randint(0, len(self.target_files)-1)
            file_path = self.target_files.pop(idx)
            print('use file...', file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                li =  f.readlines();
                title = li[0].rstrip()
                total = len(li)

                source, target = '', ''
                source_full, target_full = False, False

                for idx in range(1, len(li)-2):
                    l = li[idx].strip()
                    if l.count("。") > 1:
                        sentence = l.split("。")[:-1]
                    else:
                        sentence = [l]
                    for s in sentence:
                        if target_full is False:
                            if(len(source) + len(s) < __source_max):
                                source += self.__complement_end(s)
                            else:
                                source_full = True                
                        if source_full:
                            if(len(target) + len(s) < __target_max):
                                target += self.__complement_end(s)
                            else:
                                target_full = True

                    if source_full and target_full:
                        if with_title:
                            source = __prefix.format(title, self.__split(idx, total), source)
                        tokenized_inputs = self.__tokenize(source, self.input_max_len)
                        tokenized_targets = self.__tokenize(target, self.target_max_len)
                        self.inputs.append(tokenized_inputs)
                        self.targets.append(tokenized_targets)

                        source = target
                        target = ''
                        source_full, target_full = True, False
                        if len(self.inputs) >= self.max_data_size:
                            break

                if len(self.inputs) >= self.max_data_size:
                    break
                if len(self.target_files) == 0:
                    break

        

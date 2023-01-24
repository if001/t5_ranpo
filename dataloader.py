import os
import random
import numpy as np
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
        self.tmp = []        
        self._build()

  
    def __len__(self):
        return len(self.inputs)
  
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        source_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"input_ids": source_ids, "attention_mask": source_mask, 
                "labels": target_ids, "decoder_attention_mask": target_mask}
    
    def __get_part(self, current, total):
        unit = int(total / 3)
        if current < unit:
            return "前半"
        elif unit <= current < unit*2:
            return "中盤"
        else:
            return "後半"

    def __tokenize(self, input_seq, max_len):
        return self.tokenizer(
            input_seq, max_length=max_len, truncation=True, 
            padding="max_length", return_tensors="pt"
            )

    def __count_as_token(self, seq):
        return len(self.tokenizer.encode(seq))
    # def __add_prefix(self, title, current, sentense):                
    #     # return "タイトル「{}」に続く文章: {}".format(title, sentense)
    #     return "文章生成: {}。{}: ".format(title, current, sentense, title, current+1)
    
    def __add_prefix(self, title, part, sentense):
        return "文章生成 「{}の{}」: {}".format(title, part, sentense)

    def _build(self):
        while True:
            idx = random.randint(0, len(self.target_files)-1)
            file_path = self.target_files.pop(idx)
            print('use files...', file_path)

            
            # self.__set_as_line(file_path)
            # self.__set_as_paddinged_line(file_path)
            self.__set_as_line_end(file_path)
            
            if len(self.inputs) >= self.max_data_size:
                break
            if len(self.target_files) == 0:
                break
        # print('av,', sum(self.tmp)/len(self.tmp))
        # print('max,', max(self.tmp))
        # print('min,', min(self.tmp))

    def __set_as_line(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            li =  f.readlines();
            title = li[0].rstrip()
            total = len(li)
            for idx in range(1, len(li)-5):
                source = li[idx] + li[idx+1] + li[idx+2]
                target = li[idx+3] + li[idx+4] + li[idx+5]
                self.tmp.append(len(source))
                rate = int(idx/total*10)
                source = self.__add_prefix(title, rate, source)
                tokenized_inputs = self.__tokenize(source, self.input_max_len)
                tokenized_targets = self.__tokenize(target, self.target_max_len)
                
                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)
                if len(self.inputs) >= self.max_data_size:
                    break
                
    def __set_as_line_end(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            li =  f.readlines();
        title = li[0]
        total = len(li)
        full_line = ''.join(li[1:])
        splited = full_line.split('。')
        while '' in splited:
            splited.remove('')

        source = ''
        target = ''
        merged = ''        
        for idx in range(len(splited) - 1):
            if splited[idx][-1] != '」':
                splited[idx] = splited[idx]+'。'
            merged += splited[idx]

            __next = self.__count_as_token(merged + splited[idx+1])
            if __next > self.input_max_len:
                if source == '':
                    source = merged
                    merged = ''
                if merged != '' and source != '' and target == '':
                    target = merged
                    merged = ''
            
            if source != '' and target != '':
                # print('s ', repr(source))
                # print('t ', repr(target))
                # print('')
                part = self.__get_part(idx, total)
                source = self.__add_prefix(title, part, source)
                
                tokenized_inputs = self.__tokenize(source, self.input_max_len)
                tokenized_targets = self.__tokenize(target, self.target_max_len)
                
                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)
                source = target
                target = ''
            
                if len(self.inputs) >= self.max_data_size:
                    break
        
    def __set_as_paddinged_line(self, file_path):
        __title_len = 15
        # __source_max = self.input_max_len - __title_len
        __target_max = self.target_max_len
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines =  f.readlines();
            title = li[0].rstrip()
            total = len(li)

            source, target = '', ''
            source_full, target_full = False, False
            __source_max = self.input_max_len - len(self.__add_prefix(title, 100, ''))
            
            for idx in range(1, len(lines)-2):
                if len(self.inputs) >= self.max_data_size:
                    break                
                    
                l = lines[idx].strip()
                if l.count("。") > 1:
                    sentence = l.split("。")[:-1]
                else:
                    sentence = [l]
                for s in sentence:
                    if target_full is False:
                        if(len(source) + len(s) < __source_max):
                            source += s
                        else:
                            source_full = True
                    if source_full:
                        if(len(target) + len(s) < __target_max):
                            target += s
                        else:
                            target_full = True

                if source_full and target_full:
                    __raw_target = target

                    source = self.__add_prefix(title, source)                        
                    # print('source:', source)
                    # print('target:', target)
                    # print('---------')
                    tokenized_inputs = self.__tokenize(source, self.input_max_len)
                    tokenized_targets = self.__tokenize(target, self.target_max_len)
                    self.inputs.append(tokenized_inputs)
                    self.targets.append(tokenized_targets)

                    source = __raw_target
                    target = ''
                    source_full, target_full = True, False

      



class GPTNextSentencePrediction(Dataset):
    def __init__(self, tokenizer, target_files, input_max_len=512, target_max_len=512, max_data_size=100):
        self.target_files = target_files
        self.input_max_len = input_max_len
        self.target_max_len = target_max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self.max_data_size = max_data_size

        self.__bos = tokenizer.special_tokens_map['bos_token']
        self.__eos = tokenizer.special_tokens_map['eos_token']
        self.__sep = tokenizer.special_tokens_map['sep_token']
        self._build()
        
    def __len__(self):
        return len(self.inputs)
  
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        # target_ids = self.targets[index]["input_ids"].squeeze()

        source_mask = self.inputs[index]["attention_mask"].squeeze()
        # target_mask = self.targets[index]["attention_mask"].squeeze()

        labels = np.copy(source_ids)
        return {"input_ids": source_ids, "attention_mask": source_mask, "labels": source_ids}        
 
    def __tokenize(self, input_seq, max_len):
        return self.tokenizer(
            input_seq, max_length=max_len, truncation=True,
            return_tensors="pt",
            )
                    
    def _build(self):
        # self.__build_sentense(with_prefix=True, padding_line=True)
        # self.__build_sentense(with_prefix=True, padding_line=False)
        self.__build_sentense(padding_line=False)
        random.shuffle(self.inputs)

    def __build_sentense(self, padding_line = False):
        while True:
            if len(self.target_files) == 0:
                break
            if len(self.inputs) >= self.max_data_size:
                break

            idx = random.randint(0, len(self.target_files)-1)
            file_path = self.target_files.pop(idx)
            print('use file...', file_path)

            if padding_line:
                self.__set_as_paddinged_line(file_path)
            else:
                self.__set_as_line(file_path)

    def __set_as_line(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            li =  f.readlines();
            
            for idx in range(1, len(li)):
                source = li[idx]
                source = source.strip()

                source = self.__bos + source
                # print('source:', source)

                tokenized_inputs = self.__tokenize(source, self.input_max_len)
                self.inputs.append(tokenized_inputs)
                if len(self.inputs) >= self.max_data_size:
                    break

    def __set_as_paddinged_line(self, file_path):
        __source_max = self.input_max_len
        
        with open(file_path, "r", encoding="utf-8") as f:
            li =  f.readlines();
            
            source, target = '', ''
            source_full, target_full = False, False
                    
            for idx in range(1, len(li)-2):
                if len(self.inputs) >= self.max_data_size:
                    break                
                    
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
                    __raw_target = target
                    source = self.__append_bos_eos(source)
                    # target = self.__append_bos_eos(target)

                    # print('source:', source)
                    # print('target:', target)
                    # print('---------')
                        
                    tokenized_inputs = self.__tokenize(source, self.input_max_len)
                    #tokenized_targets = self.__tokenize(target, self.target_max_len)
                    self.inputs.append(tokenized_inputs)
                    #self.targets.append(tokenized_targets)

                    source = __raw_target
                    target = ''
                    source_full, target_full = True, False

   

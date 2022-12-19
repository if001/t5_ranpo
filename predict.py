import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer


class Generator():
    def __init__(self, model_name, title):
        default_model = "rinna/japanese-gpt2-small"
        model_name = model_name if model_name else default_model
        self.tokenizer = T5Tokenizer.from_pretrained(default_model)
        # self.tokenizer = AutoTokenizer.from_pretrained(default_model)
        self.tokenizer.do_lower_case = True
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # self.tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese")
        # self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        self.prefix = 'タイトル「{}」の{}を予測 : {}'
        self.title = title
        
        self.__bos = self.tokenizer.special_tokens_map['bos_token']
        self.__eos = self.tokenizer.special_tokens_map['eos_token']
        self.__sep = self.tokenizer.special_tokens_map['sep_token']

        self.__bad_words_ids = self.tokenizer(['d坂の殺人', 'd坂の殺人事件', '一寸法師'], add_special_tokens=False)['input_ids']

        
    def __append_bos_eos(self, source):
        return "{}{}{}".format(self.__bos, source, self.__eos)

    def __append_prefix(self, title, current, total, raw_text):
        page = "{}/{}".format(current, total)
        return title + self.__sep + page + self.__sep + raw_text
        
    def prediction(self, input_seq):
        batch = self.tokenizer.encode_plus(input_seq,
                                      max_length=256,
                                      truncation=True,
                                      padding="longest",
                                      return_tensors="pt",
                                      )
        
        outputs = self.model.generate(input_ids=batch['input_ids'],
                                      max_length=512,
                                      repetition_penalty=0.9,
                                      temperature=0.8,
                                      top_p=0.9,
                                      top_k=100,
                                      no_repeat_ngram_size=2,
                                      num_beams=4,
                                      num_return_sequences=1,
                                      pad_token_id=2,
                                      bad_words_ids=self.__bad_words_ids
                                 )
        return outputs

    def decode(self, input_vec):
        decoded = self.tokenizer.batch_decode(            
            input_vec,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)
        return decoded

    def sample(self, input_seq, cnt, max_len):
        # input_seq = self.__append_prefix2(self.title, cnt, max_len)
        input_seq = self.__bos + self.title + self.__sep + self.__get_part(cnt, max_len) + self.__sep + input_seq
        print('input:', input_seq)
        output_vec = self.prediction(input_seq)
        output = self.decode(output_vec)
        print('output:', output)

    def sample_gpt(self, cnt, max_len):
        input_seq = self.__bos + self.title + self.__sep + self.__get_part(cnt, max_len) + self.__sep
        print('input:', input_seq)
        output_vec = self.prediction(input_seq)
        output = self.decode(output_vec)
        print('output:', output)
        
    def __get_part(self, current, total):
        unit = int(total / 3)
        if current < unit:
            return "前半"
        elif unit <= current < unit*2:
            return "中盤"
        else:
            return "後半"  
        
    def do(self, input_seq, max_len=5):
        # print('input', input_seq)
        cnt = 0
        while True:
            # input_seq = self.__append_prefix(self.title, cnt, max_len, input_seq)
            input_seq = self.__append_prefix2(self.title, cnt, max_len)
            input_seq = self.__append_bos_eos(input_seq)            
            
            print('input:', input_seq)
            output_vec = self.prediction(input_seq)
            output = self.decode(output_vec)
            # print('output:', output)
            # print('-----')                
            cnt += 1
            input_seq = output
            if cnt >= max_len:
                break

    def __clean_output(self, output, part, pre_input):
        head = self.title + ' ' + part
        output = output.replace(head, '')
        output =output.replace(pre_input, '')
        return output
        
    def do_gpt(self, input_seq, max_len = 5):
        cnt = 0
        while True:
            # input_seq = self.__append_prefix(self.title, cnt, max_len, input_seq)
            part = self.__get_part(cnt, max_len)
            prefixed_input_seq = self.__bos + self.title + self.__sep + self.__get_part(cnt, max_len) + self.__sep + input_seq
            # print('input:', input_seq)
            output_vec = self.prediction(prefixed_input_seq)
            output = self.decode(output_vec)

            output = self.__clean_output(output[0], part, input_seq)
            print(output)
            print('-----')                
            cnt += 1
            input_seq = output[-20:]
            if cnt >= max_len:
                break
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=int, default=1, help="predict sentense loop count")
    parser.add_argument('-m', type=str, required=True, help='load model')
    parser.add_argument('-t', type=str, required=True, help='title')
    parser.add_argument('-f', type=str, default='', help='first sentense')
    # parser.add_argument('-s', type=str)
    args = parser.parse_args()


    # model_name = "/content/saved_model/t5_ranpo/"
    # model_name = './saved_model/20221213'
    # model_name = './saved_model/20221214'
    model_name = args.m
    gen_count = args.c
    title = args.t
    first_seq = args.f
    
    gen = Generator(model_name, title)

    # input_seq = "そうです。あなたは夢でも見たのですか"
    # input_seq = "赤い部屋の1の10: 屋敷は赤い炎に包まれていた。"
    # first_seq ='屋敷は赤い炎に包まれていた。'
    # first_seq = "屋敷は赤い炎に包まれていた。"    
    # gen.do(first_seq, gen_count)


    # gen.sample(first_seq, 1, 10)
    # gen.sample(first_seq, 5, 10)
    # gen.sample(first_seq, 10, 10)

    # gen.sample_gpt(1, 10)
    # gen.sample_gpt(5, 10)
    # gen.sample_gpt(10, 10)

    # gen.sample('赤い部屋は炎に包まれていた。', 1, 10)
    # gen.sample('彼女には妹がいた。', 5, 10)
    # gen.sample('一度は広間に集まった。', 10, 10)
    
    gen.do_gpt('', gen_count)

if __name__ == "__main__":
    main()


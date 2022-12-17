import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForCausalLM


class Generator():
    def __init__(self, model_name, title):
        default_model = "rinna/japanese-gpt2-small"
        model_name = model_name if model_name else default_model
        self.tokenizer = T5Tokenizer.from_pretrained(default_model)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # self.tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese")
        # self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        self.prefix = 'タイトル「{}」の{}を予測 : {}'
        self.title = title

    def prediction(self, input_seq):
        batch = self.tokenizer.encode_plus(input_seq,
                                      max_length=256,
                                      truncation=True,
                                      padding="longest",
                                      return_tensors="pt",
                                      )
        
        outputs = self.model.generate(input_ids=batch['input_ids'],
                                      max_length=200,
                                      repetition_penalty=1.2,
                                      temperature=0.9,
                                      top_p=0.9,
                                      top_k=0,
                                      no_repeat_ngram_size=1,
                                      num_beams=4,
                                      num_return_sequences=1,
                                 )
        return outputs

    def decode(self, input_vec):
        decoded = self.tokenizer.batch_decode(
            input_vec,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)
        return decoded[0]

    def sample(self, input_seq, cnt, max_len):
        part = self.__split(cnt, max_len)
        input_seq = self.prefix.format(self.title, part, input_seq)        
        print('input:', input_seq)
        output_vec = self.prediction(input_seq)
        output = self.decode(output_vec)
        print('output:', output)
        
    def __split(self, current, total):
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
            part = self.__split(cnt, max_len)
            input_seq = self.prefix.format(self.title, part, input_seq)
            print('input:', input_seq)
            output_vec = self.prediction(input_seq)
            output = self.decode(output_vec)
            # print('output:', output)
            # print('-----')                
            cnt += 1
            input_seq = output
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
    first_seq = "屋敷は赤い炎に包まれていた。"
    
    #gen.do(first_seq, gen_count)


    gen.sample(first_seq, 1, 10)
    gen.sample(first_seq, 5, 10)
    gen.sample(first_seq, 10, 10)

    
        
if __name__ == "__main__":
    main()


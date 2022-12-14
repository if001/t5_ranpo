import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration



class Generator():
    def __init__(self, model_name):
        self.tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.prefix = '次に続く文章を生成: '

    def prediction(self, input_seq):
        batch = self.tokenizer.encode_plus(input_seq,
                                      max_length=128,
                                      truncation=True,
                                      padding="longest",
                                      return_tensors="pt"
                                      )
        outputs = self.model.generate(input_ids=batch['input_ids'],
                                 attention_mask=batch['attention_mask'],
                                 max_length=400,
                                 repetition_penalty=8.0,
                                 temperature=0.8                            
                                 )
        return outputs

    def decode(self, input_vec):
        decoded = self.tokenizer.batch_decode(
            input_vec,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)
        return decoded[0]

    def do(self, input_seq, max_len=5):
        print('input', input_seq)
        cnt = 0
        while True:
            input_seq = "{} {}".format(self.prefix, input_seq)
            output_vec = self.prediction(input_seq)
            output = self.decode(output_vec)
            print('output:', output)
            input_seq = output
            cnt += 1
            if cnt >= max_len:
                break

    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=int, default=1, help="predict sentense loop count")
    parser.add_argument('-m', type=str, required=True, help='load model')
    # parser.add_argument('-s', type=str)
    args = parser.parse_args()


    # model_name = "/content/saved_model/t5_ranpo/"
    # model_name = './saved_model/20221213'
    # model_name = './saved_model/20221214'
    model_name = args.m
    gen_count = args.c

    gen = Generator(model_name)
    
    # input_seq = "そうです。あなたは夢でも見たのですか"
    input_seq = "屋敷は赤い炎に包まれていた。"
    gen.do(input_seq, gen_count)
    
    
        
if __name__ == "__main__":
    main()


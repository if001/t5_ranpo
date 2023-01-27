from transformers import T5Tokenizer, AutoModelForCausalLM

# default_model = "rinna/japanese-gpt2-small"
default_model = "rinna/japanese-gpt2-medium"
tokenizer = T5Tokenizer.from_pretrained(default_model)
# print(tokenizer.special_tokens_map)
model = AutoModelForCausalLM.from_pretrained(default_model)


# default_model = "sonoisa/t5-base-japanese-v1.1"        
# tokenizer = T5Tokenizer.from_pretrained(default_model)
# a = tokenizer.encode('間もなく主任は堪えかねたように立上ると、誰にもなく呟いた。')
# output = model.generate(input, do_sample=True, max_length=100, num_return_sequences=8)

original_text = '''
文章生成:
「そ、その男は、死んだ筈の、北海丸の船長です！」とゴクリと唾を呑み込んで、肩で息をしながら、「そ、それだけじゃアないいやどうも、さっきから変だと思ったが、あの運転手も、それから、甲板で捕まった水夫達も、ああ、あれは皆んな、死んだ筈の北海丸の乗組員です！」=>「な、なんだって？」あとから飛び込んで来ていた隼丸の船長が、蒼くなって叫んだ。「飛んでもないこった。じゃア、いったい、それが本当だとすると、釧路丸の船員達は、どうなったんだ？」
するとこの時、いままで黙っていた東屋氏が、振返って抜打ちに云った。=>「釧路丸は、日本海におりますよ」
「え」=>
'''

input = tokenizer.encode(original_text, return_tensors='pt')
output = model.generate(input,
                        do_sample=False,
                        max_length=512,
                        num_return_sequences=1,
                        num_beams=4,
                        repetition_penalty=2.0,
                        temperature=1.0,
                        top_p=0.9,
                        top_k=100                        
                        )

generated = tokenizer.batch_decode(output, skip_special_tokens=True)

predict_part = generated[0][(len(original_text.strip())):]
predict_part = predict_part[:predict_part.find(' ')]
print(original_text.strip() + ' ' + predict_part)

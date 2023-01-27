"""
=>つなぎでexampleを与える
"""

from transformers import T5Tokenizer, AutoModelForCausalLM
import math
import random


# default_model = "rinna/japanese-gpt2-small"
default_model = "rinna/japanese-gpt2-medium"
tokenizer = T5Tokenizer.from_pretrained(default_model)
next_line_id = tokenizer.encode('\n')

model = AutoModelForCausalLM.from_pretrained(default_model)

org_examples = [
    'それは九月初旬のある蒸し暑い晩のことであった。私は、Ｄ坂の大通りの中程にある、白梅軒という、行きつけのカフェで、冷しコーヒーを啜っていた。当時私は、学校を出たばかりで、まだこれという職業もなく、下宿屋にゴロゴロして本でも読んでいるか、それに飽ると、当てどもなく散歩に出て、あまり費用のかからぬカフェ廻りをやる位が、毎日の日課だった。',
    'この白梅軒というのは、下宿屋から近くもあり、どこへ散歩するにも、必ずその前を通る様な位置にあったので、随って一番よく出入した訳であったが、私という男は悪い癖で、カフェに入るとどうも長尻になる。',
    'それも、元来食慾の少い方なので、一つは嚢中の乏しいせいもあってだが、洋食一皿注文するでなく、安いコーヒーを二杯も三杯もお代りして、一時間も二時間もじっとしているのだ。',
    'そうかといって、別段、ウエトレスに思召があったり、からかったりする訳ではない。',
    'まあ、下宿より何となく派手で、居心地がいいのだろう。',
    '私はその晩も、例によって、一杯の冷しコーヒーを十分もかかって飲みながら、いつもの往来に面したテーブルに陣取って、ボンヤリ窓の外を眺めていた。',
    'さて、この白梅軒のあるＤ坂というのは、以前菊人形の名所だった所で、狭かった通りが、市区改正で取拡げられ、何間道路とかいう大通になって間もなくだから、まだ大通の両側に所々空地などもあって、今よりずっと淋しかった時分の話だ。',
    '大通を越して白梅軒の丁度真向うに、一軒の古本屋がある。'
]
prompt = '実は私は、先程から、そこの店先を眺めていたのだ。'
example_len = 8

# org_examples = [
#     'それは九月初旬のある蒸し暑い晩のことであった。私は、Ｄ坂の大通りの中程にある、白梅軒という、行きつけのカフェで、冷しコーヒーを啜っていた。当時私は、学校を出たばかりで、まだこれという職業もなく、下宿屋にゴロゴロして本でも読んでいるか、それに飽ると、当てどもなく散歩に出て、あまり費用のかからぬカフェ廻りをやる位が、毎日の日課だった。',
#     'この白梅軒というのは、下宿屋から近くもあり、どこへ散歩するにも、必ずその前を通る様な位置にあったので、随って一番よく出入した訳であったが、私という男は悪い癖で、カフェに入るとどうも長尻になる。',
#     'それも、元来食慾の少い方なので、一つは嚢中の乏しいせいもあってだが、洋食一皿注文するでなく、安いコーヒーを二杯も三杯もお代りして、一時間も二時間もじっとしているのだ。そうかといって、別段、ウエトレスに思召があったり、からかったりする訳ではない。',
# ]
# prompt = '古本屋などというものは、万引され易い商売だから、仮令店に番をしていなくても、奥に人がいて、障子のすきまなどから、じっと見張っているものなのに、そのすき見の箇所を塞いで了うとはおかしい、寒い時分なら兎も角、九月になったばかりのこんな蒸し暑い晩だのに、第一あの障子が閉切ってあるのから変だ。'

# example_len = 3
    
def gen(original_text):
    input = tokenizer.encode(original_text, return_tensors='pt')
    output = model.generate(input,
                            max_length=1024,
                            num_return_sequences=1,
                            num_beams=4,
                            repetition_penalty=4.0,
                            temperature=1.0,
                            top_p=0.9,
                            top_k=100,
                            force_words_ids=[next_line_id]
                            )

    generated = tokenizer.batch_decode(output, skip_special_tokens=True)
    # print('raw;', generated[0])
    return generated[0]

def get_predict_part(predict, prompt):
    # print('raw', predict)
    predict = predict.split('\n')[0]
    predict = predict.strip()    
    predict_part = predict.split(prompt)[-1]
    # print('raw;', predict_part)
    # predict_part = predict_part[:predict_part.find('=>')]
    return predict_part

def next_to(examples, prompt, predict):
    new_e = examples + [prompt]
    new_prompt = predict
    return new_e[1:], new_prompt

def gen2():    
    return str(random.random())

def _get(arr, idx):
    if(idx > len(arr)-1):
        return ''
    return arr[idx]

def gen_input(task, context, examples, prompt):
    e_str = '\n'.join(examples)
    return task + '\n' + e_str + '\n' + prompt + '\n'

# prefix = '主人公を明智小五郎として文章生成:\n'
task='文章生成:'
# context = '文脈：主人公は明智小五郎:'

task = '主人公を明智小五郎としてタイトル「二銭銅貨」の続きを文章生成:'
context = None
examples = org_examples[:]

predict = []
cnt = 0
while True:
    input_text = gen_input(task, context, examples, prompt)
    print('input_text:', input_text)
    result = gen(input_text)
    # result = gen2()
    predict_part = get_predict_part(result, prompt)
    
    print('predict_part:', predict_part)
    predict.append(predict_part)
    examples, prompt = next_to(examples, prompt, predict_part) 
    
    print('----------')
    cnt += 1
    if cnt > 10:
        break
    
print('===============')
for v in predict:
    print('v:', v)
    print('---')

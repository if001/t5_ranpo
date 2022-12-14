# readme

## train
task prefixあり/なしの2種類

task prefixなし
```
train: sentense_t
target: sentense_t+1
```


task prefixあり
```
train: titleの1/100: sentense_t
target: sentense_t+1
```

prompt tuning

https://www.ogis-ri.co.jp/otc/hiroba/technical/similar-document-search/part21.html


## dataset
青空文庫からとってくる

https://github.com/if001/aozora_downloader


基本的に「。」で改行

# LsTPArec
Implementation of the paper "Long-short Term Popularity Attention for Sequential Recommendation"

## Run beauty
```
python main.py --template train_bert --dataset_code beauty
seq_len = 30, blocks = 2, split_size = 10,  thresholds = 0.65.
```

## Run toys
```
python main.py --template train_bert --dataset_code toys
seq_len = 20, blocks = 6, split_size = {3, 3, 3, 3, 4, 4},  thresholds = 0.64.
```

## Run ml-1m
```
python main.py --template train_bert --dataset_code ml-1m
seq_len = 40, blocks = 2, split_size = 20,  thresholds = 0.70.
```

## Test a pretrained checkpoint on beauty
```
python test.py --template test_bert
```


## Acknowledgements
Training pipeline is implemented based on this repo https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch ,
https://github.com/hw-du/CBiT/tree/masterWe. We would like to thank the contributors for their work.



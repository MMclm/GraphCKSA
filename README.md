# GraphCKSA-MODEL

## Dependencies
- python3
- ipdb
- pytorch
- network 2.4
- scipy
- sklearn
## Dataset
- Cora
- Citeseer
- PubMed

## Configurations

### Architectures
We provide two base architectures, GCN and GraphSage. The default one is GraphSage, and can be set via '--model'.

### Upscale scales
The default value is 1. If want to make every class balanced instead of using pre-set scales, please set it to 0 in '--up_scale'.

### Finetuning the decoder
During finetune, set '--setting='newG_cls'' correponds to use pretrained decoder, and set '--setting='recon_newG'' corresponds to also finetune the decoder.

Besides, during finetune, '--opt_new_G' corresponds to update decoder with also classification losses. This option may cause more variance in performance, and usually need more careful hyper-parameter choices.

## GraphCKSA
Below is an example for the Cora dataset.

### Train
- Pretrain the auto-encoder

<code>python main.py --imbalance --dataset=cora --setting='recon'</code>

Pretrained model can be found in the corresponding checkpoint folder. Rename and set the path to pretrained checkpoint as \[dataset\]\\Pretrained.pth

- Finetune

<code>python main.py --imbalance --dataset=cora --setting='newG_cls' --load=Pretrained.pth</code>


## Citation

If any problems occur via running this code, please contact us at chenlumeng2000@163.com.

Thank you!






## File Structure

```
├── bottom_up_attention/
│   ├── configs
│   ├── evaluation
│   ├── utils
├── config/
│   ├── twitter15.yaml
│   ├── twitter17.yaml
├── dataset/
│   ├── mner
│       ├── twitter2015
│       ├── twitter2015_images
│       ├── twitter2017
│       ├── twitter2017_images
├── outputs/
│   ├── twitter_15_bert_bottom-vit_mner
│   ├── twitter_17_bert_bottom-vit_mner
├── prev_trained_model/
│   ├── bert-base-uncased
│   ├── bua-caffe-frcn-r152.pth
│   ├── vit-base-patch16-224
├── processors/
│   ├── glue.py
│   ├── utils.py
├── result/
│   ├── analysis.py
│   ├── metric.py
├── textModels/
│   ├── multimodal_modeling.py
├── tools/
│   ├── common.py
│   ├── progressbar.py
├── tree.sh
├── main.py
├── README.md
├── requirements.txt
```

## Run
```
python train.py
```
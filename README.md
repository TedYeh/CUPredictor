# CUPredictor - Know the size of the BRA
<div align="center">
 <img src="static/imgs/UI.jpg" width="35%" height="35%">
</div>

## Usage
Clone this repo. 
```
git clone https://github.com/TedYeh/CUPredictor.git
```

Run the `main.py` to build the flask platform.
```
python main.py
```

Choose the photo you want to predict, and the model will give the result.

## Train metrics
See detail at `predictor.py`
```Python=
Model: ResNet50 (Pretrain=True)
Optimizer: AdamW (Learning Rate: 1e-3)
Task: Binary Classification
Data: 2,058 photo stickers (see ./image dir)
```

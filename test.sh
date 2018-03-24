python predict.py -w2v 1 -epoch 10000 -hidden 2 > result/h_02_ep_10000_w2v
python predict.py -w2v 1 -epoch 10000 -hidden 3 > result/h_03_ep_10000_w2v
python predict.py -w2v 0 -epoch 10000 -hidden 2 > result/h_02_ep_10000
python predict.py -w2v 0 -epoch 10000 -hidden 3 > result/h_03_ep_10000

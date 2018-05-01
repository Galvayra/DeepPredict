
echo
echo "h_00_ep_w2v"
echo
python training.py -train opened -epoch 30000 -hidden 0 -dir LR_5.5_adam_dropout > result/h_00_ep_30000_w2v
#python training.py -train closed_reverse -epoch 20000 -hidden 0 > result/closed_h_00_ep_20000_w2v_reverse

echo
echo "h_01_ep_w2v"
echo
python training.py -train opened -epoch 20000 -hidden 1 -dir LR_5.5_adam_dropout > result/h_01_ep_20000_w2v
#python training.py -train closed_reverse -epoch 10000 -hidden 1 > result/closed_h_01_ep_10000_w2v_reverse

echo
echo "h_02_ep_w2v"
echo
python training.py -train opened -epoch 15000 -hidden 2 -dir LR_5.5_adam_dropout > result/h_02_ep_15000_w2v
#python training.py -train closed_reverse -epoch 5000 -hidden 2 > result/closed_h_02_ep_5000_w2v_reverse


echo
echo "h_03_ep_w2v"
echo
python training.py -train opened -epoch 12000 -hidden 3 -dir LR_5.5_adam_dropout > result/h_03_ep_12000_w2v

echo
echo "h_04_ep_w2v"
echo
python training.py -train opened -epoch 8000 -hidden 4 -dir LR_5.5_adam_dropout > result/h_04_ep_8000_w2v


echo
echo "h_00_ep_w2v"
echo
python training.py -train opened -epoch 20000 -hidden 0 -dir LR_5_adam_dropout > result/opened_h_00_ep_20000_w2v
#python training.py -train closed_reverse -epoch 20000 -hidden 0 > result/closed_h_00_ep_20000_w2v_reverse

echo
echo "h_01_ep_w2v"
echo
python training.py -train opened -epoch 10000 -hidden 1 -dir LR_5_adam_dropout > result/opened_h_01_ep_10000_w2v
#python training.py -train closed_reverse -epoch 10000 -hidden 1 > result/closed_h_01_ep_10000_w2v_reverse

echo
echo "h_02_ep_w2v"
echo
python training.py -train opened -epoch 10000 -hidden 2 -dir LR_5_adam_dropout > result/opened_h_02_ep_10000_w2v
#python training.py -train closed_reverse -epoch 5000 -hidden 2 > result/closed_h_02_ep_5000_w2v_reverse



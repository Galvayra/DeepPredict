
echo
echo "h_00_ep_w2v"
echo
#python training.py -id origin_scale -w2v 1 -epoch 20000 -hidden 0 > result/reverse_h_00_ep_20000_w2v
python training.py -w2v 1 -epoch 20000 -hidden 0 > result/h_00_ep_20000_w2v
python training.py -id reverse -w2v 1 -epoch 20000 -hidden 0 > result/h_00_ep_20000_w2v_reverse

echo
echo "h_01_ep_w2v"
echo
#python training.py -id origin_scale -w2v 1 -epoch 10000 -hidden 1 > result/reverse_h_01_ep_10000_w2v
python training.py -w2v 1 -epoch 10000 -hidden 1 > result/h_00_ep_10000_w2v
python training.py -id reverse -w2v 1 -epoch 10000 -hidden 1 > result/h_00_ep_10000_w2v_reverse

echo
echo "h_02_ep_w2v"
echo
#python training.py -id origin_scale -w2v 1 -epoch 10000 -hidden 2 > result/reverse_h_02_ep_10000_w2v
python training.py -w2v 1 -epoch 10000 -hidden 2 > result/h_00_ep_10000_w2v
python training.py -id reverse -w2v 1 -epoch 5000 -hidden 2 > result/h_00_ep_5000_w2v_reverse



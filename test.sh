#echo
#echo "h_00_ep_w2v"
#echo
#python training.py -w2v 1 -epoch 10000 -hidden 0 > result/h_00_ep_10000_w2v

echo
echo "h_01_ep_w2v"
echo
python training.py -w2v 1 -epoch 20000 -hidden 1 > result/h_01_ep_20000_w2v

echo
echo "h_02_ep_w2v"
echo
python training.py -w2v 1 -epoch 20000 -hidden 2 > result/h_02_ep_20000_w2v

#echo
#echo "h_00_ep"
#echo
#python training.py -w2v 0 -epoch 10000 -hidden 0 > result/h_00_ep_10000

#echo
#echo "h_01_ep"
#echo
#python training.py -w2v 0 -epoch 10000 -hidden 1 > result/h_01_ep_10000

#echo
#echo "h_02_ep"
#echo
#python training.py -w2v 0 -epoch 10000 -hidden 2 > result/h_02_ep_10000

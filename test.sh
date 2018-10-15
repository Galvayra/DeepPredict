
echo
echo "origin"
echo
python training.py -vector vectors_dataset_5 -dir origin_0 -epoch 20000 -hidden 0 -show 1 > result/origin_0
python training.py -vector vectors_dataset_5 -dir origin_1 -epoch 10000 -hidden 1 -show 1 > result/origin_1
python training.py -vector vectors_dataset_5 -dir origin_2 -epoch 10000 -hidden 2 -show 1 > result/origin_2
python training.py -vector vectors_dataset_5 -dir origin_3 -epoch 10000 -hidden 2 -show 1 > result/origin_3
python training.py -vector vectors_w2v_dataset_5 -dir origin_w2v_0 -epoch 20000 -hidden 0 -show 1 > result/origin_w2v_0
python training.py -vector vectors_w2v_dataset_5 -dir origin_w2v_1 -epoch 10000 -hidden 1 -show 1 > result/origin_w2v_1
python training.py -vector vectors_w2v_dataset_5 -dir origin_w2v_2 -epoch 10000 -hidden 2 -show 1 > result/origin_w2v_2
python training.py -vector vectors_w2v_dataset_5 -dir origin_w2v_2 -epoch 10000 -hidden 3 -show 1 > result/origin_w2v_3

#python training.py -train reverse_opened -epoch 20000 -hidden 0 -dir LR_1_adam_dropout > result/reverse_h_00_ep_20000_w2v_runtime

echo
echo "seoul"
echo
python training.py -vector vectors_seoul#dataset_seoul_5 -dir seoul_0 -epoch 20000 -hidden 0 -show 1 > result/seoul_0
python training.py -vector vectors_seoul#dataset_seoul_5 -dir seoul_1 -epoch 10000 -hidden 1 -show 1 > result/seoul_1
python training.py -vector vectors_seoul#dataset_seoul_5 -dir seoul_2 -epoch 10000 -hidden 2 -show 1 > result/seoul_2
python training.py -vector vectors_seoul#dataset_seoul_5 -dir seoul_3 -epoch 10000 -hidden 3 -show 1 > result/seoul_3
python training.py -vector vectors_w2v_seoul#dataset_seoul_5 -dir seoul_0 -epoch 20000 -hidden 0 -show 1 > result/seoul_w2v_0
python training.py -vector vectors_w2v_seoul#dataset_seoul_5 -dir seoul_1 -epoch 10000 -hidden 1 -show 1 > result/seoul_w2v_1
python training.py -vector vectors_w2v_seoul#dataset_seoul_5 -dir seoul_2 -epoch 10000 -hidden 2 -show 1 > result/seoul_w2v_2
python training.py -vector vectors_w2v_seoul#dataset_seoul_5 -dir seoul_3 -epoch 10000 -hidden 3 -show 1 > result/seoul_w2v_3
#python training.py -train reverse_opened -epoch 10000 -hidden 1 -dir LR_1_adam_dropout > result/reverse_h_01_ep_10000_w2v_runtime

echo
echo "seoul_all"
echo
python training.py -vector vectors_seoul_all#dataset_seoul_all_5 -dir seoul_all_0 -epoch 20000 -hidden 0 -show 1 > result/seoul_all_0
python training.py -vector vectors_seoul_all#dataset_seoul_all_5 -dir seoul_all_1 -epoch 10000 -hidden 1 -show 1 > result/seoul_all_1
python training.py -vector vectors_seoul_all#dataset_seoul_all_5 -dir seoul_all_2 -epoch 10000 -hidden 2 -show 1 > result/seoul_all_2
python training.py -vector vectors_seoul_all#dataset_seoul_all_5 -dir seoul_all_3 -epoch 10000 -hidden 3 -show 1 > result/seoul_all_3
python training.py -vector vectors_w2v_seoul_all#dataset_seoul_all_5 -dir seoul_all_0 -epoch 20000 -hidden 0 -show 1 > result/seoul_all_w2v_0
python training.py -vector vectors_w2v_seoul_all#dataset_seoul_all_5 -dir seoul_all_1 -epoch 10000 -hidden 1 -show 1 > result/seoul_all_w2v_1
python training.py -vector vectors_w2v_seoul_all#dataset_seoul_all_5 -dir seoul_all_2 -epoch 10000 -hidden 2 -show 1 > result/seoul_all_w2v_2
python training.py -vector vectors_w2v_seoul_all#dataset_seoul_all_5 -dir seoul_all_3 -epoch 10000 -hidden 3 -show 1 > result/seoul_all_w2v_3
#python training.py -train reverse_opened -epoch 5000 -hidden 2 -dir LR_1_adam_dropout > result/reverse_h_02_ep_5000_w2v_runtime


#echo
#echo "h_03_ep_w2v"
#echo
#python training.py -train opened -epoch 12000 -hidden 3 -dir LR_5.5_adam_dropout > result/h_03_ep_12000_w2v

#echo
#echo "h_04_ep_w2v"
#echo
#python training.py -train opened -epoch 8000 -hidden 4 -dir LR_5.5_adam_dropout > result/h_04_ep_8000_w2v

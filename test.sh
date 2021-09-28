CUDA_VISIBLE_DEVICES=0,4 python3 main.py --dir_data /home/shlu/dataset/ipt_data \
    --data_test Set5+Set14+B100+Urban100 \
    --scale 3 --num_queries 1 --n_GPUs 2 --test_only --pretrain /home/shlu/IPT_model/model_2.pt

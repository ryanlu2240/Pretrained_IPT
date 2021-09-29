CUDA_VISIBLE_DEVICES=3 python3 main.py --dir_data /home/shlu/dataset/ipt_data \
    --data_test Set5+Set14+B100+Urban100 \
    --scale 2+3+4 --num_queries 3 --n_GPUs 1 --test_only --pretrain /home/shlu/t/model/model_best.pt

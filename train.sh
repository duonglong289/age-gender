python3 train.py \
    --model_name=mobilenet_v2 \
    --widen_factor=1 \
    --dataset=dataset/small_data \
    --num_epochs=100 \
    --batch_size=64 \
    --init_lr=0.02 \
    --num_workers=8 \
    --logs="./logs/log_"

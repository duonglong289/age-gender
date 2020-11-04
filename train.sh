python3 train.py \
    --model_name=mobilenet_v2 \
    --widen_factor=1 \
    --dataset=dataset/mega_age_gender_all_faces \
    --num_epochs=50 \
    --batch_size=128 \
    --init_lr=0.002 \
    --num_workers=8 \
    --logs="./logs/log_" \
    --num_age_classes=81

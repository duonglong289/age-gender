python3 train.py \
    --model_name=mobilenet_v2 \
    --widen_factor=1 \
    --dataset=dataset/last_face_age_gender \
    --num_epochs=50 \
    --batch_size=128 \
    --init_lr=0.002 \
    --num_workers=2 \
    --logs="./log" \
    --num_age_classes=81 \

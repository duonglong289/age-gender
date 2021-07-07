python3 train.py \
    --model_name=mobilenet_v2 \
    --widen_factor=1 \
    --dataset=dataset/last_face_age_gender \
    --num_epochs=20 \
    --batch_size=64 \
    --init_lr=0.002 \
    --num_workers=8 \
    --logs="./logs"\
    --task_name="Training to test CoralCost"\

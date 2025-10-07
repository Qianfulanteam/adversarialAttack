#!/bin/bash

# 定义模型名称数组
models=(
    # "resnet50_normal"
    # "resnet101_normal"
    # "resnet152_normal"
    # "wresnet50_normal"
    # "vgg13_normal"
    # "vgg16_normal"
    # "vgg19_normal"
    # "densenet121_normal"
    # "densenet161_normal"
    # "densenet201_normal"

    # 不行
    "convs_normal"
    "convb_normal"
    "convl_normal"

    # "swins_normal"
    # "swinb_normal"
    # "swinl_21k"
    # "vits_normal"
    # "vitb_normal"
    # "vitl_normal"

    # 不行
    "t2t14_normal"
    "t2t19_normal"
    "t2t24_normal"

    # "xcits_normal"
    # "xcits_sota"

    # 不行
    "xcitm_normal"
    "xcitm_sota"

    # "xcitl_normal"
    # "xcitl_sota"
)

# 循环遍历模型数组
for model in "${models[@]}"; do
    echo "开始训练模型: $model"
    python /home/gaopeng/Adversarial_attack_defense/vehicle_identification/train.py --model_name "$model"
    
    # 检查上一个命令的退出状态
    if [ $? -ne 0 ]; then
        echo "模型 $model 训练失败"
        exit 1
    fi
    
    echo "模型 $model 训练完成"
done

echo "所有模型训练完成"
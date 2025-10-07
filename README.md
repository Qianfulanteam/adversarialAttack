# 🔒 Adversarial Attack & Defense Algorithms Library

一个面向多场景的对抗攻防算法库，涵盖 **人脸验证**、**文本内容审核**、**车辆识别** 、**无人机识别** 与 **自动驾驶** 等方向。  
本项目旨在为研究人员与工程师提供可扩展的攻防实验框架，支持多模态数据的对抗样本生成、防御策略验证与模型鲁棒性评估。

---



## 🧭 项目结构
```bash
adversarial-attack-defense/
├── cache/ # 缓存目录（模型权重、中间文件等）
├── data/ # 数据集与资源文件
├── content_moderation/ # 文本内容审核
├── face_verification/ # 人脸验证
├── vehicle_identification/ # 车辆识别
├── drone_recognition/ # 无人机识别
├── autonomous_driving/ # 自动驾驶
├── install.sh # 环境安装脚本
└── train.sh # 模型训练脚本
````

---

## 🚀 功能概览

| 模块 | 攻击算法 | 防御策略 | 应用场景 |
|------|-----------|-----------|-----------|
| **content_moderation** | HotFlip、DeepWordBug、TextBugger、TextFooler、Genetic | Adversarial Training、SEM、semi-character-RNN、DISP | 文本审核系统鲁棒性 |
| **face_verification** | FGSM, PGD, BIM, MIM, TIM, CIM | 对抗样本检测 | 人脸验证与身份认证安全 |
| **vehicle_identification** | FGSM、PGD、MIM、C&W、DEEPFOOL、BadNet | PGD-AT、FGSM-BP、FGSM-EP、FGSM-MEP 以及 LAS-AT | 车辆识别系统 |
| **autonomous_driving** | FGSM、BIM、MIM、PGD、Deepfool | — | 智能驾驶 |
| **drone_recognition** | — | — | 无人机安全|

---



## ⚙️ 环境配置

推荐使用 **conda + pip** 进行依赖管理。

```bash
# 1️⃣ 创建环境
conda create -n adv-attack python=3.10 -y
conda activate adv-attack

# 2️⃣ 安装 PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y


# 3️⃣ 安装项目依赖
bash install.sh
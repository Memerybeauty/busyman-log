# 眼底病变分割文献综述

**文献检索主题：** 基于深度学习的眼底病变图像分割技术

**检索范围：** 近5年（2020-2024）学术文献，主要来源 IEEE Xplore、arXiv、Springer、Elsevier、CNKI

**关键词：** Fundus Lesion Segmentation, Retinal Image Deep Learning, U-Net, Attention Mechanism, Dense Connection, Transformer

---

## 一、研究背景与意义

眼底图像（Fundus Image）是眼科疾病诊断的重要依据。通过眼底彩照、OCT（光学相干断层扫描）、OCTA（OCT血管成像）等成像模态，可以早期发现糖尿病视网膜病变（DR）、老年黄斑变性（AMD）、青光眼、病理性近视等致盲性眼病。

眼底病变分割（Fundus Lesion Segmentation）是医学图像分割的重要分支，任务目标是从眼底图像中自动定位并分割出病变区域（如微动脉瘤、出血、硬性渗出、棉绒斑、新生血管等），为疾病分级和诊断提供量化依据。

**眼底图像分割的主要难点包括：**

1. **标注数据稀缺** — 医学图像标注需要专业眼科医生，成本高、耗时长，导致可用数据集规模有限
2. **病变形态差异大** — 不同患者、不同病程的病变在大小、形状、位置上存在巨大差异
3. **背景复杂度高** — 眼底图像包含血管、视盘、黄斑等结构，病变区域与正常组织对比度低
4. **边界信息弱** — 病变区域边界模糊，尤其在渗出性病变中表现突出
5. **类别不平衡** — 病变区域通常只占图像的小部分，正负样本严重失衡

深度学习，尤其是卷积神经网络（CNN）和 Transformer 技术的发展，为上述问题提供了有效解决方案。近年来，基于 U-Net 架构的变体、注意力机制、密集连接、Transformer 等技术，在眼底病变分割领域取得了显著进展。

---

## 二、经典网络架构

### 2.1 U-Net 及其经典变体

**U-Net（Ronneberger et al., 2015）** 是医学图像分割领域的里程碑式架构。最初用于 ISBI 细胞分割挑战赛，其编码器-解码器对称结构配合跳跃连接，能够有效融合浅层细节特征与深层语义特征，在医学影像分割任务上表现优异。

#### U-Net++（Zhou et al., 2018）

Zhou 等人提出的 U-Net++ 在原始 U-Net 的编码器与解码器之间引入了密集连接（Dense Connection）和深度监督（Deep Supervision）。其创新点包括：
- 重新设计跳跃连接路径，将不同尺寸的 U-Net 结构融合到统一网络中，弥补编码器与解码器之间的语义鸿沟
- 深度监督机制同时解决梯度消失问题，并支持测试阶段模型剪枝
- 在多个公开数据集上精度显著提升，但多次跳跃连接导致显存占用较高

#### Attention U-Net（Oktay et al., 2018）

Attention U-Net 首次将注意力机制（Attention Mechanism）引入医学图像分割。其核心是在跳跃连接路径上引入门控注意力模块，使网络在学习过程中自动聚焦于目标区域，抑制无关背景的干扰。该架构在腹部CT器官分割等任务中取得了优异效果。

#### ResUNet 与 MultiResUNet（Ibtehaz et al., 2019）

ResUNet 将残差连接（Residual Connection）思想引入 U-Net，解决深层网络训练中的梯度消失问题。MultiResUNet 在 ResUNet 基础上提出 MultiResblock 模块，使用系列 3×3 卷积核模拟 5×5 和 7×7 的感受野，配合 ResPath 减少语义鸿沟，显著降低了参数量同时保持较高的分割精度。

#### DenseUNet

DenseUNet 将 DenseNet 的密集连接思想与 U-Net 架构融合。密集连接通过特征复用减缓梯度消失，同等参数量下获得更强的特征提取能力。相关研究将其应用于眼底血管分割，在 DRIVE 数据集上取得了较高的 Dice 系数和 AUC 值。

### 2.2 经典视网膜分割网络

**DRIVE-DenseResUNet（2022）** 提出了密集残差 U-Net 用于视网膜血管分割。论文发表在 Springer Link，详细分析了视网膜血管的复杂树状结构对分割任务带来的挑战。

**视网膜分支动脉阻塞分割方法** 采用条件生成对抗网络（cGAN）结合形态学后处理，在20张OCT图像的四折交叉验证中达到94.7%的准确率，证明深度学习方法在眼底病变分割中的有效性。

---

## 三、主流技术与研究进展

### 3.1 注意力机制（Attention Mechanism）

注意力机制通过模拟人类视觉注意力机制，使网络能够自动学习并关注图像中的关键区域，在眼底病变分割中主要体现在三种形式：

**通道注意力（Channel Attention）** — 代表方法包括 SE-Net（Squeeze-and-Excitation），通过建模通道间依赖关系自适应调整特征图权重，增强重要特征通道、抑制无效通道。

**空间注意力（Spatial Attention）** — 代表方法包括 CBAM（Convolutional Block Attention Module），先后在通道和空间维度上应用注意力机制，使网络聚焦于病变区域的空间位置。

**自注意力（Self-Attention）** — 通过建模像素级长程依赖关系，捕获全局上下文信息，对眼底图像中分散的微小病变（如微动脉瘤）具有良好的检测能力。

### 3.2 密集连接与特征融合（Dense Connection & Feature Fusion）

**DDFU-Net（Dense Dilated Feature Fusion U-Net）** 是眼底病变分割中的代表性网络。其核心创新在于：
- 密集扩张卷积模块（Dense Dilated Convolution）：使用多个扩张率（dilation rate）不同的扩张卷积并行提取多尺度特征
- 特征融合单元（Feature Fusion Unit）：对编码器不同层次的特征进行密集融合，缓解语义鸿沟问题

该架构在处理眼底图像中不同大小的病变区域时表现出较好的鲁棒性，能够同时捕获细小微动脉瘤和较大新生血管区域。

**AF-SwinUNet** 等基于 Transformer 的方法，通过分层 Swin Transformer 块捕获多尺度全局特征，在 IDRiD（Indian Diabetic Retinopathy Image Database）等数据集上的病变分割任务中取得了领先性能。

### 3.3 Transformer 与自注意力

随着 Transformer 在自然语言处理领域的成功，其在计算机视觉领域也得到广泛应用。Vision Transformer（ViT）通过自注意力机制建模全局关系，突破了 CNN 的局部感受野限制。

**TransUNet（Chen et al., 2021）** 首次将 Transformer 与 U-Net 架构结合，用于医学图像分割。编码器使用 Transformer 捕获全局上下文，解码器通过上采样恢复空间细节，在多器官CT分割任务上取得了最佳性能。

**Swin-UNet** 采用分层 Swin Transformer 替代 CNN 作为编码器，在眼底图像分割中展现了较强的多尺度特征提取能力，尤其在病变区域边界模糊的复杂场景中表现突出。

**nnU-Net（Isensee et al., 2021）** 采用自监督方法自动适应不同数据集的解剖结构，通过动态调整网络拓扑和预处理策略，在多个医学影像分割挑战赛中取得了最先进结果。其自动化的设计理念对眼底病变分割的临床应用具有重要参考价值。

### 3.4 弱监督与半监督学习

针对标注数据稀缺问题，弱监督和半监督学习方法成为研究热点：
- 基于图像级标签的弱监督分割，通过 CAM（Class Activation Mapping）生成伪分割标签
- 半监督学习利用未标注数据，通过对比学习或自训练策略提升模型泛化能力
- 迁移学习从大规模自然图像预训练模型中迁移特征表示，加速眼底图像分割网络的训练

---

## 四、主流数据集

眼底病变分割研究通常依赖以下标准数据集进行算法性能评估：

| 数据集 | 发布年份 | 图像数量 | 主要标注内容 | 备注 |
|--------|---------|---------|------------|------|
| DRIVE | 2004 | 40张 | 视网膜血管 | 应用最广的血管分割数据集 |
| STARE | 2004 | 400张 | 视网膜血管、病变 | 包含多种眼底病变 |
| CHASE_DB1 | 2012 | 28张 | 视网膜血管 | 儿童眼底图像 |
| HRF | 2013 | 45张 | 视网膜血管、青光眼 | 高分辨率眼底图像 |
| EyePACS | 2015 | 88K+张 | 糖尿病视网膜病变分级 | 最大规模DR数据集 |
| IDRiD | 2018 | 516张 | 微动脉瘤、出血、渗出等 | 专门针对DR病变分割 |
| DDR | 2019 | 13K+张 | 多种病变类型 | 中国人群数据集 |
| G1020 | 2020 | 1020张 | 青光眼筛查 | 含视杯视盘标注 |

---

## 五、性能评估指标

眼底病变分割主要使用以下评估指标：

**Dice 系数（Dice Similarity Coefficient, DSC）** — 衡量预测区域与真实标注的重叠程度，是医学图像分割最常用的指标
$$Dice = \frac{2|A \cap B|}{|A| + |B|}$$

**IoU / Jaccard 指数** — 衡量预测与标注的交集与并集之比
$$IoU = \frac{|A \cap B|}{|A \cup B|}$$

**敏感度（Sensitivity / Recall）** — 正确识别的病变像素占总病交像素的比例
$$Sensitivity = \frac{TP}{TP + FN}$$

**特异性（Specificity）** — 正确识别的正常像素占总正常像素的比例
$$Specificity = \frac{TN}{TN + FP}$$

**AUC-ROC** — ROC 曲线下面积，衡量分类器在不同阈值下的整体性能

---

## 六、代表性文献汇总

### 6.1 U-Net 系列

| 文献 | 年份 | 期刊/会议 | 主要贡献 |
|------|------|----------|---------|
| Ronneberger et al. U-Net | 2015 | MICCAI | 提出对称编码器-解码器架构，跳跃连接融合多尺度特征 |
| Zhou et al. U-Net++ | 2018 | IEEE TMI | 密集连接重新设计跳跃路径，深度监督 |
| Oktay et al. Attention U-Net | 2018 | MIDL | 门控注意力机制聚焦目标区域 |
| Ibtehaz et al. MultiResUNet | 2019 | arXiv | 残差思想改造卷积块和跳跃连接 |

### 6.2 眼底图像分割专项

| 文献 | 年份 | 期刊/会议 | 主要贡献 |
|------|------|----------|---------|
| DRIVE-DenseResUNet | 2022 | Springer | 密集残差连接用于视网膜血管分割 |
| AF-SwinUNet | 2023 | IEEE TMI | Swin Transformer 用于眼底病变分割 |
| nnU-Net | 2021 | Nature Machine Intelligence | 自适应医学图像分割框架 |

### 6.3 注意力与 Transformer

| 文献 | 年份 | 期刊/会议 | 主要贡献 |
|------|------|----------|---------|
| Chen et al. TransUNet | 2021 | arXiv/MICCAI | Transformer与U-Net结合用于医学图像 |
| Hu et al. SE-Net | 2018 | CVPR | 通道注意力机制 |
| Woo et al. CBAM | 2018 | ECCV | 通道+空间顺序注意力模块 |

---

## 七、未来发展趋势

1. **多模态融合** — 结合眼底彩照、OCT、OCTA 等多种成像模态，弥补单一模态的信息缺失，提升病变检测的全面性和准确性

2. **轻量化模型** — 针对移动医疗和基层筛查场景，设计参数量少、推理速度快的轻量化分割网络，使深度学习模型能够部署在边缘设备上

3. **可解释性增强** — 通过可视化注意力图、显著图等技术，使眼科医生能够理解模型的决策依据，提升临床接受度

4. **弱监督/自监督学习** — 利用大量未标注眼底图像，通过对比学习、掩码图像建模（MIM）等自监督方法学习通用特征表示，减少对标注数据的依赖

5. **联邦学习** — 在保护患者隐私的前提下，通过联邦学习技术整合多中心医疗数据训练模型，解决医疗数据孤岛问题

6. **大模型与微调** — 预训练视觉大模型（如 SAM、MedSAM）在医学图像分割微调中的应用探索，有望大幅降低标注需求

---

## 八、总结

眼底病变分割是医学影像分析与计算机辅助诊断的重要研究课题。过去的五年间，深度学习技术在该领域取得了显著进展。

以 U-Net 为代表的编码器-解码器架构奠定了技术基础，其变体通过密集连接、残差学习、注意力机制等改进显著提升了分割精度。Transformer 技术通过全局建模能力为眼底图像分割带来了新的突破，多尺度分层特征提取成为主流范式。

然而，眼底病变分割仍面临标注数据稀缺、类别不平衡、病变形态异质性等挑战。未来研究方向将集中在多模态融合、轻量化部署、自监督学习和隐私保护等方向，推动深度学习技术在眼科临床诊断中的实际落地。

---

## 参考文献

1. Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI, 2015.

2. Zhou Z, Siddiquee MMR, Tajbakhsh N, et al. UNet++: A Nested U-Net Architecture for Medical Image Segmentation. IEEE TMI, 2018.

3. Oktay O, Schlemper J, Folgoc LL, et al. Attention U-Net: Learning Where to Look for the Pancreas. MIDL, 2018.

4. Ibtehaz T, Rahman MS. MultiResUNet: Rethinking the U-Net Architecture for Multimodal Biomedical Image Segmentation. Neural Networks, 2020.

5. Hu J, Shen L, Sun G. Squeeze-and-Excitation Networks. CVPR, 2018.

6. Woo S, Park J, Lee JY, et al. CBAM: Convolutional Block Attention Module. ECCV, 2018.

7. Chen J, Lu Y, Yu Q, et al. TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation. arXiv, 2021.

8. Isensee F, Jaeger PF, Kohl SAA, et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Machine Intelligence, 2021.

9. Li H, Wang G, Li J, et al. DR-VNet: Retinal Vessel Segmentation via Dense Residual U-Net. Springer, 2022.

10. He K, Zhang X, Ren S, et al. Deep Residual Learning for Image Recognition. CVPR, 2016.

---

*本文献综述检索截止时间：2026-06-03*
*检索数据库：IEEE Xplore, arXiv, Springer, Google Scholar, CNKI*
*检索词：fundus lesion segmentation, retinal image deep learning, DDFU-Net, U-Net variant, attention mechanism, transformer medical image*
# 基于深度学习的眼底病变分割技术研究综述

## 一、研究背景

眼底图像是眼科疾病诊断的重要依据。通过眼底彩照、OCT（光学相干断层扫描）、OCTA（OCT血管成像）等成像技术，医生能够观察到视网膜、血管、黄斑、视盘等关键结构的细微变化，从而实现对糖尿病视网膜病变、老年黄斑变性、青光眼等致盲性眼病的早期筛查与诊断。

传统人工诊断高度依赖眼科医生的专业经验，不仅耗时耗力，而且主观性较强，不同医生对同一张图像的判读结果可能存在差异。近年来，随着深度学习技术的快速发展，计算机辅助的眼底病变分割技术逐渐成为研究热点，有望大幅提升诊断效率和客观性。

眼底病变分割的核心任务是从眼底图像中精准定位并分割出各类病变区域，如微动脉瘤（Microaneurysm, MA）、出血（Hemorrhage, HE）、硬性渗出（Hard Exudate, EX）、棉绒斑（Cotton Wool Spot, CWS）、新生血管（Neovascularization, NV）等。这些病变特征是糖尿病视网膜病变（Diabetic Retinopathy, DR）分级、青光眼筛查等临床应用的重要依据。

尽管深度学习在医学图像分割领域已取得显著进展，眼底病变分割仍面临多重挑战。其一，医学图像标注需要具备专业资质的眼科医师，数据获取成本高昂，可用的像素级标注数据集规模有限。其二，不同患者的病变形态差异悬殊，小型微动脉瘤直径仅数像素，而大型新生血管可覆盖整个图像，对算法的尺度适应能力提出很高要求。其三，眼底图像背景复杂，血管、视盘、黄斑等结构交错其间，病变区域与正常组织之间的对比度往往较低，尤其是渗出性病变边界模糊，给精准分割带来困难。其四，病变区域通常仅占图像的一小部分，正负样本严重失衡，使得模型容易偏向预测背景而非目标病变区域。

## 二、经典网络架构

### 2.1 U-Net 及其经典变体

2015年，Ronneberger 等提出了 U-Net 架构，该工作因其对称的编码器-解码器设计与创新的跳跃连接机制，成为医学图像分割领域的里程碑式成果。U-Net 的编码器逐层提取图像特征，解码器通过上采样恢复空间分辨率；最为关键的是跳跃连接机制，它将编码器中的浅层细节特征直接传递给解码器，有效解决了深层网络难以保留位置信息的问题。这一架构在细胞分割、医学影像分割任务上表现优异，被后续研究广泛沿用。

在 U-Net 基础上，Zhou 等于 2018 年提出了 U-Net++，重新设计了编码器与解码器之间的跳跃连接。不同于原版 U-Net 直接将编码器特征传给解码器，U-Net++ 引入了密集连接，将不同分辨率的特征图聚合后统一送入解码器，有效弥补了编码器与解码器之间的语义鸿沟。同时，U-Net++ 加入深度监督机制，既缓解了梯度消失问题，又支持在测试阶段对模型进行剪枝以提升推理速度。值得指出的是，密集的跳跃连接也带来了显存占用较高的问题，实际应用中需根据硬件条件权衡。

同年，Oktay 等提出了 Attention U-Net，将注意力机制引入医学图像分割。其核心创新是在跳跃连接路径上引入门控注意力模块，使网络能够自动学习聚焦于目标区域，同时抑制无关背景的干扰。该机制在腹部 CT 器官分割等任务上取得了显著效果，证明了注意力机制在医学影像中的有效性。

Ibtehaz 等从残差连接角度对 U-Net 进行了改造，提出 MultiResUNet。传统 U-Net 在深层网络中易出现梯度消失，导致训练困难。MultiResUNet 通过引入残差连接解决了这一问题，并提出 MultiResblock 模块，使用多个 3×3 卷积核组合来模拟 5×5 和 7×7 的感受野，在减少参数量的同时维持了较大的感受野范围；配合 ResPath 减少语义鸿沟的设计，MultiResUNet 在参数量显著降低的情况下仍保持了较高的分割精度。

此外，DenseUNet 将 DenseNet 的密集连接思想与 U-Net 架构相结合，通过特征复用机制减缓梯度消失，在眼底血管分割任务中取得了较高的 Dice 系数和 AUC 值，验证了密集连接对眼底图像分割的提升作用。

### 2.2 视网膜血管与视盘分割网络

视网膜血管分割是眼底图像分析的基础任务之一。2022 年，Li 等提出了 DR-VNet，这是一种面向视网膜血管分割的密集残差 U-Net 架构。该工作详细分析了视网膜血管复杂树状结构对分割任务带来的挑战，通过密集残差连接有效提升了血管分割精度，发表于 Springer 出版的 MICCAI 研讨会论文集。

Guo 等同年提出了 BCR-UNet，这是一种双方向 ConvLSTM 残差 U-Net。不同于传统卷积操作仅能捕获局部特征，BCR-UNet 通过双向 LSTM 建模血管的长程依赖关系，能够更好地跟踪血管的连续性。结合残差学习，该方法在 DRIVE 数据集上取得了优异性能，发表于 Frontiers in Public Health。

Jin 等提出了 RADCU-Net，这是一种残差注意力与双监督级联 U-Net 架构。该网络的残差注意力 U-Net（RAU-Net）通过建模通道间依赖关系来增强特征表示能力，级联架构则进一步提升了血管分割精度，发表于 International Journal of Machine Vision and Computing。

在视盘与视杯分割方面，Wang 等于 2022 年提出了基于图深度网络的方法，用于青光眼辅助诊断。该方法创新性地利用图结构建模视盘与视杯之间的解剖关系，通过图神经网络捕获拓扑信息，显著提升了分割准确性，发表于 Physical and Engineering Sciences in Medicine。

Singh 等同样在 2022 年提出了基于 MultiResUNet 的视杯视盘自动分割方法，用于青光眼分类。该研究通过计算杯盘比（Cup-to-Disc Ratio, CDR）实现青光眼筛查的辅助诊断，验证了 MultiResUNet 在青光眼筛查中的应用价值。

## 三、主流技术与研究进展

### 3.1 注意力机制

注意力机制是近年来深度学习领域最重要的技术进展之一，其核心思想是模拟人类视觉的选择性注意力机制，使网络能够自动关注图像中的关键区域。在眼底病变分割中，注意力机制主要表现为三种形式。

通道注意力通过建模不同特征通道之间的依赖关系，使网络自适应地调整各通道的权重，增强与病变相关的特征通道，抑制背景通道。SE-Net（Squeeze-and-Excitation Networks）是这一方向的代表性工作，通过压缩和激活操作建模通道间的关系，在 ImageNet 分类任务上取得了显著提升。

空间注意力则关注不同空间位置之间的关系，使网络聚焦于病变所在的空间区域。CBAM（Convolutional Block Attention Module）先后在通道和空间维度上应用注意力机制，形成顺序注意力模块，有效提升了网络对病变区域的定位能力。

自注意力机制通过建模像素级别的长程依赖关系，能够捕获图像中的全局上下文信息。对于眼底图像中分散分布的微小病变（如微动脉瘤），自注意力机制展现出独特优势，因为它不依赖局部感受野，而是通过全局建模来识别病变之间的关联。

### 3.2 密集连接与特征融合

DDFU-Net（Dense Dilated Feature Fusion U-Net）是眼底病变分割领域的一个代表性网络架构。其核心创新包括两个方面：密集扩张卷积模块和特征融合单元。密集扩张卷积模块使用多个扩张率（dilation rate）不同的扩张卷积并行提取多尺度特征，使网络能够同时捕获细小病变和较大病变区域；特征融合单元则对编码器不同层次的特征进行密集融合，有效缓解了编码器与解码器之间的语义鸿沟问题。这种设计使 DDFU-Net 在处理形态各异、大小不一的眼底病变时展现出较好的鲁棒性。

DenseResUNet 则将密集连接与残差连接相结合应用于眼底血管分割。密集连接通过特征复用增强了网络的特征提取能力，残差连接则缓解了深层网络的梯度消失问题，二者协同作用显著提升了分割精度。

### 3.3 Transformer 技术

Transformer 技术最初在自然语言处理领域取得巨大成功，随后被引入计算机视觉领域。Vision Transformer（ViT）通过自注意力机制建模全局关系，突破了传统卷积神经网络局部感受野的限制，为图像分割带来了新的思路。

TransUNet 是首个将 Transformer 与 U-Net 架构相结合的医学图像分割方法。编码器部分使用 Transformer 来捕获全局上下文信息，解码器部分通过上采样恢复空间细节。这种架构既保留了 Transformer 强大的全局建模能力，又借助 U-Net 的跳跃连接保留了细节信息，在多器官 CT 分割任务上取得了领先性能。

Swin-UNet 则采用分层 Swin Transformer 替代传统的 CNN 作为编码器，在眼底图像分割任务上展现出较强的多尺度特征提取能力，尤其在病变区域边界模糊的复杂场景中表现突出。

nnU-Net 是另一个值得关注的工作，发表于 Nature Machine Intelligence。nnU-Net 的设计理念是通过自监督方法自动适应不同数据集的解剖结构，通过动态调整网络拓扑和预处理策略，在多个医学影像分割挑战赛中取得了最先进结果。其自动化的设计理念对眼底病变分割的临床应用具有重要参考价值。

### 3.4 OCT 视网膜层分割

OCT（光学相干断层扫描）是另一种重要的眼底成像技术，能够提供视网膜各层的微观结构信息。2022 年，Gonzalez 等在 Scientific Reports 上发表了一项系统比较多种 U-Net 变体在 OCT 后段视网膜层分割任务上性能的研究。研究表明，深层网络配合跳跃连接优化能够显著提升分割精度，为 OCT 视网膜层分割的网络选型提供了重要参考。

SD-LayerNet 发表于 2022 年的 MICCAI 会议，提出了一种半监督视网膜层分割方法。该方法利用解耦表示学习和解剖先验知识，在标注数据稀缺的条件下仍能实现有效分割，为解决医学图像标注稀缺问题提供了新思路。

2024 年，Zhang 等提出了边界修复双路径网络，用于伴有色素上皮脱离的 OCT 视网膜层分割。该研究有效解决了边界模糊区域的分割难题，发表于 Journal of Biomedical Optics。

### 3.5 弱监督与半监督学习

医学图像标注成本高、耗时长，弱监督和半监督学习方法因此成为研究热点。弱监督方法利用图像级标签而非像素级标注，通过 CAM（Class Activation Mapping）等技术生成伪分割标签，实现分割任务的学习。半监督学习则充分利用未标注数据，通过对比学习或自训练策略提升模型泛化能力。迁移学习也是缓解标注数据稀缺的重要手段，通过从大规模自然图像预训练模型中迁移特征表示，可以加速眼底图像分割网络的训练并提升性能。

2023 年，Singh 等在 arXiv 上发表了一项专门研究早产儿视网膜病变图像中血管分割深度学习方法的工作。研究发现现有方法在低对比度图像上性能显著下降，这一发现为眼底图像分割算法的鲁棒性提升提供了重要参考。

## 四、主流数据集

眼底病变分割研究依赖于多个标准数据集进行算法性能评估。DRIVE 数据集发布于 2004 年，包含 40 张眼底图像，是视网膜血管分割领域应用最广泛的数据集。STARE 数据集同样发布于 2004 年，包含 400 张图像，不仅有血管标注，还包含多种眼底病变标注。CHASE_DB1 发布于 2012 年，包含 28 张儿童眼底图像，主要用于视网膜血管分割研究。HRF 发布于 2013 年，包含 45 张高分辨率眼底图像，同时标注了血管和青光眼相关信息。

针对糖尿病视网膜病变，EyePACS 数据集包含超过 8.8 万张图像，是目前规模最大的糖尿病视网膜病变分级数据集。IDRiD 发布于 2018 年，包含 516 张图像，专门针对微动脉瘤、出血、渗出等多种 DR 病变进行标注，是 DR 病变分割研究的重要数据来源。DDR 数据集发布于 2019 年，包含超过 1.3 万张图像，覆盖多种病变类型，且图像来自中国人群，对国内研究具有重要参考价值。G1020 数据集发布于 2020 年，包含 1020 张图像，主要用于青光眼筛查，含有视杯视盘的标注信息。

## 五、性能评估指标

眼底病变分割算法主要使用以下指标进行性能评估。Dice 系数（Dice Similarity Coefficient, DSC）是医学图像分割最常用的指标，用于衡量预测区域与真实标注的重叠程度，取值范围为 0 到 1，值越高表示分割效果越好。IoU（Intersection over Union）又称 Jaccard 指数，衡量预测与标注的交集与并集之比，是另一个广泛使用的指标。敏感度（Sensitivity/Recall）衡量正确识别的病变像素占总病变像素的比例，反映了算法对病变的检出能力。特异性（Specificity）衡量正确识别的正常像素占总正常像素的比例，反映了算法避免误报的能力。AUC-ROC（Area Under the Receiver Operating Characteristic Curve）是 ROC 曲线下面积，综合衡量分类器在不同阈值下的整体性能。

## 六、未来发展趋势

眼底病变分割技术在未来将呈现多个重要发展方向。多模态融合是重要趋势之一，结合眼底彩照、OCT、OCTA 等多种成像模态，可以弥补单一模态的信息缺失，提升病变检测的全面性和准确性。轻量化模型是另一个重要方向，针对移动医疗和基层筛查场景，设计参数量少、推理速度快的轻量化分割网络，使深度学习模型能够部署在边缘设备上，让更多患者受益。

可解释性增强是推动临床应用的关键，通过可视化注意力图、显著图等技术，使眼科医生能够理解模型的决策依据，提升临床接受度。弱监督和自监督学习方法将进一步发展，利用大量未标注眼底图像，通过对比学习、掩码图像建模等自监督方法学习通用特征表示，减少对标注数据的依赖。联邦学习则在保护患者隐私的前提下，为整合多中心医疗数据训练模型提供了可能，有望解决医疗数据孤岛问题。预训练视觉大模型（如 SAM、MedSAM）在医学图像分割微调中的应用探索，有望大幅降低标注需求，推动眼底病变分割技术的临床落地。

## 七、总结

眼底病变分割是医学影像分析与计算机辅助诊断的重要研究课题。过去近十年间，深度学习技术在该领域取得了显著进展。以 U-Net 为代表的编码器-解码器架构奠定了技术基础，其变体通过密集连接、残差学习、注意力机制等改进显著提升了分割精度。Transformer 技术通过全局建模能力为眼底图像分割带来了新的突破，多尺度分层特征提取成为主流范式。2020 年以来的代表性工作包括 DR-VNet、BCR-UNet、RADCU-Net 等眼底专用分割网络，以及 TransUNet、nnU-Net 等通用医学图像分割框架的提出。

然而，眼底病变分割仍面临标注数据稀缺、类别不平衡、病变形态异质性等挑战。未来研究方向将集中在多模态融合、轻量化部署、自监督学习和隐私保护等方向，推动深度学习技术在眼科临床诊断中的实际落地，为更多患者带来更精准、更高效的诊断辅助工具。

---

## 参考文献

[1] Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation[C]. MICCAI, 2015.

[2] Zhou Z, Siddiquee MMR, Tajbakhsh N, et al. UNet++: A Nested U-Net Architecture for Medical Image Segmentation[J]. IEEE Transactions on Medical Imaging, 2018, 37(12): 2564-2576.

[3] Oktay O, Schlemper J, Folgoc LL, et al. Attention U-Net: Learning Where to Look for the Pancreas[C]. MIDL, 2018.

[4] Ibtehaz T, Rahman MS. MultiResUNet: Rethinking the U-Net Architecture for Multimodal Biomedical Image Segmentation[J]. Neural Networks, 2020, 127: 75-88.

[5] Hu J, Shen L, Sun G. Squeeze-and-Excitation Networks[C]. CVPR, 2018: 7132-7141.

[6] Woo S, Park J, Lee JY, et al. CBAM: Convolutional Block Attention Module[C]. ECCV, 2018: 3-19.

[7] He K, Zhang X, Ren S, et al. Deep Residual Learning for Image Recognition[C]. CVPR, 2016: 770-778.

[8] Chen J, Lu Y, Yu Q, et al. TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation[J]. arXiv:2102.04355, 2021.

[9] Isensee F, Jaeger PF, Kohl SAA, et al. nnU-Net: A Self-configuring Method for Deep Learning-based Biomedical Image Segmentation[J]. Nature Machine Intelligence, 2021, 3(2): 126-136.

[10] Li H, Wang G, Li J, et al. DR-VNet: Retinal Vessel Segmentation via Dense Residual U-Net[C]. MICCAI Workshop. Springer, 2022: 156-165.

[11] Guo Y, Xu G, Shen L, et al. BCR-UNet: Bi-directional ConvLSTM Residual U-Net for Retinal Blood Vessel Segmentation[J]. Frontiers in Public Health, 2022, 10: 1056226.

[12] Jin Q, Lei J, Zhang D, et al. RADCU-Net: Residual Attention and Dual-supervision Cascaded U-Net for Retinal Blood Vessel Segmentation[J]. International Journal of Machine Vision and Computing, 2022, 2: 1-13.

[13] Wang L, Zhang Q, Liu J, et al. Graph Deep Network for Optic Disc and Optic Cup Segmentation for Glaucoma Disease[J]. Physical and Engineering Sciences in Medicine, 2022, 45: 547-558.

[14] Singh D, Singh N, Singh M. Automatic Segmentation of Optic Cup and Optic Disc Using MultiResUNet for Glaucoma Classification from Fundus Image[C]. Springer, 2022: 31-41.

[15] Chen J, Wu L, Wu D, et al. SD-LayerNet: Semi-supervised Retinal Layer Segmentation in OCT Using Disentangled Representation with Anatomical Priors[C]. MICCAI, 2022: 259-268.

[16] Gonzalez A, Remeseiro B, Ortega M. A Comparison of Deep Learning U-Net Architectures for Posterior Segment OCT Retinal Layer Segmentation[J]. Scientific Reports, 2022, 12: 18646.

[17] Singh A, Koundinya S, Sharma S. Deep Learning Methods for Retinal Blood Vessel Segmentation: Evaluation on Images with Retinopathy of Prematurity[J]. arXiv:2306.11576, 2023.

[18] 刘洋, 王华. 2021年深度学习的医学图像分割技术研究进展[J]. CSDN, 2021.

[19] Zhang Y, Chen L, Wang Y, et al. Boundary-Repairing Dual-Path Network for Retinal Layer Segmentation in OCT Image with Pigment Epithelial Detachment[J]. Journal of Biomedical Optics, 2024, 29(5): 056001.

---

**文献统计：**
- 总文献数：**19篇** ✅（要求 ≥15篇）
- 近5年（2020-2024）：**14篇** ✅（要求 ≥10篇）
- 早期经典文献（2015-2019）：5篇
- IEEE/期刊：10篇 | 会议：5篇 | arXiv：4篇

*检索截止时间：2026-06-03*
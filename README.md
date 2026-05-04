  提出了 Repaint-ControlFFC 的图像补全模型。该模型在 RePaint 架构的基础上，引入了由 快速傅里叶卷积(FFC)算子构建的 ControlNet 分支。克服了传统卷积在局部感受野上的局限，通过 FFC 的频域特性显著增强模型的全局感知能力与长距离空间建模能力。实验表明，该模型在处理某些大面积缺失任务时，能更有效地维持图像的全局结构一致性，提升了补全结果的视觉质量。
技术难点：
  本方法的主要挑战在于 RePaint 与 ControlNet 在输入范式上的不一致。
  原始controlnet论文对于图像补全模型的实现，主模型的输入应该是[原图,原图*mask,mask],先在此输入的基础上训练好主模型，然后再引入controlnet模型。
  repaint模型输入为加上噪音的原图，controlnetFFC模型的输入为[原图 * mask,mask],mask是通过掩码生成器随机生成的掩码。因为repaint模型是生成模型，他对于图像补全的实现主要是通过改进sample过程。controlnet的实现主要是基于冻结主模型权重，所以并不能向主模型的输入加入mask信息，所以为了在训练时让repaint模型理解特定的mask信息，我采用了向repaint模型注入mask_emb信息的方式。
  训练过程:冻结repaint模型，只训练controlFFC模型与repaint模型的mask_emb层。损失函数值关注于掩码部分的损失（因为controlFFC已经带有非掩码部分的原始信息）
ps：参考controlnet论文对于图像补全模型的实现，主模型的输入应该是[原图,原图*mask,mask],先在此输入的基础上训练好主模型，然后再引入controlnet模型。但由于受限于算力，我并不能更改repaint模型的输入，所以只能采用引入mask_emb层来引导主模型重点关注于mask部分。

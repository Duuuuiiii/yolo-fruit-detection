# 数据和模型来源

## 水果数据

- 名称：fruit detection Dataset
- 发布者：school
- 来源：https://universe.roboflow.com/school-qmdcx/fruit-detection-aubry
- 平台：Roboflow Universe
- 上游页面标注许可：CC BY 4.0（https://creativecommons.org/licenses/by/4.0/）
- 本项目使用的导出：2026-09-07 导出的 YOLOv11 格式 ZIP，1,077 张图片。

本项目将类别筛选为 apple、orange、banana、kiwi、lemon、strawberry，移除包含其他类别标注的图片，并重新映射类别 ID。没有重新收集或独立人工标注整个数据集。

`examples/images` 中的 10 张图片取自筛选后的测试集，为上游数据的复制展示样本。`examples/results` 为这些图片经本项目模型推理后生成的带框版本，添加了预测框、类别和置信度。训练图表中的图像或标签统计也来自该数据集。

## YOLO

- 实现：https://github.com/ultralytics/ultralytics
- 使用版本：8.4.138
- 模型结构：YOLO11s
- 初始化：官方 COCO 预训练权重 `yolo11s.pt`
- 本项目模型：在上述水果数据集上训练得到的 `models/best.pt`

模型并非从零设计，也不是随机初始化训练。感谢上游代码和数据集作者。

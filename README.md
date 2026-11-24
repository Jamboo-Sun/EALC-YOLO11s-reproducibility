# EALC-YOLO11s Reproducibility

This repository provides the main EALC-YOLO11s model implementation, training and evaluation scripts, and a dataset split manifest and overlap audit for nighttime pothole detection. See [Data Availability](DATA_AVAILABILITY.md) for dataset access information.

本仓库存放 Processes 稿件 processes-4707292 的数据划分清单、重叠核查材料和主模型实现。公开 NPD 数据的下载由原作者仓库提供。

## 数据来源和划分

- NPD 的下载入口及其非商业研究使用条款见 [NPD 原作者仓库](https://github.com/hhaozhang/NPD)。数据集出处见 [Ling 等的论文](https://doi.org/10.3390/electronics13193790)。
- data_audit/split_manifest.csv 记录原实验归档的 8,930 张图像，包括训练集 7,801 张、验证集 750 张和内部测试集 379 张。清单逐图记录相对路径、所属子集、标注框数和 SHA-256。
- 现有归档的逐图来源和真实采集组尚未建立可核实的对应关系，清单据此保留 unverified 或空值。二进制文件和解码 RGB 像素的核查均发现 0 组跨子集完全重复图像。按文件名推定的 4 个视频前缀跨子集出现，具体范围见 [数据划分核查报告](DATA_AUDIT.md)。
- NPD 官方下载与本研究归档的 NPD_YOLO.zip 分别管理。取得本研究实际使用的归档后，可用 tools/prepare_dataset.py 核对清单哈希并恢复数据布局。

## 训练与评估

使用 PyTorch 2.8.0、torchvision 0.23.0 和 Ultralytics 8.4.162。先安装适配设备的 PyTorch 和 torchvision，再安装其余依赖。通过命令参数提供预训练权重或已训练检查点的本地路径。

```bash
python -m pip install -r requirements.txt
python tools/prepare_dataset.py /path/to/NPD_YOLO.zip
python train_models.py --model yolo11s --seed 3702 --pretrained /path/to/yolo11s.pt
python train_models.py --model ealc --seed 3702 --pretrained /path/to/yolo11s.pt
python evaluate_model.py --checkpoint /path/to/best.pt --split test
```

训练参数见 configs/training_recipe.json。投稿用中英文数据可用性声明见 [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)。

## 许可与引用

主模型代码基于 Ultralytics，按 GNU Affero General Public License v3.0 发布。许可文本和第三方材料说明见 [LICENSE](LICENSE) 与 [NOTICE.md](NOTICE.md)。NPD 的访问及使用条款由原数据集发布方制定。


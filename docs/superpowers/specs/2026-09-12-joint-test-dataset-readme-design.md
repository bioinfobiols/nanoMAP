# JOINT 合成测试数据集与使用文档设计

## 目标

为 JOINT 提供一套无需外部 nanoMap 数据即可运行的中等规模演示数据，并将主 README 扩展为中文使用手册。数据既覆盖预处理后的 `.h5ad` 输入，也覆盖原始 `.imzML/.ibd` 输入，方便测试 I/O、峰对齐、空间处理、分割、配准和定量流程。

## 已批准范围

- 数据规模：32×32 像素，共 1024 个 MSI 观测。
- 数据类型：原始 imzML/ibd、预处理 h5ad、激光图像、细胞分割标签。
- 数据交付：生成脚本和一份已经生成、直接提交到仓库的测试数据都提供。
- 随机性：固定随机种子，脚本重复执行产生相同的数值内容和相同的元数据。
- 文档语言：主 README 以中文为主，保留必要的英文 API 名称、命令和文件格式名。

## 目录与交付物

新增目录 `tests/data/synthetic_demo/`，包含：

```text
tests/data/synthetic_demo/
├── generate_dataset.py
├── README.md
├── manifest.json
├── configs/
│   ├── synthetic-h5ad.yaml
│   └── synthetic-imzml.yaml
├── synthetic_demo.h5ad
├── synthetic_demo.imzML
├── synthetic_demo.ibd
├── laser_image.png
└── cell_segmentation.npy
```

其中：

- `generate_dataset.py` 是唯一的数据生成入口，支持显式输出目录、seed 和覆盖控制；
- `synthetic_demo.h5ad` 与原始 imzML 数据具有一致的像素坐标和 m/z 语义；
- `laser_image.png` 是 32×32 灰度图，包含可复现的激光标记模式；
- `cell_segmentation.npy` 是二维整数标签数组，0 为背景，非零标签表示四个细胞区域；
- `manifest.json` 记录 seed、形状、峰数、坐标范围、文件 SHA-256 和预期对象数量；
- 两个配置分别演示从 h5ad 和从 imzML/ibd 开始的 pipeline 输入。

## 合成数据语义

- 使用 10 个稳定的正 m/z 峰，所有谱值均为有限数；每个像素强度由空间梯度、细胞区域效应和可控噪声组成。
- 像素坐标覆盖完整的 32×32 网格，`adata.obsm["spatial"]` 与 imzML 坐标一致。
- 细胞标签按四个象限生成，并保证每个区域具有正面积、互不重叠且与 MSI 网格同形状。
- 激光图像含有可检测的行/点结构；配置默认使用稳定的 simulated laser segmentation，避免测试数据依赖某一台机器的图像阈值。
- 原始 imzML 保留每个像素的光谱数组；h5ad 使用稀疏矩阵并在 `var["mz"]` 中保存特征轴。

## 生成器接口

脚本提供命令行接口：

```bash
python tests/data/synthetic_demo/generate_dataset.py \
  --output-dir tests/data/synthetic_demo \
  --seed 20260912 \
  --overwrite
```

脚本也提供可测试的 Python 函数，至少包含：

```python
generate_dataset(output_dir: str | Path, *, seed: int = 20260912, overwrite: bool = False) -> Path
```

默认情况下，若目标文件已经存在，脚本拒绝覆盖并返回清晰错误；只有显式指定 `--overwrite` 才覆盖。生成过程先写临时文件，成功后再提升为最终文件，避免留下半成品数据集。

## README 使用手册内容

主 `README.md` 增加以下章节：

1. 软件定位与适用输入；
2. 安装和可选依赖；
3. 快速开始：生成/检查 synthetic demo 数据；
4. 使用 h5ad 配置运行 `joint run`；
5. 使用 imzML/ibd 配置运行 `joint run`；
6. `--resume`、`--overwrite` 和输出目录说明；
7. Python API 示例：`JointPipeline.from_config`、分阶段调用和 `read_imzml`；
8. 输出 artifact、manifest、日志和诊断图说明；
9. 真实 d2/d8 数据的目录约定和 `JOINT_DATA_ROOT` 配置；
10. 常见错误：缺少 `.ibd`、不可信 pickle、路径解析和可选依赖缺失。

README 中的命令必须使用仓库内实际存在的路径，不能要求用户预先拥有未提交的外部文件。真实 d2/d8 流程继续明确标记为需要外部参考数据，synthetic demo 则作为无外部依赖的默认验证入口。

## 测试与验收

新增测试覆盖：

- 生成器使用同一 seed 两次生成时，manifest、数组、h5ad 内容和 imzML 光谱内容一致；
- 生成器拒绝非空目录覆盖，`overwrite=True` 才允许替换；
- h5ad 具有 1024 个观测、10 个 m/z 特征、32×32 spatial 坐标；
- segmentation 数组为二维整数数组，标签集合为 `{0, 1, 2, 3, 4}`；
- `read_imzml` 能读回 1024 个像素和有限、正的 m/z 轴；
- 两个 synthetic 配置可被 `load_config` 解析，路径解析后指向数据目录；
- README 中的最小命令可完成 preprocess/segment/register/quantify smoke run，并生成 manifest、日志和核心 h5ad artifact；
- 运行现有非回归测试、Ruff 和 `git diff --check`，确认不破坏 d2/d8 回归接口。

## 错误处理与兼容性

- 生成器输入参数错误、目标冲突和写入失败统一提供可读的 `ValueError` 或专用生成器错误信息；
- 数据生成脚本只依赖项目已有运行时依赖和 `pyimzML`，不新增大型运行时依赖；
- 新增 fixture 不修改现有真实数据配置的默认路径和结果目录；
- README 扩展保持已有安装、CLI、API 和真实数据说明可用。

## 非目标

- 不伪造 d2/d8 的真实科学结果或替代外部参考回归；
- 不生成大规模生产数据，也不把 `.venv`、缓存、pipeline `results/` 或构建临时目录提交到仓库；
- 不把 synthetic demo 作为论文数值或方法性能基准。

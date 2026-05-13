# Qwen Select Element LoRA 微调

这个项目用于微调 `Qwen/Qwen3.6-35B-A3B`，任务是根据截图和元素列表选择正确的 `element_index`。

评估指标只看输出 JSON 里的 `element_index` 是否完全一致，`reason` 等其他字段不参与评分。

## 数据

当前默认训练数据：

- `data/train.json`
- `data/eval.json`

从数据库重新生成已审核通过数据：

```bash
venv/bin/python scripts/prepare_data.py
```

导出“已审核通过全部 + 最近 1000 条待审核”数据，并打包：

```bash
venv/bin/python scripts/export_dataset.py \
  --pending_limit 1000 \
  --eval_ratio 0.1 \
  --seed 42
```

最近一次导出结果：

- 目录：`exports/select_element_20260512_095427`
- 压缩包：`exports/select_element_20260512_095427.tar.gz`
- 训练集：`1098`
- 评估集：`122`
- 总样本：`1220`

## 新机器环境检查

先不要直接安装或训练，先查看 Python、GPU、CUDA driver、关键包版本：

```bash
python --version
which python
nvidia-smi
nvcc --version || true
```

如果已经创建并激活了虚拟环境，可以运行项目自带检查脚本：

```bash
python scripts/check_env.py
```

如果还没有虚拟环境：

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -U pip setuptools wheel
python scripts/check_env.py
```

当前本地已验证环境：

- Python：`3.12.7`
- Torch：`2.5.1+cu121`
- Torchvision：`0.20.1+cu121`
- Transformers：`5.8.0`
- PEFT：`0.19.1`
- BitsAndBytes：`0.49.2`
- NumPy：`requirements-base.txt` 使用 `2.2.6`，兼容 Python 3.10

CUDA 12.1 / 当前兼容环境安装：

```bash
python -m pip install -r requirements-cu121.txt
python -m pip install -r requirements-base.txt
python scripts/check_env.py
```

如果新机器的 driver / CUDA 明显更新，并且你想使用更新版 PyTorch，先用 PyTorch 官网对应命令安装 `torch` / `torchvision`，再安装：

```bash
python -m pip install -r requirements-base.txt
python scripts/check_env.py
```

不要在生产机器上盲装 `flash-linear-attention` / `causal-conv1d`。之前测试过，最新版 `flash-linear-attention` 会拉起 `torch 2.11 + CUDA 13`，可能和当前 driver 不兼容；`causal-conv1d` 如果没有 `nvcc` 会源码编译失败。确认环境后再单独评估这些加速依赖。

### H100 80GB x4 推荐环境

如果机器环境类似：

- Python：`3.10.x`
- Driver：`550.x`
- `nvidia-smi` 显示 CUDA：`12.4`
- `nvcc`：`12.1`
- GPU：H100 80GB

推荐使用稳定的 PyTorch CUDA 12.1 wheel：

```bash
conda create -n qwen-select python=3.10 -y
conda activate qwen-select
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements-h100-cu121.txt
python -m pip install -r requirements-base.txt
python scripts/check_env.py
```

如果 `check_env.py` 显示 `torch.cuda.is_available: True`，再开始训练。

如果你当前在 `(base)` 环境，并且里面已有 `vllm`、`xformers`、`torchaudio`，不要执行 `python -m pip install -r requirements-h100-cu121.txt`。这些包通常会锁定自己的 torch 版本，例如 `torch==2.8.0`。强行安装 `torch==2.5.1+cu121` 会破坏 base 环境。

这种情况下有两种选择：

1. 推荐：新建干净 conda 环境，按上面的 `qwen-select` 命令安装。
2. 如果必须复用 base：保留已有 torch，只安装非 torch 依赖。

复用 base 的命令：

```bash
conda activate base
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements-no-torch.txt
python scripts/check_env.py
```

如果你已经在 base 里误装了 `requirements-h100-cu121.txt`，先恢复 base 的 torch 版本，使它重新满足 vLLM 约束：

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
python -m pip check
python scripts/check_env.py
```

如果恢复命令因为 CUDA wheel 源不同失败，优先按这个镜像/容器原始文档恢复 `torch==2.8.0`，不要继续在 base 里降级 torch。

H100 质量优先训练命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/train.py \
  --output_dir output/qwen36_35b_a3b_select_element_h100 \
  --epochs 3 \
  --max_pixels 1003520 \
  --max_length 65536 \
  --lora_targets all \
  --lora_r 32 \
  --lora_alpha 64 \
  --grad_accum 16 \
  --logits_to_keep 256
```

上面的命令使用 `device_map=auto`，主要用于能跑通多卡显存分片，但不能充分利用 4 张 H100 算力。更推荐使用 DeepSpeed ZeRO-3。

安装 DeepSpeed：

```bash
python -m pip install -r requirements-deepspeed.txt
```

DeepSpeed ZeRO-3 训练命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
nohup deepspeed --num_gpus 4 scripts/train.py \
  --deepspeed configs/ds_zero3_h100.json \
  --no_4bit \
  --output_dir output/qwen36_35b_a3b_select_element_ds_zero3 \
  --epochs 3 \
  --max_pixels 1003520 \
  --max_length 65536 \
  --lora_targets all \
  --lora_r 32 \
  --lora_alpha 64 \
  --grad_accum 16 \
  --logits_to_keep 256 \
  > train_ds_zero3.log 2>&1 &
```

查看日志：

```bash
tail -f train_ds_zero3.log
```

这条命令使用 bf16 LoRA，不使用 4-bit 量化。它比 `device_map=auto` 更适合 4 卡 H100，质量也优先于 QLoRA。

如果显存仍然充足，可以尝试提高图片上限：

```bash
--max_pixels 1605632
```

如果显存不足，优先降：

```bash
--max_pixels 401408
--max_length 32768
--grad_accum 8
```

## 基线评估

评估默认基座模型：

```bash
venv/bin/python scripts/eval.py
```

评估指定基座模型：

```bash
venv/bin/python scripts/eval.py \
  --model_path Qwen/Qwen3.5-4B
```

## 质量优先训练

默认训练参数面向大显存机器，不是本地 RTX 4090 的保守配置：

```bash
venv/bin/python scripts/train.py \
  --output_dir output/qwen36_35b_a3b_select_element
```

关键默认参数：

- 模型：`Qwen/Qwen3.6-35B-A3B`
- 训练方式：QLoRA，4-bit NF4
- LoRA 目标模块：attention + MLP，即 `--lora_targets all`
- LoRA rank：`32`
- LoRA alpha：`64`
- epoch：`3`
- 有效 batch size：`16`，即 `batch_size=1`、`grad_accum=16`
- 图片上限：`max_pixels=1003520`
- 最大序列长度：`max_length=65536`
- 默认不跳过长样本
- 默认关闭 Trainer eval loss，训练后使用生成式评估

如果显存不足，按这个顺序降级：

```bash
--max_pixels 401408
--max_length 32768
--lora_targets attention
--lora_r 16 --lora_alpha 32
--max_chars 30000
```

## 训练后评估

评估训练后的 LoRA adapter：

```bash
venv/bin/python scripts/eval.py \
  --model_path output/qwen36_35b_a3b_select_element/final \
  --max_pixels 1003520
```

只跑 1 条样本做 smoke test：

```bash
venv/bin/python scripts/eval.py \
  --model_path output/qwen36_35b_a3b_select_element/final \
  --max_pixels 1003520 \
  --limit 1
```

详细结果会写入：

```text
data/eval_results.json
```

# 本机 RoboTwin 评测

在 `/mnt/miaohua/charles/codes/FasterWAM` 执行。

## 已配置的路径

| 仓库入口 | 实际位置 |
| --- | --- |
| `checkpoints/Wan-AI` | `/mnt/miaohua/charles/models/FastWAM-compat/Wan-AI` |
| `checkpoints/fasterwam_release` | `/mnt/miaohua/charles/models/fasterwam_release` |
| `data/robotwin2.0` | `/mnt/miaohua/charles/datasets/FastWAM-RoboTwin` |
| `third_party/RoboTwin/assets/background_texture`、`objects` | `/mnt/miaohua/charles/codes/benchmark/RoboTwin/assets` 下的对应目录 |

`embodiments` 已复制到本仓库，并运行上游
`script/update_embodiment_config_path.py` 生成本仓库的绝对路径。
`configs/sim_robotwin.yaml` 设置了 `model.redirect_common_files: false`，
以读取本机 Wan2.2 的 T5/VAE `.pth` 文件。

环境按 `bash scripts/setup/install_robotwin.sh` 安装。本机系统 Python 缺少
开发头文件，cuRobo 编译时复用现有 Python 3.10 环境的头文件：

```bash
CPATH=/mnt/miaohua/charles/envs/miniconda3/envs/RoboTwin/include/python3.10 \
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=8 \
uv pip install --python .venvs/robotwin/bin/python \
  --no-deps --no-build-isolation -e third_party/RoboTwin/envs/curobo
```

权重通过以下命令下载（已有完整文件时会复用）：

```bash
/opt/conda/bin/hf download hustvl/FasterWAM \
  robotwin/step_029355.pt robotwin/dataset_stats.json \
  --local-dir /mnt/miaohua/charles/models/fasterwam_release
```

权重 revision：`6bf9471ced6919a15ab8fded89f7772f5060c44b`。
`step_029355.pt` SHA-256：
`934684f2b60f78d493d14f30dba4554c0c064803f0a7f659aaf2b16e60d6c7ef`。
随权重发布的统计文件与本地数据集的 `dataset_stats.json` 内容一致。

## 启动评测

```bash
cd /mnt/miaohua/charles/codes/FasterWAM
set -eo pipefail

OUT="$PWD/evaluate_results/robotwin/step_029355/fasterwam_10step_$(date +%Y%m%d_%H%M%S)" \
NUM_GPUS=8 MAX_TASKS_PER_GPU=1 \
bash scripts/eval_fasterwam_robotwin_local.sh \
  EVALUATION.eval_num_episodes=100 \
  EVALUATION.num_inference_steps=10 \
  EVALUATION.sigma_shift=5.0
```

本机脚本调用 README 中的 `scripts/eval_fasterwam_robotwin.sh`，
使用独立的 `.venvs/robotwin/bin/python`，并配置本机 NVIDIA/OIDN 渲染库。
不需要激活 LightX2V 或导出 DMD/EMA 权重。

官方配置采用 `one_pass_future_cache`、10 步动作去噪和 `replan_steps=28`。
manager 会遍历全部任务，每个任务依次评测 `demo_clean` 和
`demo_randomized`，每种配置各 100 回合，指令类型为 `unseen`。

结果位于 `evaluate_results/robotwin/step_029355/<运行名>/`，
包括 `manager.log`、`summary.csv`、`summary.json` 和任务日志。
上游 manager 只使用 `OUT` 的最后一级目录名，结果根目录固定在本仓库。

仅检查 Hydra 配置、不启动评测：

```bash
bash scripts/eval_fasterwam_robotwin_local.sh --cfg job --resolve
```

单任务小规模试跑：

```bash
NUM_GPUS=1 bash scripts/eval_fasterwam_robotwin_local.sh \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.eval_num_episodes=1
```

该命令仍会依次运行 clean 和 randomized 两种配置。

## `left_planner` 初始化报错修复

2026-09-14 的多卡日志中，首次异常来自 Warp 0.10.1 加载共用缓存
`/root/.cache/warp/0.10.1` 中的 PTX（文件缺失、格式错误、找不到 kernel）。
规划器初始化失败后，环境仍保存了半初始化的 Robot；后续换 seed 时
调用 `reset()`，才反复出现 `left_planner` 不存在的异常。

已在规划器导入 cuRobo 前通过 Warp API 设置进程独立的临时缓存，
正常退出时清理。机器人完成规划器和关节初始化后才保存到环境中。
评测遇到意外异常会保留原始 traceback 并退出当前 worker；
`UnStableError` 仍按原逻辑跳过 seed。

旧进程需要退出后重新执行启动命令，才能加载这些修复。
无需重新下载权重，也无需删除其他程序使用的全局 Warp 缓存。
首次进入新进程会重新编译 Warp kernels。

`grab_roller` 的部分 seed 可能没有可达抓取姿态。`grasp_actor()` 现在
会将这种情况标记为 `plan_success=False` 并抛出 `PlanningError`，
由专家筛选逻辑跳过该 seed。此前规划失败后再次抓取也会立即抛出该异常，
避免 `put_bottles_dustbin` 等任务继续索引空动作列表。
重放 `grab_roller` seed `4300003` 已确认正常拒绝，seed `4300004`
的专家轨迹成功。CUDA 初始化等意外错误仍会退出并报错。

`click_alarmclock` 直接调用 `get_grasp_pose()`，也增加了空姿态检查，
以 `PlanningError` 跳过不可达场景。已重放 seed `4300028` 确认正常跳过，
前后的 `4300027`、`4300029` 均可完成专家轨迹。
修复后用新进程完成 `click_alarmclock` clean 30 回合，30/30 成功，
正常越过 `4300028` 并运行至 seed `4300033`。结果位于
`evaluate_results/robotwin/step_029355/alarmclock_fix_30trials_20260914/`。
在 11:22 UTC 启动的旧进程早于 11:25:40 UTC 的闹钟补丁，
必须重启才能加载修复，运行中的 Python 不会自动更新已导入的函数。

另已扫描全部 50 个任务的规划调用：任务代码包含 232 处 `move()` 调用，
均未检查其返回值。公共 `move()` 现在在左臂、右臂或双臂规划失败后
立即抛出 `PlanningError`，由专家筛选层换 seed，防止任务继续处理后续动作。
候选抓取姿态的搜索仍可跳过不可达候选，继续尝试其他接触点。
本次公共入口调整通过 CPU 故障注入检查，未重新启动全量 GPU smoke。

同一运行目录、同一模型和评测配置下，可用 `MULTIRUN.resume=true`
复用已完成的 clean/random 结果，只运行未完成的配置。例如：

```bash
OUT=all_tasks_smoke_20260914_8gpu NUM_GPUS=8 MAX_TASKS_PER_GPU=1 \
bash scripts/eval_fasterwam_robotwin_local.sh \
  EVALUATION.eval_num_episodes=1 MULTIRUN.resume=true
```

续跑默认关闭。更换权重、回合数或推理参数时应使用新的 `OUT`，
不要复用旧结果目录。

回归测试：

```bash
.venvs/robotwin/bin/python -m unittest discover -s tests -v
```

## 已完成的检查

- 权重 SHA-256 与 Hugging Face 下载记录一致，检查点记录步数为 29355。
- 本机脚本的 shell 语法和 Hydra 完整配置解析通过。
- policy 配置能解析本地 T5/VAE/分词器，分词器可以离线处理指令。
- SAPIEN 成功渲染 64×64 图像。
- cuRobo 五个 CUDA 扩展和 RoboTwin 任务模块可以导入。
- 8 项评测回归测试通过，覆盖单臂/双臂动作失败中止和正常执行、闹钟空姿态、
  不可达抓取、初始化失败后重试、意外异常退出、
  场景/规划失败跳过，以及续跑时保留完整结果、重试缺失或无效结果。
- 4 个并发 GPU 进程完成 cuRobo 位姿求逆，各自使用独立 Warp 缓存并正常清理。

- 修复后执行 `click_alarmclock` 完整策略 rollout，clean 和 randomized 各 1 回合，
  两回合均成功，manager 正常退出。结果位于
  `evaluate_results/robotwin/step_029355/planner_fix_smoke_20260914/summary.csv`。
- 8 卡、每卡 1 个 worker 完成全部 50 个任务的 clean/randomized 各 1 回合，
  共 100 回合。调试中修复后续跑，最终 manager 正常退出；结果位于
  `evaluate_results/robotwin/step_029355/all_tasks_smoke_20260914_8gpu/summary.csv`。

完整 100 回合评测已于 2026-09-14 21:25:58 UTC 完成：50 个任务的
clean/randomized 各 100 回合，共 10,000 回合；clean 93.20%，randomized
92.66%，合计 92.93%。结果位于
`evaluate_results/robotwin/step_029355/fasterwam_10step_20260914_114338/summary.csv`。
续跑使用 8 卡、每卡 2 个 worker，保留所有已完成阶段；中断阶段重新测满
100 回合，不合并不同尝试的部分计数。逐阶段日志与原有结果哈希的核对记录位于
`evaluate_results/robotwin/resume_20260914_140514/completed_verification.json`。

完整评测期间还修复了两处问题：专家演示生成中的 NumPy `LinAlgError`
按无效场景种子处理，不计入策略回合；`place_can_basket` 和
`place_object_basket` 的首次放置现在捕获 `PlanningError`，进入任务原有的
备用放置动作。此前 `move()` 从返回 False 改为抛异常，使该备用分支无法执行，
导致持续拒绝种子。首次抓取、备用动作失败及意外异常仍向上传播。
三个相同种子的罐子放篮演示在修复前均失败，恢复备用流程后均成功；
13 项单元测试通过。以上修复不改变策略推理或成功判定。

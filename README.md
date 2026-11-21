# audio-tape-saturate

## 题面

给人声加磁带饱和后，高频糊成硬削、整体还越来越响：软削用硬截断，谐波过强；drive 未归一，输出随驱动线性变响；谐波生成不按 amount，固定 50%；DC 阻断不做稳态，低频被吃掉；干湿混合总和不为 1，整体变响；抖动相位用帧号，采样率变化时音高变；磁滞宽度不钳制，输入反转；THD 用峰值比而非谐波能量比；批量共用磁滞状态，轨道之间串；cutoff<=0 不校验，除零。saturate/tape.py 是核心，tests/test_tape.py 15个红。要修的是：软削改平滑 tanh；drive 归一化；谐波随 amount 缩放；DC 阻断稳态归零并保留低频；干湿权重和为 1；抖动相位按时间计；磁滞宽度钳制到 0..1；THD 用谐波能量比；批量每轨独立磁滞状态；cutoff<=0 抛 TapeError。

## 运行方式

在仓库根目录执行：

```bash
python -m pytest tests -q
```

初始状态是 `15 failed`，且不得出现 collection、import 或 syntax error。修好 `saturate/tape.py` 后同一命令应当全绿；测试只依赖 Python 标准库 `math` 与 `pytest`，离线运行、不使用随机数、不 sleep。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 说明

- `saturate/tape.py` 是核心实现文件；`saturate/__init__.py` 只做导出。
- `tests/test_tape.py` 不可修改，修复时只能改 `saturate/tape.py`。
- 音频统一使用 `list[float]`，数值容差默认 `1e-6`（`mix=0` 的回归用 `1e-9`）。

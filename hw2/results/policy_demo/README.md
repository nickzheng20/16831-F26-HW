# CartPole 策略行为演示

用浏览器打开 `index.html`，可播放、暂停、拖动时间轴，并查看每一步的四维状态、左右动作概率与实际动作。`before_after.gif` 是 Gym 实际渲染的训练前后对比动画。

这是使用当前 HW2 实现额外训练的演示。原来的正式实验仅保存了 TensorBoard 分数日志，没有保存模型或录像，因此本目录不是旧实验的回放，也不属于需要放进提交包 `run_logs` 的正式实验记录。

训练配置：CartPole-v0，50 轮，batch size 1500，学习率 0.005，两层各 64 单元，reward-to-go 和 advantage 标准化开启，baseline 关闭，训练 seed 1，CPU。

评估使用随机采样动作，并在训练前后使用相同的环境及动作采样 seed。GIF 与交互动画展示预先选定的 seed 101：训练前 45 步，训练后 200 步。

| 评估 seed | 训练前回报 | 训练后回报 |
|---|---:|---:|
| 101 | 45 | 200 |
| 102 | 35 | 200 |
| 103 | 11 | 200 |

CartPole 的策略输入是 `[小车位置, 小车速度, 杆角度, 杆角速度]`，输出左右动作的概率。环境每步奖励 +1，倒杆、出界或到达 200 步上限时结束。策略梯度改变网络参数，使获得较高回报的行为更可能出现。

`before.pt`、`after.pt` 保存策略参数；`rollouts.json` 保存交互页面使用的状态、动作和概率；`training_logs/` 与 `training.log` 是本次额外演示的训练日志。GIF 中的每帧来自环境渲染，网页画布由相同真实状态重新绘制。

从 hw2 目录生成新的独立演示（输出到新目录）：

```bash
../.conda/rob831/bin/python -m rob831.scripts.make_policy_demo --output-dir results/policy_demo_new
../.conda/rob831/bin/python -m rob831.scripts.build_policy_viewer --input results/policy_demo_new/rollouts.json --output results/policy_demo_new/index.html
```

只重新生成当前交互页面：

```bash
../.conda/rob831/bin/python -m rob831.scripts.build_policy_viewer
```

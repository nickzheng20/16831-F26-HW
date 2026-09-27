# 本地开发环境

环境位于项目的 `.conda/rob831`，Python 3.10。每次打开终端，在仓库根目录运行：

```bash
conda activate "$PWD/.conda/rob831"
```

如果当前 shell 尚未初始化 Conda，先运行：

```bash
source /home/nick12138/anaconda3/etc/profile.d/conda.sh
```

已安装 PyTorch 1.12.1+cu116、NumPy 1.25.2、Gym 0.25.1（含 Box2D）、MuJoCo 2.2.1、TensorBoard 2.10.0、TensorBoardX 2.5.1 和 IPython/ipykernel。

旧版 Gym 的 `Ant-v2` 等任务通过 `free-mujoco-py==2.1.6` 提供的 `mujoco_py` 和自带 MuJoCo 2.1 运行库运行。不要再同时安装另一份 `mujoco-py`，两者会覆盖同一模块。为兼容旧依赖，另安装了 Cython 0.29.37、setuptools 69.5.1、MoviePy 1.0.3、SWIG 和 patchelf。

OSMesa/OpenGL 开发文件放在环境内的 `sysroot`，Xvfb 使用 Conda 的 `xorg-xvfb-server` 包。Conda 激活钩子会设置编译和动态库路径；退出环境时恢复原值。MuJoCo 使用 CPU 离屏渲染，PyTorch 可独立使用 CUDA。

## 运行作业

默认已将 HW1 安装为可编辑包：

```bash
cd hw1
python rob831/scripts/run_hw1.py --help
```

HW1 和 HW2 的包名都叫 `rob831`。切换到 HW2 时，在仓库根目录执行：

```bash
python -m pip install --no-build-isolation -e hw2
cd hw2
python rob831/scripts/run_hw2.py --help
```

切回 HW1 时，在仓库根目录执行 `python -m pip install --no-build-isolation -e hw1`。

使用 Notebook 时，在激活环境后启动编辑器或 Jupyter，并选择 `.conda/rob831/bin/python` 对应的内核。作业代码中的 TODO 仍需自行完成。

## GPU 构建

默认 PyPI 的 PyTorch 1.12.1 使用 CUDA 10.2，不能运行本机 RTX 4090；本环境安装的是同版本的 CUDA 11.6 构建：

```bash
python -m pip install --no-deps torch==1.12.1+cu116 --index-url https://download.pytorch.org/whl/cu116
```

## WSLg 中使用虚拟显示

WSLg 的 X11 socket 目录可能不可写。需要 `pyvirtualdisplay` 时，使用下列已验证的参数：

```python
from pyvirtualdisplay import Display
with Display(visible=False, size=(640, 480), extra_args=["-nolisten", "unix"]):
    # 在此创建环境并渲染视频
    pass
```

MuJoCo 的离屏渲染无需 Xvfb。Box2D 也可以通过 `SDL_VIDEODRIVER=dummy` 无窗口渲染。

## 验证结果

已通过 `pip check`、HW1/HW2 的 `--help`、TensorBoard 启动检查、RTX 4090 的 CUDA 前向/反向运算，以及 Ant、HalfCheetah、Hopper、Walker2d、Humanoid、CartPole、LunarLanderContinuous、InvertedPendulum 的 reset/step 检查。MuJoCo 离屏渲染、Box2D/Xvfb 渲染、TensorBoard 标量与视频写入也已通过。

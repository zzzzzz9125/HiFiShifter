# HifiShifter

[中文](README.md) | [English](README_en.md)

HifiShifter 是一个基于深度学习神经声码器（NSF-HiFiGAN）的图形化音高修正工具。它允许用户加载音频文件，在界面上直观地编辑音高曲线，并利用预训练的声码器模型实时合成修改后的音频。

## 安装

### 1. 克隆仓库
```bash
git clone https://github.com/ARounder-183/HiFiShifter.git
cd HifiShifter
```

### 2. 安装依赖
请确保已安装 Python 3.10+。

```bash
pip install -r requirements.txt
```

如果 `requirements.txt` 不存在，请手动安装以下库：

```bash
pip install PyQt6 pyqtgraph sounddevice numpy scipy torch torchaudio pyyaml
```

## 快速开始

1. **运行程序**:
   ```bash
   python run_gui.py
   ```

2. **加载模型**:
   - 点击 `文件` -> `加载模型`。
   - 选择包含 `model.ckpt` 和 `config.json` 的文件夹。
   - 默认提供的模型为：`pc_nsf_hifigan_44.1k_hop512_128bin_2025.02`。

3. **加载音频**:
   - 点击 `文件` -> `加载音频`。
   - 选择 `.wav` 或 `.flac` 文件。

4. **编辑与合成**:
   - 使用左键在钢琴卷帘上绘制音高曲线。
   - 点击 `播放` -> `合成并播放` 听取效果。
   - **复制/粘贴音高**: 在左侧轨道列表的某个轨道上点击右键，选择“复制音高”或“粘贴音高”，可在不同轨道间复用音高曲线。

## 常用快捷键

| 操作                     | 快捷键 / 鼠标动作 |
| :----------------------- | :---------------- |
| **平移视图**             | 鼠标中键拖动      |
| **横向缩放 (时间)**      | Ctrl + 鼠标滚轮   |
| **纵向缩放 (音高)**      | Alt + 鼠标滚轮    |
| **绘制音高**             | 鼠标左键          |
| **擦除音高**             | 鼠标右键          |
| **播放/暂停**            | Space (空格键)    |
| **撤销**                 | Ctrl + Z          |
| **重做**                 | Ctrl + Y          |
| **切换模式 (编辑/选择)** | Tab               |

## 最近更新

- **界面美化**: 全面优化了 UI 主题，支持深色/浅色模式切换，并对控件样式进行了精细调整。
- **视觉增强**: 优化了音高曲线、选择框和光标的显示效果，确保在不同背景下的清晰度。
- **控件优化**: 改进了轨道控制区域，将 BGM 开关改为更直观的按钮样式，并美化了音量推子和数值输入框。
- **图标支持**: 增加了应用程序图标加载功能。

## 已知问题

目前存在许多问题，如播放时无法修改音量和导入长音频有大概率会导致卡死。

## 文档

- [开发手册](DEVELOPMENT_zh.md)
- [更新计划](todo.md)

## 致谢

本项目使用了以下开源库的代码或模型结构：
- [SingingVocoders](https://github.com/openvpi/SingingVocoders)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)

## License

MIT License

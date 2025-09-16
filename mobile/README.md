# GPT-SoVITS 移动端应用

本目录包含GPT-SoVITS的Android平台实现，提供了在移动设备上运行语音合成和语音克隆的能力。

## 功能特点

- 完全本地化推理，无需网络连接
- 低延迟语音合成
- 支持语音克隆
- 针对移动设备优化的性能

## 环境要求

- Android SDK 26+（Android 8.0及以上）
- Rust开发环境
- [cargo-apk2](https://github.com/mzdk100/cargo-apk2)工具

## 构建步骤

### 1. 安装cargo-apk2

```shell
cargo install cargo-apk2
```

### 2. 下载模型文件

需要下载以下模型文件并放置到项目根目录的`assets`文件夹中：

1. **主模型**：[huggingface.co/mikv39/gpt-sovits-onnx-custom](https://huggingface.co/mikv39/gpt-sovits-onnx-custom)
2. **g2p_en模型**：[cisco-ai/mini-bart-g2p](https://huggingface.co/cisco-ai/mini-bart-g2p/tree/main/onnx)

### 3. 部署模型到设备

```shell
adb push assets /data/local/tmp/gpt-sovits
```

### 4. 构建并运行

```shell
cargo apk2 run -p gpt-sovits-mobile --release
```

## 自定义配置

可以在`Cargo.toml`文件中修改以下配置：

- `apk_name` - 应用名称
- `min_sdk_version` - 最低支持的Android版本
- `target_sdk_version` - 目标Android版本

## 性能优化建议

- 确保设备有足够的存储空间和内存
- 首次运行时，模型加载可能需要较长时间，属于正常现象

## 已知问题

- 在某些设备上，首次运行可能需要手动授予存储权限
- 目前不支持后台运行，切换应用可能导致合成中断

## 常见问题解答

**Q: 应用安装后无法运行怎么办？**  
A: 请确保已经正确部署模型文件到`/data/local/tmp/gpt-sovits`目录。

**Q: 如何使用自己的参考音频？**  
A: 目前需要通过ADB推送音频文件到设备，未来版本将支持从设备存储选择。

**Q: 支持iOS平台吗？**  
A: 理论上支持，但需要额外的配置和测试，目前尚未提供完整的iOS构建指南。
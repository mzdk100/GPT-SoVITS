# GPT-SoVITS 桌面应用

本目录包含GPT-SoVITS的桌面平台实现，提供了在Windows、macOS和Linux上运行语音合成和语音克隆的示例应用。

## 功能特点

- 高性能本地化语音合成
- 支持多种参考音频进行语音克隆
- 流式合成，实时输出音频
- 跨平台支持

## 环境要求

- Rust开发环境
- 音频输出设备
- 足够的系统内存（建议4GB以上）

## 快速开始

### 1. 下载模型文件

需要下载以下模型文件并放置到项目根目录的`assets`文件夹中：

1. **主模型**：[huggingface.co/mikv39/gpt-sovits-onnx-custom](https://huggingface.co/mikv39/gpt-sovits-onnx-custom)
2. **g2p_en模型**：[cisco-ai/mini-bart-g2p](https://huggingface.co/cisco-ai/mini-bart-g2p/tree/main/onnx)

### 2. 构建并运行

```shell
# 开发模式运行
cargo run -p gpt-sovits-desktop

# 发布模式运行（性能更好）
cargo run --release -p gpt-sovits-desktop
```

## 使用方法

示例应用展示了如何使用不同的参考音频合成相同的文本内容。您可以在`main.rs`中修改以下内容来自定义行为：

- 参考音频文件路径
- 参考音频对应的文本
- 要合成的目标文本
- 合成参数（如top_k、top_p、temperature等）

### 代码示例

```rust
// 使用参考音频合成语音
synth(
    &mut tts,
    assets_dir.join("ref.wav"),  // 参考音频路径
    "参考音频对应的文本",        // 参考音频文本
    "要合成的目标文本",          // 目标文本
    &player,                     // 音频播放器
)
.await?;
```

## 自定义配置

可以通过修改`SamplingParams`来调整合成效果：

```rust
let sampling_params = SamplingParams::builder()
    .top_k(4)               // 控制采样多样性
    .top_p(0.9)             // 概率截断阈值
    .temperature(1.0)       // 温度参数，控制随机性
    .repetition_penalty(1.35) // 重复惩罚系数
    .build();
```

## 性能优化建议

- 使用`--release`模式运行以获得最佳性能
- 首次运行时模型加载可能较慢，属于正常现象

## 常见问题解答

**Q: 如何使用自己的音频作为参考？**  
A: 将您的音频文件放入assets目录，并在代码中更新路径和对应的参考文本。

**Q: 支持哪些音频格式作为参考？**  
A: 支持WAV和MP3格式的音频文件。

**Q: 如何调整合成语音的质量？**  
A: 可以通过调整`SamplingParams`中的参数来影响合成结果，如提高temperature增加随机性，或调整top_k和top_p控制采样范围。
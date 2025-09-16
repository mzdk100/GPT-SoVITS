## 安卓平台构建方法

## 安装cargo-apk2
```shell
cargo install cargo-apk2
```

## 运行

模型下载地址：
1. https://huggingface.co/mikv39/gpt-sovits-onnx-custom
2. https://huggingface.co/cisco-ai/mini-bart-g2p/tree/main/onnx
把模型下载到根目录的assets中
运行`adb push assets /data/local/tmp/gpt-sovits

```shell
cargo apk2 run -p gpt-sovits-mobile --release
```
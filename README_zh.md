# asr2clip 语音转文字剪贴板工具

[![PyPI version](https://img.shields.io/pypi/v/asr2clip?color=green)](https://pypi.org/project/asr2clip/)
[![License](https://img.shields.io/github/license/Oaklight/asr2clip?color=green)](https://github.com/Oaklight/asr2clip/blob/master/LICENSE)

[English](README.md)

本工具旨在实时识别语音，将其转换为文字，并自动将文字复制到系统剪贴板。该工具利用 API 服务进行语音识别，并使用 Python 库进行音频捕获和剪贴板管理。

## 快速开始

```bash
pip install asr2clip       # 安装
asr2clip --edit            # 创建/编辑配置文件
asr2clip --test            # 测试配置
asr2clip                   # 开始录音和转录
```

## 前置条件

在开始之前，请确保已准备好以下内容：

- **Python 3.8 或更高版本**：该工具是用 Python 编写的。
- **API 密钥**：您需要一个语音识别服务的 API 密钥（例如 **OpenAI/Whisper** API 或兼容的 ASR API，如 [硅基流动](https://siliconflow.cn/) 或 [xinference](https://inference.readthedocs.io/en/latest/) 上的 **FunAudioLLM/SenseVoiceSmall**）。

### 系统依赖

| 依赖 | 用途 | Linux | macOS | Windows |
|------|------|-------|-------|---------|
| **ffmpeg** | 音频格式转换 | `apt install ffmpeg` | `brew install ffmpeg` | [下载](https://ffmpeg.org/download.html) |
| **PortAudio** | 音频录制 | `apt install libportaudio2` | `brew install portaudio` | 随 sounddevice 安装 |
| **剪贴板** | 复制到剪贴板 | 内置 (copykitten) | 内置 | 内置 |

## 安装

### 选项 1: 使用 pip 或 pipx 安装（推荐）

```bash
# 使用 pip 安装
pip install asr2clip

# 或使用 pipx 安装（推荐用于隔离环境）
pipx install asr2clip

# 升级到最新版本
pip install --upgrade asr2clip
```

### 选项 2: 从源码安装

```bash
git clone https://github.com/Oaklight/asr2clip.git
cd asr2clip
pip install -e .
```

## 配置

### 快速设置

使用内置编辑器配置 asr2clip 是最简单的方式：

```bash
asr2clip --edit  # 在默认编辑器中打开配置文件
```

如果配置文件不存在，将自动在 `~/.config/asr2clip/config.yaml` 创建。

### 配置文件

配置文件使用 YAML 格式：

```yaml
api_base_url: "https://api.openai.com/v1/"  # 或其他兼容的 API 地址
api_key: "YOUR_API_KEY"                     # API 密钥
model_name: "whisper-1"                     # 或其他兼容的模型
# quiet: false                              # 可选，禁用日志
# audio_device: "pulse"                     # 可选，音频输入设备
```

配置文件搜索位置（按顺序）：
1. `./asr2clip.conf`（当前目录）
2. `~/.config/asr2clip/config.yaml`
3. `~/.config/asr2clip.conf`（旧版）
4. `~/.asr2clip.conf`（旧版）

### 测试配置

使用前，请验证您的设置：

```bash
asr2clip --test
```

这将检查：
- ✓ 剪贴板支持
- ✓ 音频设备功能
- ✓ API 连接

### 音频设备选择

如果默认音频设备不工作，列出可用设备并选择一个：

```bash
asr2clip --list-devices    # 列出所有音频输入设备
asr2clip --device pulse    # 使用指定设备
```

或添加到配置文件：
```yaml
audio_device: "pulse"  # 或设备索引如 12
```

## 使用方法

### 基本用法

```bash
asr2clip                   # 录音直到 Ctrl+C，转录，复制到剪贴板
asr2clip --vad             # 持续录音，语音检测自动转录
asr2clip -i audio.mp3      # 转录音频文件
```

### 命令行选项

以下为 `asr2clip --help` 的原文（与当前程序一致）：

```
usage: asr2clip [-h] [-v] [-q] [-c FILE] [-e] [--generate-config]
                [--print-config] [--test] [-x NAME] [--list-devices] [-d DEV]
                [-i FILE] [-p NAME] [-b NAME] [-l LANG] [-r] [-C SEC] [-g]
                [--serve] [--host HOST] [--port PORT] [--model-dir MODEL_DIR]
                [--num-threads NUM_THREADS] [--download-model] [--vad]
                [--interval SEC] [--silence-threshold PROB]
                [--silence-duration SEC] [-s N] [-P NAME] [-M MODEL] [-o FILE]
                [-T NAME] [-z]

Record audio and transcribe to clipboard using ASR API

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -q, --quiet           Quiet mode — only output transcription and errors

Setup:
  -c FILE, --config FILE
                        Path to configuration file
  -e, --edit            Open configuration file in editor (creates default
                        config if missing)
  --generate-config     Write config template to
                        ~/.config/asr2clip/config.yaml
  --print-config        Print config template to stdout
  --test                Test backend connectivity and configured
                        preprocessors, then exit
  -x NAME, --preset NAME
                        Pipeline preset name (key under 'presets:' in config).
                        Presets define complete pipelines: ASR backend,
                        preprocessor, post-processor. Optional if
                        'default_preset' is set in config; CLI overrides still
                        work (-b, -p, -P).

Audio:
  --list-devices        List available audio input devices
  -d DEV, --device DEV  Audio input device (name, ALSA name, or index).
                        Overrides config.
  -i FILE, --input FILE
                        Transcribe an existing audio or video file instead of
                        recording. Supported: wav, mp3, m4a, ogg, flac, aac,
                        opus, wma, mp4, mov, mkv, webm, avi, flv, mvi
  -p NAME, --preprocessor NAME
                        Audio preprocessor: none, noisereduce, pyrnnoise,
                        deepfilter. Overrides the preprocessor in the selected
                        preset.

Transcription:
  -b NAME, --backend NAME
                        ASR backend to use (key under 'asr_backends:' in
                        config). Overrides the backend in the selected preset.
  -l LANG, --language LANG
                        Language hint for transcription (ISO-639-1, e.g. 'fi',
                        'en'). Overrides config. Omit to auto-detect.
  -r, --robust          Robust mode for -i file input: split at silence
                        boundaries, quality-check chunks, retry failures,
                        stream output (tail-f friendly).
  -C SEC, --chunk-duration SEC
                        Max chunk duration in seconds for -r/--robust mode
                        (default: 180)
  -g, --toggle          Toggle recording: first call starts, second call stops
                        and transcribes. Designed for keyboard shortcuts.

Local ASR server:
  --serve               Start the local sherpa-onnx ASR API server
  --host HOST           Server bind address (default: 127.0.0.1 or
                        local_asr.host in config)
  --port PORT           Server bind port (default: 8000)
  --model-dir MODEL_DIR
                        Path to ASR model directory
  --num-threads NUM_THREADS
                        Inference threads (default: 4)
  --download-model      Download the SenseVoice model and exit

VAD (continuous recording):
  --vad                 Continuous recording with voice activity detection.
                        Transcribes automatically when silence is detected
                        after speech. Requires sherpa-onnx: pip install
                        asr2clip[vad].
  --interval SEC        Continuous recording with fixed interval (seconds)
  --silence-threshold PROB
                        VAD speech probability threshold, 0.0-1.0 (default:
                        0.5)
  --silence-duration SEC
                        Silence duration to trigger transcription (default:
                        1.5 s)

Diarization:
  -s N, --speakers N    Speaker count hint for diarization backends (type:
                        whisperx, type: mock-diarize). Ignored by all other
                        backends. Selects a diarization backend in your preset
                        or with -b / --backend; see 'asr_backends:' in config.
                        If omitted, the backend uses its own default or auto-
                        detects speaker count.

Post-processing:
  -P NAME, --post NAME  AI post-processor name (key in 'postprocessors:'
                        config) or an inline system-prompt string. Requires
                        'postprocessor_backends:' in config. Overrides the
                        post-processor in the selected preset.
  -M MODEL, --post-model MODEL
                        AI model used for the post-processing (f. ex. claude-
                        sonnet-4-6). Overrides the post-processor config for
                        this run.

Output:
  -o FILE, --output FILE
                        Append transcripts to file
  -T NAME, --template NAME
                        Output template name from 'output_templates:' in
                        config. Controls what is written to clipboard/-o FILE.
                        Overrides the template specified in the prompt
                        definition.
  -z, --no-clipboard    Do not copy the transcript (or a file path) to the
                        system clipboard. Stdout and -o output behave as
                        usual.

Examples:
  asr2clip --edit                             # create/open config in editor
  asr2clip --test                             # verify backend and preprocessors
  asr2clip                                    # record, transcribe, copy to clipboard
  asr2clip --toggle                           # toggle recording (for keyboard shortcuts)
  asr2clip --toggle -P solo-restructure       # toggle, and produce AI-structured memo
  asr2clip -i audio.mp3                       # transcribe an existing file
  asr2clip -i m.mp3 -p deepfilter -r          # neural denoising + chunked transcription
  asr2clip -i meeting.m4a -b whisperx -s 3    # speaker diarization, 3-speaker hint
  asr2clip --serve                            # start local sherpa-onnx ASR server
  asr2clip --vad -o meeting.txt               # continuous VAD transcription to file
  asr2clip --interval 60                      # fixed-interval continuous recording

See https://github.com/sjjsy/asr2clip for full documentation and configuration examples.
```

### 示例

```bash
# 单次录音（按 Ctrl+C 停止）
asr2clip

# 转录音频文件
asr2clip -i recording.mp3

# 保存转录结果到文件
asr2clip -o transcript.txt

# 使用指定音频设备
asr2clip --device pulse
```

### 持续录音模式

适用于会议、讲座等长时间录音场景，使用 `--vad` 或 `--interval`：

```bash
# 持续录音，语音活动检测（静音时自动转录）
asr2clip --vad -o ~/meeting.txt

# 持续录音，固定间隔（每 60 秒转录一次）
asr2clip --interval 60 -o ~/meeting.txt

# 结合 VAD 和最大间隔
asr2clip --vad --interval 120 -o ~/meeting.txt
```

持续模式特点：
- 持续录音
- 自动转录（静音时或达到间隔时）
- 按一次 Ctrl+C 停止（退出前会转录剩余音频）
- 转录结果带时间戳追加到输出文件

### 语音活动检测（VAD）

VAD 使用 [Silero VAD](https://github.com/snakers4/silero-vad) 神经网络模型（通过 sherpa-onnx）进行可靠的语音检测。需要安装 `vad` 额外依赖：

```bash
pip install asr2clip[vad]
```

启用静音检测，在您停止说话时自动转录：

```bash
# 检测到静音时自动转录
asr2clip --vad

# 使用自定义设置
asr2clip --vad --silence-threshold 0.3 --silence-duration 2.0

# 保存转录结果到文件
asr2clip --vad -o ~/meeting.txt
```

VAD 选项：
- `--vad`：启用语音活动检测
- `--silence-threshold`：语音概率阈值，0.0-1.0（默认：0.5）。值越低越敏感。
- `--silence-duration`：触发转录的静音时长（秒，默认：1.5）

启用 VAD 后，转录在以下情况触发：
1. 检测到语音（音频概率高于阈值）
2. 随后是静音（低于阈值持续指定时长）

Silero VAD 模型（~629 KB）将在首次使用时自动下载。

## 故障排除

| 问题 | 解决方案 |
|------|----------|
| 音频未捕获 | 运行 `asr2clip --list-devices` 并选择可用设备 |
| 剪贴板不工作 | 安装 `xclip` (X11) 或 `wl-clipboard` (Wayland) |
| API 错误 | 检查配置中的 API 密钥和端点 |
| 静音音频 | 使用 `--device` 尝试其他音频设备 |

运行 `asr2clip --test` 诊断问题。

## 贡献

如果您想为此项目做出贡献，请 fork 仓库并提交 pull request。欢迎任何改进或新功能！

## 许可证

本项目采用 GNU Affero 通用公共许可证 v3.0。详情请参阅 [LICENSE](LICENSE) 文件。
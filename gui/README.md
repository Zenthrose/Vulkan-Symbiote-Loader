# Symbiote GUI - User-Friendly Interface

A modern, intuitive GUI for Vulkan Symbiote that makes AI inference accessible to non-technical users.

## Features

### Phase 1: Core GUI Shell ✅ (Implemented)

#### ImGui + GLFW + Vulkan Integration
- **Hardware-accelerated rendering** using Vulkan
- **Cross-platform**: Windows, macOS, Linux
- **Responsive UI** with 60 FPS target
- **Dockable panels** for customizable layout

#### Model File Picker
- **Native system dialogs** (zenity/kdialog on Linux, IFileDialog on Windows, NSOpenPanel on macOS)
- **Filter by file type**: .gguf, .ggml, .bin
- **Recent models list** for quick access
- **Drag-and-drop support** for model files

#### Basic Chat Interface
```
┌─────────────────────────────────────────────────────────┐
│  Vulkan Symbiote - AI Chat                    [≡] [×]  │
├─────────────────────────────────────────────────────────┤
│  System: Model loaded (Llama-2-70B-Q4)                  │
│  Tokens: 2,847 / 200,000                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  User: Write a story about a space explorer             │
│                                                         │
│  Assistant:                                             │
│  In the year 2187, Commander Sarah Chen piloted         │
│  the research vessel Aurora through the Kepler          │
│  system's asteroid belt. The ship's AI, VIKI, had       │
│  detected unusual readings from the fourth planet...    │
│                                                         │
│  [Generating... 42 tokens/s]                            │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  [Type your message...                    ] [Send] [📎] │
└─────────────────────────────────────────────────────────┘
```

#### Real-Time Token Counter
- **Live token count** as you type
- **Context usage bar** (visual indicator of 200K limit)
- **Generation speed** (tokens/second)
- **Estimated time remaining** for long outputs

### Phase 2: Smart Features 🚧 (Planned)

#### Drag-Drop Project Folders
```
Drag your project folder here
┌─────────────────────┐
│  📁 MyNovel/        │
│    ├── chapter1.txt │
│    ├── chapter2.txt │
│    └── outline.md   │
└─────────────────────┘

Detected: Novel project
Auto-loading chapters into context...
```

#### Auto-Context Management
- **File type detection**:
  - `.txt`, `.md` → Novel/Documentation mode
  - `.py`, `.js`, `.cpp` → Code assistant mode
  - `.json`, `.yaml` → Configuration mode
- **Smart chunking**: Automatically splits large files
- **Priority queuing**: Recent files get priority in context

#### Visual Pack Migration
```
Memory Status
VRAM (Hot)     ████████░░░░░░░░  8.2 GB / 16 GB
  Layer 12-15  ████████         (Active)
  Layer 8-11   ████             (Cached)

RAM (Warm)     ██████░░░░░░░░░░  12.4 GB / 32 GB
  Layer 4-7    ████             (Standby)

Disk (Cold)    █████████████████ 64 GB used
  Layer 0-3    ████████████████ (Stored)
```

#### Power Profile Switcher
- **⚡ High Performance**: Maximum speed, higher power
- **⚖️ Balanced**: Optimal efficiency (default)
- **🔋 Power Saver**: Extended battery life
- **🌙 Auto**: Adapts based on battery level

### Phase 3: Polish 🚧 (Planned)

#### Context Visualizer (200K Token Map)
```
Context Map (47,234 / 200,000 tokens)
├─ System Prompt        [░░░░░░░░░░] 512 tokens
├─ Chapter 1-3          [████████░░] 18,432 tokens  
├─ Chapter 4 (partial)  [████░░░░░░] 8,192 tokens
├─ Recent conversation  [██████░░░░] 12,096 tokens
└─ KV Cache             [████░░░░░░] 8,002 tokens
                        └─ Layer 28 (active)
```

#### VitalityOracle Predictions
```
[🔮 Predictive Loading]
Loading next 3 layers...
Pre-fetching: Layer 29, 30, 31
Estimated time: 1.2s

Context prediction:
- 87% probability: User will continue story
- 13% probability: User will ask question
→ Pre-loading story continuation weights
```

#### Settings Persistence
```toml
[gui]
window_width = 1600
window_height = 900
theme = "dark"
font_size = 16

[model]
last_model = "/path/to/llama-2-70b.gguf"
auto_load = true

[context]
default_mode = "novel"
max_tokens = 200000
sparse_attention = true

[power]
profile = "balanced"
auto_throttle = true
```

#### First-Run Tutorial
```
Welcome to Vulkan Symbiote! 🚀

1. Load a Model
   Click "Browse" to select a GGUF model file
   
2. Start Chatting  
   Type in the box below and press Enter
   
3. Add Context
   Drag a folder to give the AI project context
   
4. Monitor Performance
   Watch the token counter and speed indicators

[✓] Don't show this again    [Get Started]
```

## Usage

### Basic Usage
```bash
# Launch GUI
./symbiote_chat

# With model pre-selected
./symbiote_chat /path/to/model.gguf
```

### The "Novice User" Test

Your GUI succeeds if a user can:

✅ **Double-click the app** (no terminal required)
   - Desktop entry created on Linux
   - .app bundle on macOS  
   - Start menu on Windows

✅ **Drag their novel folder into the window**
   - Files are automatically detected and loaded
   - Progress shown: "Loading chapter 3 into context..."
   - Context visualizer updates in real-time

✅ **Click "Continue Story"**
   - One-click prompt templates
   - Context-aware continuation
   - Visual feedback during generation

✅ **Type naturally** (no --context 200000 flags)
   - Settings are persistent
   - Smart defaults for each mode
   - Auto-detection of use case

✅ **Understand why it's fast or slow** (visual pack status)
   - Green = Fast (in VRAM)
   - Yellow = Medium (in RAM) 
   - Red = Slow (loading from disk)
   - Hover for detailed stats

## Building

### Dependencies
```bash
# Ubuntu/Debian
sudo apt-get install libglfw3-dev libvulkan-dev

# macOS
brew install glfw vulkan-loader

# Windows
# Download GLFW and Vulkan SDK manually
```

### Build GUI
```bash
cd /path/to/Vulkan-Symbiote-Loader
mkdir build && cd build
cmake .. -DBUILD_GUI=ON
make -j$(nproc)
```

### Run
```bash
./gui/symbiote_chat
```

## Architecture

```
symbiote_chat (GUI executable)
├── SymbioteGUI (main window)
│   ├── ChatPanel (input/output)
│   ├── ContextVisualizer (200K token map)
│   ├── PackStatusPanel (VRAM/RAM/Disk)
│   └── SettingsWindow (preferences)
├── ChatSession (state management)
├── NativeFileDialog (cross-platform)
└── VulkanSymbioteEngine (backend)
    ├── GGUFLoader
    ├── VitalityOracle
    └── ShaderRuntime
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open model file |
| `Ctrl+N` | New chat |
| `Ctrl+L` | Clear context |
| `Ctrl+T` | Toggle theme |
| `Ctrl+,` | Settings |
| `F11` | Fullscreen |
| `Esc` | Exit fullscreen / Cancel generation |

## Future Roadmap

- **v1.1**: Context templates (novel, coding, research)
- **v1.2**: Multi-model comparison (side-by-side)
- **v1.3**: Export conversations (Markdown, HTML)
- **v1.4**: Plugin system for custom tools
- **v2.0**: Multiplayer mode (shared sessions)

## License

Same as Vulkan Symbiote - MIT License

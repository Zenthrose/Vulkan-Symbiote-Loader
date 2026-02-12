#!/bin/bash
# Vulkan Symbiote Engine Build Script

set -e  # Exit on error

echo "🚀 Building Vulkan Symbiote Engine..."

# Check dependencies
echo "🔍 Checking dependencies..."
if ! command -v vulkan-info >/dev/null 2>&1; then
    echo "❌ Vulkan SDK not found!"
    echo "Please install Vulkan SDK:"
    echo "  Ubuntu/Debian: sudo apt-get install vulkan-dev"
    echo "  Fedora: sudo dnf install vulkan-devel"
    echo "  Arch: sudo pacman -S vulkan-devel"
    exit 1
fi

if ! command -v cmake --version >/dev/null 2>&1; then
    echo "❌ CMake not found!"
    echo "Please install CMake:"
    echo "  Ubuntu/Debian: sudo apt-get install cmake"
    echo "  Fedora: sudo dnf install cmake"
    echo "  Arch: sudo pacman -S cmake"
    exit 1
fi

# Create build directory
echo "📁 Creating build directory..."
mkdir -p build || exit 1
cd build

# Configure build
echo "⚙️ Configuring build..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DVulkan_DIR=/usr/local/lib/cmake/vulkan

# Build
echo "🔨 Building engine..."
make -j$(nproc) || {
    echo "❌ Build failed!"
    exit 1
}

# Success
echo "✅ Build completed successfully!"
echo "📁 Binary location: ./vk_symbiote/libvk_symbiote.a"
echo "🚀 Executable location: ./vk_symbiote/vk_symbiote_example"
echo "📊 Benchmark executable: ./vk_symbiote/vk_symbiote_benchmark"

echo "🎯 Run examples:"
echo "  ./vk_symbiote_example --model <path/to/model.gguf>"
echo "  ./vk_symbiote_benchmark --model <path/to/model.gguf>"
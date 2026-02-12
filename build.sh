#!/bin/bash
# Vulkan Symbiote Engine Build Script
# Neural symbiote architecture for unquantized LLM inference

set -e  # Exit on error

echo "🚀 Vulkan Symbiote Engine - Neural Symbiote Architecture"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check dependencies
echo -e "${BLUE}🔍 Checking dependencies...${NC}"

if ! command -v vulkaninfo >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  vulkaninfo not found, trying vulkan-info...${NC}"
    if ! command -v vulkan-info >/dev/null 2>&1; then
        echo -e "${RED}❌ Vulkan SDK not found!${NC}"
        echo "Please install Vulkan SDK:"
        echo "  Ubuntu/Debian: sudo apt-get install vulkan-tools vulkan-validationlayers-dev"
        echo "  Fedora: sudo dnf install vulkan-tools vulkan-validation-layers-devel"
        echo "  Arch: sudo pacman -S vulkan-tools vulkan-validation-layers"
        exit 1
    fi
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo -e "${RED}❌ CMake not found!${NC}"
    echo "Please install CMake:"
    echo "  Ubuntu/Debian: sudo apt-get install cmake"
    echo "  Fedora: sudo dnf install cmake"
    echo "  Arch: sudo pacman -S cmake"
    exit 1
fi

# Check for VMA (Vulkan Memory Allocator)
echo -e "${BLUE}📦 Checking for Vulkan Memory Allocator...${NC}"
if [ ! -f "/usr/include/vk_mem_alloc.h" ] && [ ! -f "/usr/local/include/vk_mem_alloc.h" ]; then
    echo -e "${YELLOW}⚠️  VMA header not found in standard locations${NC}"
    echo "Installing VMA..."
    
    # Clone VMA if not present
    if [ ! -d "third_party/VulkanMemoryAllocator" ]; then
        mkdir -p third_party
        git clone --depth 1 https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git third_party/VulkanMemoryAllocator 2>/dev/null || {
            echo -e "${YELLOW}⚠️  Could not clone VMA, will try system path${NC}"
        }
    fi
    
    if [ -f "third_party/VulkanMemoryAllocator/include/vk_mem_alloc.h" ]; then
        export VMA_INCLUDE_DIR="${PWD}/third_party/VulkanMemoryAllocator/include"
        echo -e "${GREEN}✅ VMA found at ${VMA_INCLUDE_DIR}${NC}"
    else
        echo -e "${YELLOW}⚠️  VMA not available, build may fail${NC}"
    fi
else
    echo -e "${GREEN}✅ VMA found in system${NC}"
fi

# Check for glslang (for runtime shader compilation)
echo -e "${BLUE}📦 Checking for glslang...${NC}"
if ! pkg-config --exists glslang 2>/dev/null; then
    if [ ! -f "/usr/lib/libglslang.so" ] && [ ! -f "/usr/local/lib/libglslang.so" ]; then
        echo -e "${YELLOW}⚠️  glslang not found - runtime shader mutation disabled${NC}"
        echo "Install for runtime shader compilation:"
        echo "  Ubuntu/Debian: sudo apt-get install glslang-tools libglslang-dev"
    else
        echo -e "${GREEN}✅ glslang found${NC}"
    fi
else
    echo -e "${GREEN}✅ glslang found via pkg-config${NC}"
fi

# Check for Blosc2 compression
echo -e "${BLUE}📦 Checking for Blosc2...${NC}"
if ! pkg-config --exists blosc2 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Blosc2 not found - will use built-in compression${NC}"
else
    echo -e "${GREEN}✅ Blosc2 found${NC}"
fi

# Parse build type
BUILD_TYPE="Release"
if [ "$1" = "debug" ] || [ "$1" = "Debug" ]; then
    BUILD_TYPE="Debug"
    echo -e "${BLUE}🔧 Debug build selected${NC}"
fi

# Create build directory
echo -e "${BLUE}📁 Creating build directory...${NC}"
mkdir -p build
cd build

# Detect number of CPU cores
if command -v nproc >/dev/null 2>&1; then
    NUM_JOBS=$(nproc)
elif command -v sysctl >/dev/null 2>&1; then
    NUM_JOBS=$(sysctl -n hw.ncpu)
else
    NUM_JOBS=4
fi

# Configure build
echo -e "${BLUE}⚙️  Configuring build with CMake...${NC}"
CMAKE_ARGS=(
    "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
    "-DCMAKE_CXX_STANDARD=20"
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
)

# Add VMA include if found
if [ -n "$VMA_INCLUDE_DIR" ]; then
    CMAKE_ARGS+=("-DVMA_INCLUDE_DIR=${VMA_INCLUDE_DIR}")
fi

# Configure - CMakeLists.txt is in vk_symbiote subdirectory
cmake ../vk_symbiote "${CMAKE_ARGS[@]}" 2>&1 | tee cmake_output.log || {
    echo -e "${RED}❌ CMake configuration failed!${NC}"
    echo "See cmake_output.log for details"
    exit 1
}

# Build
echo -e "${BLUE}🔨 Building with ${NUM_JOBS} parallel jobs...${NC}"
make -j${NUM_JOBS} 2>&1 | tee build_output.log || {
    echo -e "${RED}❌ Build failed!${NC}"
    echo "See build_output.log for details"
    exit 1
}

# Success
echo ""
echo -e "${GREEN}✅ Build completed successfully!${NC}"
echo "============================================================"
echo ""
echo "📁 Build Artifacts:"
echo "   Library: build/libvk_symbiote.a"
echo "   Example: build/vk_symbiote_example"
echo "   Benchmark: build/vk_symbiote_benchmark"
echo ""
echo "🎯 Quick Start:"
echo "   ./vk_symbiote/vk_symbiote_example --model /path/to/model.gguf --prompt \"Hello\""
echo ""
echo "📊 Performance Testing:"
echo "   ./vk_symbiote/vk_symbiote_benchmark --model /path/to/model.gguf --tokens 100"
echo ""
echo "🔧 Configuration:"
echo "   Engine supports runtime configuration via:"
echo "   • Environment variables (VK_SYMBIOTE_VRAM_BUDGET, etc.)"
echo "   • Config file (vk_symbiote.conf)"
echo "   • Command-line arguments"
echo ""
echo "📚 Documentation:"
echo "   • architecture_design.md - Detailed system architecture"
echo "   • README.md - Usage guide"
echo "   • examples/ - Sample code"
echo ""
echo -e "${GREEN}🚀 Vulkan Symbiote Engine ready for neural symbiosis!${NC}"

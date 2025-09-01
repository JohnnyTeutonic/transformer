#!/bin/bash

# Build script for distributed transformer with MPI support
# Usage: ./scripts/build_distributed.sh [clean]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Distributed Transformer Build Script ===${NC}"

# Check if we should clean first
if [[ "$1" == "clean" ]]; then
    echo -e "${YELLOW}Cleaning previous build...${NC}"
    rm -rf build/
fi

# Create build directory
mkdir -p build
cd build

echo -e "${BLUE}Checking for MPI installation...${NC}"

# Check for MPI
if command -v mpicc &> /dev/null; then
    echo -e "${GREEN}✓ MPI found: $(mpicc --version | head -n1)${NC}"
    MPI_STATUS="enabled"
else
    echo -e "${RED}✗ MPI not found - distributed training will be disabled${NC}"
    MPI_STATUS="disabled"
fi

# Check for CUDA
echo -e "${BLUE}Checking for CUDA installation...${NC}"
if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}✓ CUDA found: $(nvcc --version | grep release)${NC}"
    CUDA_STATUS="enabled"
else
    echo -e "${YELLOW}! CUDA not found - GPU acceleration will be disabled${NC}"
    CUDA_STATUS="disabled"
fi

# Check for NCCL (if CUDA is available)
if [[ "$CUDA_STATUS" == "enabled" ]]; then
    echo -e "${BLUE}Checking for NCCL installation...${NC}"
    if ldconfig -p | grep -q libnccl; then
        echo -e "${GREEN}✓ NCCL found${NC}"
        NCCL_STATUS="enabled"
    else
        echo -e "${YELLOW}! NCCL not found - using MPI for all communication${NC}"
        NCCL_STATUS="disabled"
    fi
else
    NCCL_STATUS="disabled"
fi

echo -e "${BLUE}Configuring build with CMake...${NC}"

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=20 \
      -DCMAKE_CUDA_STANDARD=17 \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      ..

echo -e "${BLUE}Building transformer library and executables...${NC}"

# Build with all available cores
make -j$(nproc)

echo -e "${GREEN}=== Build Summary ===${NC}"
echo -e "MPI Support:        ${MPI_STATUS}"
echo -e "CUDA Support:       ${CUDA_STATUS}"
echo -e "NCCL Support:       ${NCCL_STATUS}"
echo ""

# List built executables
echo -e "${GREEN}Built executables:${NC}"
ls -la transformer* 2>/dev/null || echo "No executables found"

if [[ "$MPI_STATUS" == "enabled" ]]; then
    echo ""
    echo -e "${GREEN}=== Distributed Training Ready! ===${NC}"
    echo -e "To run distributed training:"
    echo -e "  ${YELLOW}mpirun -np 4 ./distributed_training_example${NC}"
    echo -e "  ${YELLOW}srun -N 2 -n 8 --gres=gpu:4 ./distributed_training_example${NC}"
    echo ""
    echo -e "To run with different MPI implementations:"
    echo -e "  OpenMPI: ${YELLOW}mpirun -np 4 --bind-to core ./distributed_training_example${NC}"
    echo -e "  Intel MPI: ${YELLOW}mpiexec -n 4 ./distributed_training_example${NC}"
    echo -e "  MPICH: ${YELLOW}mpiexec -n 4 ./distributed_training_example${NC}"
else
    echo ""
    echo -e "${YELLOW}=== MPI Not Available ===${NC}"
    echo -e "Install MPI to enable distributed training:"
    echo -e "  Ubuntu/Debian: ${YELLOW}sudo apt-get install libopenmpi-dev openmpi-bin${NC}"
    echo -e "  CentOS/RHEL:   ${YELLOW}sudo yum install openmpi-devel${NC}"
    echo -e "  macOS:         ${YELLOW}brew install open-mpi${NC}"
fi

echo -e "${GREEN}Build completed successfully!${NC}"

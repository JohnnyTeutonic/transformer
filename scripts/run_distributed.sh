#!/bin/bash

# Run script for distributed transformer training
# Usage: ./scripts/run_distributed.sh [num_processes] [hostfile]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Distributed Transformer Runner ===${NC}"

# Default values
NUM_PROCESSES=${1:-4}
HOSTFILE=${2:-""}
EXECUTABLE="./build/distributed_training_example"

# Check if executable exists
if [[ ! -f "$EXECUTABLE" ]]; then
    echo -e "${RED}Error: Executable not found at $EXECUTABLE${NC}"
    echo -e "${YELLOW}Run ./scripts/build_distributed.sh first${NC}"
    exit 1
fi

# Check if MPI is available
if ! command -v mpirun &> /dev/null; then
    echo -e "${RED}Error: mpirun not found${NC}"
    echo -e "${YELLOW}Install MPI first (e.g., sudo apt-get install openmpi-bin)${NC}"
    exit 1
fi

echo -e "${GREEN}Configuration:${NC}"
echo -e "  Processes: $NUM_PROCESSES"
echo -e "  Executable: $EXECUTABLE"

if [[ -n "$HOSTFILE" ]]; then
    echo -e "  Hostfile: $HOSTFILE"
    if [[ ! -f "$HOSTFILE" ]]; then
        echo -e "${RED}Error: Hostfile not found at $HOSTFILE${NC}"
        exit 1
    fi
fi

echo ""

# Detect MPI implementation
MPI_VERSION=$(mpirun --version 2>&1 | head -n1)
echo -e "${BLUE}MPI Implementation: ${MPI_VERSION}${NC}"

# Build MPI command
MPI_CMD="mpirun -np $NUM_PROCESSES"

# Add hostfile if provided
if [[ -n "$HOSTFILE" ]]; then
    MPI_CMD="$MPI_CMD --hostfile $HOSTFILE"
fi

# Add common MPI options for better performance
MPI_CMD="$MPI_CMD --bind-to core --map-by core"

# Add CUDA-aware MPI options if available
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}CUDA detected - enabling GPU support${NC}"
    MPI_CMD="$MPI_CMD --mca pml ob1 --mca btl ^openib"
fi

# Final command
FULL_CMD="$MPI_CMD $EXECUTABLE"

echo -e "${BLUE}Running command:${NC}"
echo -e "${YELLOW}$FULL_CMD${NC}"
echo ""

# Set environment variables for better MPI performance
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_mpi_warn_on_fork=0

# Run the distributed training
echo -e "${GREEN}Starting distributed training...${NC}"
exec $FULL_CMD

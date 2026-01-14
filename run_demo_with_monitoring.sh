#!/bin/bash

# Hunyuan3D-2.1 Demo Runner Script
# Runs the demo with proper environment settings and optional VRAM monitoring

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Hunyuan3D-2.1 Demo Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Warning: venv directory not found!${NC}"
    echo "Please create virtual environment first:"
    echo "  python3.10 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Check if demo.py exists
if [ ! -f "demo.py" ]; then
    echo -e "${YELLOW}Error: demo.py not found!${NC}"
    exit 1
fi

# Ask if user wants VRAM monitoring
echo -e "${BLUE}Enable VRAM monitoring? [y/N]:${NC} "
read -r ENABLE_MONITORING

MONITOR_PID=""

if [[ $ENABLE_MONITORING =~ ^[Yy]$ ]]; then
    if [ ! -f "monitor_vram.sh" ]; then
        echo -e "${YELLOW}Warning: monitor_vram.sh not found. Skipping monitoring.${NC}"
    else
        echo -e "${GREEN}Starting VRAM monitor...${NC}"
        > vram_usage.log  # Clear previous log
        ./monitor_vram.sh &
        MONITOR_PID=$!
        echo "Monitor running with PID: $MONITOR_PID"
        sleep 1
    fi
fi

# Run the demo
echo
echo -e "${GREEN}Running demo.py...${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Set environment variable for better CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run demo and capture output
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="demo_run_${TIMESTAMP}.log"

python demo.py 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

echo
echo -e "${BLUE}========================================${NC}"

# Stop VRAM monitor if running
if [ -n "$MONITOR_PID" ]; then
    echo -e "${GREEN}Stopping VRAM monitor...${NC}"
    kill $MONITOR_PID 2>/dev/null || true
    sleep 1
    
    # Analyze VRAM if analysis script exists
    if [ -f "analyze_vram.py" ]; then
        echo
        echo -e "${GREEN}VRAM Usage Analysis:${NC}"
        python analyze_vram.py
    fi
fi

# Summary
echo
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Demo completed successfully!${NC}"
    echo
    echo "Output saved to: save_dir/"
    echo "Log saved to: $LOG_FILE"
    
    # Show latest output directory
    LATEST_DIR=$(ls -td save_dir/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ]; then
        echo
        echo "Latest output:"
        ls -lh "$LATEST_DIR"
    fi
else
    echo -e "${YELLOW}⚠ Demo exited with error code: $EXIT_CODE${NC}"
    echo "Check $LOG_FILE for details"
fi

echo
echo -e "${BLUE}========================================${NC}"

#!/bin/bash
LOG_FILE="vram_usage.log"
> "$LOG_FILE"
echo "Timestamp,Used_MB,Total_MB,Utilization%" > "$LOG_FILE"

while true; do
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    VRAM_INFO=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | head -1)
    echo "$TIMESTAMP,$VRAM_INFO" >> "$LOG_FILE"
    sleep 1
done

#!/bin/bash

clear

ip_port=$(od -An -N2 -i /dev/urandom | awk '{print ($1 % 5001) + 15000}')

echo "port: $ip_port"

csv_file="process_info.csv"

current_time=$(date "+%Y-%m-%d_%H-%M-%S")

save_dir="/mnt/lzc/results3/VQ-SimCLR/imagenet/$current_time"
python_command="python main.py --save_dir $save_dir --port $ip_port"

echo "command: $python_command";

mkdir -p $save_dir

nohup $python_command > "$save_dir/run.log" 2>&1 &

pid=$!

echo "$pid,$python_command,$current_time,$ip_port" >> $csv_file
echo "Process started with PID: $pid"

watch -n 0 tail -n 30 "$save_dir/run.log"


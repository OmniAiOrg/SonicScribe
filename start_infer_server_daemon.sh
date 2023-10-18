#!/bin/bash

pkill -fc inference/flask_server.py

while true; do
    # 检查api.py是否在运行
    if ! pgrep -f "python3 inference/flask_server.py" > /dev/null; then
        echo "inference/flask_server.py已经停止，重新启动..."
        python3 inference/flask_server.py &
    fi
    sleep 10
done

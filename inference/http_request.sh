#!/bin/bash

# 设置API的URL
API_URL="http://localhost:33567/transcribe"

# 设置音频文件路径
AUDIO_FILE_PATH='assets/test_wav/output/wenbie/vocals.wav'

# 发起POST请求
response=$(curl -s -F "audio=@$AUDIO_FILE_PATH" $API_URL)

# 打印响应
echo "response: $response"
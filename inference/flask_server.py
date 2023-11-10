from flask import Flask, request, jsonify
from utils.chinese_to_pinyin import is_chinese
from utils.model_weight_initializer import initialize_whisper_official_from_checkpoint
from whisper import load_audio
from inference.transcribe import process_audio, transcribe
from model.whisper_official import WhisperOfficial
from collections import OrderedDict
import hashlib
import os
import time
import random
import string
import threading
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# 结果队列和等候队列
result_queue = OrderedDict()
wait_queue = OrderedDict()

# 最大队列大小
MAX_RESULT_QUEUE_SIZE = 100
MAX_WAIT_QUEUE_SIZE = 50

# 创建线程锁
result_queue_lock = threading.Lock()
wait_queue_lock = threading.Lock()

# 后台执行器
def executor():
    with app.app_context():
        while True:
            if len(wait_queue) > 0:
                md5, audio_file_path = wait_queue.popitem(last=False)
                retry_times = 1
                is_success = False
                while retry_times > 0 and not is_success:
                    try:
                        retry_times -= 1
                        result = process_audio(audio_file_path, model)
                        is_success = all([is_chinese(hanzi) for hanzi in result['text']])
                        if is_success:
                            print(f'success in process {audio_file_path}')
                            result_queue_lock.acquire()
                            result_queue[md5] = jsonify(result)
                            result_queue_lock.release()
                            print('result_queue2', result_queue)
                    except:
                        pass
                if not is_success:
                    print(f'failed in process {audio_file_path}')
                    result_queue_lock.acquire()
                    result_queue[md5] = jsonify({"message": "sorry, we tried our best, but still unable to deal with your input. Please try another audio file."})
                    result_queue_lock.release()
            time.sleep(1)

# 路由：音频转录
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    audio_file = request.files['audio']
    audio_extension = audio_file.filename.rsplit('.', 1)[1].lower()

    if audio_extension not in ['mp3', 'wav', 'webm', 'mp4']:
        return jsonify({"message": "Invalid audio file format."}), 400

    md5 = hashlib.md5(audio_file.read()).hexdigest()
    audio_file.seek(0)
    print('result_queue', result_queue)
    if md5 in result_queue:
        result = result_queue[md5]
        if result == "still running":
            return jsonify({"message": "Transcription in progress. Please try again later."})
        else:
            return result

    # if md5 in wait_queue:
    #     return jsonify({"message": "Transcription in progress. Please try again later."})

    if len(wait_queue) >= MAX_WAIT_QUEUE_SIZE:
        return jsonify({"message": "Too many requests. Please try again later."}), 429

    timestamp = time.strftime("%Y%m%d%H%M%S")
    hex_string = ''.join(random.choices(string.hexdigits, k=6))
    folder_name = time.strftime("%Y%m%d%H")
    folder_path = os.path.join(audio_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    audio_file_path = os.path.join(folder_path, f"{timestamp}-{hex_string}.{audio_extension}")
    audio_file.save(audio_file_path)

    if len(result_queue) >= MAX_RESULT_QUEUE_SIZE:
        result_queue_lock.acquire()
        result_queue.popitem(last=False)
        result_queue_lock.release()

    wait_queue[md5] = audio_file_path
    result_queue_lock.acquire()
    result_queue[md5] = "still running"
    result_queue_lock.release()

    return jsonify({"message": "Request added to the wait queue."})

if __name__ == '__main__':

    # 音频文件存储路径
    audio_folder = "assets/audio_files"
    # 启动后台执行器
    import threading
    executor_thread = threading.Thread(target=executor)
    executor_thread.start()
    
    # 初始化机器学习模型
    model = WhisperOfficial('tiny').to('cpu')
    initialize_whisper_official_from_checkpoint("artifacts/checkpoint/opencpop_014/last.ckpt", model, True)

    # 启动Flask应用
    app.run(port=33567, host='0.0.0.0')
import threading
import time
import psutil
from pynvml import *

def monitor_resources(
    interval=1.0,
    log_file=None,
    gpu_id=0,
    stop_event=None,
):
    """
    非阻塞监控 CPU / RAM / GPU
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(gpu_id)

    while stop_event is None or not stop_event.is_set():
        # CPU
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        info = nvmlDeviceGetMemoryInfo(handle)
        
        # GPU
        util = nvmlDeviceGetUtilizationRates(handle)
        gpu = util.gpu
        # gpu_mem = util.memory
        mem_total = round((info.total // 1048576) / 1024)
        # mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
        mem_process_used = round((info.used // 1048576) / 1024)

        line = (
            f"[{time.strftime('%H:%M:%S')}] "
            f"CPU: {cpu:5.1f}% | "
            f"RAM: {mem:5.1f}% | "
            f"GPU: {gpu:5.1f}% | "
            f"GPU-USED: {mem_process_used:5.1f}G |"
            f"GPU-TOTAL: {mem_total:5.1f}G |"
        )

        print(line)

        # if log_file is not None:
        #     with open(log_file, "a") as f:
        #         f.write(line + "\n")

        time.sleep(interval)

    nvmlShutdown()




stop_event = threading.Event()

monitor_thread = threading.Thread(
    target=monitor_resources,
    kwargs={
        "interval": 1.0,
        "log_file": "resource.log",
        "gpu_id": 0,
        "stop_event": stop_event,
    },
    daemon=True,
)

monitor_thread.start()

# ==========================
# 你的主程序（COLMAP / SLAM / GS / MVS）
# ==========================

time.sleep(10)


# 停止监控
stop_event.set()
monitor_thread.join()

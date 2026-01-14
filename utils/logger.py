# utils/logger.py
import logging
import os
import sys
from datetime import datetime

def setup_logger(name="DefectNemesis"):
    # 1. 确保日志目录存在
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 2. 定义日志文件名 (按日期)
    today_str = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f"app_{today_str}.log")

    # 3. 获取 Logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 捕获所有级别的日志

    # 避免重复添加 Handler (防止日志重复打印)
    if not logger.handlers:
        # --- 文件输出 (记录 INFO 及以上) ---
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_fmt = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

        # --- 控制台输出 (用于开发调试) ---
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_fmt = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_fmt)
        logger.addHandler(console_handler)

    return logger

# 全局单例
log = setup_logger()
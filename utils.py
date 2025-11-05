# import logging
# from logging.handlers import TimedRotatingFileHandler
# import os

# import logging
# import os
# from datetime import datetime


# # utils.py
# import os, logging
# from datetime import datetime

# def setup_logger(log_dir="logs", log_name="buoy_net", use_timestamp=False):
#     os.makedirs(log_dir, exist_ok=True)

#     if use_timestamp:
#         timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")
#     else:
#         log_file = os.path.join(log_dir, f"{log_name}.log")

#     logger = logging.getLogger(log_name)
#     logger.setLevel(logging.DEBUG)

#     if not logger.handlers:
#         fh = logging.FileHandler(log_file, encoding="utf-8")
#         ch = logging.StreamHandler()

#         formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
#         fh.setFormatter(formatter)
#         ch.setFormatter(formatter)

#         logger.addHandler(fh)
#         logger.addHandler(ch)

#     return logger
import logging
from logging.handlers import TimedRotatingFileHandler
import os
from datetime import datetime


def setup_logger(log_dir="logs", log_name="buoy_net", use_timestamp=False):
    """
    初始化日志系统
    - 支持 UTF-8 输出（防止中文乱码）
    - 支持可选时间戳文件名
    - 控制台 + 文件双输出
    - 防止重复添加 handler
    - 自动每天轮转日志（保留 7 天）
    """
    os.makedirs(log_dir, exist_ok=True)

    if use_timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")
    else:
        log_file = os.path.join(log_dir, f"{log_name}.log")

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    # ✅ 防止重复添加 handler（否则多次 import utils 会重复输出日志）
    if logger.hasHandlers():
        logger.handlers.clear()

    # ✅ 使用 TimedRotatingFileHandler 支持每天生成新日志文件
    fh = TimedRotatingFileHandler(
        log_file,
        when="midnight",  # 每天 00:00 切分
        interval=1,
        backupCount=7,  # 保留最近 7 天
        encoding="utf-8",  # ✅ 保证中文不乱码
    )

    ch = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# from utils import setup_logger

# logger = setup_logger(log_dir="logs", log_name="buoy_debug")

# logger.info("测试中文日志输出: 实体类型=buoy_node, id=0")

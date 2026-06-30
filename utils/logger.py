"""日志工具"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name='witt_lab', log_dir='./logs', level=logging.INFO):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path(log_dir) / f'{name}_{timestamp}.log'

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # File handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(level)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)

    fmt = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

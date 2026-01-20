import sys
from loguru import logger


# 配置全局logger
def setup_global_logger():
    """设置全局logger配置."""
    logger.remove()  # 移除默认处理器
    logger.add(sys.stderr,
               format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | '
               '<cyan>{file}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>',
               level='INFO',
               backtrace=True,
               diagnose=True,
               colorize=True)
    return logger


# 创建全局logger实例
global_logger = setup_global_logger()

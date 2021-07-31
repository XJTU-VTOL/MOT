# coding=utf-8
import logging
from logging import handlers

class Logger:
    level = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename: str, level='info', when='D', backCount=3, fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        """
        创建 Logger

        :param filename:
            保存的日志文件名
        :param level:
            记录等级
            参考上面的的 level
        :param when:
            日期分割标志
        :param backCount:
            备份数
        :param fmt:
            格式
        可参考 https://docs.python.org/3.5/library/logging.handlers.html#timedrotatingfilehandler
        与 https://docs.python.org/3.5/library/logging.html#logrecord-attributes
        """

        self.logger = logging.getLogger(filename)
        self.logger.setLevel(self.level[level])
        log_format = logging.Formatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(log_format)
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
        th.setFormatter(log_format)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)

logger = Logger('log.log', level='info')

if __name__=='__main__':
    # Example
    log = Logger('log.log', level='debug')
    log.logger.debug("DEBUG MESSAGE")
    log.logger.info("INFO MESSAGE")
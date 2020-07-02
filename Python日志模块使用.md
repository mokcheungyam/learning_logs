```python

import logging

# 按等级先后输出，默认是输出warning以上的
logging.debug('debug信息')
logging.warning('爆炸！')
logging.info('xx')


logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)


# 保存log文件
logging.basicConfig(level=logging.DEBUG,
                    filename='new.log',
                    filemode='a',
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    )

# 保存log文件且输出控制台
class Logger:
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    def __init__(self, filename, level='info', when='D',
        backCount=3, fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
    self.logger = logging.getLogger(filename)
    format_str = logging.Formatter(fmt)
    self.logger.setLevel(self.level_relations.get(level))
    sh = logging.StreamHandler()
    th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
    th.setFormatter(format_str)
    self.logger.addHandler(sh)
    self.logger.addHandler(th)

    

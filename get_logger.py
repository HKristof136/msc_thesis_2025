import logging

def get_logger(name=__name__):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(filename)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(name)
    return logger

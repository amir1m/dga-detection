import logging
def get_logger(name, file = False):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    logger = logging.getLogger(name)

    if (file):
        output_file_handler = logging.FileHandler(name + '.log')
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
        output_file_handler.setFormatter(formatter)
        logger.addHandler(output_file_handler)
    return logger

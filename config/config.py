import configparser


def load_config(config_path):
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    return cfg
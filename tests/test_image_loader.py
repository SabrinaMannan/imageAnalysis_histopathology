from src.config import load_config


def test_configuration():

    config = load_config()

    assert "data" in config

    assert "image" in config
from app.core.settings import Settings


def get_settings() -> Settings:
    """Instantiating the Settings object for FastAPI

    Returns
    -------
    Settings
        The Settings config object
    """
    return Settings()

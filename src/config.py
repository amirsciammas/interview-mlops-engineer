from pydantic import BaseSettings


# Settings to share across the application for configuration usingdotenv
# Or environment variables
class Settings(BaseSettings):

    """
    Settings to share across the application for configuration using
    environment variables
    """

    class Config:
        env_file = ".env"

    MODEL_PATH: str


settings = Settings()

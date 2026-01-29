"""Configuration management."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = Field(default="postgresql://fpl:fplpass@localhost:5432/fpl")

    # Redis
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_password: str | None = Field(default=None)
    redis_ssl: bool = Field(default=False)

    # Auth
    api_key: str | None = Field(default=None)


@lru_cache
def get_settings() -> Settings:
    return Settings()

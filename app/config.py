"""Configuration management for FPL MCP Server."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Database
    database_url: str = Field(
        default="postgresql://fpl:fplpass@localhost:5432/fpl",
        description="PostgreSQL connection URL",
    )

    # Redis
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_password: str | None = Field(default=None, description="Redis password")
    redis_ssl: bool = Field(default=False, description="Use SSL for Redis connection")

    # Auth
    api_key: str | None = Field(default=None, description="API key for authentication")

    # Server
    port: int = Field(default=8000, description="Server port")
    log_level: str = Field(default="INFO", description="Logging level")

    # FPL API
    fpl_user_agent: str = Field(default="fpl-mcp/1.0", description="User agent for FPL API")

    @computed_field
    @property
    def redis_url(self) -> str:
        """Construct Redis URL from components."""
        scheme = "rediss" if self.redis_ssl else "redis"
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"{scheme}://{auth}{self.redis_host}:{self.redis_port}/0"

    @computed_field
    @property
    def is_external_redis(self) -> bool:
        """Check if using external Redis (not localhost/redis container)."""
        return self.redis_host not in ("localhost", "127.0.0.1", "redis")

    @computed_field
    @property
    def is_external_postgres(self) -> bool:
        """Check if using external PostgreSQL."""
        return not any(
            host in self.database_url
            for host in ("localhost", "127.0.0.1", "@postgres:")
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

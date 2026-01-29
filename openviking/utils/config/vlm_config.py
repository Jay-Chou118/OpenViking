# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from openviking.models.vlm import VLMFactory, VLMBase


class VLMConfig(BaseModel):
    """VLM configuration, supports multiple providers (openai, volcengine)."""

    model: Optional[str] = Field(default=None, description="Model name")
    api_key: Optional[str] = Field(default=None, description="API key")
    api_base: Optional[str] = Field(default=None, description="API base URL")
    temperature: float = Field(default=0.0, description="Generation temperature")
    max_retries: int = Field(default=2, description="Maximum retry attempts")
    provider: Literal["openai", "volcengine"] = Field(
        default="volcengine", description="Provider type"
    )

    _vlm_instance: Optional[VLMBase] = None

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="before")
    @classmethod
    def apply_env_defaults(cls, data):
        """Read default values from environment variables."""
        if isinstance(data, dict):
            env_mapping = {
                "api_key": "OPENVIKING_VLM_API_KEY",
                "model": "OPENVIKING_VLM_MODEL",
                "api_base": "OPENVIKING_VLM_API_BASE",
                "provider": "OPENVIKING_VLM_PROVIDER",
            }
            for field, env_var in env_mapping.items():
                if data.get(field) is None:
                    env_val = os.getenv(env_var)
                    if env_val is not None:
                        data[field] = env_val
                
        return data

    @model_validator(mode='after')
    def validate_config(self):
        """Validate configuration completeness and consistency"""
        # VLM is optional, but if configured, must have required fields
        if self.api_key or self.model or self.api_base:
            # If any VLM config is provided, require model and api_key
            if not self.model:
                raise ValueError("VLM configuration requires 'model' to be set")
            if not self.api_key:
                raise ValueError("VLM configuration requires 'api_key' to be set")
        return self

    def get_vlm_instance(self) -> VLMBase:
        """Get  VLM instance"""
        if self._vlm_instance is None:
            config_dict = self.model_dump()
            self._vlm_instance = VLMFactory.create(config_dict)
        return self._vlm_instance

    def get_completion(self, prompt: str) -> str:
        """Get LLM completion."""
        return self.get_vlm_instance().get_completion(prompt)

    async def get_completion_async(self, prompt: str, max_retries: int = 0) -> str:
        """Get LLM completion asynchronously, max_retries=0 means no retry."""
        return await self.get_vlm_instance().get_completion_async(prompt, max_retries)

    def is_available(self) -> bool:
        """Check if LLM is configured."""
        return self.api_key is not None or self.api_base is not None

    def get_vision_completion(
        self,
        prompt: str,
        images: list,
    ) -> str:
        """Get LLM completion with images."""
        return self.get_vlm_instance().get_vision_completion(prompt, images)

    async def get_vision_completion_async(
        self,
        prompt: str,
        images: list,
    ) -> str:
        """Get LLM completion with images asynchronously."""
        return await self.get_vlm_instance().get_vision_completion_async(prompt, images)

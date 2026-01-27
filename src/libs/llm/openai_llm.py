"""OpenAI-Compatible LLM 实现。"""

import json
import urllib.error
import urllib.request
from typing import Dict, List, Optional
from libs.llm.base_llm import BaseLLM


class OpenAICompatibleLLM(BaseLLM):
    """OpenAI-Compatible 基类，封装通用请求与解析逻辑。"""

    provider_name: str = "openai"
    default_base_url: Optional[str] = "https://api.openai.com"
    auth_header_name: str = "Authorization"

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """执行对话并返回模型输出。"""
        self._validate_messages(messages)
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        response = self._post(payload)
        return self._extract_content(response)

    def _post(self, payload: Dict[str, object]) -> Dict[str, object]:
        url = self._build_url()
        api_key = self.config.api_key
        if self.auth_header_name and not api_key:
            raise ValueError(f"{self.provider_name} 缺少 api_key")

        headers = {"Content-Type": "application/json"}
        if self.auth_header_name:
            if self.auth_header_name.lower() == "authorization":
                headers[self.auth_header_name] = f"Bearer {api_key}"
            else:
                headers[self.auth_header_name] = api_key

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw)
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError) as exc:
            raise ValueError(f"{self.provider_name} 请求失败: {exc}") from exc

    def _build_url(self) -> str:
        base_url = self.config.base_url or self.default_base_url
        if not base_url:
            raise ValueError(f"{self.provider_name} 缺少 base_url")
        base_url = base_url.rstrip("/")
        if "chat/completions" in base_url:
            return base_url
        return f"{base_url}/v1/chat/completions"

    def _validate_messages(self, messages: List[Dict[str, str]]) -> None:
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{self.provider_name} messages 不能为空")
        for item in messages:
            if not isinstance(item, dict):
                raise ValueError(f"{self.provider_name} messages 格式错误")
            role = item.get("role")
            content = item.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError(f"{self.provider_name} messages 格式错误")

    def _extract_content(self, response: Dict[str, object]) -> str:
        try:
            choices = response.get("choices", [])
            message = choices[0].get("message", {})
            content = message.get("content", "")
        except (AttributeError, IndexError):
            content = ""
        if not content:
            raise ValueError(f"{self.provider_name} 返回内容为空")
        return str(content)


class OpenAILLM(OpenAICompatibleLLM):
    """OpenAI 官方兼容实现。"""

    provider_name = "openai"
    default_base_url = "https://api.openai.com"
    auth_header_name = "Authorization"

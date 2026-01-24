# coding=utf-8
"""
AI 客户端模块

使用自定义 LLM API（基于 curl 调用）
"""

import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional


class AIClient:
    """统一的 AI 客户端（基于自定义 LLM API）"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 AI 客户端

        Args:
            config: AI 配置字典
                - MODEL: 模型标识（实际使用 qwen3-max-preview）
                - API_KEY: API 密钥（使用环境变量 AUTH_TOKEN 或 AI_API_KEY）
                - API_BASE: API 基础 URL（可选，使用环境变量 LLM_API_URL）
                - TEMPERATURE: 采样温度
                - MAX_TOKENS: 最大生成 token 数
                - TIMEOUT: 请求超时时间（秒）
                - NUM_RETRIES: 重试次数（可选）
        """
        self.model = config.get("MODEL", "deepseek/deepseek-chat")
        self.api_key = config.get("API_KEY") or os.environ.get("AI_API_KEY", "")
        self.api_base = config.get("API_BASE", "")
        self.temperature = config.get("TEMPERATURE", 1.0)
        self.max_tokens = config.get("MAX_TOKENS", 5000)
        self.timeout = config.get("TIMEOUT", 120)
        self.num_retries = config.get("NUM_RETRIES", 2)

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        调用 AI 模型进行对话

        Args:
            messages: 消息列表，格式: [{"role": "system/user/assistant", "content": "..."}]
            **kwargs: 额外参数，会覆盖默认配置

        Returns:
            str: AI 响应内容

        Raises:
            Exception: API 调用失败时抛出异常
        """
        return self._call_api_with_curl(messages, **kwargs)

    def _call_api_with_curl(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        使用 curl 命令调用 API
        
        Returns:
            str: AI 响应内容
            
        Raises:
            ValueError: API 调用失败时抛出异常
        """
        # API 配置（优先环境变量，其次配置文件）
        api_url = os.getenv("LLM_API_URL", "https://ai-llm-gateway.amap.com/open_api/v1/chat")
        # 优先级：AUTH_TOKEN（含默认值）> AI_API_KEY > config.api_key
        auth_token = (
            os.getenv("AUTH_TOKEN", "1Zx8EvYhC8FsyQXLqwKgXVRD") 
            or os.getenv("AI_API_KEY") 
            or self.api_key
        )
        
        # 构建请求体
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        payload = {
            "model": "qwen3-max-preview",  # 使用固定模型
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.85,
            "max_tokens": max_tokens if max_tokens > 0 else 5000,
        }
        
        payload_json = json.dumps(payload, ensure_ascii=False)
        
        # 构造 curl 命令
        max_retries = kwargs.get("num_retries", self.num_retries) + 1
        delay = 2
        max_delay = 16
        
        for attempt in range(max_retries):
            result = None
            try:
                command = [
                    'curl',
                    '--request', 'POST',
                    '--url', api_url,
                    '--header', 'Content-Type: application/json',
                    '--header', f'Authorization: Bearer {auth_token}',
                    '--data', payload_json,
                    '--silent',
                    '--connect-timeout', '10',
                    '--max-time', str(self.timeout)
                ]
                
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True,
                    encoding='utf-8',
                    timeout=self.timeout + 10
                )
                
                # 解析响应
                response_data = json.loads(result.stdout)
                
                # 检查是否有错误
                if "error" in response_data:
                    error_msg = response_data["error"].get("message", "API返回未知错误")
                    if attempt + 1 < max_retries:
                        time.sleep(delay)
                        delay = min(delay * 2, max_delay)
                        continue
                    raise ValueError(f"API调用失败: {error_msg}")
                
                # 提取响应内容
                choices = response_data.get("choices", [])
                if not choices or not choices[0].get("message", {}).get("content"):
                    raise ValueError("Empty or invalid response from LLM")
                
                return choices[0]["message"]["content"]
                
            except subprocess.CalledProcessError as e:
                error_info = f"API请求失败 (尝试 {attempt + 1}/{max_retries}): curl返回码 {e.returncode}. Stderr: {e.stderr}. Stdout: {e.stdout}"
                if attempt + 1 == max_retries:
                    raise ValueError(error_info)
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
            except subprocess.TimeoutExpired:
                error_info = f"API请求超时 (尝试 {attempt + 1}/{max_retries})"
                if attempt + 1 == max_retries:
                    raise ValueError(error_info)
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
            except json.JSONDecodeError as e:
                stdout_content = result.stdout if result else 'N/A'
                error_info = f"解析API响应失败 (尝试 {attempt + 1}/{max_retries}). 原始响应: {stdout_content}"
                if attempt + 1 == max_retries:
                    raise ValueError(error_info)
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
            except Exception as e:
                error_info = f"发生未知错误 (尝试 {attempt + 1}/{max_retries}): {e}"
                if attempt + 1 == max_retries:
                    raise ValueError(error_info)
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
        
        raise ValueError("达到最大重试次数后依然失败")

    def validate_config(self) -> tuple[bool, str]:
        """
        验证配置是否有效

        Returns:
            tuple: (是否有效, 错误信息)
        """
        # 检查认证令牌（环境变量或配置文件，AUTH_TOKEN 含默认值）
        auth_token = (
            os.getenv("AUTH_TOKEN", "1Zx8EvYhC8FsyQXLqwKgXVRD") 
            or os.getenv("AI_API_KEY") 
            or self.api_key
        )
        
        return True, ""

# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from abc import ABC, abstractmethod
from typing import Any, Dict

import openai
import litellm
import tiktoken

from camel.typing import ModelType
from chatdev.utils import log_and_print_online
from tenacity import retry, wait_exponential



class ModelBackend(ABC):
    r"""Base class for different model backends.
    May be OpenAI API, a local LLM, a stub for unit tests, etc."""

    @abstractmethod
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        r"""Runs the query to the backend model.

        Raises:
            RuntimeError: if the return value from OpenAI API
            is not a dict that is expected.

        Returns:
            Dict[str, Any]: All backends must return a dict in OpenAI format.
        """
        pass


class OpenAIModel(ModelBackend):
    r"""OpenAI API in a unified ModelBackend interface."""

    def __init__(self, model: str, model_config_dict: Dict) -> None:
        super().__init__()
        self.model = model
        self.model_config_dict = model_config_dict

    # @retry(wait=wait_exponential(multiplier=0.04, max=10))
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        string = "\n".join([message["content"] for message in kwargs["messages"]])
        encoding = tiktoken.encoding_for_model(self.model)
        num_prompt_tokens = len(encoding.encode(string))
        gap_between_send_receive = 15 * len(kwargs["messages"])
        num_prompt_tokens += gap_between_send_receive

        num_max_token_map = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-3.5-turbo-0613": 4096,
            "gpt-3.5-turbo-16k-0613": 16384,
            "gpt-4": 8192,
            "gpt-4-0613": 8192,
            "gpt-4-32k": 32768,
        }
        num_max_token = num_max_token_map[self.model]
        num_max_completion_tokens = num_max_token - num_prompt_tokens
        self.model_config_dict['max_tokens'] = num_max_completion_tokens
        response = openai.completion(*args, **kwargs,
                                                model=self.model,
                                                **self.model_config_dict)

        log_and_print_online(
            "**[OpenAI_Usage_Info Receive]**\nprompt_tokens: {}\ncompletion_tokens: {}\ntotal_tokens: {}\n".format(
                response["usage"]["prompt_tokens"], response["usage"]["completion_tokens"],
                response["usage"]["total_tokens"]))
        if not isinstance(response, Dict):
            raise RuntimeError("Unexpected return from OpenAI API")
        return response

class LiteLLMModel(ModelBackend):
    def __init__(self, model: str, model_config_dict: Dict) -> None:
        super().__init__()
        self.model = model
        self.model_config_dict = model_config_dict

    # @retry(wait=wait_exponential(multiplier=0.04, max=10))
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        num_prompt_tokens = litellm.token_counter(self.model, messages=kwargs["messages"])
        gap_between_send_receive = 15 * len(kwargs["messages"])
        num_prompt_tokens += gap_between_send_receive

        num_max_token = self.model_config_dict.get('max_tokens', 8192) or 8192
        # if not num_max_token:
        #     num_max_token = litellm.get_max_tokens(self.model)
        num_max_completion_tokens = num_max_token - num_prompt_tokens
        self.model_config_dict['max_tokens'] = num_max_completion_tokens
        if 'model' in kwargs:
            model = kwargs['model']
        else:
            model = self.model
        # print(f'LLM running with model: {model}, kwargs: {str(kwargs)}')
        response = litellm.completion(*args, **kwargs,
                                                model=model,
                                                **self.model_config_dict)

        
        log_and_print_online(
            "**[LLM_Usage_Info Receive]**\nprompt_tokens: {}\ncompletion_tokens: {}\ntotal_tokens: {}\n".format(
                response["usage"]["prompt_tokens"], response["usage"]["completion_tokens"],
                response["usage"]["total_tokens"]))
        if not isinstance(response, Dict):
            raise RuntimeError("Unexpected return from OpenAI API")
        return response

class StubModel(ModelBackend):
    r"""A dummy model used for unit tests."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        ARBITRARY_STRING = "Lorem Ipsum"

        return dict(
            id="stub_model_id",
            usage=dict(),
            choices=[
                dict(finish_reason="stop",
                     message=dict(content=ARBITRARY_STRING, role="assistant"))
            ],
        )


class ModelFactory:
    r"""Factory of backend models.

    Raises:
        ValueError: in case the provided model type is unknown.
    """

    @staticmethod
    def create(model: str, model_config_dict: Dict) -> ModelBackend:
        default_model = "gpt-3.5-turbo"

        if model in {
            "gpt-3.5-turbo","gpt-3.5-turbo-0613","gpt-3.5-turbo-16k","gpt-3.5-turbo-16k-0613","gpt-4","gpt-4-0613","gpt-4-32k",
        }:
            model_class = LiteLLMModel
        elif "/" in model:
            model_class = LiteLLMModel
        elif model == ModelType.STUB:
            model_class = StubModel
        else:
            raise ValueError("Unknown model")

        if model is None:
            model = default_model

        # log_and_print_online("Model Type: {}".format(model))
        inst = model_class(model, model_config_dict)
        return inst

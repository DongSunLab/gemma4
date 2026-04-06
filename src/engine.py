import asyncio
import inspect
import json
import logging
import os
import time
from typing import AsyncGenerator

from dotenv import load_dotenv
from vllm import AsyncLLMEngine
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.protocol import BaseModelPath, LoRAModulePath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels

try:
    from vllm.entrypoints.serve.render.serving import OpenAIServingRender
except ImportError:
    OpenAIServingRender = None

from constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_SIZE_GROWTH_FACTOR,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MIN_BATCH_SIZE,
)
from engine_args import get_engine_args
from tokenizer import TokenizerWrapper
from utils import BatchSize, DummyRequest, JobInput, create_error_response


class vLLMEngine:
    def __init__(self, engine=None):
        load_dotenv()
        self.engine_args = get_engine_args()
        logging.info(f"Engine args: {self.engine_args}")

        self.llm = self._initialize_llm() if engine is None else engine.llm

        if self.engine_args.tokenizer_mode != "mistral":
            self.tokenizer = TokenizerWrapper(
                self.engine_args.tokenizer or self.engine_args.model,
                self.engine_args.tokenizer_revision,
                self.engine_args.trust_remote_code,
            )
        else:
            self.tokenizer = None

        self.max_concurrency = int(
            os.getenv("MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)
        )
        self.default_batch_size = int(
            os.getenv("DEFAULT_BATCH_SIZE", DEFAULT_BATCH_SIZE)
        )
        self.batch_size_growth_factor = int(
            os.getenv("BATCH_SIZE_GROWTH_FACTOR", DEFAULT_BATCH_SIZE_GROWTH_FACTOR)
        )
        self.min_batch_size = int(
            os.getenv("MIN_BATCH_SIZE", DEFAULT_MIN_BATCH_SIZE)
        )

    def _get_tokenizer_for_chat_template(self):
        if self.tokenizer is not None:
            return self.tokenizer

        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.engine_args.tokenizer or self.engine_args.model,
                revision=self.engine_args.tokenizer_revision or "main",
                trust_remote_code=self.engine_args.trust_remote_code,
            )

            class MinimalTokenizerWrapper:
                def __init__(self, tokenizer):
                    self.tokenizer = tokenizer
                    self.custom_chat_template = os.getenv("CUSTOM_CHAT_TEMPLATE")
                    self.has_chat_template = bool(
                        self.tokenizer.chat_template
                    ) or bool(self.custom_chat_template)
                    if (
                        self.custom_chat_template
                        and isinstance(self.custom_chat_template, str)
                    ):
                        self.tokenizer.chat_template = self.custom_chat_template

                def apply_chat_template(self, input_data, chat_template_kwargs=None):
                    if isinstance(input_data, list):
                        if not self.has_chat_template:
                            raise ValueError(
                                "Chat template does not exist for this model, "
                                "you must provide a single string input instead of a list of messages"
                            )
                    elif isinstance(input_data, str):
                        input_data = [{"role": "user", "content": input_data}]
                    else:
                        raise ValueError("Input must be a string or a list of messages")

                    chat_template_kwargs = chat_template_kwargs or {}
                    return self.tokenizer.apply_chat_template(
                        input_data,
                        tokenize=False,
                        add_generation_prompt=True,
                        **chat_template_kwargs,
                    )

            return MinimalTokenizerWrapper(tokenizer)
        except Exception as e:
            logging.error(f"Failed to create fallback tokenizer: {e}")
            raise e

    async def generate(self, job_input: JobInput):
        try:
            async for batch in self._generate_vllm(
                llm_input=job_input.llm_input,
                validated_sampling_params=job_input.sampling_params,
                batch_size=job_input.max_batch_size,
                stream=job_input.stream,
                apply_chat_template=job_input.apply_chat_template,
                chat_template_kwargs=job_input.chat_template_kwargs,
                request_id=job_input.request_id,
                batch_size_growth_factor=job_input.batch_size_growth_factor,
                min_batch_size=job_input.min_batch_size,
            ):
                yield batch
        except Exception as e:
            yield {"error": create_error_response(str(e)).model_dump()}

    async def _generate_vllm(
        self,
        llm_input,
        validated_sampling_params,
        batch_size,
        stream,
        apply_chat_template,
        chat_template_kwargs,
        request_id,
        batch_size_growth_factor,
        min_batch_size,
    ) -> AsyncGenerator[dict, None]:
        if apply_chat_template or isinstance(llm_input, list):
            tokenizer_wrapper = self._get_tokenizer_for_chat_template()
            llm_input = tokenizer_wrapper.apply_chat_template(
                llm_input,
                chat_template_kwargs=chat_template_kwargs,
            )

        results_generator = self.llm.generate(
            llm_input,
            validated_sampling_params,
            request_id,
        )

        n_responses = validated_sampling_params.n
        n_input_tokens = 0
        is_first_output = True
        last_output_texts = ["" for _ in range(n_responses)]
        token_counters = {"batch": 0, "total": 0}

        batch = {"choices": [{"tokens": []} for _ in range(n_responses)]}

        max_batch_size = batch_size or self.default_batch_size
        batch_size_growth_factor = (
            batch_size_growth_factor or self.batch_size_growth_factor
        )
        min_batch_size = min_batch_size or self.min_batch_size
        batch_size_obj = BatchSize(
            max_batch_size,
            min_batch_size,
            batch_size_growth_factor,
        )

        async for request_output in results_generator:
            if is_first_output:
                n_input_tokens = len(request_output.prompt_token_ids)
                is_first_output = False

            for output in request_output.outputs:
                output_index = output.index
                token_counters["total"] += 1

                if stream:
                    new_output = output.text[len(last_output_texts[output_index]) :]
                    batch["choices"][output_index]["tokens"].append(new_output)
                    token_counters["batch"] += 1

                    if token_counters["batch"] >= batch_size_obj.current_batch_size:
                        batch["usage"] = {
                            "input": n_input_tokens,
                            "output": token_counters["total"],
                        }
                        yield batch
                        batch = {
                            "choices": [{"tokens": []} for _ in range(n_responses)]
                        }
                        token_counters["batch"] = 0
                        batch_size_obj.update()

                last_output_texts[output_index] = output.text

        if not stream:
            for output_index, output_text in enumerate(last_output_texts):
                batch["choices"][output_index]["tokens"] = [output_text]
            token_counters["batch"] += 1

        if token_counters["batch"] > 0:
            batch["usage"] = {
                "input": n_input_tokens,
                "output": token_counters["total"],
            }
            yield batch

    def _initialize_llm(self):
        try:
            start = time.time()
            engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            end = time.time()
            logging.info(f"Initialized vLLM engine in {end - start:.2f}s")
            return engine
        except Exception as e:
            logging.error("Error initializing vLLM engine: %s", e)
            raise e


class OpenAIvLLMEngine(vLLMEngine):
    def __init__(self, vllm_engine):
        super().__init__(vllm_engine)
        self.served_model_name = (
            os.getenv("OPENAI_SERVED_MODEL_NAME_OVERRIDE")
            or self.engine_args.served_model_name
            or self.engine_args.model
        )
        self.response_role = os.getenv("OPENAI_RESPONSE_ROLE") or "assistant"
        self.lora_adapters = self._load_lora_adapters()
        self._engines_initialized = False

        if self.lora_adapters:
            logging.info(
                f"LoRA mode: {len(self.lora_adapters)} adapter(s) will load on first request"
            )
            for adapter in self.lora_adapters:
                logging.info(f"  - {adapter.name}: {adapter.path}")
        else:
            logging.info("OpenAI engines will initialize on first request")

        raw_output_env = os.getenv("RAW_OPENAI_OUTPUT", "1")
        if raw_output_env.lower() in ("true", "false"):
            self.raw_openai_output = raw_output_env.lower() == "true"
        else:
            self.raw_openai_output = bool(int(raw_output_env))

        self.chat_engine = None
        self.completion_engine = None
        self.serving_models = None
        self.model_config = None

    def _load_lora_adapters(self):
        adapters = []
        try:
            adapters = json.loads(os.getenv("LORA_MODULES", "[]"))
        except Exception as e:
            logging.info(f"---Initialized adapter json load error: {e}")

        for i, adapter in enumerate(adapters):
            try:
                adapters[i] = LoRAModulePath(**adapter)
                logging.info(f"---Initialized adapter: {adapter}")
            except Exception as e:
                logging.info(f"---Initialized adapter not worked: {e}")
                continue
        return adapters

    def _filter_kwargs_for_callable(self, cls_or_fn, kwargs: dict) -> dict:
        sig = inspect.signature(cls_or_fn)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}

    async def _ensure_engines_initialized(self):
        if not self._engines_initialized:
            logging.info("Initializing OpenAI serving engines...")
            await self._initialize_engines()
            self._engines_initialized = True
            logging.info("OpenAI serving engines initialized successfully")

    async def _initialize_engines(self):
        self.model_config = self.llm.model_config
        self.base_model_paths = [
            BaseModelPath(
                name=self.served_model_name,
                model_path=self.engine_args.model,
            )
        ]

        self.serving_models = OpenAIServingModels(
            engine_client=self.llm,
            base_model_paths=self.base_model_paths,
            lora_modules=self.lora_adapters,
        )
        await self.serving_models.init_static_loras()

        chat_template = None
        if self.tokenizer and hasattr(self.tokenizer, "tokenizer"):
            chat_template = self.tokenizer.tokenizer.chat_template

        openai_serving_render = None
        if OpenAIServingRender is not None:
            try:
                render_kwargs = {
                    "model_config": getattr(self.llm, "model_config", None),
                    "renderer": getattr(self.llm, "renderer", None),
                    "io_processor": getattr(self.llm, "io_processor", None),
                    "model_registry": getattr(self.serving_models, "registry", None),
                    "request_logger": None,
                    "chat_template": chat_template,
                    "chat_template_content_format": "auto",
                    "trust_request_chat_template": (
                        os.getenv("TRUST_REQUEST_CHAT_TEMPLATE", "false").lower()
                        == "true"
                    ),
                }
                render_kwargs = self._filter_kwargs_for_callable(
                    OpenAIServingRender,
                    render_kwargs,
                )
                openai_serving_render = OpenAIServingRender(**render_kwargs)
                logging.info("OpenAIServingRender initialized")
            except Exception as e:
                logging.warning(f"OpenAIServingRender init skipped: {e}")
                openai_serving_render = None

        chat_kwargs = {
            "engine_client": self.llm,
            "models": self.serving_models,
            "response_role": self.response_role,
            "request_logger": None,
            "chat_template": chat_template,
            "chat_template_content_format": "auto",
            "trust_request_chat_template": (
                os.getenv("TRUST_REQUEST_CHAT_TEMPLATE", "false").lower() == "true"
            ),
            "return_tokens_as_token_ids": (
                os.getenv("RETURN_TOKENS_AS_TOKEN_IDS", "false").lower() == "true"
            ),
            "reasoning_parser": os.getenv("REASONING_PARSER", "") or "",
            "enable_auto_tools": (
                os.getenv("ENABLE_AUTO_TOOL_CHOICE", "false").lower() == "true"
            ),
            "exclude_tools_when_tool_choice_none": (
                os.getenv("EXCLUDE_TOOLS_WHEN_TOOL_CHOICE_NONE", "false").lower()
                == "true"
            ),
            "tool_parser": os.getenv("TOOL_CALL_PARSER", "") or None,
            "enable_prompt_tokens_details": (
                os.getenv("ENABLE_PROMPT_TOKENS_DETAILS", "false").lower() == "true"
            ),
            "enable_force_include_usage": (
                os.getenv("ENABLE_FORCE_INCLUDE_USAGE", "false").lower() == "true"
            ),
            "enable_log_outputs": (
                os.getenv("ENABLE_LOG_OUTPUTS", "false").lower() == "true"
            ),
            "openai_serving_render": openai_serving_render,
        }
        chat_kwargs = self._filter_kwargs_for_callable(
            OpenAIServingChat,
            chat_kwargs,
        )
        self.chat_engine = OpenAIServingChat(**chat_kwargs)

        completion_kwargs = {
            "engine_client": self.llm,
            "models": self.serving_models,
            "request_logger": None,
            "return_tokens_as_token_ids": (
                os.getenv("RETURN_TOKENS_AS_TOKEN_IDS", "false").lower() == "true"
            ),
            "enable_prompt_tokens_details": (
                os.getenv("ENABLE_PROMPT_TOKENS_DETAILS", "false").lower() == "true"
            ),
            "enable_force_include_usage": (
                os.getenv("ENABLE_FORCE_INCLUDE_USAGE", "false").lower() == "true"
            ),
            "openai_serving_render": openai_serving_render,
        }
        completion_kwargs = self._filter_kwargs_for_callable(
            OpenAIServingCompletion,
            completion_kwargs,
        )
        try:
            self.completion_engine = OpenAIServingCompletion(**completion_kwargs)
        except Exception as e:
            logging.warning(f"OpenAIServingCompletion init skipped: {e}")
            self.completion_engine = None

        warmup_fn = getattr(self.chat_engine, "warmup", None)
        if callable(warmup_fn):
            warmup_result = warmup_fn()
            if inspect.isawaitable(warmup_result):
                await warmup_result

    async def generate(self, openai_request: JobInput):
        await self._ensure_engines_initialized()

        if openai_request.openai_route == "/v1/models":
            model_result = self._handle_model_request()
            if inspect.isawaitable(model_result):
                model_result = await model_result
            yield model_result
            return

        if openai_request.openai_route == "/v1/chat/completions":
            async for response in self._handle_chat_or_completion_request(
                openai_request
            ):
                yield response
            return

        if openai_request.openai_route == "/v1/completions":
            if self.completion_engine is None:
                yield create_error_response(
                    "Completions route unavailable in this build; use /v1/chat/completions"
                ).model_dump()
                return
            async for response in self._handle_chat_or_completion_request(
                openai_request
            ):
                yield response
            return

        yield create_error_response("Invalid route").model_dump()

    async def _handle_model_request(self):
        models = await self.serving_models.show_available_models()
        return models.model_dump()

    async def _handle_chat_or_completion_request(self, openai_request: JobInput):
        if openai_request.openai_route == "/v1/chat/completions":
            request_class = ChatCompletionRequest
            generator_function = self.chat_engine.create_chat_completion
        elif openai_request.openai_route == "/v1/completions":
            if self.completion_engine is None:
                yield create_error_response(
                    "Completions route unavailable in this build; use /v1/chat/completions"
                ).model_dump()
                return
            request_class = CompletionRequest
            generator_function = self.completion_engine.create_completion
        else:
            yield create_error_response("Invalid route").model_dump()
            return

        try:
            request = request_class(**openai_request.openai_input)
        except Exception as e:
            yield create_error_response(str(e)).model_dump()
            return

        dummy_request = DummyRequest()
        response_or_coro = generator_function(
            request,
            raw_request=dummy_request,
        )
        response_generator = (
            await response_or_coro
            if inspect.isawaitable(response_or_coro)
            else response_or_coro
        )

        if (
            not openai_request.openai_input.get("stream")
            or isinstance(response_generator, ErrorResponse)
        ):
            if hasattr(response_generator, "model_dump"):
                yield response_generator.model_dump()
            else:
                yield response_generator
            return

        batch = []
        batch_token_counter = 0
        batch_size_obj = BatchSize(
            self.default_batch_size,
            self.min_batch_size,
            self.batch_size_growth_factor,
        )

        async for chunk_str in response_generator:
            if "data" in chunk_str:
                if self.raw_openai_output:
                    data = chunk_str
                elif "[DONE]" in chunk_str:
                    continue
                else:
                    data = json.loads(
                        chunk_str.removeprefix("data: ").rstrip("\n\n")
                    )

                batch.append(data)
                batch_token_counter += 1

                if batch_token_counter >= batch_size_obj.current_batch_size:
                    if self.raw_openai_output:
                        batch = "".join(batch)
                    yield batch
                    batch = []
                    batch_token_counter = 0
                    batch_size_obj.update()

        if batch:
            if self.raw_openai_output:
                batch = "".join(batch)
            yield batch

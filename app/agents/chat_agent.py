"""Chat agent factory and tool registration."""

from typing import TYPE_CHECKING

from pydantic_ai import Agent
from pydantic_ai import RunContext
from pydantic_ai.models import Model

from app.agents.deps import AgentDeps
from app.config import Settings
from app.config import get_settings


# Placeholder API key for local providers (e.g., Ollama) that don't require authentication.
# This constant is passed to satisfy the SDK's non-optional api_key parameter.
# Never use this for cloud providers - those must have a real API key configured.
_LOCAL_PROVIDER_DUMMY_KEY = "LOCAL-PROVIDER-DOES-NOT-USE-API-KEY"


if TYPE_CHECKING:
    pass


def build_model(settings: Settings) -> Model:
    """Build a Model instance from settings based on the provider.

    Supports multiple providers: openai, anthropic, ollama, groq.
    The provider is determined from the llm_model format "provider:model".

    Provider-specific behavior:
        - openai: Uses OpenAI API or Azure OpenAI (if llm_base_url provided)
        - anthropic: Uses Anthropic Claude API
        - groq: Uses Groq's fast inference API
        - ollama: Uses local Ollama server (defaults to http://localhost:11434/v1)

    API key handling:
        - Cloud providers (openai, anthropic, groq): llm_api_key is required
        - Local providers (ollama): llm_api_key is optional (uses dummy key)

    Args:
        settings: Settings instance with LLM configuration (llm_model, llm_api_key, llm_base_url).

    Returns:
        Configured Model instance for the specified provider.

    Raises:
        ValueError: If the provider is not supported or configuration is invalid.

    Example:
        >>> from app.config import get_settings
        >>> settings = get_settings()
        >>> model = build_model(settings)
        >>> # Model is now ready for use with Pydantic AI Agent
    """
    # Extract provider and model from llm_model format "provider:model"
    llm_model = settings.llm_model
    if ":" in llm_model:
        provider_name, model_name = llm_model.split(":", 1)
    else:
        # Default to openai if no provider specified
        provider_name = "openai"
        model_name = llm_model

    provider_name = provider_name.lower()

    # Provider-specific initialization
    if provider_name == "anthropic":
        # Lazy import to avoid requiring anthropic package when not needed
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        # AnthropicModel uses AnthropicProvider to configure API key
        # Similar pattern to OpenAI provider
        # Task 16.7: Extract secret value from SecretStr
        if settings.llm_api_key:
            provider = AnthropicProvider(api_key=settings.llm_api_key.get_secret_value())
            return AnthropicModel(model_name, provider=provider)
        else:
            # Use default provider (reads from ANTHROPIC_API_KEY environment variable)
            return AnthropicModel(model_name)

    elif provider_name == "groq":
        # Lazy import to avoid requiring groq package when not needed
        from pydantic_ai.models.groq import GroqModel
        from pydantic_ai.providers.groq import GroqProvider

        # GroqModel uses GroqProvider to configure API key
        # Similar pattern to OpenAI provider
        # Task 16.7: Extract secret value from SecretStr
        if settings.llm_api_key:
            provider = GroqProvider(api_key=settings.llm_api_key.get_secret_value())
            return GroqModel(model_name, provider=provider)
        else:
            # Use default provider (reads from GROQ_API_KEY environment variable)
            return GroqModel(model_name)

    elif provider_name == "ollama":
        # Ollama uses OpenAI-compatible API with custom base_url
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        base_url = settings.llm_base_url or "http://localhost:11434/v1"
        # Task 16.7: Extract secret value from SecretStr or use dummy key for local provider
        api_key = (
            settings.llm_api_key.get_secret_value()
            if settings.llm_api_key
            else _LOCAL_PROVIDER_DUMMY_KEY
        )
        provider = OpenAIProvider(
            base_url=str(base_url),
            api_key=api_key,
        )
        return OpenAIChatModel(model_name, provider=provider)

    elif provider_name == "openai":
        # Lazy import to avoid requiring openai package when not needed
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        if settings.llm_base_url:
            # Custom base URL (e.g., Azure OpenAI)
            # Task 16.7: Extract secret value from SecretStr or use dummy key
            api_key = (
                settings.llm_api_key.get_secret_value()
                if settings.llm_api_key
                else _LOCAL_PROVIDER_DUMMY_KEY
            )
            provider = OpenAIProvider(
                base_url=str(settings.llm_base_url),
                api_key=api_key,
            )
            return OpenAIChatModel(model_name, provider=provider)
        elif settings.llm_api_key:
            # Task 16.7: Extract secret value from SecretStr
            provider = OpenAIProvider(api_key=settings.llm_api_key.get_secret_value())
            return OpenAIChatModel(model_name, provider=provider)
        else:
            # Use default provider (reads from environment)
            return OpenAIChatModel(model_name)

    else:
        raise ValueError(
            f"Unsupported provider: {provider_name}. "
            f"Supported providers are: openai, anthropic, groq, ollama"
        )


async def _build_system_prompt(ctx: RunContext[AgentDeps]) -> str:
    """Build dynamic system prompt for the chat agent.

    Args:
        ctx: RunContext with AgentDeps providing access to settings.

    Returns:
        System prompt string.
    """
    return (
        "You are a helpful AI assistant with access to tools. "
        "Use the available tools when needed to answer user questions accurately. "
        "Be concise and informative in your responses."
    )


def build_chat_agent(model: Model | str | None = None) -> Agent[AgentDeps, str]:
    """Build a Pydantic AI chat agent with tool-calling capabilities.

    This factory creates an Agent instance configured with:
    - AgentDeps for dependency injection into tools
    - str as the output type (simple text responses)
    - Configurable output retries for validation failures
    - Dynamic system prompt builder
    - Registered mock tools (when enabled in non-production environments)

    The agent is instrumented with Logfire for observability, automatically
    tracking all agent runs, tool calls, and token usage.

    Args:
        model: Optional model to use. If None, builds from Settings.
            Accepts Model instance or string identifier.

    Returns:
        Configured Agent[AgentDeps, str] instance ready for use.

    Example:
        >>> # Build with default settings
        >>> agent = build_chat_agent()
        >>> # Run the agent
        >>> from app.agents.deps import AgentDeps
        >>> deps = AgentDeps(...)
        >>> result = await agent.run("Hello!", deps=deps)
        >>> print(result.output)
        "Hello! How can I help you today?"
    """
    settings = get_settings()
    resolved_model = model if model is not None else build_model(settings)

    agent: Agent[AgentDeps, str] = Agent(
        model=resolved_model,
        deps_type=AgentDeps,
        output_type=str,
        output_retries=settings.max_output_retries,
    )

    # Register dynamic system prompt builder
    agent.system_prompt(_build_system_prompt)

    # Register mock tools only in non-production environments
    # CRITICAL: Mock tools are separated into tools_mock.py and only imported
    # when app_env is NOT production. This prevents accidental mock tool usage
    # in production even if enable_mock_tools is misconfigured.
    if settings.app_env != "production" and settings.enable_mock_tools:
        from app.agents.tools_mock import register_mock_tools

        register_mock_tools(agent)

    # TODO: Implement real web search integration
    # Example for SerpAPI:
    # if hasattr(settings, 'serpapi_key') and settings.serpapi_key:
    #     @agent.tool
    #     async def web_search(ctx: RunContext[AgentDeps], query: str) -> str:
    #         """Search the web using SerpAPI."""
    #         async with ctx.deps.http_client.get(
    #             "https://serpapi.com/search",
    #             params={"q": query, "api_key": settings.serpapi_key}
    #         ) as response:
    #             data = await response.json()
    #             # Process and return formatted results
    #             ...

    return agent

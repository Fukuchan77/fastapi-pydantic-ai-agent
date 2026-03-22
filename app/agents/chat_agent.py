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


def _build_model(settings: Settings) -> Model:
    """Build a Model instance from settings based on the provider.

    Supports multiple providers: openai, anthropic, ollama, groq.
    The provider is determined from the llm_model format "provider:model".

    Args:
        settings: Settings instance with LLM configuration.

    Returns:
        Configured Model instance for the specified provider.

    Raises:
        ValueError: If the provider is not supported.
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
        if settings.llm_api_key:
            provider = AnthropicProvider(api_key=settings.llm_api_key)
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
        if settings.llm_api_key:
            provider = GroqProvider(api_key=settings.llm_api_key)
            return GroqModel(model_name, provider=provider)
        else:
            # Use default provider (reads from GROQ_API_KEY environment variable)
            return GroqModel(model_name)

    elif provider_name == "ollama":
        # Ollama uses OpenAI-compatible API with custom base_url
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        base_url = settings.llm_base_url or "http://localhost:11434/v1"
        provider = OpenAIProvider(
            base_url=str(base_url),
            api_key=settings.llm_api_key or _LOCAL_PROVIDER_DUMMY_KEY,
        )
        return OpenAIChatModel(model_name, provider=provider)

    elif provider_name == "openai":
        # Lazy import to avoid requiring openai package when not needed
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        if settings.llm_base_url:
            # Custom base URL (e.g., Azure OpenAI)
            provider = OpenAIProvider(
                base_url=str(settings.llm_base_url),
                api_key=settings.llm_api_key or _LOCAL_PROVIDER_DUMMY_KEY,
            )
            return OpenAIChatModel(model_name, provider=provider)
        elif settings.llm_api_key:
            provider = OpenAIProvider(api_key=settings.llm_api_key)
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
    - ChatOutput as the structured response type
    - Configurable output retries for validation failures
    - Dynamic system prompt builder
    - Registered tools (example_web_search)

    Args:
        model: Optional model to use. If None, builds from Settings.
            Accepts Model instance or string identifier.

    Returns:
        Configured Agent[AgentDeps, ChatOutput] instance.
    """
    settings = get_settings()
    resolved_model = model if model is not None else _build_model(settings)

    agent: Agent[AgentDeps, str] = Agent(
        model=resolved_model,
        deps_type=AgentDeps,
        output_type=str,
        output_retries=settings.max_output_retries,
    )

    # Register dynamic system prompt builder
    agent.system_prompt(_build_system_prompt)

    # Register web search tool only if explicitly enabled in settings
    # In production, this should be replaced with real search API (SerpAPI/Tavily)
    if hasattr(settings, "enable_mock_tools") and settings.enable_mock_tools:

        @agent.tool
        async def mock_web_search(ctx: RunContext[AgentDeps], query: str) -> str:
            """Mock web search tool - placeholder that returns stub data.

            ⚠️ WARNING: This is a MOCK implementation for development only!
            It does NOT perform actual web searches and returns stub data.

            Args:
                ctx: RunContext providing access to AgentDeps (http_client, settings).
                query: The search query string.

            Returns:
                Mock search results as a formatted string (not real search data).
            """
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                "Mock web search tool called with query: %s. "
                "This returns stub data only. Replace with real search API for production.",
                query[:100],
            )

            # Stub implementation - returns mock data only
            return (
                f"Mock search results for '{query}':\n"
                f"1. Example result about {query}\n"
                f"2. Another relevant link about {query}\n"
                f"3. More information on {query}"
            )

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

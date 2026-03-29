"""Chat agent factory and tool registration."""

from pydantic_ai import Agent
from pydantic_ai import RunContext
from pydantic_ai.models import Model
from pydantic_ai_litellm import LiteLLMModel

from app.agents.deps import AgentDeps
from app.config import Settings
from app.config import get_settings


def build_model(settings: Settings) -> Model:
    """Build a LiteLLMModel instance from settings.

    Converts the internal "provider:model" format to LiteLLM's "provider/model"
    format (e.g. "openai:gpt-4o" → "openai/gpt-4o", "ollama:granite3.3:latest" →
    "ollama/granite3.3:latest") and delegates all provider routing to LiteLLM.

    API key handling:
        - Cloud providers (openai, anthropic, groq): llm_api_key is required
        - Local providers (ollama): llm_api_key is optional; LiteLLM omits it
          automatically

    Args:
        settings: Settings instance with LLM configuration.

    Returns:
        Configured LiteLLMModel instance ready for use with Pydantic AI Agent.

    Example:
        >>> from app.config import get_settings
        >>> model = build_model(get_settings())
    """
    llm_model = settings.llm_model
    if ":" in llm_model:
        provider_name, model_name = llm_model.split(":", 1)
        provider_name = provider_name.lower()  # Normalize to lowercase ()
    else:
        provider_name, model_name = "openai", llm_model

    # LiteLLM uses "provider/model" format
    litellm_model_name = f"{provider_name}/{model_name}"

    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None

    # Build optional settings dict (litellm_api_base for custom / Ollama endpoints)
    model_settings = {}
    if settings.llm_base_url:
        model_settings["litellm_api_base"] = str(settings.llm_base_url)
    elif provider_name == "ollama":
        # Ollama base URL does NOT include /v1 suffix because LiteLLM
        # automatically appends /v1 when making requests to Ollama endpoints.
        # This differs from OllamaEmbeddingVectorStore which calls the Ollama API
        # directly (not through LiteLLM) and therefore needs the full URL with /v1.
        # See: https://docs.litellm.ai/docs/providers/ollama
        model_settings["litellm_api_base"] = "http://localhost:11434"

    return LiteLLMModel(
        model_name=litellm_model_name,
        api_key=api_key,
        settings=model_settings if model_settings else None,  # type: ignore[arg-type]
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


def build_chat_agent(
    model: Model | str | None = None,
    settings: Settings | None = None,
) -> Agent[AgentDeps, str]:
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
        settings: Optional settings to use. If None, loads from environment.

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
    settings = settings or get_settings()
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

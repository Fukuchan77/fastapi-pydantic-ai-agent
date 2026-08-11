"""Pydantic-ai API-surface lock (Req 15.1, 15.2, 15.3; L5.4).

Locks the pydantic-ai / `pydantic-ai-litellm` API surface this project
depends on, independently of application code (no `app/` imports here), so a
dependency upgrade that removes, renames, or changes the kind of a symbol
this project uses fails *here* - naming the offending symbol - instead of
surfacing as an opaque application-test failure.

Four subset-only layers, per ADR-8 in `research.md`:

- Layer A - symbol existence per module (`importlib` + `hasattr`).
- Layer B - parameter-name subsets (`inspect.signature(...).parameters`).
- Layer B' - dataclass-field / `__annotations__` subsets, for members
  `inspect.signature` cannot reach (`ModelProfile.__init__` reports
  `(*args, **kwargs)` due to a custom `__new__`; `FallbackModel.models` is an
  instance attribute declared only in `__annotations__`).
- Layer C - kind assertions where the kind is load-bearing (property vs.
  plain attribute, staticmethod, `cached_property`, dataclass vs. type alias).

Every assertion is a *subset* check (`expected <= actual` / `expected -
actual`), never a full-signature or `__all__` snapshot: a benign upstream
*addition* (a new parameter, a new field, a new export) must stay green.
Only a removal, rename, or kind change may fail a test in this module.

`TestAntiFalseGreen` below guards the guard: a subset check against an
accidentally-emptied table passes vacuously, so it separately asserts the
expected-surface tables stay non-empty and every module they name is still
importable - mirroring
`test_file_size_policy.py::test_scan_covers_at_least_one_file_in_each_directory`'s
guard against a silently-empty scan.

Known blind spots (deliberate, not gaps for this module to close):

- Subset assertions can only ever catch a removal, rename, or kind change to
  a symbol already listed below - never an *addition*. A new parameter,
  field, or export this project doesn't use yet is invisible to every layer,
  including the anti-false-green twin.
- The `str(UsageLimitExceeded)` substring parsing in
  `app/agents/guardrails.py::classify_usage_limit_exceeded` has no
  importable symbol or kind to lock - it is behaviour, not shape - so it is
  coverable only behaviourally, by the pinned-template regression test in
  `tests/unit/agents/test_usage_limit_templates.py`, not by anything here.
"""

import dataclasses
import functools
import inspect
import typing
from importlib import import_module

import pytest
from pydantic import TypeAdapter
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets import ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset
from pydantic_ai.usage import RunUsage
from pydantic_ai.usage import UsageLimits
from pydantic_ai_litellm import LiteLLMModel
from pydantic_ai_litellm import LiteLLMModelSettings


# Layer A - symbol existence per module. Every module this project actually
# imports `pydantic_ai`/`pydantic_ai_litellm` symbols from, mapped to the
# subset of symbols used.
_EXPECTED_SYMBOLS: dict[str, frozenset[str]] = {
    "pydantic_ai": frozenset(
        {
            "Agent",
            "RunContext",
            "NativeOutput",
            "CombinedToolset",
            "RunUsage",
            "UsageLimits",
            "UsageLimitExceeded",
            "ModelHTTPError",
        }
    ),
    "pydantic_ai.messages": frozenset(
        {
            "ModelMessage",
            "ModelRequest",
            "ModelResponse",
            "ModelRequestPart",
            "ModelResponsePart",
            "TextPart",
            "TextPartDelta",
            "ToolCallPart",
            "ToolReturnPart",
            "UserPromptPart",
            "SystemPromptPart",
            "RetryPromptPart",
            "BaseToolCallPart",
            "BaseToolReturnPart",
            "FunctionToolCallEvent",
            "PartDeltaEvent",
            "PartStartEvent",
            "ModelMessagesTypeAdapter",
        }
    ),
    "pydantic_ai.models": frozenset({"Model", "infer_model", "ModelRequestParameters"}),
    "pydantic_ai.models.fallback": frozenset({"FallbackModel"}),
    "pydantic_ai.models.function": frozenset({"FunctionModel", "AgentInfo", "DeltaToolCall"}),
    "pydantic_ai.models.test": frozenset({"TestModel"}),
    "pydantic_ai.toolsets": frozenset({"AbstractToolset", "ToolsetTool", "CombinedToolset"}),
    "pydantic_ai.toolsets.wrapper": frozenset({"WrapperToolset"}),
    "pydantic_ai.profiles": frozenset({"ModelProfile"}),
    "pydantic_ai.settings": frozenset({"ModelSettings"}),
    "pydantic_ai.usage": frozenset({"UsageLimits", "RunUsage"}),
    "pydantic_ai.exceptions": frozenset({"FallbackExceptionGroup", "ModelHTTPError"}),
    "pydantic_ai_litellm": frozenset({"LiteLLMModel", "LiteLLMModelSettings"}),
}


# Layer B - parameter-name subsets. Keyed on a human-readable label; targets
# resolved lazily inside the test so a class-attribute lookup failure (an
# even more severe break than a missing parameter) still names the label.
_EXPECTED_PARAMS: dict[str, frozenset[str]] = {
    "Agent.__init__": frozenset(
        {"model", "output_type", "deps_type", "retries", "tools", "toolsets", "system_prompt"}
    ),
    "Agent.run": frozenset({"user_prompt", "deps", "message_history", "usage_limits"}),
    "Agent.iter": frozenset({"user_prompt", "deps", "message_history", "usage_limits"}),
    "Agent.run_stream": frozenset({"user_prompt", "deps", "message_history", "usage_limits"}),
    "Agent.override": frozenset({"tools", "toolsets"}),
    "AbstractToolset.call_tool": frozenset({"name", "tool_args", "ctx", "tool"}),
    "LiteLLMModel.__init__": frozenset({"model_name", "api_key", "settings"}),
    "Model.request": frozenset({"messages", "model_settings", "model_request_parameters"}),
}

_PARAM_TARGETS: dict[str, typing.Callable[..., object]] = {
    "Agent.__init__": Agent.__init__,
    "Agent.run": Agent.run,
    "Agent.iter": Agent.iter,
    "Agent.run_stream": Agent.run_stream,
    "Agent.override": Agent.override,
    "AbstractToolset.call_tool": AbstractToolset.call_tool,
    "LiteLLMModel.__init__": LiteLLMModel.__init__,
    "Model.request": Model.request,
}


class TestSymbolExistence:
    """Layer A: every relied-upon symbol still exists in its module."""

    def test_every_expected_symbol_exists(self) -> None:
        """Fail with the fully-qualified name of every symbol that vanished."""
        missing: list[str] = []
        for module_name, symbols in _EXPECTED_SYMBOLS.items():
            module = import_module(module_name)
            for symbol in sorted(symbols):
                if not hasattr(module, symbol):
                    missing.append(f"{module_name}.{symbol}")
        assert not missing, f"pydantic-ai symbol(s) no longer exist: {missing}"


class TestParameterNameSubsets:
    """Layer B: relied-upon parameter names are still accepted."""

    def test_every_expected_parameter_is_still_accepted(self) -> None:
        """Fail naming the callable and the parameter(s) it lost."""
        mismatches: dict[str, frozenset[str]] = {}
        for label, expected in _EXPECTED_PARAMS.items():
            actual = set(inspect.signature(_PARAM_TARGETS[label]).parameters)
            missing = expected - actual
            if missing:
                mismatches[label] = frozenset(missing)
        assert not mismatches, f"pydantic-ai parameter(s) no longer present: {mismatches}"

    def test_abstract_toolset_call_tool_parameter_order_is_unchanged(self) -> None:
        """`call_tool`'s four params are positional-or-keyword; order is load-bearing."""
        params = list(inspect.signature(AbstractToolset.call_tool).parameters)
        assert params[1:] == ["name", "tool_args", "ctx", "tool"]


class TestDataclassFieldAndAnnotationSubsets:
    """Layer B': fields/annotations for members `inspect.signature` can't reach."""

    def test_usage_limits_has_the_fields_guardrails_relies_on(self) -> None:
        """`UsageLimits` fields (not its signature, which advertises deprecated params)."""
        expected = frozenset({"request_limit", "total_tokens_limit", "tool_calls_limit"})
        actual = set(UsageLimits.__dataclass_fields__)
        missing = expected - actual
        assert not missing, f"UsageLimits field(s) missing: {missing}"

    def test_model_profile_has_the_field_the_known_v2_break_lands_on(self) -> None:
        """`ModelProfile.__init__` reports `(*args, **kwargs)`; fields come from here."""
        expected = frozenset({"supports_json_schema_output"})
        actual = set(ModelProfile.__dataclass_fields__)
        missing = expected - actual
        assert not missing, f"ModelProfile field(s) missing: {missing}"

    def test_run_usage_total_tokens_is_reachable_across_the_mro(self) -> None:
        """`total_tokens` is inherited from `UsageBase`; `vars(RunUsage)` alone misses it."""
        assert inspect.getattr_static(RunUsage, "total_tokens") is not None

    def test_fallback_model_declares_models_only_in_annotations(self) -> None:
        """`FallbackModel.models` is an instance attribute; `hasattr` on the class is False."""
        assert "models" in FallbackModel.__annotations__

    def test_litellm_model_settings_has_the_field_the_factory_relies_on(self) -> None:
        """`LiteLLMModelSettings` is a `TypedDict`; check `__annotations__`, not fields."""
        expected = frozenset({"litellm_api_base"})
        actual = set(LiteLLMModelSettings.__annotations__)
        missing = expected - actual
        assert not missing, f"LiteLLMModelSettings annotation(s) missing: {missing}"

    def test_model_settings_has_the_field_health_checks_rely_on(self) -> None:
        """`ModelSettings` is a `TypedDict`; check `__annotations__`, not fields."""
        expected = frozenset({"max_tokens"})
        actual = set(ModelSettings.__annotations__)
        missing = expected - actual
        assert not missing, f"ModelSettings annotation(s) missing: {missing}"


class TestKindAssertions:
    """Layer C: assertions where *how* a symbol is implemented is load-bearing."""

    @pytest.mark.parametrize("name", ["toolsets", "output_type", "model"])
    def test_agent_run_configuration_attributes_are_properties(self, name: str) -> None:
        """`app/agents/guardrails.py` reads these as computed properties, not plain data."""
        attr = inspect.getattr_static(Agent, name)
        assert isinstance(attr, property), f"Agent.{name} is no longer a property"

    def test_model_profile_is_a_cached_property(self) -> None:
        """`app/llm/factory.py::supports_native_output` relies on lazy, cached computation."""
        attr = inspect.getattr_static(Model, "profile")
        assert isinstance(attr, functools.cached_property), (
            "Model.profile is no longer a functools.cached_property"
        )

    @pytest.mark.parametrize("name", ["is_model_request_node", "is_call_tools_node", "is_end_node"])
    def test_agent_node_predicates_are_staticmethods_returning_typeis(self, name: str) -> None:
        """`_stream.py` narrows graph nodes via these; a `TypeIs` return keeps narrowing."""
        raw = inspect.getattr_static(Agent, name)
        assert isinstance(raw, staticmethod), f"Agent.{name} is no longer a staticmethod"
        hints = typing.get_type_hints(raw.__func__)
        assert typing.get_origin(hints["return"]) is typing.TypeIs, (
            f"Agent.{name} no longer returns a TypeIs narrowing type"
        )

    def test_model_messages_type_adapter_is_a_type_adapter_instance(self) -> None:
        """`RedisSessionStore` calls `.validate_json`/`.dump_json` on this instance."""
        assert isinstance(ModelMessagesTypeAdapter, TypeAdapter)
        assert hasattr(ModelMessagesTypeAdapter, "validate_json")
        assert hasattr(ModelMessagesTypeAdapter, "dump_json")

    def test_model_message_is_a_type_alias_not_a_class(self) -> None:
        """`ModelMessage` is `Annotated[Union[...], ...]`; `isinstance` against it is impossible."""
        assert not inspect.isclass(ModelMessage)
        assert typing.get_origin(ModelMessage) is not None

    def test_wrapper_toolset_still_extends_abstract_toolset(self) -> None:
        """`_GuardedToolset` subclasses `WrapperToolset`; guardrails relies on this MRO."""
        mro_names = {cls.__name__ for cls in WrapperToolset.__mro__}
        assert "AbstractToolset" in mro_names
        assert "ABC" in mro_names

    def test_toolset_tool_fields_are_keyword_only(self) -> None:
        """`_GuardedToolset.call_tool` receives a `ToolsetTool`; its fields are kw-only."""
        expected = frozenset(
            {"toolset", "tool_def", "max_retries", "args_validator", "args_validator_func"}
        )
        actual = set(ToolsetTool.__dataclass_fields__)
        missing = expected - actual
        assert not missing, f"ToolsetTool field(s) missing: {missing}"
        not_kw_only = [
            field.name
            for field in dataclasses.fields(ToolsetTool)
            if field.name in expected and not field.kw_only
        ]
        assert not not_kw_only, f"ToolsetTool field(s) no longer keyword-only: {not_kw_only}"


class TestAntiFalseGreen:
    """Guards the guard against a silently-emptied expected-surface table.

    Every layer above is a subset check, which passes vacuously if its table
    were ever emptied by accident (e.g. a bad merge or an over-eager
    cleanup). Mirrors
    `test_file_size_policy.py::test_scan_covers_at_least_one_file_in_each_directory`'s
    guard against a silently-empty scan.
    """

    def test_expected_symbols_table_is_non_empty_and_every_entry_is_non_empty(
        self,
    ) -> None:
        """A dropped table or entry would make Layer A's subset check pass on nothing."""
        assert _EXPECTED_SYMBOLS, "_EXPECTED_SYMBOLS must not be empty"
        empty_modules = [module for module, symbols in _EXPECTED_SYMBOLS.items() if not symbols]
        assert not empty_modules, f"module(s) with an empty symbol set: {empty_modules}"

    def test_every_expected_symbols_module_is_still_importable(self) -> None:
        """A module key that no longer resolves is a bigger break than a missing symbol."""
        unimportable: list[str] = []
        for module_name in _EXPECTED_SYMBOLS:
            try:
                import_module(module_name)
            except ImportError:
                unimportable.append(module_name)
        assert not unimportable, f"module(s) no longer importable: {unimportable}"

    def test_expected_params_table_is_non_empty_and_every_entry_is_non_empty(
        self,
    ) -> None:
        """A dropped table or entry would make Layer B's subset check pass on nothing."""
        assert _EXPECTED_PARAMS, "_EXPECTED_PARAMS must not be empty"
        empty_labels = [label for label, params in _EXPECTED_PARAMS.items() if not params]
        assert not empty_labels, f"label(s) with an empty parameter set: {empty_labels}"

    def test_every_expected_params_label_has_a_resolvable_target(self) -> None:
        """A label present in one table but not the other silently drops that check."""
        missing_targets = set(_EXPECTED_PARAMS) - set(_PARAM_TARGETS)
        assert not missing_targets, f"label(s) with no resolvable target: {missing_targets}"

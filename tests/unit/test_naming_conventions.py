"""Unit tests to enforce test naming conventions.

Task 16.3: Verify that test names accurately reflect their purpose.
Tests should not have names like "test_something_fails" if they actually
verify that something succeeds.
"""

import ast
from pathlib import Path


def test_no_misleading_fail_names_in_test_main() -> None:
    """Verify test_main.py doesn't have tests with misleading 'fails' in name.

    Task 16.3: Tests named with "_fails" should actually test failure scenarios.
    If a test verifies success, it should not have "fails" in the name.
    """
    # Read the test file
    test_file = Path("tests/unit/test_main.py")
    content = test_file.read_text()

    # Parse the AST to find all test functions
    tree = ast.parse(content)
    test_functions = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]

    # Find tests with "fail" or "fails" in the name
    misleading_names = [name for name in test_functions if "_fail" in name.lower()]

    # For this task, we know test_main_app_import_fails is misleading
    # because it tests that import SUCCEEDS, not fails
    assert len(misleading_names) == 0, (
        f"Found test(s) with potentially misleading 'fail/fails' in name: {misleading_names}. "
        "Tests named with '_fails' should test failure scenarios. "
        "If they test success, rename them (e.g., 'test_main_app_can_be_imported')."
    )


def test_no_misleading_fail_names_in_test_health() -> None:
    """Verify test_health.py doesn't have tests with misleading 'fails' in name.

    Task 16.3: Tests named with "_fails" should actually test failure scenarios.
    If a test verifies success, it should not have "fails" in the name.
    """
    # Read the test file
    test_file = Path("tests/unit/api/test_health.py")
    content = test_file.read_text()

    # Parse the AST to find all test functions
    tree = ast.parse(content)
    test_functions = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]

    # Find tests with "fail" or "fails" in the name
    misleading_names = [name for name in test_functions if "_fail" in name.lower()]

    # For this task, we know test_health_endpoint_import_fails is misleading
    # because it tests that import SUCCEEDS, not fails
    assert len(misleading_names) == 0, (
        f"Found test(s) with potentially misleading 'fail/fails' in name: {misleading_names}. "
        "Tests named with '_fails' should test failure scenarios. "
        "If they test success, rename them (e.g., 'test_health_router_can_be_imported')."
    )

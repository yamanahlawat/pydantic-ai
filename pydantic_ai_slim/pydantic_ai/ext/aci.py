# Checking whether aci-sdk is installed
try:
    from aci import ACI
except ImportError as _import_error:
    raise ImportError('Please install `aci-sdk` to use ACI.dev tools') from _import_error

from typing import Any

from aci import ACI

from pydantic_ai import Tool


def _clean_schema(schema):
    if isinstance(schema, dict):
        # Remove non-standard keys (e.g., 'visible')
        return {k: _clean_schema(v) for k, v in schema.items() if k not in {'visible'}}
    elif isinstance(schema, list):
        return [_clean_schema(item) for item in schema]
    else:
        return schema


def tool_from_aci(aci_function: str, linked_account_owner_id: str) -> Tool:
    """Creates a Pydantic AI tool proxy from an ACI function.

    Args:
        aci_function: The ACI function to wrao.
        linked_account_owner_id: The ACI user ID to execute the function on behalf of.

    Returns:
        A Pydantic AI tool that corresponds to the ACI.dev tool.
    """
    aci = ACI()
    function_definition = aci.functions.get_definition(aci_function)
    function_name = function_definition['function']['name']
    function_description = function_definition['function']['description']
    inputs = function_definition['function']['parameters']

    json_schema = {
        'additionalProperties': inputs.get('additionalProperties', False),
        'properties': inputs.get('properties', {}),
        'required': inputs.get('required', []),
        # Default to 'object' if not specified
        'type': inputs.get('type', 'object'),
    }

    # Clean the schema
    json_schema = _clean_schema(json_schema)

    def implementation(*args: Any, **kwargs: Any) -> str:
        if args:
            raise TypeError('Positional arguments are not allowed')
        return aci.handle_function_call(
            function_name,
            kwargs,
            linked_account_owner_id=linked_account_owner_id,
            allowed_apps_only=True,
        )

    return Tool.from_schema(
        function=implementation,
        name=function_name,
        description=function_description,
        json_schema=json_schema,
    )

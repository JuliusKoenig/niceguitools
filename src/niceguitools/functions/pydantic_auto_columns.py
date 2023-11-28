from typing import Any

from pydantic_core import PydanticUndefinedType


def pydantic_auto_columns(model: Any) -> list[dict[str, Any]]:
    """
    Generate columns for a nicegui table from a pydantic model.

    :param model: A pydantic model.
    :return: A list of columns.
    """

    # check if model has model_fields attribute
    if not hasattr(model, 'model_fields'):
        raise ValueError(f'{model} is not a pydantic model')

    columns = []
    for field_name, field in model.model_fields.items():
        # check if required field
        required = False
        if type(field.default) is PydanticUndefinedType and field.default_factory is None:
            required = True

        columns.append({
            'name': field_name,
            'label': field.description or field_name,
            'field': field_name,
            'required': required,
            'align': 'left',
        })

    return columns

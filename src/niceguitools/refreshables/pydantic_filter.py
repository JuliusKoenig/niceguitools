import asyncio
import inspect
import json
import logging
import time
from copy import deepcopy
from typing import Any, Optional, Union, Callable, Awaitable, Coroutine, Type, Literal

from nicegui.elements.mixins.text_element import TextElement
from nicegui.events import UiEventArguments, ValueChangeEventArguments
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from nicegui import ui, nicegui

logger = logging.getLogger(__name__)

OperatorAttr1 = Any
OperatorAttr2 = Any
OperatorResult = bool
_OperatorCallable = Callable[[int, OperatorAttr1, OperatorAttr2], Awaitable[OperatorResult]]
OperatorCallable = Callable[[OperatorAttr1, OperatorAttr2], Awaitable[OperatorResult]]
OperatorSyncCallable = Callable[[OperatorAttr1, OperatorAttr2], OperatorResult]
NewOperatorCallable = Union[OperatorSyncCallable, OperatorCallable]

ExternalReloadAttr = Union[list[Any], list[dict[str, Any]], str]
ExternalReloadResult = None
ExternalReloadAwaitable = Awaitable[ExternalReloadResult]
ExternalReloadAsyncCallable = Callable[[ExternalReloadAttr], Awaitable[ExternalReloadResult]]
ExternalReloadSyncCallable = Callable[[ExternalReloadAttr], ExternalReloadResult]
ExternalReloadCallable = Union[ExternalReloadSyncCallable, ExternalReloadAsyncCallable, ExternalReloadAwaitable]

_BEFORE_CALLED: Optional[dict[id, bool]] = None
_BEFORE_CALLABLE: Optional[dict[id, ExternalReloadCallable]] = None
_ON_RELOAD_EXTERNAL_CALLED: Optional[dict[id, bool]] = None
_ON_RELOAD_EXTERNAL_CALLABLE: Optional[dict[id, ExternalReloadCallable]] = None
_AFTER_CALLED: Optional[dict[id, bool]] = None
_AFTER_CALLABLE: Optional[dict[id, ExternalReloadCallable]] = None


class PydanticFilter:
    def __init__(self,
                 get_models: Callable[[], list[BaseModel]],
                 model: Type[BaseModel],
                 name: Optional[str] = None,
                 current_filter: Optional[list[tuple[str, str, str, bool]]] = None,
                 operator: Optional[dict[str, NewOperatorCallable]] = None,
                 dense: bool = False,
                 reload_btn: bool = True,
                 auto_reload: bool = True,
                 mode: Literal["pydantic", "dict", "str"] = "pydantic",
                 stack_level: int = 2):
        """
        A filter for a list of models

        :param get_models: A function that returns a list of models
        :param name: The name of the filter
        :param current_filter: A list of tuples of the form (key, operator, value, value_type, is_new) where is_new is True if the filter is new and False otherwise
        :param operator: A dictionary of operators. The key is the operator and the value is a function that takes a value and returns True if the value matches the filter
        and False otherwise
        :param dense: True if the filter should be dense, False otherwise
        :param reload_btn: True if the filter should have a reload button, False otherwise
        :param auto_reload: True if the filter should automatically reload, False otherwise
        :param stack_level: The stack level of the page
        :return: PydanticFilter
        """

        stack = inspect.stack()
        client = None
        for current_stack_level in range(stack_level, len(stack)):
            caller_locals = stack[current_stack_level].frame.f_locals

            # check if client is in the locals
            if "client" not in caller_locals.keys():
                continue
            client = caller_locals["client"]
            if not isinstance(client, nicegui.Client):
                continue
            break

        if client is None:
            raise ValueError("Could not find a nicegui.Client")

        self._client = client

        self._on_init_or_call = True
        self._called = False

        if name is None:
            name = self.__class__.__name__
        self._name: str = name
        if type(self._name) is not str:
            raise TypeError(f"Invalid type '{type(self._name)}'")

        self._get_models: Callable[[], list[BaseModel]] = get_models
        self._model: Type[BaseModel] = model
        self._models: list[BaseModel] = []
        self._filtered_models: list[BaseModel] = []
        if current_filter is None:
            current_filter = []
        self._current_filter: list[tuple[str, str, str, bool]] = current_filter
        if operator is None:
            operator = {}

        # add default operators
        if "==" not in operator.keys():
            operator["=="] = lambda field_value, value: field_value == value
        if "!=" not in operator.keys():
            operator["!="] = lambda field_value, value: field_value != value
        if "<" not in operator.keys():
            operator["<"] = lambda field_value, value: field_value < value
        if "<=" not in operator.keys():
            operator["<="] = lambda field_value, value: field_value <= value
        if ">" not in operator.keys():
            operator[">"] = lambda field_value, value: field_value > value
        if ">=" not in operator.keys():
            operator[">="] = lambda field_value, value: field_value >= value

        self._operator: dict[str, OperatorCallable] = operator
        self._is_reload_btn_enabled: bool = reload_btn
        self._auto_reload: bool = auto_reload

        self._reload_timer: Optional[ui.timer] = None
        self._should_reload_timer: Optional[ui.timer] = None
        self._should_reload_last_time: Optional[float] = 0
        self._should_reload: Union[UiEventArguments, str, None] = None

        self._on_reload_state: Literal["ready", "before", "reload", "after"] = "ready"
        self._on_reload_before: Optional[ExternalReloadCallable] = None
        self._on_reload_after: Optional[ExternalReloadCallable] = None
        self._on_reload_after_mode: Literal["pydantic", "dict", "str"] = mode

        self._reads: Optional[list[tuple[TextElement, ui.button, ui.button]]] = None
        self._edits: Optional[list[tuple[ui.select, ui.select, ui.input, ui.button, ui.button]]] = None
        self._dense: bool = dense
        self._bt_height: Optional[int] = None
        self._le_props: Optional[str] = None
        self._on_init_or_call = False
        self._disable_matching = False

        logger.debug(f"{self} created")

    def __str__(self):
        id = self.id
        name = self.name
        return f"{self.__class__.__name__}(id={id}, name={name})"

    @ui.refreshable
    def __call__(self,
                 get_models: Optional[Callable[[], list[BaseModel]]] = None,
                 current_filter: Optional[list[tuple[str, str, str, bool]]] = None,
                 operator: Optional[dict[str, NewOperatorCallable]] = None,
                 dense: Optional[bool] = None,
                 reload_btn: Optional[bool] = None,
                 auto_reload: Optional[bool] = None):
        """
        Draws a filter for a list of models

        :param get_models: A function that returns a list of models
        :param current_filter: A list of tuples of the form (key, operator, value, value_type, is_new) where is_new is True if the filter is new and False otherwise
        :param operator: A dictionary of operators. The key is the operator and the value is a function that takes a value and returns True if the value matches the filter
        and False otherwise
        :param dense: True if the filter should be dense, False otherwise
        :param reload_btn: True if the filter should have a reload button, False otherwise
        :param auto_reload: True if the filter should automatically reload, False otherwise
        :return:
        """

        refresh = self._called

        logger.debug(f"{self}.__call__({refresh=})")

        if self._on_init_or_call:
            raise ValueError(f"Call {self.__class__.__name__}.__call__ only once")
        self._on_init_or_call = True

        # get_models
        if get_models is None:
            get_models = self._get_models
        if not callable(get_models):
            raise TypeError(f"Invalid type '{type(get_models)}' for get_models, expected 'Callable[[], list[Dictable]]'")
        self._get_models = get_models

        if current_filter is None:
            current_filter = self._current_filter
        self.current_filter = current_filter

        if operator is None:
            operator = self._operator
        self.operator = operator

        if dense is None:
            dense = self.dense
        self.dense = dense

        if reload_btn is None:
            reload_btn = self.is_reload_btn_enabled
        self.is_reload_btn_enabled = reload_btn

        if auto_reload is None:
            auto_reload = self._auto_reload
        self.auto_reload = auto_reload

        self._reads = []
        self._edits = []

        # # should_reload_timer
        # if not refresh or not self._should_reload_running:
        self._should_reload_timer = ui.timer(0.1, self._should_reload_timer_func)
        # else:
        #     if self._should_reload_timer is None:
        #         raise ValueError(f"{self}._should_reload_timer is None")

        if not self._called and self._auto_reload:  # load first time
            self._should_reload = "init"

        with ui.row().classes("gap-x-0 gap-y-0"):
            if self.is_reload_btn_enabled:
                self._add_reload_btn()
            self._add_add_btn()
            index = 0
            for c_filter in self._current_filter:
                c_filter_new = c_filter[3]
                if type(c_filter_new) is not bool:
                    raise TypeError(f"Invalid type '{type(c_filter_new)}'")

                # read
                self._add_read(i=index)

                # edit
                self._add_edit(i=index)

                # if new
                if c_filter_new:
                    self.set_visibility(index=index, visible=True)
                else:
                    self.set_visibility(index=index, visible=False)
                    self._set_read(i=index)

                index += 1

        if not self._on_init_or_call:
            raise ValueError(f"Call {self}.__call__ already executed")
        self._on_init_or_call = False
        self._called = True

        logger.debug(f"{self}.__call__({refresh=}) finished")

    @property
    def id(self) -> id:
        """
        The id of the filter

        :return: id - The id of the filter
        """

        return id(self)

    @property
    def name(self) -> str:
        """
        The name of the filter

        :return: str - The name of the filter
        """

        return self._name

    @property
    def client(self) -> nicegui.Client:
        """
        The client of the filter

        :return: nicegui.Client - The client of the filter
        """

        return self._client

    @property
    def should_reload(self) -> Union[UiEventArguments, str, None]:
        """
        True if the filter should reload, False otherwise

        :return: bool - True if the filter should reload, False otherwise
        """

        return self._should_reload

    @property
    def _should_reload_running(self) -> bool:
        """
        True if the filter should reload, False otherwise

        :return: bool - True if the filter should reload, False otherwise
        """

        delta = time.perf_counter() - self._should_reload_last_time
        if delta < 0.1:
            return True
        return False

    @property
    def models(self) -> list[BaseModel]:
        """
        The models of the filter
        :return: list[BaseModel] - A copy of the models
        """

        return deepcopy(self._models)

    @property
    def models_dict(self) -> list[dict]:
        """
        The models of the filter as a list of dictionaries
        :return: list[dict] - A copy of the models as a list of dictionaries
        """

        models_dict = []
        for model in self._models:
            models_dict.append(model.model_dump())
        return models_dict

    @property
    def models_json(self) -> str:
        """
        The models of the filter as a json string
        :return: str - A copy of the models as a json string
        """

        json_list = []
        for model in self.models_dict:
            json_list.append(model)
        json_str = json.dumps(json_list)
        return json_str

    @property
    def models_json_pretty(self) -> str:
        """
        The models of the filter as a pretty json string
        :return: str - A copy of the models as a pretty json string
        """

        json_list = []
        for model in self.models_dict:
            json_list.append(model)
        json_str = json.dumps(json_list, indent=4)
        return json_str

    @property
    def model_types(self) -> list[Type[BaseModel]]:
        """
        The types of the models of the filter
        :return: list[Type[BaseModel]] - A copy of the types of the models
        """

        model_types = []

        if self._model is not None:
            model_types.append(self._model)

        for model in self._models:
            type_model = type(model)
            if type_model not in model_types:
                model_types.append(type_model)

        return model_types

    @property
    def model_fields(self) -> dict[str, FieldInfo]:
        """
        The fields of the models of the filter
        :return: dict[str, FieldInfo] - A copy of the fields of the models
        """

        fields = {}
        for model_types in self.model_types:
            for field_name, field in model_types.model_fields.items():
                if field_name in fields.keys():
                    continue
                fields[field_name] = field
        return fields

    @property
    def filtered_models(self) -> list[BaseModel]:
        """
        The filtered models of the filter
        :return: list[BaseModel] - A copy of the filtered models
        """

        return deepcopy(self._filtered_models)

    @property
    def filtered_models_dict(self) -> list[dict]:
        """
        The filtered models of the filter as a list of dictionaries
        :return: list[dict] - A copy of the filtered models as a list of dictionaries
        """

        models_dict = []
        for model in self._filtered_models:
            models_dict.append(model.model_dump())
        return models_dict

    @property
    def filtered_models_json(self) -> str:
        """
        The filtered models of the filter as a json string
        :return: str - A copy of the filtered models as a json string
        """

        json_list = []
        for model in self.filtered_models_dict:
            json_list.append(model)
        json_str = json.dumps(json_list)
        return json_str

    @property
    def filtered_models_json_pretty(self) -> str:
        """
        The filtered models of the filter as a pretty json string
        :return: str - A copy of the filtered models as a pretty json string
        """

        json_list = []
        for model in self.filtered_models_dict:
            json_list.append(model)
        json_str = json.dumps(json_list, indent=4)
        return json_str

    @property
    def current_filter(self) -> list[tuple[str, str, str, bool]]:
        """
        The current filter

        :return: list[tuple[str, str, str, bool]] - A copy of the current filter
        """

        if self._current_filter is None:
            raise ValueError(f"Call {self.__class__.__name__}.__call__ first")
        return deepcopy(self._current_filter)

    @current_filter.setter
    def current_filter(self, value: list[tuple[str, str, str, bool]]):
        logger.debug(f"Setting {self}.current_filter({value=})")
        for c_filter in value:
            c_filter_key = c_filter[0]
            if type(c_filter_key) is not str:
                raise TypeError(f"Invalid type '{type(c_filter_key)}'")
            c_filter_operator = c_filter[1]
            if type(c_filter_operator) is not str:
                raise TypeError(f"Invalid type '{type(c_filter_operator)}'")
            c_filter_value = c_filter[2]
            if type(c_filter_value) is not str:
                raise TypeError(f"Invalid type '{type(c_filter_value)}'")
            c_filter_new = c_filter[3]
            if type(c_filter_new) is not bool:
                raise TypeError(f"Invalid type '{type(c_filter_new)}'")
        self._current_filter = deepcopy(value)
        if not self._on_init_or_call and self._called:
            self.__call__.refresh()

    def _parse_operator(self, operator: dict[str, Any]) -> dict[str, OperatorCallable]:
        """
        Parses the operator parameter

        :param operator: A dictionary of operators. The key is the operator and the value is a function that takes a value and returns True if the value matches the filter
        and False otherwise
        :return: dict[str, ExternalReloadCallable]
        """

        for k, v in operator.items():
            if type(k) is not str:
                raise TypeError(f"Invalid type '{type(k)}'")
            if callable(v):
                continue
            elif asyncio.iscoroutinefunction(v):
                continue
            else:
                raise TypeError(f"Invalid type '{type(v)}'")

        def wrap_operator(operator_name: str, operator_func: NewOperatorCallable):
            # check if the operator is a awaitable
            if asyncio.iscoroutinefunction(operator_func):
                async def wrapped_operator(index: int, field_value: Any, value: Any) -> bool:
                    result = await self.validate_filter(index=index, value=value)
                    if type(result) is ValueError:
                        return False
                    result = await operator_func(field_value, value)
                    logger.debug(f"{operator_name}({field_value=}, {value=}) -> {result=}")
                    return result

                return wrapped_operator
            else:
                async def wrapped_operator(index: int, field_value: Any, value: Any) -> bool:
                    result = await self.validate_filter(index=index, value=value)
                    if type(result) is ValueError:
                        return False
                    result = operator_func(field_value, value)
                    logger.debug(f"{operator_name}({field_value=}, {value=}) -> {result=}")
                    return result

                return wrapped_operator

        wrap_operator.__wrapped__ = True

        # wrap all operators in a logging function
        _wrap_operator = {}
        for _operator_name, _operator_func in operator.items():
            if getattr(_operator_func, "__wrapped__", False):
                continue
            _wrap_operator[_operator_name] = wrap_operator(_operator_name, _operator_func)
        return deepcopy(_wrap_operator)

    @property
    def operator(self) -> dict[str, _OperatorCallable]:
        """
        The operator of the filter
        :return: dict[str, ExternalReloadCallable] - A copy of the operator
        """

        if self._operator is None:
            raise ValueError(f"Call {self.__class__.__name__}.__call__ first")
        return self._parse_operator(self._operator)

    @operator.setter
    def operator(self, value: dict[str, Union[Callable[[Any], bool], Coroutine[Any, Any, bool]]]):
        logger.debug(f"Setting {self}.operator({value=})")
        self._operator = value
        if not self._on_init_or_call and self._called:
            self.__call__.refresh()

    @property
    def dense(self) -> bool:
        """
        True if the filter is dense, False otherwise
        :return: bool
        """

        if self._dense is None:
            raise ValueError(f"Call {self.__class__.__name__}.__call__ first")
        return self._dense

    @dense.setter
    def dense(self, value: bool):
        if type(value) is not bool:
            raise TypeError(f"Invalid type '{type(value)}'")
        if value:
            self._dense = True
            self._bt_height = 40
            self._le_props = "dense"
        else:
            self._dense = False
            self._bt_height = 56
            self._le_props = ""
        logger.debug(f"Setting {self}.dense({value=})")
        if not self._on_init_or_call and self._called:
            self.__call__.refresh()

    @property
    def is_reload_btn_enabled(self) -> bool:
        """
        True if the reload button is enabled, False otherwise
        :return: bool
        """

        if self._is_reload_btn_enabled is None:
            raise ValueError(f"Call {self.__class__.__name__}.__call__ first")
        return self._is_reload_btn_enabled

    @is_reload_btn_enabled.setter
    def is_reload_btn_enabled(self, value: bool):
        if type(value) is not bool:
            raise TypeError(f"Invalid type '{type(value)}'")
        self._is_reload_btn_enabled = value
        logger.debug(f"Setting {self}.is_reload_btn_enabled({value=})")
        if not self._on_init_or_call and self._called:
            self.__call__.refresh()

    @property
    def auto_reload(self) -> bool:
        """
        True if the filter automatically reloads, False otherwise
        :return: bool
        """

        if self._auto_reload is None:
            raise ValueError(f"Call {self.__class__.__name__}.__call__ first")
        return self._auto_reload

    @auto_reload.setter
    def auto_reload(self, value: bool):
        if type(value) is not bool:
            raise TypeError(f"Invalid type '{type(value)}'")
        self._auto_reload = value
        logger.debug(f"Setting {self}.auto_reload({value=})")
        if not self._on_init_or_call and self._called:
            self.__call__.refresh()

    @property
    def on_reload_before(self) -> Optional[ExternalReloadCallable]:
        """
        The on_reload_before of the filter
        :return: ExternalReloadCallable - A copy of the on_reload_external_before
        """

        return self._on_reload_before

    @on_reload_before.setter
    def on_reload_before(self, value: ExternalReloadCallable):
        logger.debug(f"Setting {self}.on_reload_before({value=})")
        if not callable(value):
            raise TypeError(f"Invalid type '{type(value)}'")
        self._on_reload_before = value

        if not self._on_init_or_call and self._called:
            self.__call__.refresh()

    @property
    def on_reload_after(self) -> Optional[ExternalReloadCallable]:
        """
        The on_reload_after of the filter
        :return: ExternalReloadCallable - A copy of the on_reload_external_after
        """

        return self._on_reload_after

    @on_reload_after.setter
    def on_reload_after(self, value: ExternalReloadCallable):
        logger.debug(f"Setting {self}.on_reload_after({value=})")
        if not callable(value):
            raise TypeError(f"Invalid type '{type(value)}'")
        if isinstance(value, ui.refreshable):
            real_func = getattr(value, "refresh")
            real_func_name = f"{value.__class__.__name__}.refresh"
        else:
            real_func = value
            real_func_name = value.__name__

        real_func_sig_params = dict(inspect.signature(real_func).parameters)
        if len(real_func_sig_params) != 1:
            raise ValueError(f"Invalid number of parameters '{len(real_func_sig_params)}' for {real_func_name}, expected 1")
        mode_set = False
        if real_func_sig_params[list(real_func_sig_params.keys())[0]].annotation is not inspect.Parameter.empty:
            attr_type = list(real_func_sig_params.values())[0].annotation
            attr_type_str = str(attr_type)
            if attr_type_str == "str":
                self.on_reload_after_mode = "json"
                mode_set = True
            elif attr_type_str == "list[dict]":
                self.on_reload_after_mode = "dict"
                mode_set = True
            elif attr_type_str.startswith("list[") and attr_type_str.endswith("]"):
                self.on_reload_after_mode = "pydantic"
                mode_set = True
            else:
                logger.warning(f"Invalid type '{attr_type_str}' for {real_func_name}, expected 'str', 'list[dict]' or 'list[pydantic.BaseModel]'")

        if not mode_set:
            self.on_reload_after_mode = "pydantic"
        self._on_reload_after = value

        if not self._on_init_or_call and self._called:
            self.__call__.refresh()

    @property
    def on_reload_after_mode(self) -> Literal["pydantic", "dict", "str"]:
        """
        The mode of the on_reload_external_after parameter. Can be "pydantic", "dict" or "str"
        :return: Literal["pydantic", "dict", "str"]
        """

        return self._on_reload_after_mode

    @on_reload_after_mode.setter
    def on_reload_after_mode(self, value: Literal["pydantic", "dict", "str"]):
        logger.debug(f"Setting {self}.on_reload_external_after_mode({value=})")
        if type(value) is not str:
            raise TypeError(f"Invalid type '{type(value)}'")
        if value not in ["pydantic", "dict", "str"]:
            raise ValueError(f"Invalid value '{value}'")
        self._on_reload_after_mode = value

    @property
    def reads(self) -> list[tuple[TextElement, ui.button, ui.button]]:
        """
        The reads of the filter
        :return: list[tuple[TextElement, ui.button, ui.button]] - All reads
        """

        if self._reads is None:
            raise ValueError(f"Call {self.__class__.__name__}.__call__ first")
        return self._reads

    @property
    def edits(self) -> list[tuple[ui.select, ui.select, ui.input, ui.button, ui.button]]:
        """
        The edits of the filter
        :return: list[tuple[ui.select, ui.select, ui.input, ui.button, ui.button]] - All edits
        """

        if self._edits is None:
            raise ValueError(f"Call {self.__class__.__name__}.__call__ first")
        return self._edits

    @property
    def bt_height(self) -> int:
        """
        The height of the buttons
        :return: int - The height of the buttons
        """

        if self._bt_height is None:
            raise ValueError(f"Call {self.__class__.__name__}.__call__ first")
        return self._bt_height

    @property
    def le_props(self) -> str:
        """
        The properties of the line edits
        :return: str - The properties of the line edits
        """

        if self._le_props is None:
            raise ValueError(f"Call {self.__class__.__name__}.__call__ first")
        return self._le_props

    @property
    def matching_disabled(self) -> bool:
        """
        True if the matching is disabled, False otherwise
        :return: bool
        """

        return self._disable_matching

    def _get_index(self, event: Optional[Any]) -> int:
        """
        Gets the index of the event sender
        :param event: The event
        :return: int - The index of the event sender
        """

        if event is None:
            return len(self._reads) - 1
        for _reads in self._reads:
            for item in _reads:
                if item is event.sender:
                    return self._reads.index(_reads)
        for _edits in self._edits:
            for item in _edits:
                if item is event.sender:
                    return self._edits.index(_edits)
        raise ValueError("Event sender not found")

    def _add_add_btn(self) -> None:
        """
        Adds the add button
        :return: None
        """

        logger.debug(f"Adding {self}.{self._add_add_btn.__name__}()")
        with TextElement(tag="q-badge", text=f"") as label:
            label.props("color=grey-8")
            label.classes("pr-0")
            label.classes("pl-0")
            label.classes("mr-2")
            label.classes("mb-2")

            bt_add = ui.button(icon="add", color="primary-textcolor", on_click=self._on_add_button)
            bt_add.props("flat")
            bt_add.props("size=sm")
            bt_add.props("padding=xs")

    def _add_reload_btn(self) -> None:
        """
        Adds the reload button
        :return: None
        """

        logger.debug(f"Adding {self}.{self._add_reload_btn.__name__}()")
        with TextElement(tag="q-badge", text=f"") as label:
            label.props("color=grey-8")
            label.classes("pr-0")
            label.classes("pl-0")
            label.classes("mr-2")
            label.classes("mb-2")

            bt_add = ui.button(icon="refresh", color="primary-textcolor", on_click=self._on_reload_button)
            bt_add.props("flat")
            bt_add.props("size=sm")
            bt_add.props("padding=xs")

    def _add_read(self, i: int) -> None:
        """
        Adds the read
        :param i: The index of the read
        :return: None
        """

        logger.debug(f"Adding {self}.{self._add_read.__name__}({i=})")
        # read
        with TextElement(tag="q-badge", text="UNSET") as label:
            label.props("color=grey-8")
            label.classes("pr-0")
            label.classes("mr-2")
            label.classes("mb-2")

            bt_edit = ui.button(icon="edit", color="primary-textcolor", on_click=self._on_edit_button)
            bt_edit.props("flat")
            bt_edit.props("size=sm")
            bt_edit.props("padding=xs")
            bt_edit.classes("ml-1")

            bt_remove = ui.button(icon="delete", color="primary-textcolor", on_click=self._on_remove_button)
            bt_remove.props("flat")
            bt_remove.props("size=sm")
            bt_remove.props("padding=xs")

        self._reads.append((label, bt_edit, bt_remove))

    def _set_read(self, i: int):
        """
        Sets the read

        :param i: The index of the read
        :return: None
        """

        label_str = ""

        key = self._current_filter[i][0]
        if key not in self.model_fields.keys():
            raise ValueError(f"Invalid key '{key}'")
        label_str += f"{key} "

        operator = self._current_filter[i][1]
        if operator not in self.operator.keys():
            raise ValueError(f"Invalid operator '{operator}'")
        label_str += f"{operator} "

        value = self._current_filter[i][2]
        label_str += f"'{value}'"

        logger.debug(f"Setting {self}.{self._set_read.__name__}({i=}) to '{label_str}'")

        self._reads[i][0].text = label_str

    def _add_edit(self, i: int) -> None:
        """
        Adds the edit
        :param i: The index of the edit
        :return: None
        """

        logger.debug(f"Adding {self}.{self._add_edit.__name__}({i=})")
        # edit
        with ui.row().classes("gap-x-0"):
            keys = list(self.model_fields.keys())
            key_select = ui.select(options=keys, on_change=self._on_key_changed)
            key_select.props(self._le_props)
            key_select.set_visibility(False)

            operators = list(self.operator.keys())
            operator_select = ui.select(options=operators, on_change=self._on_operator_changed)
            operator_select.props(self._le_props)
            operator_select.set_visibility(False)

            value_input = ui.input(on_change=self._on_value_changed)
            value_input.props(self._le_props)
            value_input.set_visibility(False)

            confirm_button = ui.button(icon="done", color="positive", on_click=self._on_confirm_button)
            confirm_button.style(f"height: {self._bt_height}px;")
            confirm_button.props("flat")
            confirm_button.props("padding=xs")
            confirm_button.set_visibility(False)

            cancel_button = ui.button(icon="close", color="negative", on_click=self._on_cancel_button)
            cancel_button.style(f"height: {self._bt_height}px;")
            cancel_button.props("flat")
            cancel_button.props("padding=xs")
            cancel_button.set_visibility(False)

        self._edits.append((key_select, operator_select, value_input, confirm_button, cancel_button))

    def _set_edit(self, i: int) -> None:
        """
        Sets the edit
        :param i: The index of the edit
        :return: None
        """

        key_select = self._edits[i][0]
        key_select.set_value(self._current_filter[i][0])

        operator_select = self._edits[i][1]
        operator_select.set_value(self._current_filter[i][1])

        value_input = self._edits[i][2]
        value_input.set_value(self._current_filter[i][2])

        logger.debug(f"Setting {self}.{self._set_edit.__name__}({i=}) to '{self._current_filter[i]}'")

    async def _on_add_button(self, event: UiEventArguments) -> None:
        """
        Internal called when the add button is pressed
        :param event: The event
        :return: None
        """

        _ = event
        if not await self.on_add_button(event):
            return
        self.add_filter()

    async def on_add_button(self, event: UiEventArguments) -> bool:
        """
        Called when the add button is pressed
        :param event: The event
        :return: bool - True if the filter should be added, False otherwise
        """

        logger.debug(f"{self}.{self.on_add_button.__name__}({event}) called")
        return True

    async def _on_edit_button(self, event: UiEventArguments) -> None:
        """
        Internal called when the edit button is pressed
        :param event: The event
        :return: None
        """

        i = self._get_index(event)
        if not await self.on_edit_button(event, index=i):
            return
        self.set_visibility(index=i, visible=True)

    async def on_edit_button(self, event: UiEventArguments, index: int) -> bool:
        """
        Called when the edit button is pressed
        :param event: The event
        :param index: The index of the filter
        :return: bool - True if the filter should be edited, False otherwise
        """

        logger.debug(f"{self}.{self.on_edit_button.__name__}({event}, {index=}) called")
        return True

    async def _on_remove_button(self, event: UiEventArguments) -> None:
        """
        Internal called when the remove button is pressed
        :param event: The event
        :return: None
        """

        i = self._get_index(event)
        if not await self.on_remove_button(event, index=i):
            return
        self.delete_filter(i=i)
        if self._auto_reload:
            self._should_reload = event

    async def on_remove_button(self, event: UiEventArguments, index: int) -> bool:
        """
        Called when the remove button is pressed
        :param event: The event
        :param index: The index of the filter
        :return: bool - True if the filter should be removed, False otherwise
        """

        logger.debug(f"{self}.{self.on_remove_button.__name__}({event}, {index=}) called")
        return True

    async def _on_confirm_button(self, event: UiEventArguments) -> None:
        """
        Internal called when the confirm button is pressed
        :param event: The event
        :return: None
        """

        i = self._get_index(event)
        if not await self.on_confirm_button(event, index=i):
            return
        backup = self._current_filter[i]
        self.set_filter(index=i, key=self._edits[i][0]._value, operator=self._edits[i][1]._value, value=self._edits[i][2]._value)
        value = self._current_filter[i][2]
        result = await self.validate_filter(index=i, value=value)
        if type(result) is ValueError:
            # restore
            self._current_filter[i] = backup
        else:
            self.set_visibility(index=i, visible=False)
            self._set_read(i=i)

            if self._auto_reload:
                self._should_reload = event

    async def on_confirm_button(self, event: UiEventArguments, index: int) -> bool:
        """
        Called when the confirm button is pressed
        :param event: The event
        :param index: The index of the filter
        :return: bool - True if the filter should be confirmed, False otherwise
        """

        logger.debug(f"{self}.{self.on_confirm_button.__name__}({event}, {index=}) called")
        return True

    async def _on_cancel_button(self, event: UiEventArguments):
        """
        Called when the cancel button is pressed
        :param event: The event
        :return: None
        """

        i = self._get_index(event)
        if not await self.on_cancel_button(event, index=i):
            return
        c_f = self._current_filter[i]
        c_f_new = c_f[3]
        if type(c_f_new) is not bool:
            raise TypeError(f"Invalid type '{type(c_f_new)}'")
        if c_f_new:
            self.delete_filter(i=i)
        else:
            self.set_visibility(index=i, visible=False)
            self._set_read(i=i)

    async def on_cancel_button(self, event: UiEventArguments, index: int) -> bool:
        """
        Called when the cancel button is pressed
        :param event: The event
        :param index: The index of the filter
        :return: bool - True if the filter should be canceled, False otherwise
        """

        logger.debug(f"{self}.{self.on_cancel_button.__name__}({event}, {index=}) called")
        return True

    async def _on_reload_button(self, event: UiEventArguments) -> None:
        """
        Internal called when the reload button is pressed
        :param event: The event
        :return: None
        """

        if not await self.on_reload_button(event):
            return

        if self._on_reload_state != "ready":
            return
        self._should_reload = event

    async def on_reload_button(self, event: UiEventArguments) -> bool:
        """
        Called when the reload button is pressed
        :param event: The event
        :return: bool - True if the filter should be reloaded, False otherwise
        """

        logger.debug(f"{self}.{self.on_reload_button.__name__}({event}) called")
        return True

    async def _should_reload_timer_func(self) -> None:
        self._should_reload_last_tick = time.perf_counter()
        if self._should_reload is not None:
            await self._on_reload(self._should_reload)

    async def _on_reload(self, event: Union[UiEventArguments, str]):
        """
        Internal called when the filter should be reloaded
        :param event: The event
        :return: None
        """

        if self._should_reload is None:
            raise RuntimeError(f"Invalid call!")

        if self._on_reload_state == "ready":
            if not await self.on_reload(event):
                self._should_reload = None
            # refresh
            self.__call__.refresh()
            self._on_reload_state = "before"
        elif self._on_reload_state == "before":
            if self.on_reload_before is not None:
                if asyncio.iscoroutinefunction(self.on_reload_before):
                    logger.debug(f"{self}.{self._on_reload.__name__}({event}) calling {self.on_reload_before.__name__} as coroutine function")
                    await self.on_reload_before()
                else:
                    logger.debug(f"{self}.{self._on_reload.__name__}({event}) calling {self.on_reload_before.__name__} as function")
                    self.on_reload_before()
            self._on_reload_state = "reload"
        elif self._on_reload_state == "reload":
            self._models = self.get_models()

            # match
            await self._match()
            self._on_reload_state = "after"
        elif self._on_reload_state == "after":
            if self.on_reload_after is not None:
                if self.on_reload_after_mode == "pydantic":
                    attr = self.filtered_models
                elif self.on_reload_after_mode == "dict":
                    attr = self.filtered_models_dict
                elif self.on_reload_after_mode == "json":
                    attr = self.filtered_models_json
                elif self.on_reload_after_mode == "json_pretty":
                    attr = self.filtered_models_json_pretty
                else:
                    raise ValueError(f"Invalid value '{self.on_reload_after_mode}' for on_reload_external_mode, expected 'pydantic', 'dict', 'json' or 'json_pretty'")
                if asyncio.iscoroutinefunction(self.on_reload_after):
                    logger.debug(f"{self}.{self._on_reload.__name__}({event}) calling {self.on_reload_after.__name__} as coroutine function")
                    await self.on_reload_after(attr)
                else:
                    logger.debug(f"{self}.{self._on_reload.__name__}({event}) calling {self.on_reload_after.__name__} as function")
                    self.on_reload_after(attr)
            self._on_reload_state = "ready"
            self._should_reload = None

    async def on_reload(self, event: Union[UiEventArguments, str]) -> bool:
        """
        Called when the filter should be reloaded
        :param event: The event
        :return: bool - True if the filter should be reloaded, False otherwise
        """

        logger.debug(f"{self}.{self.on_reload.__name__}({event}) called")
        return True

    async def _on_key_changed(self, event: ValueChangeEventArguments) -> None:
        i = self._get_index(event)
        self._current_filter[i] = (await self.on_key_changed(event=event, index=i), self._current_filter[i][1], self._current_filter[i][2], self._current_filter[i][3])

        c_f = self._current_filter[i]
        value = self._current_filter[i][2]
        result = await self.validate_filter(index=i, value=value)
        if type(result) is ValueError:
            edit_input = self._edits[i][2]
            edit_input.props(f'error error-message="{result}"')
            return

    async def on_key_changed(self, event: ValueChangeEventArguments, index: int) -> str:
        key = event._value
        logger.debug(f"{self}.{self.on_key_changed.__name__}({event}, {index=}) -> {key=}")

        return key

    async def _on_operator_changed(self, event: ValueChangeEventArguments) -> None:
        i = self._get_index(event)
        self._current_filter[i] = (self._current_filter[i][0], await self.on_operator_changed(event=event, index=i), self._current_filter[i][2], self._current_filter[i][3])

        value = self._current_filter[i][2]
        result = await self.validate_filter(index=i, value=value)
        if type(result) is ValueError:
            edit_input = self._edits[i][2]
            edit_input.props(f'error error-message="{result}"')
            return

    async def on_operator_changed(self, event: ValueChangeEventArguments, index: int) -> str:
        operator = event._value
        logger.debug(f"{self}.{self.on_operator_changed.__name__}({event}, {index=}) -> {operator=}")

        return operator

    async def _on_value_changed(self, event: ValueChangeEventArguments) -> None:
        i = self._get_index(event)
        value = await self.on_value_changed(event=event, index=i)
        result = await self.validate_filter(index=i, value=value)
        if type(result) is ValueError:
            edit_input = self._edits[i][2]
            edit_input.props(f'error error-message="{result}"')
            return

    async def on_value_changed(self, event: ValueChangeEventArguments, index: int) -> str:
        value = event._value
        logger.debug(f"{self}.{self.on_value_changed.__name__}({event}, {index=}) -> {value=}")

        return value

    async def _match(self) -> None:
        if self.matching_disabled:
            return
        if await self.on_match():
            return
        logger.debug(f"{self}.{self._match.__name__}() matching")

        # match
        filtered_models = []
        for model in self._models:
            filter_out = False
            index = 0
            for c_filter in self._current_filter:
                operator = c_filter[1]
                value = c_filter[2]
                field_value = getattr(model, c_filter[0])
                if not await self.operator[operator](index, field_value, value):
                    filter_out = True
                    break
                index += 1
            if not filter_out:
                filtered_models.append(model)
        self._filtered_models = filtered_models

        return

    async def on_match(self) -> bool:
        """
        Called when the models are matched
        :return: bool - True if the models are matched, False otherwise
        """

        logger.debug(f"{self}.{self.on_match.__name__}() called")
        return False

    async def validate_filter(self, index: int, value: Any) -> Union[Any, ValueError]:
        """
        Validates the filter
        :param index: The index of the filter
        :param value: The value of the filter
        :return: Union[Any, ValueError] - The value of the filter if the validation was successful, ValueError otherwise
        """

        c_filter = self._current_filter[index]
        field = self.model_fields[c_filter[0]]
        field_value = c_filter[2]

        try:
            if not await self.on_validate(index=index, field=field, field_value=field_value, value=value):
                return value
            if field.annotation == str:
                value = str(value)
            elif field.annotation == int:
                for char in value:
                    if char not in "0123456789":
                        raise ValueError(f"Invalid integer '{char}'")
                value = int(value)
            elif field.annotation == float:
                for char in value:
                    if char not in "0123456789.":
                        raise ValueError(f"Invalid float '{char}'")
                value = float(value)
            elif field.annotation == bool:
                if value.lower() in ["true", "1"]:
                    value = True
                elif value.lower() in ["false", "0"]:
                    value = False
                elif value.lower() == "none":
                    value = None
                else:
                    raise ValueError(f"Invalid boolean '{value}'")
                value = bool(value)
            else:
                raise TypeError(f"Invalid type '{field.annotation}'")
        except ValueError as e:
            await self.on_validate_failure(index=index, field=field, field_value=field_value, value=value, error=e)
            return ValueError(e)
        logger.debug(f"{self}.{self.validate_filter.__name__}(index={index}, field={field}, field_value={field_value}, value={value}) -> {value=}")
        return value

    async def on_validate(self, index: int, field: FieldInfo, field_value: Any, value: Any) -> bool:
        """
        Validates the filter

        :param index: The index of the filter
        :param field: The field of the filter
        :param field_value: The value of the field
        :param value: The value of the filter
        :return: bool - True if the filter should be validated, False otherwise
        """

        logger.debug(f"{self}.{self.on_validate.__name__}({index=}, {field=}, {field_value=}, {value=}) called")
        return True

    async def on_validate_failure(self, index: int, field: FieldInfo, field_value: Any, value: Any, error: ValueError) -> None:
        """
        Called when the validation of the filter fails

        :param index: The index of the filter
        :param field: The field of the filter
        :param field_value: The value of the field
        :param value: The value of the filter
        :param error: The error that occurred
        :return:
        """

        logger.debug(f"{self}.{self.on_validate_failure.__name__}(index={index}, field={field}, field_value={field_value}, value={value}, error={error}) called")

    def get_models(self) -> list[BaseModel]:
        """
        Calling this function returns the models of the filter
        :return: list[BaseModel] - The models of the filter
        """

        models = self._get_models()

        logger.debug(f"Getting {self}.{self.get_models.__name__}()")
        if type(models) is list:
            _models: list[BaseModel] = []
            for model in models:
                if not isinstance(model, BaseModel):
                    raise TypeError(f"Invalid type '{type(model)}'")
                _models.append(model)
            return _models
        else:
            if not isinstance(models, BaseModel):
                raise TypeError(f"Invalid type '{type(models)}'")
            return [models]

    def set_visibility(self, index: int, visible: bool) -> None:
        """
        Sets the visibility of the filter
        :param index: The index of the filter
        :param visible: True if the filter should be visible, False otherwise
        :return: None
        :return:
        """

        logger.debug(f"Setting {self}.{self.set_visibility.__name__}({index=}, {visible=})")
        for r in self._reads[index]:
            r.set_visibility(not visible)
        for e in self._edits[index]:
            e.set_visibility(visible)
        self._set_edit(i=index)

    def add_filter(self, key: str = "", operator: str = "", value: str = "") -> None:
        """
        Adds a filter
        :param key: The key of the filter
        :param operator: The operator of the filter
        :param value: The value of the filter
        :return: None
        """

        if key == "":
            key = list(self.model_fields.keys())[0]
        if operator == "":
            operator = list(self.operator.keys())[0]
        if value == "":
            value = ""

        logger.debug(f"Adding {self}.{self.add_filter.__name__}({key=}, {operator=}, {value=})")

        self._current_filter.append((key, operator, value, True))
        self.__call__.refresh()

    def set_filter(self, index: int, key: str = "", operator: str = "", value: str = "") -> None:
        """
        Sets the filter
        :param index: The index of the filter
        :param key: The key of the filter
        :param operator: The operator of the filter
        :param value: The value of the filter
        :return: None
        """

        # check if index is valid
        if index < 0 or index >= len(self._current_filter):
            raise IndexError(f"Invalid index '{index}'")

        if key == "":
            key = list(self.model_fields.keys())[0]
        if operator == "":
            operator = list(self.operator.keys())[0]
        if value == "":
            value = ""

        logger.debug(f"Setting {self}.{self.set_filter.__name__}({index=}, {key=}, {operator=}, {value=})")

        self._current_filter[index] = (key, operator, value, False)

    def delete_filter(self, i: int) -> None:
        """
        Deletes the filter
        :param i: The index of the filter
        :return: None
        """

        logger.debug(f"Deleting {self}.{self.delete_filter.__name__}({i=})")
        for r in self.reads[i]:
            r.set_visibility(False)
        for e in self.edits[i]:
            e.set_visibility(False)
        self.reads.pop(i)
        self.edits.pop(i)
        self._current_filter.pop(i)

    def reload(self, force: bool = False) -> None:
        """
        Reloads the filter

        :param force: True if the filter should be reloaded even if it is not ready, False otherwise
        :return: None
        """

        logger.debug(f"Reloading {self}.{self.reload.__name__}()")

        if force:
            self._should_reload = "reload"
        else:
            if self._on_reload_state == "ready":
                self._should_reload = "reload"
            else:
                logger.warning(f"Invalid state '{self._on_reload_state}'. Try again later or call {self}.{self._on_reload.__name__}(force=True)")

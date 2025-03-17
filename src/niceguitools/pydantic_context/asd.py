import logging
from copy import deepcopy
from typing import Any, ClassVar, Callable, Literal, Union, Optional, Type, TypeVar, Generic

from pydantic import BaseModel, Field, create_model

from pydantic.fields import FieldInfo as FieldInfo

logger = logging.getLogger(__name__)

CONTEXT_BASE_MODEL_METHODS = ["_get_context_model", "context_delta", "to_context_base"]

# raise error if BaseModel has a method name from CONTEXT_BASE_MODEL_METHODS
for method_name in CONTEXT_BASE_MODEL_METHODS:
    if hasattr(BaseModel, method_name):
        raise ValueError(f"BaseModel has a method '{method_name}'")


class ContextBaseModel(BaseModel):

    def _get_context_model(self,
                           context: str = "",
                           model_name: str = None,
                           model_kwargs: dict = None) -> Type["ContextModel"]:
        # get context info
        _context_info = self.__context_info__

        # get context
        context = _context_info.context(context)

        # get model
        if model_kwargs is None:
            model_kwargs = {}
        model: Type["ContextModel"] = context.model(model_name=model_name, **model_kwargs)

        return model

    def context_delta(self,
                      context: str = "",
                      model_name: str = None,
                      model_kwargs: dict = None) -> dict[str, Any]:
        _context_model = self._get_context_model(context=context, model_name=model_name, model_kwargs=model_kwargs)

        # get context dict
        _context_model_dict = _context_model.from_context_base(context_base_dict=self, create=False)

        # get self dict
        self_dict = self.model_dump()

        # get delta
        delta = {}
        for key, value in self_dict.items():
            if key not in _context_model_dict:
                delta[key] = value
            elif value != _context_model_dict[key]:
                delta[key] = value

        return delta

    def to_context_base(self,
                        context: str = "",
                        model_name: str = None,
                        model_kwargs: dict = None,
                        context_base_dict: Union[dict[str, Any], "ContextModelType"] = None,
                        create: bool = True) -> Union["ContextModel", dict[str, Any]]:
        _context_model = self._get_context_model(context=context, model_name=model_name, model_kwargs=model_kwargs)

        if context_base_dict is None:
            context_base_dict = self
        out = _context_model.from_context_base(context_base_dict=context_base_dict, create=create)

        return out


ContextModelType = TypeVar("ContextModelType", bound=Union[BaseModel, Any])
ContextBaseModelType = TypeVar("ContextBaseModelType", bound=Union[ContextBaseModel])


class _BaseProperty(Generic[ContextModelType]):
    def __init__(self, value: Any = None, factory: Callable[[str, Any, Any], Any] = None):
        self.value: Any = value

        # check if factory is callable
        if factory is not None and not callable(factory):
            raise ValueError("Factory must be callable")
        self.factory: Callable[[str, Any, Any], Any] = factory

        # check if value or value_factory is set
        if self.value is None and self.factory is None:
            raise ValueError("Value or factory must be set")

    def __repr__(self):
        return f"{self.__class__.__name__}(value={self.value}, factory={self.factory})"

    def __str__(self):
        value = self.value
        if value is None:
            value = self.factory
        return f"{self.__class__.__name__}(value={value}, type={type(self.value)})"

    def build(self, name: str, context_info_or_context: Union["ContextInfo", "Context"]) -> Any:
        # make a copy of self
        self_copy = deepcopy(self)

        if self_copy.factory is None:
            result = self_copy.value
        else:
            result = self_copy.factory(name, self_copy.value, context_info_or_context)

        # warn if result is None
        if result is None:
            logger.warning(f"Property {name}({self_copy}) returned None")

        self_copy.value = result

        return self_copy


class ContextInfoProperty(_BaseProperty):
    def __init__(self, value: Any = None, factory: Callable[[str, Any, "ContextInfo"], Any] = None):
        super().__init__(value=value, factory=factory)

        self.factory: Callable[[str, Any, "ContextInfo"], Any] = factory


class ContextProperty(_BaseProperty):
    def __init__(self, value: Any = None, factory: Callable[[str, Any, "Context"], Any] = None, context: str | list[str] = ""):
        super().__init__(value=value, factory=factory)

        self.factory: Callable[[str, Any, "Context"], Any] = factory

        if type(context) is not list:
            context = [context]
        for context_item in context:
            if type(context_item) is not str:
                raise ValueError(f"Context '{context_item}' must be a string")
        self.context: list[str] = context

    def __repr__(self):
        repr_str = super().__repr__()
        return f"{repr_str[:-1]}, context={self.context})"

    def __str__(self):
        str_str = super().__str__()
        return f"{str_str[:-1]}, context={self.context})"


class ContextField(Generic[ContextModelType]):
    def __init__(self,
                 field_factories: Callable[[str, FieldInfo, "Context"], FieldInfo] | list[Callable[[str, FieldInfo, "Context"], FieldInfo]] = None,
                 name: Optional[str] = None,
                 alias: Optional[str] = None,
                 map_to: Optional[str] = None,
                 mapping_priority: Optional[int] = None,
                 context: str | list[str] = ""):
        if type(field_factories) is not list:
            field_factories = [field_factories]
        self.field_factories: list[Callable[[str, FieldInfo, Context], FieldInfo]] = field_factories

        self._name: Optional[str] = name
        self._alias: Optional[str] = alias
        self._map_to: Optional[str] = map_to
        self.mapping_priority: Optional[int] = mapping_priority

        if type(context) is not list:
            context = [context]
        for context_item in context:
            if type(context_item) is not str:
                raise ValueError(f"Context '{context_item}' must be a string")
        self.context: list[str] = context

        self._field_info: FieldInfo | None = None

    def __repr__(self):
        return f"{self.__class__.__name__}(field_factories={self.field_factories}, context={self.context})"

    def __str__(self):
        if self._field_info is None:
            return f"{self.__class__.__name__}(context={self.context})"
        return f"{self.__class__.__name__}(context={self.context}), field_info={self.field_info})"

    @property
    def name(self):
        if self._name is None:
            raise ValueError("Name is not set")
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def alias(self) -> str:
        if self._alias is None:
            return self.name
        return self._alias

    @alias.setter
    def alias(self, value: str):
        self._alias = value

    @property
    def map_to(self) -> str:
        if self._map_to is None:
            return self.name
        return self._map_to

    @map_to.setter
    def map_to(self, value: str):
        self._map_to = value

    @property
    def field_info(self) -> FieldInfo:
        if self._field_info is None:
            raise ValueError("Field info is not set")
        return self._field_info

    @field_info.setter
    def field_info(self, value: FieldInfo):
        self._field_info = value

    def build(self, field_name: str, context: "Context") -> "ContextField":
        # make a copy of self
        self_copy = deepcopy(self)

        # set name
        if self_copy._name is None:
            self_copy._name = field_name

        # build field info
        result = self_copy.field_info
        for factory in self_copy.field_factories:
            result = factory(field_name, result, context)

            # raise error if result is not a FieldInfo
            if not isinstance(result, FieldInfo):
                raise ValueError(f"Field factory '{factory}' must return a FieldInfo")

        self_copy.field_info = result

        return self_copy


class _BaseFactory(Generic[ContextModelType]):
    def __init__(self,
                 factory: Callable[[Any], Any],
                 mode: Literal["before", "after"] = "before"):
        if not callable(factory):
            raise ValueError("Factory must be callable")
        self.factory: Callable[[Any], Any] = factory

        self.mode: Literal["before", "after"] = mode

    def __repr__(self):
        return f"{self.__class__.__name__}(factory={self.factory}, mode={self.mode})"

    def __str__(self):
        return f"{self.__class__.__name__}(factory={self.factory}, mode={self.mode})"


class ContextFactory(_BaseFactory):
    def __init__(self,
                 factory: Callable[[str, "Context"], "Context"],
                 mode: Literal["before", "after"] = "before",
                 context: str | list[str] = ""):
        super().__init__(factory=factory, mode=mode)

        self.factory: Callable[[str, "Context"], "Context"] = factory

        if type(context) is not list:
            context = [context]
        for context_item in context:
            if type(context_item) is not str:
                raise ValueError(f"Context '{context_item}' must be a string")
        self.context: list[str] = context

    def __repr__(self):
        repr_str = super().__repr__()
        return f"{repr_str[:-1]}, context={self.context})"

    def __str__(self):
        str_str = super().__str__()
        return f"{str_str[:-1]}, context={self.context})"


class ContextModelFactory(_BaseFactory):
    def __init__(self,
                 factory: Callable[["ContextInfo"], "ContextInfo"] = None,
                 mode: Literal["before", "after"] = "before"):
        super().__init__(factory=factory, mode=mode)

        self.factory: Callable[["ContextInfo"], "ContextInfo"] = factory


class ContextModel(BaseModel, Generic[ContextModelType]):
    @classmethod
    def from_context_base(cls, context_base_dict: dict[str, Any] | ContextModelType, create: bool = True) -> Union["ContextModel", dict[str, Any]]:
        _mapping = cls.__context__.mapping(direction="from_context_base")

        if _is_pydantic_model(context_base_dict):
            context_base_dict = context_base_dict.model_dump()

        # create new dict
        new_dict = {}
        for mapping_key, mapping_value in _mapping.items():
            if mapping_value not in context_base_dict:
                raise ValueError(f"Mapping value '{mapping_value}' not found in context base dict")
            new_dict[mapping_key] = context_base_dict[mapping_value]

        if not create:
            return new_dict

        # create new model
        new_model = cls(**new_dict)

        return new_model

    def to_context_base(self, model_kwargs: dict[str, Any] = None, create: bool = True) -> Union[ContextModelType | dict[str, Any]]:
        if model_kwargs is None:
            model_kwargs = {}

        _context_info: ContextInfo = self.__context_base_model__.__context_info__
        _mapping = self.__context__.mapping(direction="to_context_base")

        # create new dict
        new_dict = {}
        for mapping_key, mapping_value in _mapping.items():
            if mapping_value not in self.__dict__:
                raise ValueError(f"Mapping value '{mapping_value}' not found in model")
            new_dict[mapping_key] = self.__dict__[mapping_value]

        # add model_kwargs
        for model_kwarg_name, model_kwarg_value in model_kwargs.items():
            new_dict[model_kwarg_name] = model_kwarg_value

        if not create:
            return new_dict

        # create new model
        new_model = _context_info.model(**new_dict)

        return new_model


class Context(Generic[ContextModelType]):
    def __init__(self,
                 properties: dict[str, ContextProperty] = None,
                 fields: dict[str, dict[str, ContextField]] = None,
                 before_factories: list[Callable[[str, "Context"], "Context"]] = None,
                 after_factories: list[Callable[[str, "Context"], "Context"]] = None):

        if properties is None:
            properties = {}
        self.properties: dict[str, ContextProperty] = properties

        if fields is None:
            fields = {}
        self.fields: dict[str, dict[str, ContextField]] = fields

        if before_factories is None:
            before_factories = []
        self.before_factories: list[Callable[[str, "Context"], "Context"]] = before_factories

        if after_factories is None:
            after_factories = []
        self.after_factories: list[Callable[[str, "Context"], "Context"]] = after_factories

        self._current_context_name: Optional[str] = None
        self._current_context_info: Optional["ContextInfo"] = None

    def __repr__(self):
        return f"{self.__class__.__name__}(properties={self.properties}, fields={self.fields})"

    def __str__(self):
        return f"{self.__class__.__name__}(properties={self.properties}, fields={self.fields})"

    def __enter__(self) -> "Context":
        if self._current_context_name is None:
            raise ValueError("First select a context with context()")
        if self._current_context_info is None:
            raise ValueError("First select a context with context()")

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @property
    def current_context_name(self) -> str:
        if self._current_context_name is None:
            raise ValueError("Current context name is not set")
        return self._current_context_name

    @property
    def current_context_info(self) -> "ContextInfo":
        if self._current_context_info is None:
            raise ValueError("Current context model is not set")
        return self._current_context_info

    def set_current_ctx(self, context_name: str, ci: "ContextInfo"):
        self._current_context_name = context_name
        self._current_context_info = ci

    def build(self, context_name: str = "", ci: "ContextInfo" = None) -> "Context":
        # make a copy of self
        self_copy = deepcopy(self)

        # set current context
        if self_copy._current_context_name is None and self_copy._current_context_info is None:
            self_copy.set_current_ctx(context_name, ci)

        # before factories
        for factory in self_copy.before_factories:
            before_result = factory(context_name, self_copy)
            if not isinstance(before_result, Context):
                raise ValueError(f"Context factory '{factory}' must return a Context")
            self_copy = before_result

        # build properties
        for prop_name, prop in self_copy.properties.items():
            prop_result = prop.build(prop_name, self_copy)
            if not isinstance(prop_result, ContextProperty):
                raise ValueError(f"Context property '{prop}' must return a ContextProperty")
            self_copy.properties[prop_name] = prop_result

        # build fields
        new_fields: dict[str, dict[str, ContextField]] = {}
        for field_name, field in self_copy.fields.items():
            # check if field is already in new_fields
            if field_name not in new_fields:
                new_fields[field_name] = {}

            for random_alias, f in field.items():
                f_result = f.build(field_name, self_copy)
                if not isinstance(f_result, ContextField):
                    raise ValueError(f"Context field '{field}' must return a ContextField")

                # check if alias is already in use
                if f.alias in new_fields[field_name]:
                    raise ValueError(f"Alias '{f.alias}' of field '{field_name}' is already in use")

                # set alias
                new_fields[field_name][f.alias] = f_result

        self_copy.fields = new_fields

        # after factories
        for factory in self_copy.after_factories:
            after_result = factory(context_name, self_copy)
            if not isinstance(after_result, Context):
                raise ValueError(f"Context factory '{factory}' must return a Context")
            self_copy = after_result

        return self_copy

    def model(self, model_name: Optional[str] = None, **kwargs) -> Type[ContextModel[ContextModelType]]:
        model = self.current_context_info.model

        # get context name
        context_name = self.current_context_name
        if context_name == "":
            context_name = "Default"

        # get name
        if model_name is None:
            model_name = model.__name__ + "_" + context_name + "_ContextModel"

        # create model_copy type
        model_fields = {}
        for field_name, field in model.model_fields.items():
            # check if field is in fields
            if field_name not in self.fields:
                continue

            for alias, f in self.fields[field_name].items():
                # check if alias is in fields
                if alias not in self.fields[field_name]:
                    continue

                # add alias to model_fields - format: (<type>, <FieldInfo>)
                model_fields[alias] = (f.field_info.annotation, f.field_info)

        # add ContextModel to __base__
        if "__base__" not in kwargs:
            kwargs["__base__"] = ContextModel
        else:
            kwargs["__base__"] = (ContextModel, *kwargs["__base__"])

        new_model = create_model(model_name, **model_fields, **kwargs)

        # set context
        new_model.__context__ = self

        # set context base model
        new_model.__context_base_model__ = model

        return new_model

    def mapping(self, direction: Literal["to_context_base", "from_context_base"] = "to_context_base") -> dict[str, str]:
        mapping: dict[str, str] = {}
        if direction == "from_context_base":
            for field_name, field in self.fields.items():
                for alias, f in field.items():
                    # check if alias is already in mapping
                    if alias in mapping:
                        raise ValueError(f"Alias '{alias}' of field '{field_name}' is already in mapping")

                    # mapping
                    mapping[alias] = f.map_to
        elif direction == "to_context_base":
            priority_mapping: dict[str, int] = {}
            for field_name, field in self.fields.items():
                for alias, f in field.items():
                    # check mapping_priority
                    if f.mapping_priority is None:
                        raise ValueError(f"Field '{field_name}' has no mapping_priority")

                    # check if map_to is already in mapping
                    if f.map_to in mapping:
                        # check if mapping_priority is higher
                        if priority_mapping[f.map_to] < f.mapping_priority:
                            continue
                        elif priority_mapping[f.map_to] == f.mapping_priority:
                            raise ValueError(f"Field '{field_name}' has the same mapping_priority as '{mapping[f.map_to]}'")

                        # remove old mapping
                        del mapping[f.map_to]

                    # mapping
                    mapping[f.map_to] = alias
                    priority_mapping[f.map_to] = f.mapping_priority
        else:
            raise ValueError(f"Unknown direction '{direction}'")

        return mapping


class ContextInfo(Generic[ContextModelType]):
    def __init__(self,
                 properties: dict[str, ContextInfoProperty] = None,
                 contexts: dict[str, Context] = None,
                 before_factories: list[Callable[["ContextInfo"], "ContextInfo"]] = None,
                 after_factories: list[Callable[["ContextInfo"], "ContextInfo"]] = None):
        if properties is None:
            properties = {}
        self.properties: dict[str, ContextInfoProperty] = properties

        if contexts is None:
            contexts = {}
        self.contexts: dict[str, Context] = contexts

        if before_factories is None:
            before_factories = []
        self.before_factories: list[Callable[["ContextInfo"], "ContextInfo"]] = before_factories

        if after_factories is None:
            after_factories = []
        self.after_factories: list[Callable[["ContextInfo"], "ContextInfo"]] = after_factories

        self.model = None

    def __repr__(self):
        return f"{self.__class__.__name__}(contexts={self.contexts})"

    def __str__(self):
        return f"{self.__class__.__name__}(contexts={self.contexts})"

    def __add__(self, other: "ContextInfo") -> "ContextInfo":
        # add properties
        for prop_name, prop in self.properties.items():
            self.properties[prop_name] = prop

        # add contexts
        for context_name, context in other.contexts.items():
            if context_name not in self.contexts:
                self.contexts[context_name] = context
            else:
                # add properties
                for prop_name, prop in context.properties.items():
                    self.contexts[context_name].properties[prop_name] = prop

                # add fields
                for field_name, field in context.fields.items():
                    self.contexts[context_name].fields[field_name] = field

                # add before factories
                for factory in context.before_factories:
                    self.contexts[context_name].before_factories.append(factory)

                # add after factories
                for factory in context.after_factories:
                    self.contexts[context_name].after_factories.append(factory)

        # add before factories
        for factory in other.before_factories:
            self.before_factories.append(factory)

        # add after factories
        for factory in other.after_factories:
            self.after_factories.append(factory)

        return self

    def context(self, context_name: str = "") -> "Context":
        """
        Select a context.

        :param context_name:
        :return: Selected context.
        """

        context = self.contexts.get(context_name, None)
        if context is None:
            raise ValueError(f"Context '{context_name}' not found")

        # set current context
        context.set_current_ctx(context_name, self)

        return context

    def build(self) -> "ContextInfo":
        # make a copy of self
        self_copy = deepcopy(self)

        # before factories
        for factory in self_copy.before_factories:
            before_result = factory(self_copy)
            if not isinstance(before_result, ContextInfo):
                raise ValueError(f"Context model factory '{factory}' must return a ContextModel")
            self_copy = before_result

        # build properties
        for prop_name, prop in self_copy.properties.items():
            prop_result = prop.build(prop_name, self_copy)
            if not isinstance(prop_result, ContextInfoProperty):
                raise ValueError(f"Context model property '{prop}' must return a ContextModelProperty")
            self_copy.properties[prop_name] = prop_result

        # build contexts
        for context_name, context in self_copy.contexts.items():
            context_result = context.build(context_name, self_copy)
            if not isinstance(context_result, Context):
                raise ValueError(f"Context '{context}' must return a Context")
            self_copy.contexts[context_name] = context_result

        # after factories
        for factory in self_copy.after_factories:
            after_result = factory(self_copy)
            if not isinstance(after_result, ContextInfo):
                raise ValueError(f"Context model factory '{factory}' must return a ContextModel")
            self_copy = after_result

        return self_copy


def _is_pydantic_model(cls: BaseModel) -> bool:
    if not hasattr(cls, "model_fields"):
        return False
    return True


def _lookup_model(cls: BaseModel) -> ContextInfo:
    ctx_model = ContextInfo()

    # get context model properties
    for prop_name, prop in cls.__dict__.items():
        if not isinstance(prop, ContextInfoProperty):
            continue

        # add property to ctx_model
        ctx_model.properties[prop_name] = prop

    # get context model factories
    for prop_name, prop in cls.__dict__.items():
        if not isinstance(prop, ContextModelFactory):
            continue

        # add factory to ctx_model
        if prop.mode == "before":
            ctx_model.before_factories.append(prop.factory)
        elif prop.mode == "after":
            ctx_model.after_factories.append(prop.factory)
        else:
            raise ValueError(f"Unknown mode '{prop.mode}'")

    # get context factories
    for prop_name, prop in cls.__dict__.items():
        if not isinstance(prop, ContextFactory):
            continue

        # check if context is already in ctx_model
        for context_item in prop.context:
            if context_item not in ctx_model.contexts:
                ctx_model.contexts[context_item] = Context()

        # add factory to ctx_model
        if prop.mode == "before":
            for context_item in prop.context:
                ctx_model.contexts[context_item].before_factories.append(prop.factory)
        elif prop.mode == "after":
            for context_item in prop.context:
                ctx_model.contexts[context_item].after_factories.append(prop.factory)
        else:
            raise ValueError(f"Unknown mode '{prop.mode}'")

    # get context properties
    for prop_name, prop in cls.__dict__.items():
        if not isinstance(prop, ContextProperty):
            continue

        # check if context is already in ctx_model
        for context_item in prop.context:
            if context_item not in ctx_model.contexts:
                ctx_model.contexts[context_item] = Context()

        # add property to ctx_model
        for context_item in prop.context:
            ctx_model.contexts[context_item].properties[prop_name] = prop

    # get context fields
    for field_name, field_info in cls.model_fields.items():
        if field_info.json_schema_extra is None:
            continue
        context_fields: list[ContextField] = field_info.json_schema_extra.get("context_fields", [ContextField()])

        # check if context is a list
        if type(context_fields) is not list:
            if isinstance(context_fields, ContextField):
                context_fields = [context_fields]
            else:
                raise ValueError(f"Context fields of field '{field_name}' in model '{cls}' is not a ContextField or list")
        else:
            # check if context is a list of strings
            for context_fields_item in context_fields:
                if type(context_fields_item) is not ContextField:
                    raise ValueError(f"Context fields of field '{field_name}' in model '{cls}' is not a ContextField or list of ContextFields")

        # check if context is already in ctx_model
        for context_fields_item in context_fields:
            for context_item in context_fields_item.context:
                if context_item not in ctx_model.contexts:
                    ctx_model.contexts[context_item] = Context()

        # add field_info to ctx_model
        mapping_priorities = []  # save mapping priorities, to check if a mapping priority is already in use
        last_auto_mapping_priority = 0  # save last auto mapping priority
        for context_fields_item in context_fields:
            # check if priority is set
            if context_fields_item.mapping_priority is None:
                context_fields_item.mapping_priority = last_auto_mapping_priority + 1
                last_auto_mapping_priority = context_fields_item.mapping_priority

            # add field_info to context_fields_item
            context_fields_item.field_info = field_info

            # set field name
            context_fields_item.name = field_name

            for context_item in context_fields_item.context:
                if field_name not in ctx_model.contexts[context_item].fields:
                    ctx_model.contexts[context_item].fields[field_name] = {}

                # check if priority is already in use
                if context_fields_item.mapping_priority in mapping_priorities:
                    raise ValueError(f"Mapping priority '{context_fields_item.mapping_priority}' of field '{field_name}' in model '{cls}' is already in use")

                # take field name + priority as alias
                alias = field_name + "_" + str(context_fields_item.mapping_priority)

                # check if alias is already in ctx_model
                if alias in ctx_model.contexts[context_item].fields[field_name]:
                    raise ValueError(f"Alias '{alias}' of field '{field_name}' in model '{cls}' is already in use")

                # save mapping priority
                mapping_priorities.append(context_fields_item.mapping_priority)

                ctx_model.contexts[context_item].fields[field_name][alias] = context_fields_item

    return deepcopy(ctx_model)


def context_info(model: ContextModelType | None = None) -> Callable[[ContextModelType], ContextBaseModelType] | ContextInfo:
    def decorator(cls: ContextModelType) -> ContextBaseModelType:
        if not _is_pydantic_model(cls):
            raise ValueError(f"'{cls}' is not a pydantic model")

        lookup_models: list[BaseModel] = [cls]

        # looking for base classes
        for base_cls in cls.__bases__:
            if _is_pydantic_model(base_cls):
                lookup_models.insert(0, base_cls)

        # lookup context models
        build_ctx_model = ContextInfo()
        for lookup_model in lookup_models:
            ctx_model = _lookup_model(lookup_model)
            build_ctx_model += ctx_model.build()

        # set model
        build_ctx_model.model = cls

        # set context model
        cls.__context_info__ = build_ctx_model

        def set_method(method_name: str):
            method = getattr(ContextBaseModel, method_name)
            setattr(cls, method_name, method)
            setattr(getattr(cls, method_name), "__set_on_decorator__", True)
            return method

        cls._get_context_model = set_method("_get_context_model")
        cls.context_delta = set_method("context_delta")
        cls.to_context_base = set_method("to_context_base")

        return cls

    # check if decorator is called with arguments
    if model is None:
        return decorator
    else:
        if not hasattr(model, "__context_info__"):
            raise ValueError(f"'{model}' is not a context model")
        _context_info = model.__context_info__
        return _context_info


def _context_info_factory_before(ctx_model: ContextInfo) -> ContextInfo:
    # add context model propertiy
    ctx_model.properties["context_info_factory_var"] = ContextInfoProperty("Test123")
    return ctx_model


def _my_var1_factory(name: str, value: Any, ctx_model: ContextInfo) -> Any:
    return value + name


def _context_factory_before(context_name: str, context: Context) -> Context:
    # add context propertiy
    context.properties["my_context_var"] = ContextProperty("Test123Context1")
    return context


def _context_field_factory(field_name: str, field: FieldInfo, context: Context) -> FieldInfo:
    # change field description
    field.description = field.description + "123" + context.current_context_name + field_name
    return field


def _context_factory_after(context_name: str, context: Context) -> Context:
    # change context property
    context.properties["my_context_var"] = ContextProperty("Test456Context1")
    return context


def _context1_factory_before(context_name: str, context: Context) -> Context:
    # add context propertiy
    context.properties["my_context_var_context1"] = ContextProperty("Test123Context2")
    return context


def _context1_field_factory(field_name: str, field: FieldInfo, context: Context) -> FieldInfo:
    # change field description
    field.description = field.description + "456" + context.current_context_name + field_name
    return field


def _context1_factory_after(context_name: str, context: Context) -> Context:
    # change context property
    context.properties["my_context_var_context1"] = ContextProperty("Test456Context2")
    return context


def _context2_factory_before(context_name: str, context: Context) -> Context:
    # add context propertiy
    context.properties["my_context_var_context2"] = ContextProperty("Test123Context3")
    return context


def _context2_field_factory(field_name: str, field: FieldInfo, context: Context) -> FieldInfo:
    # change field description
    field.description = field.description + "789" + context.current_context_name + field_name
    return field


def _context2_factory_after(context_name: str, context: Context) -> Context:
    # change context property
    context.properties["my_context_var_context2"] = ContextProperty("Test456Context3")
    return context


def _context_info_factory_after(ctx_model: ContextInfo) -> ContextInfo:
    # change context model propertiy
    ctx_model.properties["context_info_factory_var"] = ContextInfoProperty("Test456")
    return ctx_model


@context_info()
class Model(BaseModel):
    my_var1: ClassVar = ContextInfoProperty("Test123", factory=_my_var1_factory)
    context_info_factory_before: ClassVar = ContextModelFactory(factory=_context_info_factory_before, mode="before")
    context_info_factory_after: ClassVar = ContextModelFactory(factory=_context_info_factory_after, mode="after")

    context_factory_before: ClassVar = ContextFactory(factory=_context_factory_before, mode="before")
    context_factory_after: ClassVar = ContextFactory(factory=_context_factory_after, mode="after")
    my_context_var1: ClassVar = ContextProperty("my_context_var")
    test_str: str = Field(..., description="Test string 1", context_fields=[ContextField(field_factories=_context_field_factory),
                                                                            ContextField(field_factories=_context_field_factory,
                                                                                         alias="test_str_alias1")])
    test_int: int = Field(..., description="Test integer 1", context_fields=ContextField(field_factories=_context_field_factory))
    test_float: float = Field(..., description="Test float 1", context_fields=ContextField(field_factories=_context_field_factory))
    test_bool: bool = Field(..., description="Test boolean 1", context_fields=ContextField(field_factories=_context_field_factory))

    my_context_var1_context1: ClassVar = ContextProperty("my_context_var_context1", context="context1")
    context1_factory_before: ClassVar = ContextFactory(factory=_context1_factory_before, mode="before", context="context1")
    context1_factory_after: ClassVar = ContextFactory(factory=_context1_factory_after, mode="after", context="context1")
    test1_str_context1: str = Field(..., description="Test string 2", context_fields=ContextField(context="context1", field_factories=_context1_field_factory))
    test1_int_context1: int = Field(..., description="Test integer 2", context_fields=ContextField(context="context1", field_factories=_context1_field_factory))
    test1_float_context1: float = Field(..., description="Test float 2", context_fields=ContextField(context="context1", field_factories=_context1_field_factory))
    test1_bool_context1: bool = Field(..., description="Test boolean 2", context_fields=ContextField(context="context1", field_factories=_context1_field_factory))

    @classmethod
    def random(cls):
        data = {
            "test_str": "test",
            "test_int": 1,
            "test_float": 1.0,
            "test_bool": True,
            "test1_str_context1": "test",
            "test1_int_context1": 1,
            "test1_float_context1": 1.0,
            "test1_bool_context1": True
        }
        return cls(**data)


@context_info()
class Model2(Model):
    ...


with context_info(Model2).context() as ctx:
    model2 = Model2.random()
    m = ctx.model().from_context_base(model2)
    m_base = m.to_context_base(model2.context_delta())
    print()
    for a in ctx.properties:
        print(a)

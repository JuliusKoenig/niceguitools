import logging
from typing import Optional, Union, Type
from nicegui import ui
from nicegui.elements.mixins.visibility import Visibility
from nicegui.elements.spinner import SpinnerTypes

logger = logging.getLogger(__name__)

SpinnerTypes_list = eval(str(SpinnerTypes).replace("typing.Literal", "Literal").replace("Literal", ""))


class AsyncLoader:
    class LoadingSpinner(ui.spinner):
        def __init__(self,
                     instance: "AsyncLoader",
                     t: Optional[SpinnerTypes] = None,
                     *,
                     size: Optional[str] = None,
                     color: Optional[str] = None,
                     thickness: Optional[float] = None) -> None:
            """
            Creates a new LoadingSpinner

            :param instance: An instance of the AsyncLoader
            :param t: SpinnerType - Type of the spinner - Default: 'default'
            :param size: str - Size of the spinner - Default: '1em'
            :param color: str - Color of the spinner - Default: 'primary'
            :param thickness: float - Thickness of the spinner - Default: 5.0
            """

            if not isinstance(instance, AsyncLoader):
                raise TypeError(f"Invalid type '{type(instance)}'")
            self._instance: AsyncLoader = instance

            if t is None:
                t = SpinnerTypes_list[0]
            if t not in SpinnerTypes_list:
                raise ValueError(f"Invalid value '{t}' for t, expected {SpinnerTypes_list}")

            if size is None:
                size = 'xl'
            if not isinstance(size, str):
                raise TypeError(f"Invalid type '{type(size)}'")

            if color is None:
                color = 'primary'
            if not isinstance(color, str):
                raise TypeError(f"Invalid type '{type(color)}'")

            if thickness is None:
                thickness = 5.0
            if not isinstance(thickness, float):
                raise TypeError(f"Invalid type '{type(thickness)}'")

            super().__init__(type=t, size=size, color=color, thickness=thickness)

        @property
        def instance(self) -> "AsyncLoader":
            return self._instance

    def __init__(self,
                 name: Optional[str] = None,
                 *,
                 visibility: Optional[Type[Visibility]] = None,
                 direct_call: bool = True,
                 spinner_type: Optional[SpinnerTypes] = None,
                 spinner_size: Optional[str] = None,
                 spinner_color: Optional[str] = None,
                 spinner_thickness: Optional[float] = None,
                 **kwargs) -> None:
        """
        Creates a new AsyncLoader

        :param name: The name of the AsyncLoader
        :param visibility: The visibility of the AsyncLoader
        :param direct_call: True if the AsyncLoader should be called directly, False otherwise
        :param kwargs: The kwargs of for the visibility
        :return: AsyncLoader
        """

        if name is None:
            name = self.__class__.__name__
        self._name: str = name
        if type(self._name) is not str:
            raise TypeError(f"Invalid type '{type(self._name)}'")

        self._called: bool = False
        self._kwargs: dict = kwargs
        self._visibility: Type[Visibility] = visibility
        self._loading: bool = True
        self._loading_spinner: Optional["AsyncLoader.LoadingSpinner"] = None
        self._spinner_type: Optional[SpinnerTypes] = spinner_type
        self._spinner_size: Optional[str] = spinner_size
        self._spinner_color: Optional[str] = spinner_color
        self._spinner_thickness: Optional[float] = spinner_thickness

        logger.debug(f"{self} created")

        if direct_call:
            if not callable(self.__call__):
                raise TypeError(f"Invalid type '{type(self.__call__)}'")  # Should never happen but for type checking
            self.__call__()

    def __str__(self):
        id = self.id
        name = self.name
        kwargs_str = ""
        for key, value in self.kwargs.items():
            kwargs_str += f"{key}={value}, "
        kwargs_str = kwargs_str[:-2]
        return f"{self.__class__.__name__}({id=}, {name=}) -> {self._visibility.__name__}({kwargs_str})"

    @ui.refreshable
    def __call__(self,
                 visibility: Optional[Type[Visibility]] = None,
                 spinner_type: Optional[SpinnerTypes] = None,
                 spinner_size: Optional[str] = None,
                 spinner_color: Optional[str] = None,
                 spinner_thickness: Optional[float] = None,
                 **kwargs) -> None:
        """
        Draws the AsyncLoader

        :param visibility: The visibility of the AsyncLoader
        :param kwargs: The kwargs of for the visibility
        :return:
        """

        refresh = self._called

        logger.debug(f"{self}.__call__({refresh=})")

        if len(kwargs) > 0:
            self._kwargs = kwargs
        if visibility is not None:
            self._visibility = visibility
        if self._visibility is None:
            raise ValueError(f"No visibility set")

        if spinner_type is not None:
            self._spinner_type = spinner_type
        if spinner_size is not None:
            self._spinner_size = spinner_size
        if spinner_color is not None:
            self._spinner_color = spinner_color
        if spinner_thickness is not None:
            self._spinner_thickness = spinner_thickness

        self._loading_spinner = self.LoadingSpinner(self,
                                                    self._spinner_type,
                                                    size=self._spinner_size,
                                                    color=self._spinner_color,
                                                    thickness=self._spinner_thickness)
        if not self.loading:
            # draw visibility
            self._visibility(**self._kwargs)
            self._loading_spinner.set_visibility(False)

        self._called = True

        logger.debug(f"{self}.__call__({refresh=}) finished")

    @property
    def id(self) -> id:
        """
        The id of the AsyncLoader

        :return: id - The id of the AsyncLoader
        """

        return id(self)

    @property
    def name(self) -> str:
        """
        The name of the AsyncLoader
        :return: str - The name of the AsyncLoader
        """

        return self._name

    @property
    def kwargs(self) -> dict:
        """
        The kwargs of the AsyncLoader
        :return: dict - The kwargs of the AsyncLoader
        """

        return self._kwargs

    @property
    def visibility(self) -> Type[Visibility]:
        """
        The visibility of the AsyncLoader
        :return: Visibility - The visibility of the AsyncLoader
        """

        return self._visibility

    @property
    def loading(self) -> bool:
        """
        True if the AsyncLoader is loading, False otherwise
        :return: bool
        """

        return self._loading

    def set_loading(self, new: bool = False) -> None:
        """
        Sets the AsyncLoader to loading
        :return: None
        """

        # set loading
        self._loading = new

        # refresh
        self.__call__.refresh()

    @classmethod
    def get_instance(cls, obj: Union[ui.refreshable, ui.page]) -> "AsyncLoader":
        """
        Returns the instance of the AsyncLoader
        :param obj: The object
        :return: AsyncLoader
        """

        if isinstance(obj, ui.refreshable):
            loading_label = obj.targets[0].container.slots["default"].children[0].slots["default"].children[0]
        elif isinstance(obj, ui.page):
            raise NotImplementedError()
        else:
            raise TypeError(f"Invalid type '{type(obj)}'")
        if not isinstance(loading_label, cls.LoadingSpinner):
            raise TypeError(f"Invalid loading_label type '{type(loading_label)}'")

        async_loader = loading_label.instance

        if not isinstance(async_loader, AsyncLoader):
            raise TypeError(f"Invalid type '{type(async_loader)}'")
        return async_loader

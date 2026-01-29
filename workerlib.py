#
# Note: this library is intended for PyScript apps, it is useless outside the browser
#
# Tested on PyScript 26.1.1 / Pyodide 0.29.1 / Python 3.13.2
#
from collections.abc import Coroutine, Buffer, Callable, Iterable, Iterator, Mapping, Sequence
from functools import wraps
from importlib import import_module
from inspect import isclass, isfunction, iscoroutinefunction, signature
from itertools import chain
from numbers import Real
from re import search
from sys import flags, platform, version as _pythonVersion, _getframe
from time import time
from types import ModuleType
from typing import cast, Any, Final, NoReturn, TypeAlias

from pyscript import config, RUNNING_IN_WORKER  # Yes, this module is PyScript-only and won't work outside the browser

# We name all internal stuff starting with `_` underscore to minimize the chance of a conflict with exported user functions
type _Args = tuple[object, ...]
type _Kwargs = dict[str, object]
type _ArgsKwargs = tuple[_Args, _Kwargs]
type _Coroutine[T = object] = Coroutine[None, None, T]
type _CoroutineFunction[T = object] = Callable[..., _Coroutine[T]]
type _CallableOrCoroutine[T = object] = Callable[..., T | _Coroutine[T]]
type _Adapter[T = object] = tuple[type[T], Callable[[T], object | _Coroutine[object]] | None, Callable[[object], T | _Coroutine[T]] | None]
type _Time = float | int
type _Timed[T] = Mapping[str, str | _Time | T]
_Transferable: TypeAlias = Real | Iterable | None  # type: ignore[type-arg]  # Using `type` or adding `[Any]` breaks `isinstance()`  # noqa: UP040

try:  # Getting CPU count
    from os import process_cpu_count
    _cpus: int | str | None = process_cpu_count()
    if not _cpus:  # pylint: disable=consider-using-assignment-expr
        raise ValueError  # noqa: TRY301
except (ImportError, ValueError):
    try:
        from os import cpu_count
        _cpus = cpu_count()
        if not _cpus:  # pylint: disable=consider-using-assignment-expr
            raise ValueError  # noqa: TRY301  # pylint: disable=raise-missing-from
    except (ImportError, ValueError):
        _cpus = "UNKNOWN"

try:  # Getting platform configuration
    from sys import _emscripten_info  # type: ignore[attr-defined]  # pylint: disable=ungrouped-imports
    assert platform == 'emscripten'
    _emscriptenVersion: str | None = '.'.join(str(v) for v in _emscripten_info.emscripten_version)
    _runtime = _emscripten_info.runtime
    _pthreads = _emscripten_info.pthreads
    _sharedMemory = _emscripten_info.shared_memory
except ImportError:
    _emscriptenVersion = _runtime = _sharedMemory = None
    try:
        from os import sysconf  # Not available on Windows  # pylint: disable=ungrouped-imports
        _pthreads = (sysconf('SC_THREADS') > 0)  # pylint: disable=superfluous-parens
    except (ImportError, ValueError, AttributeError):
        _pthreads = False

try:
    from pyscript.web import page  # This would break in a worker if CORS is not configured  # pylint: disable=import-error, no-name-in-module
except AttributeError as ex:
    if RUNNING_IN_WORKER:
        page = None  # type: ignore[assignment]  # This is only needed for PyScript version detection and getting worker name, not mission-critical
    else:  # This shouldn't ever happen
        raise

try:  # Getting PyScript version
    from pyscript import version as _pyscriptVersion  # type: ignore[attr-defined]
except ImportError:
    try:
        from pyscript import __version__ as _pyscriptVersion  # type: ignore[attr-defined]
    except ImportError:
        try:
            scriptElement = next(e for e in page.find('script') if search(r'pyscript.*core\.js$', e.src))
            _pyscriptVersion = scriptElement.getAttribute('data-version') or \
                    next(word for word in scriptElement.src.split('/') if search(r'\d', word))
        except Exception:  # noqa: BLE001
            _pyscriptVersion = "UNKNOWN"

try:  # Getting Pyodide version
    from pyodide_js import version as _pyodideVersion  # type: ignore[import-not-found]
except ImportError:
    _pyodideVersion = "UNKNOWN"

try:
    from pyodide.ffi import JsProxy
except ImportError:
    type JsProxy = object  # type: ignore[no-redef]

try:  # To turn runtime typechecking on, add `"beartype"` to the `packages` array in your `workerlib.toml`
    from beartype import beartype as typechecked, __version__ as _beartypeVersion
    from beartype.roar import BeartypeException
except ImportError:
    _beartypeVersion = None  # type: ignore[assignment]

    # Public API
    def typechecked(func: _CallableOrCoroutine) -> _CallableOrCoroutine:  # type: ignore[no-redef]
        return func

_builtins = cast(_Kwargs, __builtins__)
if not isinstance(_builtins, dict):
    _builtins = _builtins.__dict__  # type: ignore[unreachable]

_NAME = ''

__EXPORT__: Final[str] = '__export__'

_ADAPTERS_SECTION: Final[str] = 'adapters'
_EXPORTS_SECTION: Final[str] = 'exports'

_WORKERLIB_PREFIX: Final[str] = '_workerlib_'
_ADAPTER_MARKER: Final[str] = _WORKERLIB_PREFIX + 'adapter'  # Used to differential adapter data from other mappings
_ADAPTER_VALUE: Final[str] = 'value'
_START_TIME: Final[str] = 'startTime'
_START_TIME_ADAPTER: Final[str] = _WORKERLIB_PREFIX + _START_TIME  # Used to differentiate start time data from other adapters
_CONNECT_REQUEST: Final[str] = _WORKERLIB_PREFIX + 'connectRequest'
_CONNECT_RESPONSE: Final[str] = _WORKERLIB_PREFIX + 'connectResponse'

_adapters: Sequence[_Adapter] = ()

__export__: Sequence[str]

__all__: Sequence[str] = (  # This is for static checkers, in runtime it will be reduced depending on context
    'Worker',
    'connectToWorker',
    'diagnostics',
    'elapsedTime',
    'export',
    'systemVersions',
    'typechecked',
)

@typechecked
def _log(*args: object) -> None:
    """Logs a message to the console."""
    print(f"[{_NAME}]", *args)

@typechecked
def _error(message: str) -> NoReturn:
    """
    Logs an error message to the console and stops further operations
    by raising a `RuntimeError`.
    """
    raise RuntimeError(f"[{_NAME}] {message}")

@typechecked  # Public API
def elapsedTime(startTime: _Time) -> str:  # noqa: PYI041  # beartype is right enforcing this: https://github.com/beartype/beartype/issues/66
    """
    Returns a diagnostic string mentioning time,
    in milliseconds or seconds, since `startTime`.
    `startTime` should be initialized with the `time.time()` call beforehand.
    """
    dt = time() - startTime
    return f"{round(dt)}s" if dt >= 1 else f"{round(dt * 1000)}ms"

@typechecked
def _importModule(moduleName: str) -> ModuleType:
    """
    Imports a module by its name and returns the module object.

    If the module has already been imported, returns it.
    """
    if not (module := globals().get(moduleName)):
        _log("Importing module", moduleName)
        globals()[moduleName] = module = import_module(moduleName)
    return module

@typechecked
def _importFromModule(module: str | ModuleType, qualNames: str | Iterable[str]) -> Iterator[object]:
    """
    Imports the specified `qualNames` from the specified `module`.

    `module` can be either a string or a module object.

    `qualNames` can be a string or an iterable of strings.

    Returns an `Iterator` of imported objects.
    """
    if isinstance(module, str):
        module = _importModule(module)
    if isinstance(qualNames, str):
        qualNames = (qualNames,)
    _log(f"Importing from module {module.__name__}: {', '.join(qualNames)}")
    for qualName in qualNames:
        try:
            ret = _builtins[qualName]
        except KeyError:
            obj = module
            for name in qualName.split('.'):
                obj = getattr(obj, name)
            ret = obj
        yield ret

@typechecked
def _adaptersFromSequence(module: ModuleType, names: Sequence[str | Sequence[str]], allowSubSequences: bool = True) -> Iterator[_Adapter]:
    """
    Given a `module` and an adapter description triple
    `["className", "encoderName", "decoderName"]`
    or a sequence of them,
    returns `Iterator` of adapter triples
    `[typeObject, encoderFunction, decoderFunction]`.
    """
    if not isinstance(names, Sequence):
        _error(f"""Bad adapter tuple type {type(names)} for module {module.__name__}""")
    if not names:
        _error(f"""Empty adapter tuple for module {module.__name__}""")
    if isinstance(names[0], str):
        if len(names) != 3:
            _error(f"""Bad adapter settings for module {module.__name__}, must be '["className", "encoderFunction", "decoderFunction"]', got {names!r}""")
        (cls, encoder, decoder) = _importFromModule(module, cast(Sequence[str], names))
        if not isclass(cls):
            _error(f'Bad adapter class "{names[0]}" for module {module.__name__}, must be a type, got {type(cls)}')
        if not (encoder is None or isfunction(encoder) or iscoroutinefunction(encoder)):
            _error(f'Bad adapter encoder "{names[1]}" for module {module.__name__}, must be a function/coroutine/None, got {type(encoder)}')
        if not (decoder is None or isfunction(decoder) or iscoroutinefunction(decoder)):
            _error(f'Bad adapter encoder "{names[2]}" for module {module.__name__}, must be a function/coroutine/None, got {type(decoder)}')
        _log(f"Adapter created: {decoder} => {cls} => {encoder}")
        yield (cls, encoder, decoder)
    elif allowSubSequences:
        for subSequence in names:
            yield from _adaptersFromSequence(module, subSequence, allowSubSequences = False)
    else:
        _error("""Adapter specification should be either [strings] or [[strings], ...], third level of inclusion is not needed""")

@typechecked
def _importAdapters() -> None:
    """
    Reads `[adapters]` config section, imports the necessary modules
    and functions, and makes them available as the global `_adapters` list.
    """
    global _adapters  # noqa: PLW0603  # pylint: disable=global-statement
    if _adapters:
        return  # Initialize adapters only once
    if not (mapping := config.get(_ADAPTERS_SECTION)):
        return
    adapters: list[_Adapter] = []
    for (moduleName, names) in mapping.items():
        module = _importModule(moduleName)
        adapters.extend(_adaptersFromSequence(module, names))
    _adapters = tuple(adapters)

@typechecked
async def _to_js(obj: object) -> object:
    """
    Converts an object to a data structure transferable to another process.

    Recurses into collections, uses adapters, and makes every effort possible
    to produce a transferable result.
    """
    for (cls, encoder, _decoder) in _adapters:
        if isinstance(obj, cls):
            if encoder:  # noqa: SIM108
                value = await encoder(obj) if iscoroutinefunction(encoder) else encoder(obj)
            else:
                value = obj  # Trying to pass the object as is, hoping it would work, like it does for `Enum`s
            return {_ADAPTER_MARKER: cls.__name__, _ADAPTER_VALUE: value}  # Encoded class name is NOT the name of the type of object being encoded, but the name of the adapter thas has to be used to decode the object on the other side
    if not isinstance(obj, _Transferable):
        _log(f"WARNING: No adapter found for class {type(obj)}, and transport layer (JavaScript structured clone) would probably not accept it as is, see https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Structured_clone_algorithm")
    if isinstance(obj, str | Buffer):  # Save it from being converted to `tuple` as an `Iterable`
        return obj
    if isinstance(obj, Mapping):  # Should be checked before `Iterable`, as `Mapping` is an `Iterable` too
        return {await _to_js(k): await _to_js(v) for (k, v) in obj.items()}
    if isinstance(obj, Iterable):
        return tuple([await _to_js(v) for v in obj])  # pylint: disable=consider-using-generator
    return obj  # Trying to pass the object as is, hoping it would work

@typechecked
async def _to_py(obj: object) -> object:
    """
    Converts an object received from another process to a usable Python object.

    Unpacks `JsProxy` objects, recurses into collections, uses adapters,
    and makes every effort possible to produce a result the recipient expects.
    """
    if hasattr(obj, 'to_py'):  # JsProxy
        return await _to_py(obj.to_py())
    if isinstance(obj, str | Buffer):  # Save it from being converted to `tuple` as an `Iterable`
        return obj
    if not isinstance(obj, Mapping):
        if isinstance(obj, Iterable):
            return tuple([await _to_py(v) for v in obj])  # pylint: disable=consider-using-generator
        return obj
    # isinstance(obj, Mapping)
    obj = {await _to_py(k): await _to_py(v) for (k, v) in obj.items()}

    # Checking for adapter
    if not (adapterName := obj.get(_ADAPTER_MARKER)):
        return obj
    if adapterName == _START_TIME_ADAPTER:
        return obj
    value = obj[_ADAPTER_VALUE]
    for (cls, _encoder, decoder) in _adapters:
        if cls.__name__ == adapterName:
            if decoder is None:
                return cast(type[Any], cls)(value)  # e.g. Enum
            return await decoder(value) if iscoroutinefunction(decoder) else decoder(value)
    # The data EXPLICITLY requests an adapter, so the unconverted object is definitely not expected
    _error(f"No adapter found to decode class '{adapterName}'")

@typechecked
def _diagnostics() -> Sequence[str]:
    """
    Returns a `tuple` of strings with detailed diagnostics of system components
    that can be nicely printed to the browser console, line by line,
    at the process startup.
    """
    ret: list[str] = []
    ret.append(f"PyScript {_pyscriptVersion}")
    ret.append(f"Pyodide {_pyodideVersion}")

    if platform == 'emscripten':
        ret.append(f"Emscripten {_emscriptenVersion}")
        ret.append(f"Runtime: {_runtime}")
        ret.append(f"CPUs: {_cpus}  pthreads: {_pthreads}  SharedMemory: {_sharedMemory}")
    else:
        ret.append(f"Platform: {platform}")
        ret.append(f"CPUs: {_cpus}  pthreads: {_pthreads}")

    ret.append(f"Python {_pythonVersion}")
    ret.append(f"DevMode {flags.dev_mode}  Optimized {flags.optimize or False}")
    assert __debug__ == (flags.optimize < 1)

    try:
        assert str()  # noqa: UP018
        ret.append("Assertions are DISABLED")
    except AssertionError:
        ret.append("Assertions are enabled")

    if _beartypeVersion:
        try:
            @typechecked
            def _test() -> int:
                return 'notInt'  # type: ignore[return-value]
            _test()
            raise RuntimeError(f"Beartype {_beartypeVersion} is not operating properly")
        except BeartypeException:
            ret.append(f"Beartype {_beartypeVersion} is up and watching, remove it from PyScript configuration to make things faster")
    else:
        ret.append("Runtime type checking is off")

    return tuple(ret)

# Public API
diagnostics = _diagnostics()
"""
Contains a sequence of strings with detailed diagnostics of system components
that can be nicely printed to the browser console, line by line, at the process startup.
"""

# Public API
systemVersions = {
    'PyScript': _pyscriptVersion,
    'Pyodide': _pyodideVersion,
    'Python': _pythonVersion.split()[0],
    'Emscripten': _emscriptenVersion,
}
"""
Mapping of system components and their detected runtime versions
(PyScript, Pyodide, Python, Emscripten).
May be useful for diagnostic purposes.
"""

                       ##
if RUNNING_IN_WORKER:  ##
                       ##

    _NAME = "worker"

    _log("Starting worker, sync_main_only =", config.get('sync_main_only', False))

    for info in diagnostics:
        _log(info)

    @typechecked
    def _workerSerialized[T](func: _CallableOrCoroutine[T]) -> _CoroutineFunction[object]:  # pylint: disable=redefined-outer-name
        """
        Used to wrap exported functions.

        Must be the outermost decorator, as it implements JS transport for calls from another process.

        Uses `_to_py()` to convert arguments, and `_to_js()` to convert returned value.
        Coordinates with `@_mainSerialized`.
        """
        @wraps(func)
        @typechecked
        async def _workerSerializedWrapper(*args: object, **kwargs: object) -> object:
            @typechecked
            def _extractKwargs(*args: object) -> _ArgsKwargs:
                """
                If the last of `args` is a `Mapping`, check whether it is possible
                for that to be the `kwargs` for the `func`. If so, extract it.

                Returns `(args, kwargs)` tuple to be used for calling `func`.
                """
                argsOnly: _ArgsKwargs = (args, {})
                if not args:
                    return argsOnly
                lastArg = args[-1]
                if not isinstance(lastArg, Mapping):
                    return argsOnly
                argsKwargs = cast(_ArgsKwargs, (args[:-1], lastArg))
                if _START_TIME_ADAPTER in lastArg:
                    return argsKwargs
                if not all(isinstance(key, str) for key in lastArg):
                    return argsOnly
                funcParams = signature(func).parameters
                if any(param.kind == param.VAR_KEYWORD for param in funcParams.values()):
                    return argsKwargs  # If `func` has a `**kwargs` argument, any `str`-keyed dictionary would fit
                paramNames = {paramName for (paramName, param) in funcParams.items() if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)}
                return argsKwargs if all(argName in paramNames for argName in lastArg) else argsOnly  # If `func` accepts keyword arguments, make sure all the keys in `lastArg` are among them

            assert not kwargs  # `kwargs` get passed to workers as the last of `args`, of type `dict`
            args = cast(_Args, await _to_py(args))
            (args, kwargs) = _extractKwargs(*args)  # If `func` accepts keyword arguments, extract them from the last of `args`
            # vv WRAPPED CALL vv
            ret = await func(*args, **kwargs) if iscoroutinefunction(func) else func(*args, **kwargs)
            # ^^ WRAPPED CALL ^^
            return await _to_js(ret)

        return _workerSerializedWrapper

    @typechecked
    def _workerLogged[T](func: _CallableOrCoroutine[T]) -> _CoroutineFunction[_Timed[T]]:  # pylint: disable=redefined-outer-name
        """
        Used to wrap exported functions.

        Uses `elapsedTime()` to measure wrapped function execution time
        and also time used to transfer the arguments and return value from
        and to the main process, coordinates with `@_mainLogged` for that.

        Prints diagnostic messages to the console before and after the call.
        """
        @wraps(func)
        @typechecked
        async def _workerLoggedWrapper(*args: object, **kwargs: object) -> _Timed[T]:
            try:
                elapsed = elapsedTime(cast(_Time, argStartTime)) if (argStartTime := kwargs.pop(_START_TIME_ADAPTER, None)) else ''
                callStartTime = time()
                if iscoroutinefunction(func):
                    _log(f"{f"Transferred arguments {elapsed}, awaiting" if elapsed else "Awaiting"} {func.__name__}(): {args} {kwargs}")
                    # vv WRAPPED CALL vv
                    ret: T = await func(*args, **kwargs)
                    # ^^ WRAPPED CALL ^^
                else:
                    assert isfunction(func), func
                    _log(f"{f"Transferred arguments {elapsed}, calling" if elapsed else "Calling"} {func.__name__}(): {args} {kwargs}")
                    # vv WRAPPED CALL vv
                    ret = func(*args, **kwargs)
                    # ^^ WRAPPED CALL ^^
                _log(f"Returned {elapsedTime(callStartTime)} from {func.__name__}(): {ret}")
                return {
                    _ADAPTER_MARKER: _START_TIME_ADAPTER,
                    _ADAPTER_VALUE: ret,
                    _START_TIME: time(),  # Starting calculating time it would take to transfer the return value to the main thread
                }
            except BaseException as ex:
                _log(f"Exception at {func.__name__}: {ex}")
                raise

        return _workerLoggedWrapper

    @typechecked
    def _workerWrap[T](func: _CallableOrCoroutine[T]) -> _CoroutineFunction[object]:  # pylint: disable=redefined-outer-name
        """Wraps a function to be exported with all the necessary decorators."""
        return _workerSerialized(_workerLogged(typechecked(func)))  # type: ignore[arg-type]  # It looks like there's some bug in `mypy` in this matter

    @_workerSerialized
    @typechecked
    def _connectFromMain(request: str, name: str) -> Sequence[str]:
        """
        Service function; is exported to be called by `connectToWorker()`
        in the main thread right after a connection to this worker is made.

        Assigns this worker a name sent from the main thread –
        usually the same name used to connect to this worker.
        Received name is further used as a prefix to logging messages.

        Returns the list of names of other exported functions so that
        they can be wrapped properly in the main thread.
        """
        if request == _CONNECT_REQUEST:
            _log(f'Connected to the main thread, assigned name "{name}", ready for requests')
            global _NAME  # noqa: PLW0603  # pylint: disable=global-statement
            _NAME = name
            assert __export__, __export__
            return tuple(chain((_CONNECT_RESPONSE,), (funcName for funcName in __export__ if funcName != _connectFromMain.__name__)))
        _error(f"Connection to main thread is misconfigured, can't continue: {type(request)}({request!r})")

    @typechecked  # Public API
    def export(*functions: _CallableOrCoroutine) -> None:
        """
        Must be called by the importing module with the list of functions
        to be exported.

        Wraps them with necessary decorators and puts them
        into the `__export__` list of the calling module.
        """
        target = _getframe(1).f_globals  # globals of the calling module
        target[_connectFromMain.__name__] = _connectFromMain
        exportNames: list[str] = []
        if _connectFromMain.__name__ not in target.get(__EXPORT__, ()):
            exportNames.append(_connectFromMain.__name__)

        for func in functions:
            target[func.__name__] = _workerWrap(func)
            exportNames.append(func.__name__)

        exportNamesTuple = tuple(chain(target.get(__EXPORT__, ()), exportNames))
        target[__EXPORT__] = exportNamesTuple
        globals()[__EXPORT__] = exportNamesTuple  # This is only needed to make `_connectFromMain()` code simple and universal for both `export()` and `_autoExport()`
        _importAdapters()  # Importing adapters late so that potential errors would not interrupt startup too early
        _log("Started worker, providing functions:", ', '.join(name for name in exportNamesTuple if name != _connectFromMain.__name__))

    @typechecked
    def _autoExport() -> None:
        """
        Gets called automatically if this module itself is loaded as a worker.

        Wraps and exports functions listed in the config.
        """
        exportNames = [_connectFromMain.__name__,]
        target = globals()

        if mapping := config.get(_EXPORTS_SECTION):
            for (moduleName, funcNames) in mapping.items():
                for (funcName, func) in zip(funcNames, _importFromModule(moduleName, funcNames), strict = True):
                    if not callable(func):
                        _error(f'Bad exported function "{funcName}", must be a class/function/coroutine, got {type(func)}')
                    target[funcName] = _workerWrap(func)
                    exportNames.append(funcName)
        else:
            _log("WARNING: no functions found to export, check `[exports]` section in the config")

        target[__EXPORT__] = tuple(exportNames)
        _importAdapters()
        _log("Started worker, providing functions:", ', '.join(name for name in exportNames if name != _connectFromMain.__name__))

    if __name__ == '__main__':
        # If this module itself is used as a worker, it imports functions mentioned in the config and exports them automatically
        del export
        __all__ = ()
        __export__ = ()
        _autoExport()
        assert __export__, __export__
    else:
        # If a user is importing this module into a worker, they MUST call `export()` explicitly
        del _autoExport
        __all__ = (
            'diagnostics',
            'elapsedTime',
            'export',
            'systemVersions',
            'typechecked',
        )

       ##
else:  ##  MAIN THREAD
       ##

    from pyscript import workers  # pylint: disable=ungrouped-imports

    _NAME = "main"

    @typechecked  # Public API
    class Worker:
        """Stores PyScript `worker` object and decorated functions exported by a worker."""
        def __init__(self, worker: JsProxy) -> None:
            self.worker = worker

    @typechecked
    def _mainSerialized[T](func: JsProxy, looksLike: _CallableOrCoroutine[T] | str) -> _CoroutineFunction[T]:  # pylint: disable=redefined-outer-name
        """
        Used to wrap worker functions.

        Must be the innermost decorator, as it implements JS transport for calls to another process.

        Uses `_to_js()` to convert arguments, and `_to_py()` to convert returned value.
        Coordinates with `@_mainSerialized`.
        """
        @typechecked
        async def _mainSerializedWrapper(*args: object, **kwargs: object) -> T:
            assert isinstance(func, JsProxy), type(func)
            args = cast(_Args, await _to_js(args))
            kwargs = cast(_Kwargs, await _to_js(kwargs))
            # vv WRAPPED CALL vv
            ret = await cast(_CoroutineFunction[T], func)(*args, kwargs)  # Passing `kwargs` as a positional argument because `**kwargs` don't get serialized properly and are sent as the last of `args` anyway
            # ^^ WRAPPED CALL ^^
            return cast(T, await _to_py(ret))

        if isfunction(looksLike) or iscoroutinefunction(looksLike):
            return wraps(looksLike)(_mainSerializedWrapper)
        ret: _CoroutineFunction[T] = wraps(cast(_CoroutineFunction[T], func))(_mainSerializedWrapper)
        assert isinstance(looksLike, str)
        ret.__name__ = looksLike
        return ret

    @typechecked
    def _mainLogged[T](func: _CoroutineFunction[T]) -> _CoroutineFunction[T]:  # pylint: disable=redefined-outer-name
        """
        Used to wrap worker functions.

        Uses `elapsedTime()` to measure wrapped function's execution time
        and also time used to transfer the arguments and return value from
        and to the worker, coordinates with `@_workerLogged` for that.

        Prints return value transfer time to the console.
        """
        @wraps(func)
        @typechecked
        async def _mainLoggedWrapper(*args: object, **kwargs: object) -> T:
            kwargs[_START_TIME_ADAPTER] = time()  # Starting calculating time it would take to transfer the function arguments to the worker
            # vv WRAPPED CALL vv
            ret: T | _Timed[T] = await func(*args, **kwargs)
            # ^^ WRAPPED CALL ^^
            if isinstance(ret, Mapping) and ret.get(_ADAPTER_MARKER) == _START_TIME_ADAPTER:
                startTime = cast(_Time, ret[_START_TIME])
                ret = cast(T, ret[_ADAPTER_VALUE])
                _log(f"Transferred return value {elapsedTime(startTime)} from {func.__name__}()")
            return cast(T, ret)

        return _mainLoggedWrapper

    @typechecked  # Public API
    async def connectToWorker(workerName: str | None = None) -> Worker:
        """
        Connects to a named worker with the specified `name`.
        That worker would use this `name` as a prefix to console logging messages.

        If `name` is not specified, looks for the
        `<script type="py" worker name="...">` tag in the page.
        If no such a tag is found, or more than one is found,
        raises an exception.
        """
        if not workerName:
            if not (names := tuple(element.getAttribute('name') for element in page.find('script[type="py"][worker][name]'))):
                _error("Could not find any named workers in DOM")
            if len(names) > 1:
                _error(f"Found the following named workers in DOM: ({', '.join(names)}), which one to connect to?..")
            workerName = names[0]

        _log(f'Looking for a worker named "{workerName}"')
        # vv WRAPPED CALL vv
        worker: JsProxy = await workers[workerName]
        # ^^ WRAPPED CALL ^^
        _log("Got the worker, connecting")
        data: Sequence[str] = await _mainSerialized(worker._connectFromMain, '_connectFromMain')(_CONNECT_REQUEST, workerName)  # type: ignore[attr-defined]  # noqa: SLF001  # pylint: disable=protected-access
        if not data or data[0] != _CONNECT_RESPONSE:
            _error(f"Connection to worker is misconfigured, can't continue: {type(data)}: {data!r}")

        ret = Worker(worker)  # We can't return `worker`, as it is a `JsProxy` and we can't reassign its fields – `setattr()` is not working, so we have to create a class of our own to store wrapped functions.
        for funcName in data[1:]:  # We can't get a list of exported functions from the `worker` object to wrap them, so we have to pass it over in our own way.
            assert funcName != '_connectFromMain'
            if not (func := getattr(worker, funcName, None)):
                _error(f"Function {funcName} is not exported from the worker")
            setattr(ret, funcName, _mainLogged(_mainSerialized(func, funcName)))

        _importAdapters()  # Importing adapters late so that potential errors would not interrupt startup too early
        _log("Connected to worker")
        return ret

    assert not globals().get(__EXPORT__), globals()[__EXPORT__]

    __all__ = (
        'Worker',
        'connectToWorker',
        'diagnostics',
        'elapsedTime',
        'systemVersions',
        'typechecked',
    )

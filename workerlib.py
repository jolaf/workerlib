#
# Note: this library is intended for PyScript / Pyodide apps, it is useless outside the browser
#
# Tested on PyScript 26.1.1 / Pyodide 0.29.1 / Python 3.13.2
#
from asyncio import create_task, gather
from collections.abc import Coroutine, Buffer, Callable, Iterable, Iterator, Mapping, Sequence
from functools import partial, wraps
from importlib import import_module
from inspect import isclass, iscoroutinefunction, signature
from itertools import chain
from numbers import Real
from pickle import dumps as pickleDump, loads as pickleLoad
from re import search
from sys import flags, modules, platform, version as _pythonVersion, _getframe
from time import time
from types import ModuleType
from typing import cast, ClassVar, Final, NoReturn, ReadOnly, Self, TypeAlias, TypedDict

from pyscript import config, RUNNING_IN_WORKER  # Yes, this module is indeed PyScript-only and won't work outside the browser

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
        page = None  # type: ignore[assignment]  # This would only inhibit PyScript version detection, not mission-critical
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
    def typechecked[T](func: _CallableOrCoroutine[T]) -> _CallableOrCoroutine[T]:  # type: ignore[no-redef]  # pylint: disable=redefined-outer-name
        return func

# We name all internal global stuff starting with `_` underscore to minimize the chance of a conflict with exported user functions

type _Args[T = object] = tuple[T, ...]
type _Kwargs[T = object] = dict[str, T]
type _ArgsKwargs[AT = object, KT = object] = tuple[_Args[AT], _Kwargs[AT]]
type _Coroutine[T = object] = Coroutine[None, None, T]
type _CoroutineFunction[T = object] = Callable[..., _Coroutine[T]]
type _CallableOrCoroutine[T = object] = Callable[..., T | _Coroutine[T]]
type _Time = float | int
type _Timed[T] = Mapping[str, str | _Time | T]

_BasicTypes: TypeAlias = Real | str | Buffer | None  # Using `type` breaks `isinstance()`
_Transferable: TypeAlias = _BasicTypes | Iterable  # type: ignore[type-arg]  # Using `type` or adding `[Any]` breaks `isinstance()`  # noqa: UP040

_NAME = ''

__EXPORT__: Final[str] = '__export__'

_ADAPTERS_SECTION: Final[str] = 'adapters'
_EXPORTS_SECTION: Final[str] = 'exports'
_IMPORTS_SECTION: Final[str] = 'imports'

_WORKERLIB_PREFIX: Final[str] = '_workerlib_'
_ADAPTER_MARKER: Final[str] = _WORKERLIB_PREFIX + 'adapter'  # Used to differential adapter data from other mappings
_ADAPTER_VALUE: Final[str] = 'value'
_ADAPTER_FULLNAME: Final[str] = 'className'
_START_TIME: Final[str] = 'startTime'
_START_TIME_ADAPTER: Final[str] = _WORKERLIB_PREFIX + _START_TIME  # Used to differentiate start time data from other adapters
_DEFAULT = '__default'
_PICKLE = "pickle"

_BUILTINS = 'builtins'
_BUILTINS_NAMES: Final[Sequence[str | None]] = (_BUILTINS, '', 'None', None)
_builtins = modules[_BUILTINS].__dict__

_BUILTIN_SPECIALS: Final[Mapping[str, type[object] | str | None]] = {
    '': None,
    'None': None,
    'Any': object,
    'object': object,
    _PICKLE: _PICKLE,
}

_CONNECT_REQUEST: Final[str] = _WORKERLIB_PREFIX + 'connectRequest'
_CONNECT_RESPONSE: Final[str] = _WORKERLIB_PREFIX + 'connectResponse'

_TAG_ATTR = 'tag'
_NAME_ATTR = 'name'

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
    raise RuntimeError(message)

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
def _fullName(obj: object) -> str:
    if not callable(obj):
        obj = type(obj)
    assert hasattr(obj, '__qualname__'), obj
    qualName: str = obj.__qualname__
    if (moduleName := obj.__module__) in _BUILTINS_NAMES:
        return qualName
    return f"{moduleName}.{qualName}"

@typechecked
def _importModule(moduleName: str | None) -> ModuleType:
    """
    Imports a module by its name and returns the module object.

    If the module has already been imported, returns it.
    """
    if moduleName in _BUILTINS_NAMES:
        return modules[_BUILTINS]
    if moduleName in (__name__, 'workerlib'):
        return modules[__name__]
    assert moduleName
    if module := modules.get(moduleName):
        return module
    _log("Importing module", moduleName)
    return import_module(moduleName)

@typechecked
def _importFromModule(module: str | ModuleType, names: str | Iterable[str]) -> Iterator[object]:
    """
    Imports the specified `names` from the specified `module`.

    `module` can be either a string or a module object.

    `names` can be a string or an iterable of strings.

    Returns an `Iterator` of imported objects.
    """
    if isinstance(module, str):  # ToDo: Find a way to import from '.' – user's module; see `target` in `export()`
        module = _importModule(module)
    if isinstance(names, str):
        names = (names,)
    if printNames := [n for n in names if '.' in n or (n not in _BUILTIN_SPECIALS and (module.__name__ == _BUILTINS or n not in _builtins))]:
        _log(f"Importing from module {'workerlib' if module.__name__ == __name__ else module.__name__}: {', '.join(printNames)}")

    for name in names:
        y: object
        if '.' not in name:
            if (y := _BUILTIN_SPECIALS.get(name, _DEFAULT)) != _DEFAULT:
                yield y
                continue
            if (y := _builtins.get(name, _DEFAULT)) != _DEFAULT:
                yield y
                continue
        y = module
        for n in name.split('.'):
            y = getattr(y, n)
        yield y

@typechecked
class _AnyCallArg(TypedDict):
    funcName: ReadOnly[str]
    args: ReadOnly[Sequence[object]]
    kwargs: ReadOnly[Mapping[str, object]]

@typechecked
class _Adapter:  # ToDo: Find a way to declare a "total" adapter for total pickle

    type _AdapterEncoding = Mapping[str, object]
    type _EncoderType[T = object] = Callable[[T], object | _Coroutine[object]] | None
    type _DecoderType[T = object] = Callable[[object], T | _Coroutine[T]] | Callable[[object, str], T | _Coroutine[T]]

    cls: type
    encoder: _EncoderType
    decoder: _DecoderType

    firstAdapters: ClassVar[dict[str, '_Adapter']] = {}  # Should be checked before collections, e.g., `Enum`
    secondAdapters: ClassVar[dict[str, '_Adapter']] = {}  # Should be checked after collections, e.g., custom classes and `object`

    def __init__[T](self, cls: type[T], encoder: _EncoderType[T], decoder: _DecoderType[T] | None) -> None:  # pylint: disable=redefined-outer-name
        if encoder:
            encoderParams = signature(encoder).parameters.values()
            if not (numEncoderArguments := sum(1 for p in encoderParams if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL))):
                _error(f"Adapter encoder for type {self.cls} accepts no positional arguments: {self.encoder}")
            if numEncoderArguments > 1:
                if (n := sum(1 for p in encoderParams if p.kind == p.POSITIONAL_ONLY)) > 1:  # ToDo: create function
                    _error(f"Adapter encoder for type {self.cls} has too many positional arguments: {self.encoder}, expected 1, got {n}")

        if decoder is None:
            decoder = cls

        decoderParams = signature(decoder).parameters.values()
        if not (numDecoderArguments := sum(1 for p in decoderParams if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL))):
            _error(f"Adapter decoder for type {cls} accepts no positional arguments: {decoder}")
        if numDecoderArguments > 1:
            if (n := sum(1 for p in decoderParams if p.kind == p.POSITIONAL_ONLY)) > 2:
                _error(f"Adapter decoder for type {cls} has too many positional arguments: {decoder}, expected 1 or 2, got {n}")

        self.cls = cls
        self.adapterName = _fullName(cls)
        self.encoder = encoder  # type: ignore[assignment]
        self.decoder = decoder
        self.numDecoderArguments = numDecoderArguments

        _log(f"Adapter created: {cls} => {encoder} => {decoder}")

    async def _encode(self, obj: object) -> _AdapterEncoding | None:
        if not isinstance(obj, self.cls):
            return None
        if self.encoder:  # noqa: SIM108  # pylint: disable=consider-ternary-expression
            value = await self.encoder(obj) if iscoroutinefunction(self.encoder) else self.encoder(obj)
        else:
            value = obj  # Trying to pass the object as is, hoping it would work, like it does for `Enum`s
        return {
            _ADAPTER_MARKER: self.adapterName,  # This is NOT the name of the type of object being encoded, but the name of the adapter thas has to be used to decode the object on the other side
            _ADAPTER_FULLNAME: _fullName(obj),  # This is the actual name of the type being transferred, but it's rarely used in decoding
            _ADAPTER_VALUE: value,
        }

    async def _decode(self, obj: object, fullName: str) -> object:
        if self.numDecoderArguments > 1:
            # If the decoder accepts two arguments, pass `fullName` as a second argument to give the decoder a chance to reconstruct an object precisely
            return await self.decoder(obj, fullName) if iscoroutinefunction(self.decoder) else self.decoder(obj, fullName)  # type: ignore[call-arg]
        return await self.decoder(obj) if iscoroutinefunction(self.decoder) else self.decoder(obj)  # type: ignore[call-arg]

    @classmethod
    async def _checkAndEncode(cls, adapters: Mapping[str, Self], obj: object) -> _AdapterEncoding | None:
        for adapter in adapters.values():
            if ret := await adapter._encode(obj):  # pylint: disable=protected-access
                return ret
        return None

    @classmethod
    async def checkAndEncodeFirst(cls, obj: object) -> _AdapterEncoding | None:
        return await cls._checkAndEncode(cls.firstAdapters, obj)

    @classmethod
    async def checkAndEncodeSecond(cls, obj: object) -> _AdapterEncoding | None:
        return await cls._checkAndEncode(cls.secondAdapters, obj)

    @classmethod
    async def findAndDecode(cls, obj: Mapping[object, object]) -> object:
        if not (adapterName := cast(str | None, obj.get(_ADAPTER_MARKER))):
            return obj
        if adapterName == _START_TIME_ADAPTER:
            return obj  # It's not an adapter, but a similar format used to measure duration of data transfer
        fullName = cast(str, obj[_ADAPTER_FULLNAME])
        value = obj[_ADAPTER_VALUE]

        if adapter := cls.firstAdapters.get(adapterName) or cls.secondAdapters.get(adapterName):  # It must be exact equality check, not `isinstance()`; the idea is to find the adapter that encoded this data – no more, no less
            return await adapter._decode(value, fullName)  # pylint: disable=protected-access

        # The data EXPLICITLY requests an adapter, so the unconverted object is definitely not expected
        _error(f"No adapter found to decode class '{adapterName}'")

    @classmethod
    def _fromModule(cls, module: ModuleType, names: Sequence[str]) -> Self:  # ToDo: Find a way to import from '.' – user's module; see `target` in `export()`
        """
        Imports the specified `qualNames` from the specified `module`.

        Returns an `_Adapter` instance constructed from imported objects.
        """
        if len(names) < 1 or len(names) > 3:
            _error(f"""Bad adapter descriptor for module {module.__name__}, must be '["className", "encoderFunction", "decoderFunction"]', got {names!r}""")
        args = list(_importFromModule(module, names))
        _cls = args[0]
        encoder = args[1] if len(args) > 1 else None
        decoder = args[2] if len(args) > 2 else None

        if encoder == _PICKLE:
            encoder = pickleDump
        if decoder == _PICKLE:
            decoder = pickleLoad

        if not isclass(_cls):
            _error(f'Bad adapter class "{names[0]}" for module {module.__name__}, must be a type, got {type(_cls)}')
        if not (encoder is None or callable(encoder)):
            _error(f'Bad adapter encoder "{names[1]}" for module {module.__name__}, must be a function/coroutine/None, got {type(encoder)}')
        if not (decoder is None or callable(decoder)):
            _error(f'Bad adapter decoder "{names[2]}" for module {module.__name__}, must be a function/coroutine/None, got {type(decoder)}')

        return cls(_cls, encoder, decoder)

    @classmethod
    def _fromSequence(cls,
                      module: ModuleType,
                      names: Sequence[str] | Sequence[Sequence[str]],
                      allowSubSequences: bool = True) \
            -> Iterator[Self]:
        """
        Given a `module` and an adapter descriptor triple
        `["className", "encoderName", "decoderName"]`
        or a sequence of them,
        returns `Iterator` of `_Adapter` instances constructed from imported objects.
        """
        if not isinstance(names, Sequence):
            _error(f"Bad adapter descriptor type for module {module.__name__}, must be a sequence, got {type(names)}")
        if not names:
            _error(f"Empty adapter descriptor for module {module.__name__}")
        if isinstance(names[0], str):
            yield cls._fromModule(module, cast(Sequence[str], names))
        elif allowSubSequences:
            for subSequence in names:
                yield from cls._fromSequence(module, subSequence, allowSubSequences = False)
        else:
            _error("Adapter specification should be either [strings] or [[strings], …], the third level of inclusion is not needed")

    @classmethod
    def fromConfig(cls) -> None:
        """
        Reads `[adapters]` config section, imports the necessary modules
        and functions, and makes them available as the global `_adapters` list.
        """
        if cls.firstAdapters or cls.secondAdapters:
            return  # Initialize adapters only once
        if not (mapping := cast(Mapping[str, Sequence[str]], config.get(_ADAPTERS_SECTION))):
            return  # No adapters defined in config
        for (moduleName, names) in mapping.items():
            module = _importModule(moduleName)
            for adapter in cls._fromSequence(module, names):
                m = cls.firstAdapters if issubclass(adapter.cls, _Transferable) else cls.secondAdapters
                m[adapter.adapterName] = adapter

@typechecked
async def _to_js(obj: object) -> object:
    """
    Converts an object to a data structure transferable to another process.

    Recurses into collections, uses adapters, and makes every effort possible
    to produce a transferable result.
    """
    if ret := await _Adapter.checkAndEncodeFirst(obj):
        return ret
    if isinstance(obj, _BasicTypes):
        return obj  # Save it from being converted to `tuple` as an `Iterable`
    if isinstance(obj, Mapping):  # Should be checked before `Iterable`, as `Mapping` is an `Iterable` too
        return {k: v for (k, v) in await gather(*(gather(*(_to_js(x) for x in k_v)) for k_v in obj.items()))}  # ToDo: make functions for these  # pylint: disable=unnecessary-comprehension
    if isinstance(obj, Iterable):
        return await gather(*(_to_js(v) for v in obj))
    if ret := await _Adapter.checkAndEncodeSecond(obj):
        return ret
    _log(f"WARNING: No adapter found for class {type(obj)}, and transport layer (JavaScript structured clone) would probably not accept it as is, see https://developer.mozilla.org/docs/Web/API/Web_Workers_API/Structured_clone_algorithm")
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
    if isinstance(obj, _BasicTypes):
        return obj  # Save it from being converted to `tuple` as an `Iterable`
    if not isinstance(obj, Mapping):
        if isinstance(obj, Iterable):
            return tuple(await gather(*(_to_py(v) for v in obj)))
        return obj  # Not `Iterable`
    # isinstance(obj, Mapping)
    obj = {k: v for (k, v) in await gather(*(gather(*(_to_py(x) for x in k_v)) for k_v in obj.items()))}  # pylint: disable=unnecessary-comprehension
    return await _Adapter.findAndDecode(obj)

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

__export__: Sequence[str]

__all__: Sequence[str] = (  # This is for static checkers, in runtime it will be reduced depending on the context
    'Worker',
    'anyCall',
    'connectToWorker',
    'diagnostics',
    'elapsedTime',
    'export',
    'systemVersions',
    'typechecked',
)

                       ##
if RUNNING_IN_WORKER:  ##
                       ##

    _NAME = "worker"

    _log("Starting worker")

    for info in diagnostics:
        _log(info)

    @typechecked
    def _workerSerialized[T = object](func: _CallableOrCoroutine[T]) -> _CoroutineFunction:  # pylint: disable=redefined-outer-name
        """
        Used to wrap exported functions.

        Must be the outermost decorator, as it implements JS transport for calls from another process.

        Uses `_to_py()` to convert arguments, and `_to_js()` to convert returned value.
        Coordinates with `@_mainSerialized`.
        """
        @wraps(func)
        @typechecked
        async def workerSerializedWrapper(*args: object, **kwargs: object) -> object:
            @typechecked
            def extractKwargs(*args: object) -> _ArgsKwargs:
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
            (args, kwargs) = extractKwargs(*args)  # If `func` accepts keyword arguments, extract them from the last of `args`
            # vv WRAPPED CALL vv
            ret = await func(*args, **kwargs) if iscoroutinefunction(func) else func(*args, **kwargs)
            # ^^ WRAPPED CALL ^^
            return await _to_js(ret)

        return workerSerializedWrapper

    @typechecked
    def _workerLogged[T = object](func: _CallableOrCoroutine[T]) -> _CoroutineFunction[_Timed[T]]:  # pylint: disable=redefined-outer-name
        """
        Used to wrap exported functions.

        Uses `elapsedTime()` to measure wrapped function execution time
        and also time used to transfer the arguments and return value from
        and to the main process, coordinates with `@_mainLogged` for that.

        Prints diagnostic messages to the console before and after the call.
        """
        @typechecked
        @wraps(func)
        async def workerLoggedWrapper(*args: object, **kwargs: object) -> _Timed[T]:
            try:
                elapsed = elapsedTime(cast(_Time, argStartTime)) if (argStartTime := kwargs.pop(_START_TIME_ADAPTER, None)) else ''
                callStartTime = time()
                if iscoroutinefunction(func):
                    _log(f"{f"Transferred arguments {elapsed}, awaiting" if elapsed else "Awaiting"} {func.__name__}(): {args} {kwargs}")
                    # vv WRAPPED CALL vv
                    ret: T = await func(*args, **kwargs)
                    # ^^ WRAPPED CALL ^^
                else:
                    _log(f"{f"Transferred arguments {elapsed}, calling" if elapsed else "Calling"} {func.__name__}(): {args} {kwargs}")
                    # vv WRAPPED CALL vv
                    ret = cast(T, func(*args, **kwargs))
                    # ^^ WRAPPED CALL ^^
                _log(f"Returned {elapsedTime(callStartTime)} from {func.__name__}(): {ret}")
                return {
                    _ADAPTER_MARKER: _START_TIME_ADAPTER,
                    _ADAPTER_FULLNAME: _fullName(ret),
                    _ADAPTER_VALUE: ret,
                    _START_TIME: time(),  # Starting calculating time it would take to transfer the return value to the main thread
                }
            except BaseException as ex:
                _log(f"Exception at {func.__name__}: {ex}")
                raise

        return workerLoggedWrapper

    @typechecked
    def _workerWrap[T = object](func: _CallableOrCoroutine[T]) -> _CoroutineFunction:  # pylint: disable=redefined-outer-name
        """Wraps a function to be exported with all the necessary decorators."""
        return _workerSerialized(_workerLogged(typechecked(func)))

    @typechecked
    def _imports(target: dict[str, object] | None = None) -> None:
        if target is None:
            target = globals()
        for (moduleName, names) in config.get(_IMPORTS_SECTION, {}).items():
            for (name, obj) in zip(names, _importFromModule(moduleName, names), strict = True):
                target[name] = obj

    @typechecked  # Public API
    async def anyCall(funcNameAndArgs: _AnyCallArg) -> object:
        funcName = funcNameAndArgs['funcName']
        args = funcNameAndArgs['args']
        kwargs = funcNameAndArgs['kwargs']
        if not (func := globals().get(funcName)):
            _error(f"Function '{funcName}()' is not found in the worker")
        if not callable(func):
            _error(f"Variable '{funcName}' in the worker is not callable: {type(func)}")
        return await func(*args, **kwargs) if iscoroutinefunction(func) else func(*args, **kwargs)

    @_workerSerialized
    @typechecked
    def _connectFromMain(request: str, name: str, attributes: Mapping[str, str]) -> Sequence[str]:
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
            _log(f'Assigned name "{name}"')
            global _NAME  # noqa: PLW0603  # pylint: disable=global-statement
            _NAME = name
            config['attributes'] = attributes
            _log("Tag:", attributes.get(_TAG_ATTR, "''"))
            _log(f"Connected to the main thread, sync_main_only = {config.get('sync_main_only', False)}, ready for requests")
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

        _imports(target)  # ToDo: Call it only once!!
        _Adapter.fromConfig()  # Importing adapters late so that potential errors would not interrupt startup too early

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
        _log("Started worker, providing functions:", ', '.join(name for name in exportNamesTuple if name != _connectFromMain.__name__))  # ToDo: Print "started" once, prevent export() calls after connection

    @typechecked
    async def _autoExport() -> None:
        """
        Gets called automatically if this module itself is loaded as a worker.

        Wraps and exports functions listed in the config.
        """
        _imports()
        _Adapter.fromConfig()

        exportNames = [_connectFromMain.__name__,]
        target = globals()

        if mapping := config.get(_EXPORTS_SECTION):
            for (moduleName, funcNames) in mapping.items():  # ToDo: Create function
                for (funcName, func) in zip(funcNames, _importFromModule(moduleName, funcNames), strict = True):
                    if not callable(func):
                        _error(f'Bad exported function "{funcName}", must be a class/function/coroutine, got {type(func)}')
                    target[funcName] = _workerWrap(func)
                    exportNames.append(funcName)
        else:
            _log("WARNING: no functions found to export, check `[exports]` section in the config")

        target[__EXPORT__] = tuple(exportNames)
        _log("Started worker, providing functions:", ', '.join(name for name in exportNames if name != _connectFromMain.__name__))

    if __name__ == '__main__':
        # If this module itself is used as a worker, it imports functions mentioned in the config and exports them automatically
        del export
        __all__ = ()
        __export__ = ()
        create_task(_autoExport())  # Make sure this module gets fully loaded before `_autoExport()` is called – to avoid circular import errors if anything being imported tries to import something from `workerlib`, e.g., `anyCall`
    else:
        # If a user is importing this module into a worker, they MUST call `export()` explicitly
        del _autoExport
        __all__ = (
            'anyCall',
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

    @typechecked
    class _AnyCall:
        def __init__(self, wrappedAnyCallCoroutine: _CoroutineFunction) -> None:
            self.anyCallCoroutine = wrappedAnyCallCoroutine

        async def __call__(self, funcName: str, *args: object, **kwargs: object) -> object:
            funcNameAndArgs = _AnyCallArg(funcName = funcName, args = args, kwargs = kwargs)
            return await self.anyCallCoroutine(funcNameAndArgs)

        def __getattr__(self, funcName: str) -> _CoroutineFunction:
            return partial(self.__call__, funcName)

    @typechecked  # Public API
    class Worker:
        """Stores PyScript `worker` object and decorated functions exported by a worker."""
        def __init__(self, worker: JsProxy) -> None:
            self.worker = worker

    @typechecked
    def _mainSerialized[T = object](func: JsProxy, looksLike: _CallableOrCoroutine[T] | str) -> _CoroutineFunction[T]:  # pylint: disable=redefined-outer-name
        """
        Used to wrap worker functions.

        Must be the innermost decorator, as it implements JS transport for calls to another process.

        Uses `_to_js()` to convert arguments, and `_to_py()` to convert returned value.
        Coordinates with `@_workerSerialized`.
        """
        @typechecked
        async def mainSerializedWrapper(*args: object, **kwargs: object) -> T:
            assert isinstance(func, JsProxy), type(func)
            args = await gather(*(_to_js(arg) for arg in args))  # type: ignore[assignment]
            kwargs = dict(zip(kwargs.keys(), await gather(*(_to_js(v) for v in kwargs.values())), strict = True))
            # vv WRAPPED CALL vv
            ret = await cast(_CoroutineFunction[T], func)(*args, kwargs)  # Passing `kwargs` as a positional argument because `**kwargs` don't get serialized properly and are sent as the last of `args` anyway
            # ^^ WRAPPED CALL ^^
            return cast(T, await _to_py(ret))

        if callable(looksLike):
            return wraps(looksLike)(mainSerializedWrapper)
        ret: _CoroutineFunction[T] = wraps(cast(_CoroutineFunction[T], func))(mainSerializedWrapper)
        assert isinstance(looksLike, str)
        ret.__qualname__ = ret.__name__ = looksLike
        return ret

    @typechecked
    def _mainLogged[T = object](func: _CoroutineFunction[T]) -> _CoroutineFunction[T]:  # pylint: disable=redefined-outer-name
        """
        Used to wrap worker functions.

        Uses `elapsedTime()` to measure wrapped function's execution time
        and also time used to transfer the arguments and return value from
        and to the worker, coordinates with `@_workerLogged` for that.

        Prints return value transfer time to the console.
        """
        @wraps(func)
        @typechecked
        async def mainLoggedWrapper(*args: object, **kwargs: object) -> T:
            kwargs[_START_TIME_ADAPTER] = time()  # Starting calculating time it would take to transfer the function arguments to the worker
            # vv WRAPPED CALL vv
            ret: T | _Timed[T] = await func(*args, **kwargs)
            # ^^ WRAPPED CALL ^^
            if isinstance(ret, Mapping) and ret.get(_ADAPTER_MARKER) == _START_TIME_ADAPTER:
                retStartTime = cast(_Time, ret[_START_TIME])
                #className = cast(str, ret[_ADAPTER_CLASSNAME])
                ret = cast(T, ret[_ADAPTER_VALUE])
                _log(f"Transferred return value {elapsedTime(retStartTime)} from {func.__name__}()")
            return cast(T, ret)

        return mainLoggedWrapper

    @typechecked
    def _mainWrap[T = object](func: JsProxy, looksLike: _CallableOrCoroutine[T] | str) -> _CoroutineFunction[T]:  # pylint: disable=redefined-outer-name
        """Wraps an exported function with all the necessary decorators."""
        return _mainLogged(_mainSerialized(func, looksLike))

    @typechecked  # Public API
    async def connectToWorker(workerName: str | None = None) -> Worker:  # ToDo: Refactor it somehow to be compatible with other languages
        """
        Connects to a named worker with the specified `name`.
        That worker would use this `name` as a prefix to console logging messages.

        If `name` is not specified, looks for the
        `<script type="py" worker name="…">` tag in the page.
        If no such a tag is found, or more than one is found,
        raises an exception.
        """
        if workerName:
            elements = page.find(f'script[type="py"][worker][name={workerName}]')
            if not (element := elements[0] if elements else None):
                _log(f'WARNING: Could not find a worker named "{workerName}" in DOM')
        else:
            if not (elements := page.find('script[type="py"][worker][name]')):
                pyScripts = '\n'.join(tag.outerHTML for tag in page.find('script[type="py"]'))  # ToDo: Create constant?
                _error(f"Could not find any named workers in DOM{', only found the following PyScript tags:\n' + pyScripts if pyScripts else ''}")
            if len(elements) > 1:
                _error(f"Found the following named workers in DOM: {', '.join([element.getAttribute(_NAME_ATTR) for element in elements])}; which one to connect to?..")
            element = elements[0]

        attributes: dict[str, str] = {}
        if element:
            attributes[_TAG_ATTR] = element.outerHTML

            for attributeName in element.getAttributeNames():
                value = element.getAttribute(attributeName)
                assert isinstance(value, str), type(value)
                attributes[attributeName] = value

            assert not workerName or workerName == attributes[_NAME_ATTR], (workerName, attributes[_NAME_ATTR])

        if not workerName:
            workerName = attributes[_NAME_ATTR]

        assert workerName

        _Adapter.fromConfig()

        _log(f'Looking for a worker named "{workerName}"')
        # vv WRAPPED CALL vv
        worker = await workers[workerName]
        # ^^ WRAPPED CALL ^^
        _log("Got the worker, connecting")
        data = cast(Sequence[str], await _mainSerialized(worker._connectFromMain, '_connectFromMain')(_CONNECT_REQUEST, workerName, attributes))  # noqa: SLF001  # pylint: disable=protected-access
        if not data or data[0] != _CONNECT_RESPONSE:
            _error(f"Connection to worker is misconfigured, can't continue: {type(data)}: {data!r}")

        ret = Worker(worker)  # We can't return `worker`, as it is a `JsProxy` and we can't reassign its fields – `setattr()` is not working, so we have to create a class of our own to store wrapped functions.
        for funcName in data[1:]:  # We can't get a list of exported functions from the `worker` object to wrap them, so we have to pass it over in our own way.
            assert funcName != '_connectFromMain'
            if not (func := getattr(worker, funcName, None)):
                _error(f"Function {funcName} is not exported from the worker")
            wrappedFunc = _mainWrap(func, funcName)
            setattr(ret, funcName, _AnyCall(wrappedFunc) if funcName == 'anyCall' else wrappedFunc)

        _log("Connected to the worker")
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

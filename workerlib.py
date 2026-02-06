#
# Note: this library is intended for PyScript / Pyodide apps, it is useless outside the browser
#
# Tested on PyScript 26.2.1 / Pyodide 0.29.3 / Python 3.13.2
#
# pylint: disable=wrong-import-position
#
try:
    from pyscript import config, RUNNING_IN_WORKER
except ImportError as ex:
    raise RuntimeError("\n\nThis module can only be used in a browser with PyScript / Pyodide\n") from ex

from sys import version_info

if version_info < (3, 13):  # noqa: UP036
    raise RuntimeError("This module requires Python 3.13+")

from asyncio import create_task, get_running_loop, gather, AbstractEventLoop
from collections.abc import Buffer, Callable, Coroutine, Iterable, Iterator, Mapping, Sequence
from contextlib import suppress
from functools import partial, wraps
from importlib import import_module
from inspect import currentframe, getmodule, isclass, iscoroutinefunction, signature, Parameter as P, _ParameterKind
from itertools import chain
from numbers import Real
from pickle import dumps as pickleDump, loads as pickleLoad
from re import search
import sys
from sys import flags, modules, platform, stderr, version as _pythonVersion
import threading
from _thread import _ExceptHookArgs
from time import time
from traceback import extract_tb, StackSummary
from types import ModuleType, TracebackType  # noqa: TC003
from typing import cast, final, Any, ClassVar, Final, NoReturn, ReadOnly, Required, Self, TypeAlias, TypedDict, Union

if version_info < (3, 13):  # noqa: UP036
    raise RuntimeError("This module requires Python 3.13+")

try:
    from pyscript.web import page  # This would break in a worker if CORS is not configured  # pylint: disable=import-error, no-name-in-module
except AttributeError as ex:
    if RUNNING_IN_WORKER:
        page = None  # type: ignore[assignment]  # This would only inhibit PyScript version detection, not mission-critical
    else:  # This shouldn't ever happen
        raise

try:
    from pyodide.ffi import JsProxy
except ImportError:
    type JsProxy = object  # type: ignore[no-redef]

from js import console  # pyodide, doesn't require CORS to work

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

try:  # Getting PyScript version
    from pyscript import version as _pyscriptVersion  # type: ignore[attr-defined]  # pylint: disable=ungrouped-imports
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

# We name all internal global stuff starting with `_` underscore to minimize the chance of a conflict with exported user functions

type _Callable[T = object] = Callable[..., T]
type _Coroutine[T = object] = Coroutine[None, None, T]
type _CoroutineFunction[T = object] = _Callable[_Coroutine[T]]
type _CallableOrCoroutine[T = object] = _Callable[T | _Coroutine[T]]

try:  # To turn runtime typechecking on, add `"beartype"` to the `packages` array in your `workerlib.toml`
    from beartype import beartype as typechecked, __version__ as _beartypeVersion
    from beartype.roar import BeartypeException
except ImportError:
    _beartypeVersion = None  # type: ignore[assignment]

    # Public API
    def typechecked[T](func: _CallableOrCoroutine[T]) -> _CallableOrCoroutine[T]:  # type: ignore[no-redef]  # pylint: disable=redefined-outer-name
        return func

type _Args[T = object] = tuple[T, ...]
type _Kwargs[T = object] = dict[str, T]
type _ArgsKwargs[AT = object, KT = AT] = tuple[_Args[AT], _Kwargs[KT]]

type _Time = float | int
type _Timed[T] = Mapping[str, str | _Time | T]

_BasicTypes: TypeAlias = Real | str | Buffer | None  # Using `type` breaks `isinstance()`
_Transferable: TypeAlias = _BasicTypes | Iterable  # type: ignore[type-arg]  # Using `type` or adding `[Any]` breaks `isinstance()`  # noqa: UP040


_NAME = ''

_PY_SCRIPT: Final[str] = 'script[type="py"]'
_PY_SCRIPT_WORKER: Final[str] = _PY_SCRIPT + '[worker]'
__EXPORT__: Final[str] = '__export__'

_ADAPTERS_SECTION: Final[str] = 'adapters'
_EXPORTS_SECTION: Final[str] = 'exports'
_IMPORTS_SECTION: Final[str] = 'imports'

_WORKERLIB_PREFIX: Final[str] = '_workerlib_'
_ADAPTER_MARKER: Final[str] = _WORKERLIB_PREFIX + 'adapter'  # Used to differentiate adapter data from other mappings
_ADAPTER_VALUE: Final[str] = 'value'
_ADAPTER_FULLNAME: Final[str] = 'className'
_START_TIME: Final[str] = 'startTime'
_START_TIME_ADAPTER: Final[str] = _WORKERLIB_PREFIX + _START_TIME  # Used to differentiate start time data from real adapters
_DEFAULT: Final[object] = object()
_PICKLE: Final[str] = 'pickle'

_BUILTINS: Final[str] = 'builtins'
_BUILTINS_NAMES: Final[Sequence[str | None]] = (_BUILTINS, '', 'None', None)
_builtins: Final[Mapping[str, object]] = modules[_BUILTINS].__dict__
_callingModule: ModuleType | None = None

_BUILTIN_SPECIALS: Final[Mapping[str, type[object] | str | None]] = {
    '': None,
    'None': None,
    'Any': object,
    'object': object,
    _PICKLE: _PICKLE,
}

_CONNECT_REQUEST: Final[str] = _WORKERLIB_PREFIX + 'connectRequest'
_CONNECT_RESPONSE: Final[str] = _WORKERLIB_PREFIX + 'connectResponse'

_TAG_ATTR: Final[str] = 'tag'
_NAME_ATTR: Final[str] = 'name'

@typechecked
def _log(*args: object, **kwargs: object) -> None:
    """Logs a message to the console."""
    print(f"[{_NAME}]", *args, flush = True, **kwargs)  # type: ignore[call-overload]

@typechecked
def _warn(*args: object) -> None:
    """Logs a warning message to the console."""
    console.warn(f"[{_NAME}]", *args)

@typechecked
def _error(message: str) -> NoReturn:
    """
    Logs an error message to the console and stops further operations
    by raising a `RuntimeError`.
    """
    raise RuntimeError(f"[{_NAME}] ERROR: {message}")

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
def fullName(obj: object) -> str:
    if not callable(obj):
        obj = type(obj)
    assert hasattr(obj, '__qualname__'), obj
    qualName: str = obj.__qualname__
    if (moduleName := obj.__module__) in _BUILTINS_NAMES:
        return qualName
    return f"{moduleName}.{qualName}"

@typechecked
def _getArgNames(func: _CallableOrCoroutine, *paramKinds: _ParameterKind) -> Iterable[str]:
    params = signature(func).parameters
    return (name for (name, param) in params.items() if param.kind in paramKinds) if paramKinds else params.keys()

@typechecked
def _countArgs(func: _CallableOrCoroutine, *paramKinds: _ParameterKind) -> int:
    return len(list(_getArgNames(func, *paramKinds)))

@typechecked
async def _gatherList[S, V](coro: Callable[[S], _Coroutine[V]], iterable: Iterable[S]) -> list[V]:
    return await gather(*(coro(s) for s in iterable))

@typechecked
async def _gatherMap[S, V](coro: Callable[[S], _Coroutine[V]], mapping: Mapping[S, S]) -> dict[V, V]:
    return {k: v for (k, v) in await gather(*(gather(*(coro(s) for s in k_v)) for k_v in mapping.items()))}  # pylint: disable=unnecessary-comprehension

@typechecked
async def _gatherValues[K, S, V](coro: Callable[[S], _Coroutine[V]], mapping: Mapping[K, S]) -> dict[K, V]:
    return dict(zip(mapping.keys(), await gather(*(coro(s) for s in mapping.values())), strict = True))

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
            if (y := _BUILTIN_SPECIALS.get(name, _DEFAULT)) is not _DEFAULT:
                yield y
                continue
            if (y := _builtins.get(name, _DEFAULT)) is not _DEFAULT:
                yield y
                continue
        y = module
        for n in name.split('.'):
            y = getattr(y, n)
        yield y

@typechecked
class _AnyCallArg(TypedDict, total = False):
    funcName: Required[ReadOnly[str]]
    args: ReadOnly[Sequence[object]]
    kwargs: ReadOnly[Mapping[str, object]]

@typechecked
@final
class All(tuple[object]):
    """
    Special class to use in "All" adapter to process everything,
    bypassing JS-compatible data conversion.
    """
    def __init__(self) -> NoReturn:
        _error('"All" class is just a marker, it should never be instantiated')

@typechecked
class _Adapter:

    type _EncoderType[T = object] = Callable[[T], object | _Coroutine[object]] | None
    type _DecoderType[T = object] = Callable[[object], T | _Coroutine[T]] | Callable[[object, str | None], T | _Coroutine[T]]
    type _AdapterEncoding = Mapping[str, object] | object

    cls: type
    encoder: _EncoderType
    decoder: _DecoderType

    totalAdapter: ClassVar[Union['_Adapter', None]] = None  # We have to use `Union` as `beartype` breaks at `str | None`, as of v0.22.9  # pylint: disable=consider-alternative-union-syntax
    firstAdapters: ClassVar[dict[str, '_Adapter']] = {}  # Should be checked before collections, e.g., `Enum`
    secondAdapters: ClassVar[dict[str, '_Adapter']] = {}  # Should be checked after collections, e.g., custom classes and `object`

    def __init__[T](self, cls: type[T], encoder: _EncoderType[T], decoder: _DecoderType[T] | None) -> None:  # pylint: disable=redefined-outer-name
        if encoder:
            if not (numEncoderArguments := _countArgs(encoder, P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD, P.VAR_POSITIONAL)):
                _error(f"Adapter encoder for type {self.cls} accepts no positional arguments: {self.encoder}")
            if numEncoderArguments > 1:
                if (n := _countArgs(encoder, P.POSITIONAL_ONLY)) > 1:
                    _error(f"Adapter encoder for type {self.cls} has too many positional arguments: {self.encoder}, expected 1, got {n}")

        self.isTotal = issubclass(cls, All)

        if decoder is None:
            if self.isTotal:
                _error('"All" adapter must have a decoder')  # `All` is a special class used to designate a total adapter; it shouldn't get to the user
            decoder = cls

        if not (numDecoderArguments := _countArgs(decoder, P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD, P.VAR_POSITIONAL)):
            _error(f"Adapter decoder for type {cls} accepts no positional arguments: {decoder}")
        if numDecoderArguments > 1:
            n = _countArgs(decoder, P.POSITIONAL_ONLY)
            if self.isTotal or n > 2:
                _error(f"Adapter decoder for type {cls} has too many positional arguments: {decoder}, expected {'1' if self.isTotal else '1 or 2'}, got {n}")

        self.cls = cls
        self.adapterName = fullName(cls)
        self.encoder = encoder  # type: ignore[assignment]
        self.decoder = decoder
        self.numDecoderArguments = numDecoderArguments

        _log(f"Adapter created: {fullName(cls)} => {fullName(encoder) + '()' if encoder else None} => {fullName(decoder) + '()' if decoder else None}")

    async def _encode(self, obj: object) -> _AdapterEncoding:
        if not self.isTotal and not isinstance(obj, self.cls):
            return _DEFAULT
        if self.encoder:  # noqa: SIM108  # pylint: disable=consider-ternary-expression
            value = await self.encoder(obj) if iscoroutinefunction(self.encoder) else self.encoder(obj)
        else:
            value = obj  # Trying to pass the object as is, hoping it would work, like it does for `Enum`s
        return value if self.isTotal else {
            _ADAPTER_MARKER: self.adapterName,  # This is NOT the name of the type of object being encoded, but the name of the adapter that has to be used to decode the object on the other side
            _ADAPTER_FULLNAME: fullName(obj),  # This is the actual name of the type being transferred, but it's rarely used in decoding
            _ADAPTER_VALUE: value,
        }

    async def _decode(self, obj: object, name: str | None = None) -> object:
        if self.numDecoderArguments > 1:
            assert not self.isTotal
            assert name
            # If the decoder accepts two arguments, pass `fullName` as a second argument to give the decoder a chance to reconstruct an object precisely
            return await self.decoder(obj, name) if iscoroutinefunction(self.decoder) else self.decoder(obj, name)  # type: ignore[call-arg]
        return await self.decoder(obj) if iscoroutinefunction(self.decoder) else self.decoder(obj)  # type: ignore[call-arg]

    @classmethod
    async def _findAndEncode(cls, adapters: Mapping[str, Self], obj: object) -> _AdapterEncoding:
        for adapter in adapters.values():
            if (ret := await adapter._encode(obj)) is not _DEFAULT:  # pylint: disable=protected-access
                return ret
        return _DEFAULT

    @classmethod
    async def encodeFirst(cls, obj: object) -> _AdapterEncoding:
        if cls.totalAdapter:
            return await cls.totalAdapter._encode(obj)  # pylint: disable=protected-access
        return await cls._findAndEncode(cls.firstAdapters, obj)

    @classmethod
    async def encodeSecond(cls, obj: object) -> _AdapterEncoding:
        return await cls._findAndEncode(cls.secondAdapters, obj)

    @classmethod
    async def decodeFirst(cls, obj: object) -> object:
        return await cls.totalAdapter._decode(obj) if cls.totalAdapter else _DEFAULT  # pylint: disable=protected-access

    @classmethod
    async def decodeSecond(cls, obj: Mapping[object, object]) -> object:
        if not (adapterName := cast(str | None, obj.get(_ADAPTER_MARKER))):
            return obj
        if adapterName == _START_TIME_ADAPTER:
            return obj  # It's not an adapter, but a similar format used to measure duration of data transfer
        name = cast(str, obj[_ADAPTER_FULLNAME])
        value = obj[_ADAPTER_VALUE]

        if adapter := (cls.firstAdapters.get(adapterName) or cls.secondAdapters.get(adapterName)):  # It must be exact equality check, not `isinstance()`; the idea is to find the adapter that encoded this data – no more, no less
            return await adapter._decode(value, name)  # pylint: disable=protected-access

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
        if cls.totalAdapter or cls.firstAdapters or cls.secondAdapters:
            return  # Initialize adapters only once
        if not (mapping := cast(Mapping[str, Sequence[str]], config.get(_ADAPTERS_SECTION))):
            return  # No adapters defined in config
        for (moduleName, names) in mapping.items():
            module = _importModule(moduleName)
            for adapter in cls._fromSequence(module, names):
                if cls.totalAdapter or (adapter.isTotal and (cls.firstAdapters or cls.secondAdapters)):
                    _error('If "All" adapter is specified, it must be the only adapter specified')
                if adapter.isTotal:
                    cls.totalAdapter = adapter
                    continue
                m = cls.firstAdapters if issubclass(adapter.cls, _Transferable) else cls.secondAdapters
                m[adapter.adapterName] = adapter

@typechecked
async def _to_js(obj: object) -> object:
    """
    Converts an object to a data structure transferable to another process.

    Recurses into collections, uses adapters, and makes every effort possible
    to produce a transferable result.
    """
    if (ret := await _Adapter.encodeFirst(obj)) is not _DEFAULT:
        return ret
    if isinstance(obj, _BasicTypes):
        return obj  # Save it from being converted to `tuple` as an `Iterable`
    if isinstance(obj, Mapping):  # Should be checked before `Iterable`, as `Mapping` is an `Iterable` too
        return await _gatherMap(_to_js, obj)
    if isinstance(obj, Iterable):
        return await _gatherList(_to_js, obj)
    if (ret := await _Adapter.encodeSecond(obj)) is not _DEFAULT:
        return ret
    _warn(f"No adapter found for class {type(obj)}, and transport layer (JavaScript structured clone) would probably not accept it as is, see https://developer.mozilla.org/docs/Web/API/Web_Workers_API/Structured_clone_algorithm")
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
    if (ret := await _Adapter.decodeFirst(obj)) is not _DEFAULT:
        return ret
    if isinstance(obj, _BasicTypes):
        return obj  # Save it from being converted to `tuple` as an `Iterable`
    if not isinstance(obj, Mapping):
        if isinstance(obj, Iterable):
            return tuple(await _gatherList(_to_py, obj))
        return obj  # Not `Iterable`
    # isinstance(obj, Mapping)
    obj = await _gatherMap(_to_py, obj)
    return await _Adapter.decodeSecond(obj)

@typechecked
def exceptionHandler(problem: str,
                     exceptionType: type[BaseException | None] | None = None,
                     exception: BaseException | None = None,
                     traceback: TracebackType | None = None,
                     suffix: str | None = None,
                     displayFunction: _Callable | None = None,
                     **kwargs: Any,
                     ) -> object | str:
    if not exceptionType:
        exceptionType = type(exception)  # May be `None`
    assert exceptionType
    if not traceback and exception:
        traceback = exception.__traceback__
    if not displayFunction:
        displayFunction = partial(_log, file = stderr)
    tracebackStr = "\nTraceback (most recent call last):\n" + '\n'.join(extract_tb(traceback).format()) if traceback else ''
    ret = f"\nERROR: {problem}, type {fullName(exceptionType)}:\n{exception}{tracebackStr}{f"\n\n{suffix}\n" if suffix else ''}"
    return displayFunction(ret, **kwargs) or ret

@typechecked
def improveExceptionHandling(suffix: str | None = None,
                             displayFunction: Callable[..., Any] | None = None,
                             **kwargs: Any,
                             ) -> None:
    stackSummary_format: Final[Callable[[StackSummary], list[str]]] = StackSummary.format
    def patchedFormat(self: StackSummary) -> list[str]:  # Patching global traceback formatting routine
        def showExec(line: str) -> str:
            return '>>' + line[2:] if '<exec>' in line else line  # Make `<exec>` lines more visible
        return [showExec(line.rstrip()) for line in stackSummary_format(self) if line and not line.isspace()]  # Fix Pyodide bug with inserting empty lines into traceback
    StackSummary.format = patchedFormat  # type: ignore[method-assign]

    @typechecked
    def mainExceptionHandler(exceptionType: type[BaseException] | None = None,
                             exception: BaseException | None = None,
                             traceback: TracebackType | None = None) -> None:
        exceptionHandler("Uncaught exception in the main thread",
                         exceptionType, exception, traceback,
                         suffix, displayFunction, **kwargs)

    @typechecked
    def threadExceptionHandler(arg: _ExceptHookArgs) -> None:
        exceptionHandler(f"Uncaught exception in thread {arg.thread and arg.thread.name}",
                         arg.exc_type, arg.exc_value, arg.exc_traceback,
                         suffix, displayFunction, **kwargs)

    @typechecked
    def loopExceptionHandler(_loop: AbstractEventLoop,
                             context: dict[str, Any]) -> None:
        exceptionHandler(f"Uncaught exception in async loop{f":\n{message}" if (message := context.get('message')) else ''}",  # pylint: disable=used-before-assignment
                         None, context.get('exception'), None,
                         suffix, displayFunction, **kwargs)

    # Setting `exceptionHandler()` to handle uncaught exceptions
    sys.excepthook = mainExceptionHandler
    threading.excepthook = threadExceptionHandler
    with suppress(RuntimeError):
        get_running_loop().set_exception_handler(loopExceptionHandler)

    if not RUNNING_IN_WORKER:
        # Now when uncaught exceptions are controlled, errors printed by default into DOM may be hidden
        from pyscript import document  # pylint: disable=import-outside-toplevel
        from js import CSSStyleSheet  # pylint: disable=import-outside-toplevel

        styleSheet = CSSStyleSheet.new()
        styleSheet.replaceSync('.py-error { display: None; }')
        document.adoptedStyleSheets.push(styleSheet)

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
    'exceptionHandler',
    'fullName',
    'improveExceptionHandling',
    'export',
    'systemVersions',
    'typechecked',
)

                       ##
if RUNNING_IN_WORKER:  ##
                       ##

    _NAME = "worker"

    _log("Starting worker")

    _connected = False
    _callingModule: ModuleType | None = None

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
        async def workerSerializedWrapper(*args_: object, **kwargs: object) -> object:
            assert not kwargs  # `kwargs` get passed to workers as the last of `args`, of type `dict`
            assert args_
            args: Sequence[object]
            (args, kwargs) = (args_[:-1], cast(dict[str, object], cast(JsProxy, args_[-1]).to_py()))
            assert isinstance(kwargs, Mapping)
            (args, kwargs) = await gather(_gatherList(_to_py, args), _gatherValues(_to_py, kwargs))
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
                    _ADAPTER_FULLNAME: fullName(ret),
                    _ADAPTER_VALUE: ret,
                    _START_TIME: time(),  # Starting calculating time it would take to transfer the return value to the main thread
                }
            except BaseException as ex:
                _log(f"Exception at {func.__name__}: {ex}", file = stderr)
                raise

        return workerLoggedWrapper

    @typechecked
    def _workerWrap[T = object](func: _CallableOrCoroutine[T]) -> _CoroutineFunction:  # pylint: disable=redefined-outer-name
        """Wraps a function to be exported with all the necessary decorators."""
        return _workerSerialized(_workerLogged(typechecked(func)))

    @typechecked
    def _importSection(section: str) -> Iterator[tuple[str, object]]:
        for (moduleName, names) in config.get(section, {}).items():
            for (name, obj) in zip(names, _importFromModule(moduleName, names), strict = True):
                yield (name, obj)

    @typechecked
    def _imports(targetModule: ModuleType | None = None) -> None:
        target = globals() if targetModule is None else targetModule.globals()
        for (name, obj) in _importSection(_IMPORTS_SECTION):
            target[name] = obj

    @typechecked  # Public API
    async def anyCall(funcNameAndArgs: _AnyCallArg) -> object:
        """
        Special service function that can be exported the same way as any other function.

        After this function is exported, it may be used to call any other function
        in the same worker, like this: `worker.anyCall.funcName(*args, **kwargs)`.
        Anything, including `eval()` may be called this way.

        So, this function should only be exported when the main thread
        is completely controlled.

        Another benefit of this function is that all the arguments passed
        will be passed to a worker as a single argument. If that argument
        is encoded into a single `Buffer`, it may be much faster than
        encoding each argument individually, particularly if lists and maps
        are involved.

        One of the great ways to encode an arbitrary object as a `Buffer`
        is `pickle`. So, this function works the best with an adapter like this:

        [adapters]
        "workerlib" = ["All", "pickle", "pickle"]

        One exported function, one adapter – and everything works, and very fast.
        """
        funcName = funcNameAndArgs['funcName']
        args = funcNameAndArgs.get('args', ())
        kwargs = funcNameAndArgs.get('kwargs', {})
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
        global _connected, _NAME  # noqa: PLW0603  # pylint: disable=global-statement
        if _connected:
            _error("Connection to the main thread was already established earlier")
        if request != _CONNECT_REQUEST:
            _error(f"Connection to main thread is misconfigured, can't continue: {type(request)}({request!r})")
        _log(f'Assigned name "{name}"')
        _NAME = name
        config['attributes'] = attributes
        _log("Tag:", attributes.get(_TAG_ATTR, "''"))
        assert __export__, __export__
        ret = tuple(chain((_CONNECT_RESPONSE,), (funcName for funcName in __export__ if funcName != _connectFromMain.__name__)))
        _connected = True
        _log(f"Connected to the main thread, sync_main_only = {config.get('sync_main_only', False)}, ready for requests")
        return ret

    @typechecked  # Public API
    def export(*functions: _CallableOrCoroutine) -> None:
        """
        Must be called by the importing module with the list of functions to be exported.

        Wraps them with necessary decorators and puts them
        into the `__export__` list of the calling module.

        May be called multiple times until a connection from the main thread is made.
        """
        global _callingModule  # noqa: PLW0603  # pylint: disable=global-statement

        if _connected:
            _error("Connection to the main thread already established, can't export new functions after that")

        firstExport = not _callingModule

        if not _callingModule:  # Do this only once
            currentFrame = currentframe()
            assert currentFrame
            _callingModule = getmodule(currentFrame.f_back)  # Calling module

            assert _callingModule
            _imports(_callingModule)  # Doing imports this late so that potential errors would not interrupt startup too early
            _Adapter.fromConfig()

            _callingModule._connectFromMain = _connectFromMain  # type: ignore[attr-defined]  # pylint: disable=protected-access
            _callingModule.__export__ = (_connectFromMain.__name__,)  # type: ignore[attr-defined]

        exportNames: list[str] = []
        for func in functions:
            setattr(_callingModule, func.__name__, _workerWrap(func))
            exportNames.append(func.__name__)

        exportNamesTuple = tuple(chain(_callingModule.__export__, exportNames))
        _callingModule.__export__ = exportNamesTuple  # type: ignore[attr-defined]
        modules[__name__].__export__ = exportNamesTuple  # type: ignore[attr-defined]  # This is only needed to make `_connectFromMain()` code simple and universal for both `export()` and `_autoExport()`
        _log("Providing functions:", ', '.join(name for name in exportNamesTuple if name != _connectFromMain.__name__))
        if firstExport:
            _log("Started worker, waiting for connection from the main thread")

    @typechecked
    async def _autoExport() -> None:
        """
        Gets called automatically if this module itself is loaded as a worker.

        Wraps and exports functions listed in the config.
        """
        global _callingModule  # noqa: PLW0603  # pylint: disable=global-statement
        assert not _connected
        assert not _callingModule
        _callingModule = modules[__name__]

        _imports()
        _Adapter.fromConfig()

        exportNames = [_connectFromMain.__name__,]
        for (funcName, func) in _importSection(_EXPORTS_SECTION):
            if not callable(func):
                _error(f'Bad exported function "{funcName}", must be a class/function/coroutine, got {type(func)}')
            setattr(_callingModule, funcName, _workerWrap(func))
            exportNames.append(funcName)

        if len(exportNames) < 2:
            _warn("No functions found to export, check `[exports]` section in the config")

        _callingModule.__export__ = tuple(exportNames)  # type: ignore[attr-defined]
        _log("Providing functions:", ', '.join(name for name in exportNames if name != _connectFromMain.__name__))
        _log("Started worker, waiting for connection from the main thread")

    if __name__ == '__main__':
        # If this module itself is used as a worker, it imports functions mentioned in the config and exports them automatically
        del export
        __all__ = ()
        __export__ = ()
        improveExceptionHandling()
        create_task(_autoExport())  # Make sure this module gets fully loaded before `_autoExport()` is called – to avoid circular import errors if anything being imported tries to import something from `workerlib`, e.g., `anyCall`
    else:
        # If a user is importing this module into a worker, they MUST call `export()` explicitly
        del _autoExport
        __all__ = (
            'anyCall',
            'diagnostics',
            'elapsedTime',
            'exceptionHandler',
            'fullName',
            'improveExceptionHandling',
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
    def _mainSerialized[T = object](func: JsProxy, looksLike: _CallableOrCoroutine[T] | str) -> _CoroutineFunction[T]:  # pylint: disable=redefined-outer-name
        """
        Used to wrap worker functions.

        Must be the innermost decorator, as it implements JS transport for calls to another process.

        Uses `_to_js()` to convert arguments, and `_to_py()` to convert returned value.
        Coordinates with `@_workerSerialized`.
        """
        @typechecked
        async def mainSerializedWrapper(*args_: object, **kwargs: object) -> T:
            assert isinstance(func, JsProxy), type(func)
            (args, kwargs) = await gather(_gatherList(_to_js, args_), _gatherValues(_to_js, kwargs))
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
    async def connectToWorker(workerName: str | None = None) -> Worker:  # ToDo: Refactor it somehow to be compatible with other languages
        """  # ToDo: connectToPythonWorker() ? connectTo_Worker() ?
        Connects to a named worker with the specified `name`.
        That worker would use this `name` as a prefix to console logging messages.

        If `name` is not specified, looks for the
        `<script type="py" worker name="…">` tag in the page.
        If no such a tag is found, or more than one is found,
        raises an exception.
        """
        if workerName:
            elements = page.find(_PY_SCRIPT_WORKER + f'[name={workerName}]')
            if not (element := elements[0] if elements else None):
                _warn(f'Could not find a worker named "{workerName}" in DOM')
        else:
            if not (elements := page.find(_PY_SCRIPT_WORKER + '[name]')):
                pyScripts = '\n'.join(tag.outerHTML for tag in page.find(_PY_SCRIPT))
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
        'exceptionHandler',
        'fullName',
        'improveExceptionHandling',
        'systemVersions',
        'typechecked',
    )

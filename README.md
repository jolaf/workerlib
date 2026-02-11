# workerlib

`workerlib` is a pure-[Python](https://python.org)
library that provides transparent Python-to-Python calls
from the main thread to workers in a
[PyScript](https://pyscript.net)/[Pyodide](https://pyodide.org) environment.

`workerlib` is written on Python 3.13 and thus is not usable
with [MicroPython](https://micropython.org);
instead, Pyodide must be used for both main thread and workers.

## Background

You may skip this introduction and jump right to
[usage instructions](#usage-instructions)
if you already know what PyScript and
[Web Workers](https://developer.mozilla.org/docs/Web/API/Web_Workers_API) are.

Modern browsers provide an elaborate and capable API ecosystem
that traditionally was only accessible via
[JavaScript](https://developer.mozilla.org/docs/Web/JavaScript).

But with the introduction of
[Wasm](https://webassembly.org), supported by
[LLVM](https://llvm.org), and
[Emscripten](https://emscripten.org),
that ecosystem became available to other languages.

[PyScript](https://pyscript.net) is an open source
[Python](https://python.org)-in-browser platform,
based on
[Pyodide](https://pyodide.org), Emscripten, and Wasm,
with a mission of bringing Python everywhere.

One of the important parts of the modern browser ecosystem is
[Web Workers](https://developer.mozilla.org/docs/Web/API/Web_Workers_API) —
the ability to have separate background processes that can do
heavy calculations without blocking the main thread
and thus making the browser page unresponsive.

Being JavaScript-based, this mechanism is essentially cross-language –
the main thread may be running one language, e.g., JavaScript or Python,
and workers may be running other languages, like Python or
[Lua](https://lua.org),
and the browser would provide the cross-process call capabilities.

## PyScript Web Workers interface

For the reasons mentioned above, PyScript
[interface](https://docs.pyscript.net/2026.1.1/user-guide/workers/)
for Web Workers is mostly JavaScript-oriented.
It assumes the process on the other side of the call
may not be running Python but JavaScript or any other language.

This language independence makes PyScript extremely flexible
and compatible, and yet creates two important complications
on a way to transparent Python-to-Python calls.

### Complication #1: `JsProxy` instead of Python objects

When a Python function is being called from another process
and is passed an argument, it expects some Python object but gets a
[`JsProxy`](https://pyodide.org/en/stable/usage/api/python-api/ffi.html#pyodide.ffi.JsProxy)
instance instead. There is a way to extract an actual Python object
from that proxy, but it requires writing some additional code,
which makes the whole interprocess communication cumbersome.
The same thing happens when a function call to another process
returns a value to the Python caller — it also comes back as a `JsProxy`.

This situation is particularly painful if both processes,
main and worker, are running Python!
You're calling a Python function from Python code,
and you're getting some `JsProxy` instead of expected value!

In particular this means that any call to a function in another process
must be wrapped with some code.
And any existing functions that need to be called from another process cannot be
[exported](https://docs.pyscript.net/2026.1.1/user-guide/workers/#defining-workers)
as is but also need some wrappers to parse arguments for them.

That's, well, non-pythonic.
And `workerlib` is called upon to
[fix this](#usage-instructions).

### Complication #2: Limited data types

JavaScript Web Workers interface only allows a
[limited number](https://developer.mozilla.org/docs/Web/API/Web_Workers_API/Structured_clone_algorithm)
of JavaScript data types through.
And most of them are JavaScript-specific
and therefore useless for Python-to-Python calls.
What's left are basic types and simple collections.

That's what PyScript uses to convert Python objects to JavaScript objects
before the call, and converting them back to `JsProxy` instances wrapping
the Python objects on the other side. In fact, the only Python types that
[can go through](https://docs.pyscript.net/2026.1.1/user-guide/workers/#understanding-limitations)
to another process are:

* primitive types (`int`, `float`, `bool`, `None`)
* `str`ings and `bytes`
* basic collections (`list`, `dict`).

Other types are converted automatically as follows:

* `tuple`, `set`, `Iterable`, `Sequence` -> `list`
* `bytes`, `bytearray`, `Buffer` -> `memoryview`

So, if you send a `tuple`, the other side
would receive a `JsProxy` wrapping a `list`.

Collections may be nested, e.g., you may use `list[dict[str, Buffer]]`
as a function argument or return type.

The following typing annotations are not exactly correct,
so treat them as just an illustration:

```Python3
from collections.abc import Buffer, Iterable, Mapping
from numbers import Real
from pyodide.ffi import JsProxy

type _S = Real | str | None  # Note: `Real` includes `Enum`s!
type Sendable = _S | Buffer | Iterable[Sendable] | Mapping[_S, Sendable]
type _K = int | float | bool | str | None 
type _V = _K | memoryview | list[_V] | dict[_K, _V]
type Receivable = JsProxy[_V]
```

Functions, classes, objects, [`NumPy`](https://numpy.org) arrays etc.
can't be converted automatically and would result in exception if you
try to use them in a worker call or return value.

So if you need to send some object to another side, you have to convert
it to some combination of "sendable" types and then convert back
to the Python object you need on the other side of the call.

In many cases this two-way conversion is possible,
but it cannot be generalized –
instead, this task has to be solved for each class separately.

`workerlib` tries to help with this complication
[as much as it can](#adapters).

## Usage instructions

`workerlib` is designed to be imported into both main and worker Python modules.

Potentially, it can be used by only one side to ease communication
with other languages, too, but this was never tried.

`workerlib` provides a set of
[decorators](https://docs.python.org/3/glossary.html#term-decorator),
for both sides, for functions
[exported](https://docs.pyscript.net/2026.1.1/user-guide/workers/#defining-workers)
by worker, so that main thread code can call worker functions seamlessly,
without visible code overhead.
Those decorators would silently do most of the
Python <–> `JsProxy` conversions for you.

### Disclaimers

`workerlib` is just a decorative sugar.
It's not a magic bullet, it won't help you transfer objects
that are impossible to transfer between processes –
unless you convert those objects to something that can be transferred yourself.

`workerlib` only provides for calls from the main thread to workers,
NOT vice versa.

`workerlib` is a single pure-Python module without external dependencies.

`workerlib` does not need DOM access and may be used in
[`sync_main_only = true`](https://docs.pyscript.net/2026.1.1/user-guide/configuration/#sync_main_only)
workers.
And no, it won't provide workers with DOM access if
[CORS](https://docs.pyscript.net/2026.1.1/user-guide/workers/#understanding-the-problem)
is not configured appropriately.

Both synchronous and `async` functions can be exported from workers.
In both cases, calling them from the main thread is done using `await`.

### Worker configuration, Option #1

If you have some already existing Python modules with functions that you want
to run in a worker and be callable from the main thread, this is your way to go.

`index.html`:

```HTML
<script type="py" worker name="YourWorkerName" src="./workerlib.py" config="./worker.toml"></script>
```

`worker.toml`:

```TOML
[files]
"./YourModule1.py" = ""
"./YourModule2.py" = ""

[exports]
"YourModule1" = ["func1", "func2", …, "funcK"]
"YourModule2" = ["func3", "func4", …, "funcN"]
```

This is it. `workerlib` would start as a worker,
import your modules,
`__export__` the functions you've listed,
and wrap them with decorators providing argument and return values conversions.
Your existing modules would not need any adjustments.

If you prefer using
[`pyscript.create_named_worker()`](https://docs.pyscript.net/2026.1.1/api/workers/#pyscript.workers.create_named_worker),
that would also work, with the same arguments as `<script>` tag above.

### Worker configuration, Option #2

If you're writing some complicated worker module
and would like to simplify calls to its functions, this is your way.

`index.html`:

```HTML
<script type="py" worker name="YourWorkerName" src="./YourModule1.py" config="./worker.toml"></script>
```

`worker.toml`:

```TOML
experimental_remote_packages = true

packages = [
    "https://jolaf.github.io/workerlib/workerlib.toml",
]

[files]
"./YourModule2.py" = ""
```

`YourModule1.py`:

```Python3
from workerlib import export
from YourModule2 import func3, func4, …, funcN

# Define sync or async `func1`, `func2`, …, `funcK` here

export(func1, func2, …, funcK)
export(func3, func4, …, funcN)
```

This is it. In other aspects, everything would work exactly as in
[Option #1](#worker-configuration-option-1).

### Main configuration

Whatever your worker configuration is, in the main thread you go like this:

`main.py`:

```Python3
from workerlib import connectToWorker, Worker
yourWorker: Worker = await connectToWorker()
ret = await yourWorker.func1(arg1, arg2, *args, kwarg1 = value, **kwargs)
```

If you only have one `<script type="py" worker name="…">` tag in your page,
`connectToWorker()` would find it, extract the `name` attribute, and
[connect](https://docs.pyscript.net/2026.1.1/user-guide/workers/#accessing-workers)
to the worker by that name.

If you have multiple
[named workers](https://docs.pyscript.net/2026.1.1/api/workers/)
in your page, you should specify the worker name explicitly:

```Python3
yourWorker = await connectToWorker("YourWorkerName")
```

If you ever need to access the original PyScript
[worker](https://docs.pyscript.net/2026.1.1/api/workers/)
object, it's accessible, as a `JsProxy` of course,
at `yourWorker.worker` field.

### Adapters

Everything described above would only successfully transfer the
[basic data types](#complication-2-limited-data-types).
Any class objects, including those you've created yourself,
would not pass through and would cause the error like this if you try:

> `JsException: DataCloneError: Failed to execute 'postMessage' on 'MessagePort': [object Object] could not be cloned.`

That's not a problem of this library, and not a problem of PyScript or Pyodide –
it's an inherent limitation of the browser ecosystem.

Transfer of data from one process to another uses
[structured cloning](https://developer.mozilla.org/docs/Web/API/Web_Workers_API/Structured_clone_algorithm)
and thus is
[strictly limited](#complication-2-limited-data-types)
to basic data types and structures.

Anything beyond those may be transferred only if it's converted
to basic data types and structures first.
And here `workerlib` has you covered!
For each class of objects you need to transfer between processes, go like this:

```TOML
[files]
"./moduleName.py" = ""

[adapters]
"moduleName" = ["TypeName", "typeEncoder", "typeDecoder"]
```

Make sure this section is present in BOTH main and worker's configurations.

Here `TypeName` is a type that you want to transfer.

`moduleName` is a name of the module that contains
`TypeName`, `typeEncoder` and `typeDecoder`.
Make sure to also list that module in the `[files]` section in BOTH configs.

`typeEncoder` is a name of the function, sync or `async`,
converting an object of type `TypeName` into some Python type that
[can be transferred](#complication-2-limited-data-types)
to another process.

`typeDecoder` is a name of the function, also sync or `async`,
that can understand the data structure created by the `typeEncoder()`
and construct a `TypeName` instance from that data.
The input this function would receive would be the same
that `typeEncoder()` created on the other side.

So approximate signatures would be like this:

`moduleName.py`:

```Python3
class TypeName: ...
def typeEncoder(obj: TypeName) -> Sendable: ...
def typeDecoder(data: Sendable) -> TypeName: ...
```

Note that `JsProxy` is never mentioned in these signatures –
`workerlib` takes care of that for you!

#### Note 1. Type, encoder, and decoder in the same module

If `TypeName` is some third party class,
you may need to import and reexport it from your module
containing `typeEncoder()` and `typeDecoder()`.

#### Note 2. Handling subclasses

`workerlib` does not transfer the name of the actual type
of the object passed into `typeEncoder()`. So if you have an adapter
for `Superclass`, and the data you're trying to get to another process
contains an instance of `Subclass`, on receiving side a `Superclass`
instance would be constructed. Unless `superclassEncoder()`
makes an effort to send actual type too, and `superclassDecoder()`
makes an effort to import `Subclass` and construct `Subclass` instance.

This limitation is due to the fact that `Subclass` may be located in
a different module than `Superclass` and there's no reliable way to find
and import it.

So if you need `Subclass` instances to be decoded as `Subclass` instances
in another process, make sure to create a separate adapter for `Subclass`.

#### Note 3. Adapter ordering

`superclassEncoder()` function would be called for any object
that passes `isinstance(obj, Superclass)` check.
So if you want to have different encoders/decoders for
`Superclass` and `Subclass`, make sure to list the adapter for `Subclass`
higher in the config, because objects are checked against adapters
in the order adapters are listed in the config.

#### Note 4. Multiple adapters from the same module

`[adapters]` is a map, it can't have duplicate keys.

So if you need to create multiple adapters from the same module,
do it like this:

```TOML
[adapters]
"moduleName" = [
    ["Type1Name", "type1Encoder", "type1Decoder"],
    ["Type2Name", "type2Encoder", "type2Decoder"],
]
```

#### Note 5. Use `memoryview` for faster calls

If you're thinking which "sendable" data structure to use in your encoder,
choose `memoryview`. Or at least something that is a `Buffer` –
it gets converted into a `memoryview` on send, and arrives on another side
also as a `memoryview`.

PyScript, Pyodide, and the underlying JavaScript/Wasm platform would do
their best to convert `memoryview` instance in one process into a `memoryview`
instance in another process with **zero copying**, which would make
the data "transfer" effectively instantaneous (**O(1)** complexity).

Sometimes zero copying is not applicable, but anyway as few copies will be made
during the transfer as possible. So `memoryview` would be the fastest way
to transfer data from one process to another in any case.

#### Example adapter for `PIL.Image`

Here's the possible operational adapter configuration for
[`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html)
module:

```TOML
packages = ["Pillow",]

[files]
"./imageProcessor.py" = ""

[adapters]
"imageProcessor" = [
    ["Image", "imageRawEncoder", "imageFromBufferDecoder"],
    ["Transpose", "None", "None"],
]
```

`imageProcessor.py`:

```Python3
from collections.abc import Iterable
from typing import Any

from PIL.Image import Image, Transpose, frombuffer

def imageRawEncoder(image: Image) -> tuple[str, tuple[int, int], bytes, str, str, int, int]:
    return (image.mode, image.size, image.tobytes(), 'raw', image.mode, 0, 1)

def imageFromBufferDecoder(data: Iterable[Any]) -> Image:
    return frombuffer(*data)
```

See
[`frombuffer()` documentation](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.frombuffer)
for details.

Unfortunately, not all classes are that easy to create encoders/decoders for.

Some types, for example
[`Enum`](https://docs.python.org/3/library/enum.html#enum.Enum)s
like
[`Transpose`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Transpose)
in the example above, may not require any encoders/decoders.

PyScript converts `Enum` instances to their values automatically when sending.

And if the decoder is not specified, `workerlib` would try to use
the class itself (i.e., its constructor) as one.

In situations like this you may specify `"None"` as encoder/decoder.

## API Reference

`workerlib` provides the following APIs:

### In the main thread

##### `class Worker`

Simple class wrapping PyScript `JsProxy` of a `worker` object
and providing worker functions to be called.

See [Main configuration](#main-configuration) for usage.

##### `async def connectToWorker(workerName: str | None = None) -> Worker`

Connects to the worker with the specified name
or to the only named worker in your page if the name is not specified.

See [Main configuration](#main-configuration) for usage.

### In a worker

##### `def export(*functions: CallableOrCoroutine) -> None`

Must be imported from `workerlib` and called
to make `functions` available to the main thread.

If `workerlib.py` is mentioned explicitly in a
`<script type="py" worker name="…">` tag
or a `pyscript.create_named_worker()` call, no explicit exports are needed;
everything mentioned in the config would be exported automatically.

### Auxiliary stuff in both the main and workers

Besides the described essential functionality,
`workerlib` also has some useful internals
that could be imported from it and used if needed:

##### `diagnostics: Sequence[str]`

Sequence of strings with detailed diagnostics of system components
that can be nicely printed to the browser console, line by line.
`workerlib` does exactly that if started as a worker.

##### `def elapsedTime(startTime: float | int) -> str`

Returns a diagnostic string mentioning time,
in milliseconds or seconds, since `startTime`.

Use [`time.time()`](https://docs.python.org/3/library/time.html#time.time)
to initialize `startTime` beforehand.

##### `systemVersions: Mapping[str, str]`

Lists detected runtime versions of system components
(PyScript, Pyodide, Python, Emscripten).

May be useful for diagnostic purposes.

##### `@typechecked`

Decorator that may be used for runtime type checking
functions or the whole classes.

If [`"beartype"`](https://beartype.readthedocs.io)
is included in config `[packages]`,
its [`@beartype`](https://beartype.readthedocs.io/en/stable/api_decor/#beartype-decorator-api)
decorator is used. Otherwise, `@typechecked` does nothing.

## Under the hood

If you want to understand how this is working,
the main pieces of [code](./workerlib.py)
you should look into are
`_mainSerialized()` and `_workerSerialized()`
decorators that wrap the exported functions
in the main thread and workers respectively.

Also check `_to_js()` and `_to_py()` converter coroutines those decorators use.

## Please ask and comment

If you have any questions, comments, or suggestions on this library, please
[file an issue](https://github.com/jolaf/workerlib/issues)
or contact me directly
via [Telegram](https://t.me/jolaf)
or [e-mail](mailto:vmzakhar@gmail.com).

Thanks! Have nice PyScripting!

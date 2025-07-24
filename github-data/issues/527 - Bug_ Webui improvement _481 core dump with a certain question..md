### 🐛 [#527](https://github.com/ikawrakow/ik_llama.cpp/issues/527) - Bug: Webui improvement [#481](https://github.com/ikawrakow/ik_llama.cpp/issues/481) core dump with a certain question.

| **Author** | `ycat3` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-14 |
| **Updated** | 2025-06-14 |

---

#### Description

### What happened?

i asked a certain question in Japanese, then fatal error, core dump.
Both new and legacy Webui causes fatal error.
Another question in Japanese works fine.
unsloth/UD-Q3_K_XL
Probably UTF-8 code problem.
llama.cpp/llama-server works with this Japanese question.
The following question means "Tell me about Shostakovich’s symphony 11"
-------------------------------------------------------------------------------------------------------------
User: ショスタコーヴィチの交響曲第11番について教えてください。
--------------------------------------------------------------------------------------------------------------
/home/mycat7/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:630: Fatal error
/home/mycat7/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:630: Fatal error
/home/mycat7/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:630: Fatal error
/home/mycat7/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:630: Fatal error
/home/mycat7/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:630: Fatal error
/home/mycat7/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:630: Fatal error
/home/mycat7/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:630: Fatal error
/home/mycat7/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:630: Fatal error
  File "<string>", line 1
  File "<string>", line 1
    import sys
    import sys
SyntaxError: source code cannot contain null bytes
SyntaxError: source code cannot contain null bytes
Error in sys.excepthook:
Error in sys.excepthook:
Traceback (most recent call last):
Traceback (most recent call last):
  File "/usr/lib/python3.12/typing.py", line 897, in __init__
  File "/usr/lib/python3.12/typing.py", line 897, in __init__
  File "<string>", line 1
    import sys
SyntaxError: source code cannot contain null bytes
Error in sys.excepthook:
Traceback (most recent call last):
  File "/usr/lib/python3.12/typing.py", line 897, in __init__
  File "<string>", line 1
    import sys
SyntaxError: source code cannot contain null bytes
  File "<string>", line 1
Error in sys.excepthook:
    import sys
Traceback (most recent call last):
  File "<string>", line 1
SyntaxError: source code cannot contain null bytes
  File "/usr/lib/python3.12/typing.py", line 897, in __init__
    import sys
Error in sys.excepthook:
SyntaxError: source code cannot contain null bytes
Traceback (most recent call last):
Error in sys.excepthook:
  File "/usr/lib/python3.12/typing.py", line 897, in __init__
Traceback (most recent call last):
  File "/usr/lib/python3.12/typing.py", line 897, in __init__
  File "<string>", line 1
    import sys
SyntaxError: source code cannot contain null bytes
Error in sys.excepthook:
Traceback (most recent call last):
  File "/usr/lib/python3.12/typing.py", line 897, in __init__
    code = compile(arg_to_compile, '<string>', 'eval')
    code = compile(arg_to_compile, '<string>', 'eval')
    code = compile(arg_to_compile, '<string>', 'eval')
    code = compile(arg_to_compile, '<string>', 'eval')
    code = compile(arg_to_compile, '<string>', 'eval')
    code = compile(arg_to_compile, '<string>', 'eval')
    code = compile(arg_to_compile, '<string>', 'eval')
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 1
  File "<string>", line 1
  File "<string>", line 1
  File "<string>", line 1
    SourcesList
    SourcesList
SyntaxError: source code cannot contain null bytes
  File "<string>", line 1
    SourcesList
    SourcesList

SyntaxError: source code cannot contain null bytes
    SourcesList
SyntaxError: source code cannot contain null bytes
SyntaxError: source code cannot contain null bytes
During handling of the above exception, another exception occurred:
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SyntaxError: source code cannot contain null bytes



During handling of the above exception, another exception occurred:

  File "<string>", line 1
During handling of the above exception, another exception occurred:
During handling of the above exception, another exception occurred:
During handling of the above exception, another exception occurred:
Traceback (most recent call last):

    SourcesList



SyntaxError: source code cannot contain null bytes
  File "/usr/lib/python3/dist-packages/apport_python_hook.py", line 228, in partial_apport_excepthook
Traceback (most recent call last):
Traceback (most recent call last):
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Traceback (most recent call last):
Traceback (most recent call last):

  File "/usr/lib/python3/dist-packages/apport_python_hook.py", line 228, in partial_apport_excepthook
  File "/usr/lib/python3/dist-packages/apport_python_hook.py", line 228, in partial_apport_excepthook
  File "<string>", line 1
  File "/usr/lib/python3/dist-packages/apport_python_hook.py", line 228, in partial_apport_excepthook
  File "/usr/lib/python3/dist-packages/apport_python_hook.py", line 228, in partial_apport_excepthook
During handling of the above exception, another exception occurred:
    SourcesList

SyntaxError: source code cannot contain null bytes
Traceback (most recent call last):

During handling of the above exception, another exception occurred:
  File "/usr/lib/python3/dist-packages/apport_python_hook.py", line 228, in partial_apport_excepthook

Traceback (most recent call last):
  File "/usr/lib/python3/dist-packages/apport_python_hook.py", line 228, in partial_apport_excepthook
  File "<string>", line 1
    import sys
SyntaxError: source code cannot contain null bytes
Error in sys.excepthook:
Traceback (most recent call last):
  File "/usr/lib/python3.12/typing.py", line 897, in __init__
    return apport_excepthook(binary, exc_type, exc_obj, exc_tb)
    return apport_excepthook(binary, exc_type, exc_obj, exc_tb)
    return apport_excepthook(binary, exc_type, exc_obj, exc_tb)
    return apport_excepthook(binary, exc_type, exc_obj, exc_tb)
    return apport_excepthook(binary, exc_type, exc_obj, exc_tb)
    return apport_excepthook(binary, exc_type, exc_obj, exc_tb)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/apport_python_hook.py", line 66, in apport_excepthook
    return apport_excepthook(binary, exc_type, exc_obj, exc_tb)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/apport_python_hook.py", line 66, in apport_excepthook
  File "/usr/lib/python3/dist-packages/apport_python_hook.py", line 66, in apport_excepthook
  File "/usr/lib/python3/dist-packages/apport_python_hook.py", line 66, in apport_excepthook
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/apport_python_hook.py", line 66, in apport_excepthook
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/apport_python_hook.py", line 66, in apport_excepthook
    import apport.report
    code = compile(arg_to_compile, '<string>', 'eval')
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/apport/__init__.py", line 7, in <module>
  File "/usr/lib/python3/dist-packages/apport_python_hook.py", line 66, in apport_excepthook
    import apport.report
    import apport.report
    import apport.report
    import apport.report
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/apport/__init__.py", line 7, in <module>
  File "/usr/lib/python3/dist-packages/apport/__init__.py", line 7, in <module>
  File "<string>", line 1
  File "/usr/lib/python3/dist-packages/apport/__init__.py", line 7, in <module>
  File "/usr/lib/python3/dist-packages/apport/__init__.py", line 7, in <module>
    SourcesList
    import apport.report
SyntaxError: source code cannot contain null bytes

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3/dist-packages/apport/__init__.py", line 7, in <module>
  File "/usr/lib/python3/dist-packages/apport_python_hook.py", line 228, in partial_apport_excepthook
    import apport.report
  File "/usr/lib/python3/dist-packages/apport/__init__.py", line 7, in <module>
    from apport.packaging_impl import impl as packaging
    from apport.packaging_impl import impl as packaging
    from apport.packaging_impl import impl as packaging
    from apport.packaging_impl import impl as packaging
    from apport.packaging_impl import impl as packaging
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/__init__.py", line 33, in <module>
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/__init__.py", line 33, in <module>
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/__init__.py", line 33, in <module>
    from apport.packaging_impl import impl as packaging
    from apport.packaging_impl import impl as packaging
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/__init__.py", line 33, in <module>
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/__init__.py", line 33, in <module>
    return apport_excepthook(binary, exc_type, exc_obj, exc_tb)
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/__init__.py", line 33, in <module>
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/__init__.py", line 33, in <module>
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/apport_python_hook.py", line 66, in apport_excepthook
    import apport.report
  File "/usr/lib/python3/dist-packages/apport/__init__.py", line 7, in <module>
    from apport.packaging_impl import impl as packaging
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/__init__.py", line 33, in <module>
    impl = load_packaging_implementation()
    impl = load_packaging_implementation()
    impl = load_packaging_implementation()
    impl = load_packaging_implementation()
    impl = load_packaging_implementation()
    impl = load_packaging_implementation()
    impl = load_packaging_implementation()
    impl = load_packaging_implementation()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/__init__.py", line 27, in load_packaging_implementation
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/__init__.py", line 27, in load_packaging_implementation
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/__init__.py", line 27, in load_packaging_implementation
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/__init__.py", line 27, in load_packaging_implementation
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/__init__.py", line 27, in load_packaging_implementation
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/__init__.py", line 27, in load_packaging_implementation
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/__init__.py", line 27, in load_packaging_implementation
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/__init__.py", line 27, in load_packaging_implementation
    module = importlib.import_module(
    module = importlib.import_module(
    module = importlib.import_module(
             ^^^^^^^^^^^^^^^^^^^^^^^^
             ^^^^^^^^^^^^^^^^^^^^^^^^
    module = importlib.import_module(
             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/importlib/__init__.py", line 90, in import_module
  File "/usr/lib/python3.12/importlib/__init__.py", line 90, in import_module
    module = importlib.import_module(
  File "/usr/lib/python3.12/importlib/__init__.py", line 90, in import_module
             ^^^^^^^^^^^^^^^^^^^^^^^^
             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/importlib/__init__.py", line 90, in import_module
  File "/usr/lib/python3.12/importlib/__init__.py", line 90, in import_module
    module = importlib.import_module(
    module = importlib.import_module(
    module = importlib.import_module(
             ^^^^^^^^^^^^^^^^^^^^^^^^
             ^^^^^^^^^^^^^^^^^^^^^^^^
             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/importlib/__init__.py", line 90, in import_module
  File "/usr/lib/python3.12/importlib/__init__.py", line 90, in import_module
  File "/usr/lib/python3.12/importlib/__init__.py", line 90, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
    return _bootstrap._gcd_import(name[level:], package, level)
    return _bootstrap._gcd_import(name[level:], package, level)
    return _bootstrap._gcd_import(name[level:], package, level)
    return _bootstrap._gcd_import(name[level:], package, level)
    return _bootstrap._gcd_import(name[level:], package, level)
    return _bootstrap._gcd_import(name[level:], package, level)
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/apt_dpkg.py", line 51, in <module>
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/apt_dpkg.py", line 51, in <module>
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/apt_dpkg.py", line 51, in <module>
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/apt_dpkg.py", line 51, in <module>
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/apt_dpkg.py", line 51, in <module>
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/apt_dpkg.py", line 51, in <module>
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/apt_dpkg.py", line 51, in <module>
  File "/usr/lib/python3/dist-packages/apport/packaging_impl/apt_dpkg.py", line 51, in <module>
    import aptsources.sourceslist as apt_sl
    import aptsources.sourceslist as apt_sl
    import aptsources.sourceslist as apt_sl
    import aptsources.sourceslist as apt_sl
    import aptsources.sourceslist as apt_sl
  File "/usr/lib/python3/dist-packages/aptsources/sourceslist.py", line 158, in <module>
  File "/usr/lib/python3/dist-packages/aptsources/sourceslist.py", line 158, in <module>
    import aptsources.sourceslist as apt_sl
    import aptsources.sourceslist as apt_sl
  File "/usr/lib/python3/dist-packages/aptsources/sourceslist.py", line 158, in <module>
  File "/usr/lib/python3/dist-packages/aptsources/sourceslist.py", line 158, in <module>
  File "/usr/lib/python3/dist-packages/aptsources/sourceslist.py", line 158, in <module>
    import aptsources.sourceslist as apt_sl
  File "/usr/lib/python3/dist-packages/aptsources/sourceslist.py", line 158, in <module>
  File "/usr/lib/python3/dist-packages/aptsources/sourceslist.py", line 158, in <module>
  File "/usr/lib/python3/dist-packages/aptsources/sourceslist.py", line 158, in <module>
    class Deb822SourceEntry:
    class Deb822SourceEntry:
    class Deb822SourceEntry:
  File "/usr/lib/python3/dist-packages/aptsources/sourceslist.py", line 163, in Deb822SourceEntry
    class Deb822SourceEntry:
    class Deb822SourceEntry:
  File "/usr/lib/python3/dist-packages/aptsources/sourceslist.py", line 163, in Deb822SourceEntry
  File "/usr/lib/python3/dist-packages/aptsources/sourceslist.py", line 163, in Deb822SourceEntry
  File "/usr/lib/python3/dist-packages/aptsources/sourceslist.py", line 163, in Deb822SourceEntry
  File "/usr/lib/python3/dist-packages/aptsources/sourceslist.py", line 163, in Deb822SourceEntry
    class Deb822SourceEntry:
    class Deb822SourceEntry:
  File "/usr/lib/python3/dist-packages/aptsources/sourceslist.py", line 163, in Deb822SourceEntry
    class Deb822SourceEntry:
  File "/usr/lib/python3/dist-packages/aptsources/sourceslist.py", line 163, in Deb822SourceEntry
  File "/usr/lib/python3/dist-packages/aptsources/sourceslist.py", line 163, in Deb822SourceEntry
    list: Optional["SourcesList"] = None,
    list: Optional["SourcesList"] = None,
    list: Optional["SourcesList"] = None,
    list: Optional["SourcesList"] = None,
          ~~~~~~~~^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 395, in inner
          ~~~~~~~~^^^^^^^^^^^^^^^
          ~~~~~~~~^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 395, in inner
    list: Optional["SourcesList"] = None,
          ~~~~~~~~^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 395, in inner
    list: Optional["SourcesList"] = None,
  File "/usr/lib/python3.12/typing.py", line 395, in inner
          ~~~~~~~~^^^^^^^^^^^^^^^
          ~~~~~~~~^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 395, in inner
    list: Optional["SourcesList"] = None,
    list: Optional["SourcesList"] = None,
  File "/usr/lib/python3.12/typing.py", line 395, in inner
    return _caches[func](*args, **kwds)
          ~~~~~~~~^^^^^^^^^^^^^^^
          ~~~~~~~~^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 395, in inner
  File "/usr/lib/python3.12/typing.py", line 395, in inner
    return _caches[func](*args, **kwds)
    return _caches[func](*args, **kwds)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 510, in __getitem__
    return _caches[func](*args, **kwds)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 510, in __getitem__
  File "/usr/lib/python3.12/typing.py", line 510, in __getitem__
    return _caches[func](*args, **kwds)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 510, in __getitem__
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 510, in __getitem__
    return _caches[func](*args, **kwds)
    return self._getitem(self, parameters)
    return self._getitem(self, parameters)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    return self._getitem(self, parameters)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 510, in __getitem__
  File "/usr/lib/python3.12/typing.py", line 743, in Optional
    return _caches[func](*args, **kwds)
    return _caches[func](*args, **kwds)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 743, in Optional
  File "/usr/lib/python3.12/typing.py", line 743, in Optional
    return self._getitem(self, parameters)
    return self._getitem(self, parameters)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 510, in __getitem__
  File "/usr/lib/python3.12/typing.py", line 510, in __getitem__
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 743, in Optional
  File "/usr/lib/python3.12/typing.py", line 743, in Optional
    arg = _type_check(parameters, f"{self} requires a single type.")
    return self._getitem(self, parameters)
    arg = _type_check(parameters, f"{self} requires a single type.")
    arg = _type_check(parameters, f"{self} requires a single type.")
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 193, in _type_check
  File "/usr/lib/python3.12/typing.py", line 743, in Optional
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 193, in _type_check
  File "/usr/lib/python3.12/typing.py", line 193, in _type_check
    arg = _type_check(parameters, f"{self} requires a single type.")
    arg = _type_check(parameters, f"{self} requires a single type.")
    return self._getitem(self, parameters)
    return self._getitem(self, parameters)
    arg = _type_convert(arg, module=module, allow_special_forms=allow_special_forms)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    arg = _type_convert(arg, module=module, allow_special_forms=allow_special_forms)
    arg = _type_convert(arg, module=module, allow_special_forms=allow_special_forms)
  File "/usr/lib/python3.12/typing.py", line 193, in _type_check
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 193, in _type_check
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 743, in Optional
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 743, in Optional
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 171, in _type_convert
  File "/usr/lib/python3.12/typing.py", line 171, in _type_convert
  File "/usr/lib/python3.12/typing.py", line 171, in _type_convert
    arg = _type_convert(arg, module=module, allow_special_forms=allow_special_forms)
    arg = _type_convert(arg, module=module, allow_special_forms=allow_special_forms)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 171, in _type_convert
    return ForwardRef(arg, module=module, is_class=allow_special_forms)
    return ForwardRef(arg, module=module, is_class=allow_special_forms)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    return ForwardRef(arg, module=module, is_class=allow_special_forms)
  File "/usr/lib/python3.12/typing.py", line 171, in _type_convert
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 899, in __init__
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    arg = _type_check(parameters, f"{self} requires a single type.")
  File "/usr/lib/python3.12/typing.py", line 899, in __init__
    return ForwardRef(arg, module=module, is_class=allow_special_forms)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 899, in __init__
    return ForwardRef(arg, module=module, is_class=allow_special_forms)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 899, in __init__
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 193, in _type_check
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 899, in __init__
    arg = _type_check(parameters, f"{self} requires a single type.")
    arg = _type_check(parameters, f"{self} requires a single type.")
    arg = _type_convert(arg, module=module, allow_special_forms=allow_special_forms)
    raise SyntaxError(f"Forward reference must be an expression -- got {arg!r}")
    raise SyntaxError(f"Forward reference must be an expression -- got {arg!r}")
SyntaxError: Forward reference must be an expression -- got 'SourcesList'
SyntaxError: Forward reference must be an expression -- got 'SourcesList'

Original exception was:
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Original exception was:
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 1
  File "/usr/lib/python3.12/typing.py", line 193, in _type_check
  File "<string>", line 1
    import sys
  File "/usr/lib/python3.12/typing.py", line 193, in _type_check
    import sys
SyntaxError: source code cannot contain null bytes
SyntaxError: source code cannot contain null bytes
    raise SyntaxError(f"Forward reference must be an expression -- got {arg!r}")
    raise SyntaxError(f"Forward reference must be an expression -- got {arg!r}")
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 171, in _type_convert
SyntaxError: Forward reference must be an expression -- got 'SourcesList'

Original exception was:
    raise SyntaxError(f"Forward reference must be an expression -- got {arg!r}")
  File "<string>", line 1
SyntaxError: Forward reference must be an expression -- got 'SourcesList'
    import sys

Original exception was:
SyntaxError: source code cannot contain null bytes
  File "<string>", line 1
    import sys
SyntaxError: Forward reference must be an expression -- got 'SourcesList'
SyntaxError: source code cannot contain null bytes

Original exception was:
  File "<string>", line 1
    import sys
SyntaxError: source code cannot contain null bytes
    arg = _type_convert(arg, module=module, allow_special_forms=allow_special_forms)
    arg = _type_convert(arg, module=module, allow_special_forms=allow_special_forms)
    return ForwardRef(arg, module=module, is_class=allow_special_forms)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 899, in __init__
  File "/usr/lib/python3.12/typing.py", line 171, in _type_convert
  File "/usr/lib/python3.12/typing.py", line 171, in _type_convert
    return ForwardRef(arg, module=module, is_class=allow_special_forms)
    return ForwardRef(arg, module=module, is_class=allow_special_forms)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/typing.py", line 899, in __init__
  File "/usr/lib/python3.12/typing.py", line 899, in __init__
    raise SyntaxError(f"Forward reference must be an expression -- got {arg!r}")
SyntaxError: Forward reference must be an expression -- got 'SourcesList'

Original exception was:
  File "<string>", line 1
    import sys
SyntaxError: source code cannot contain null bytes
    raise SyntaxError(f"Forward reference must be an expression -- got {arg!r}")
    raise SyntaxError(f"Forward reference must be an expression -- got {arg!r}")
SyntaxError: Forward reference must be an expression -- got 'SourcesList'
SyntaxError: Forward reference must be an expression -- got 'SourcesList'

Original exception was:

Original exception was:
  File "<string>", line 1
  File "<string>", line 1
    import sys
    import sys
SyntaxError: source code cannot contain null bytes
SyntaxError: source code cannot contain null bytes
Python Exception Python Exception Python Exception Python Exception Python Exception Python Exception <class 'SyntaxError'><class 'SyntaxError'><class 'SyntaxError'><class 'SyntaxError'><class 'SyntaxError'>: : <class 'SyntaxError'>: : : source code cannot contain null bytes (__init__.py, line 16)source code cannot contain null bytes (__init__.py, line 16): source code cannot contain null bytes (__init__.py, line 16)source code cannot contain null bytes (__init__.py, line 16)source code cannot contain null bytes (__init__.py, line 16)
Python Exception 
source code cannot contain null bytes (__init__.py, line 16)


<class 'SyntaxError'>
gdb: warning: gdb: warning: : gdb: warning: gdb: warning: 
Could not load the Python gdb module from `gdb: warning: 
Could not load the Python gdb module from `source code cannot contain null bytes (__init__.py, line 16)
Could not load the Python gdb module from `gdb: warning: 
Could not load the Python gdb module from `
Could not load the Python gdb module from `/usr/share/gdb/python
Could not load the Python gdb module from `Python Exception /usr/share/gdb/python
/usr/share/gdb/python/usr/share/gdb/python/usr/share/gdb/python'.
Limited Python support is available from the _gdb module.
Suggest passing --data-directory=/path/to/gdb/data-directory./usr/share/gdb/python'.
Limited Python support is available from the _gdb module.
Suggest passing --data-directory=/path/to/gdb/data-directory.<class 'SyntaxError'>'.
Limited Python support is available from the _gdb module.
Suggest passing --data-directory=/path/to/gdb/data-directory.'.
Limited Python support is available from the _gdb module.
Suggest passing --data-directory=/path/to/gdb/data-directory.'.
Limited Python support is available from the _gdb module.
Suggest passing --data-directory=/path/to/gdb/data-directory.gdb: warning: 
'.
Limited Python support is available from the _gdb module.
Suggest passing --data-directory=/path/to/gdb/data-directory.
Could not load the Python gdb module from `
/usr/share/gdb/python
: 


'.
Limited Python support is available from the _gdb module.
Suggest passing --data-directory=/path/to/gdb/data-directory.source code cannot contain null bytes (__init__.py, line 16)

gdb: warning: 
Could not load the Python gdb module from `/usr/share/gdb/python'.
Limited Python support is available from the _gdb module.
Suggest passing --data-directory=/path/to/gdb/data-directory.
Could not attach to process.  If your uid matches the uid of the target
Could not attach to process.  If your uid matches the uid of the target
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
Could not attach to process.  If your uid matches the uid of the target
Could not attach to process.  If your uid matches the uid of the target
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
Could not attach to process.  If your uid matches the uid of the target
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
Could not attach to process.  If your uid matches the uid of the target
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: 許可されていない操作です.ptrace: 許可されていない操作です.ptrace: 許可されていない操作です.

ptrace: 許可されていない操作です.ptrace: 許可されていない操作です.
ptrace: 許可されていない操作です.ptrace: 許可されていない操作です.
ptrace: 許可されていない操作です.



No stack.
No stack.No stack.No stack.

No stack.No stack.

No stack.
No stack.The program is not being run.The program is not being run.The program is not being run.



The program is not being run.
The program is not being run.The program is not being run.


The program is not being run.The program is not being run.

abort (core dump)


### Name and Version

version: 3748 (066ed4fd)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu


### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-06-14** at **02:54:09**:<br>

Should be fixed now via PR #528.

---

👤 **ikawrakow** commented the **2025-06-14** at **10:56:13**:<br>

Closed via #528
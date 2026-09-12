# Contributing

## Hard rules

1. **Standard library only.** This package ships with zero third-party runtime
   dependencies and that is a hard constraint, not a preference. Do not add an
   import that is not in the Python standard library, and do not add anything to
   a requirements file. If a format needs a serializer we do not have, write it.
2. **Every renderer raises `PluginError`**, never a built-in exception, on input
   it cannot render. Callers catch `PluginError` and nothing else.
3. **Every renderer is registered** with `@register("<name>")` and its class name
   is added to `__all__` in `plugins/__init__.py`. The registry is what the CLI
   enumerates; an unregistered renderer is invisible.
4. **Every renderer has a case in the existing parametrized test** in
   `tests/test_plugins.py`. Add to the parameter list; do not write a new test
   function.

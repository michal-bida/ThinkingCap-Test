"""Runs KodCode's pytest-style tests against model-written code, one forked child per item.

    python kodcode_worker.py < items.jsonl > verdicts.jsonl

Each input line is {"code", "test"}; each output line is {"verdict": "pass" | "fail" |
"unsupported" | "timeout" | "crash"}. Started by behaviour_check.py inside bubblewrap (no
network, read-only filesystem, memory and CPU limits). pytest itself is not installed, so a
small stand-in provides the parts KodCode's tests use: raises, approx and mark.parametrize.
A test that needs a fixture is "unsupported" and the item is left ungraded.
"""

import contextlib
import inspect
import io
import json
import math
import os
import signal
import sys
import types

SECONDS = 10


class Unsupported(Exception):
    pass


def fake_pytest():
    module = types.ModuleType("pytest")

    @contextlib.contextmanager
    def raises_context(expected, match=None):
        info = types.SimpleNamespace(value=None, type=None)
        try:
            yield info
        except expected as error:
            info.value, info.type = error, type(error)
            if match is not None:
                import re
                if not re.search(match, str(error)):
                    raise AssertionError(f"{error!r} does not match {match!r}")
            return
        raise AssertionError(f"DID NOT RAISE {expected}")

    def raises(expected, *args, match=None, **kwargs):
        if args:
            with raises_context(expected):
                args[0](*args[1:], **kwargs)
            return None
        return raises_context(expected, match)

    class approx:
        def __init__(self, expected, rel=1e-6, abs=1e-12):
            self.expected, self.rel, self.abs = expected, rel, abs

        def __eq__(self, actual):
            def close(a, b):
                if isinstance(b, (list, tuple)):
                    return isinstance(a, (list, tuple)) and len(a) == len(b) and all(map(close, a, b))
                if isinstance(b, dict):
                    return isinstance(a, dict) and a.keys() == b.keys() and all(close(a[k], b[k]) for k in b)
                try:
                    return math.isclose(a, b, rel_tol=self.rel, abs_tol=self.abs)
                except TypeError:
                    return a == b
            return close(actual, self.expected)

    class Mark:
        def parametrize(self, names, values, **kwargs):
            names = [n.strip() for n in names.split(",")] if isinstance(names, str) else list(names)

            def decorate(function):
                function.__dict__.setdefault("_parametrize", []).append((names, list(values)))
                return function
            return decorate

        def __getattr__(self, name):  # skip, xfail, timeout, ...: not supported
            def decorate(*args, **kwargs):
                def mark(function):
                    function._unsupported = True
                    return function
                return mark(args[0]) if len(args) == 1 and callable(args[0]) and not kwargs else mark
            return decorate

    def fixture(*args, **kwargs):
        raise Unsupported("fixture")

    module.raises, module.approx, module.mark, module.fixture = raises, approx, Mark(), fixture
    return module


def calls(function):
    """Every argument dict a (possibly parametrized) test is called with."""
    cases = [{}]
    for names, values in reversed(function.__dict__.get("_parametrize", [])):
        expanded = []
        for case in cases:
            for value in values:
                value = value if len(names) > 1 else (value,)
                expanded.append({**case, **dict(zip(names, value))})
        cases = expanded
    wanted = [p for p in inspect.signature(function).parameters if p != "self"]
    if any(p not in cases[0] for p in wanted):
        raise Unsupported("fixture argument")
    return cases


def run(code, test):
    sys.modules["pytest"] = fake_pytest()
    solution = types.ModuleType("solution")
    sys.modules["solution"] = solution
    exec(compile(code, "solution.py", "exec"), solution.__dict__)
    # Some KodCode tests have no `from solution import ...` and expect the solution in scope.
    namespace = {**solution.__dict__, "__name__": "test_solution"}
    exec(compile(test, "test_solution.py", "exec"), namespace)
    tests = []
    for name, value in list(namespace.items()):
        if name.startswith("test") and inspect.isfunction(value):
            tests.append(value)
        elif name.startswith("Test") and inspect.isclass(value):
            instance = value()
            tests += [getattr(instance, n) for n in dir(value) if n.startswith("test")]
    if not tests:
        raise Unsupported("no tests")
    for function in tests:
        if getattr(function, "_unsupported", False):
            raise Unsupported("mark")
        for arguments in calls(function):
            function(**arguments)


def child(item):
    signal.alarm(SECONDS)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            run(item["code"], item["test"])
        except Unsupported:
            os._exit(2)
        except BaseException:
            os._exit(1)
    os._exit(0)


def main():
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    for line in sys.stdin:
        item = json.loads(line)
        pid = os.fork()
        if pid == 0:
            child(item)
        _, status = os.waitpid(pid, 0)
        if os.WIFEXITED(status):
            verdict = {0: "pass", 1: "fail", 2: "unsupported"}.get(os.WEXITSTATUS(status), "crash")
        elif os.WIFSIGNALED(status) and os.WTERMSIG(status) in (signal.SIGALRM, signal.SIGXCPU):
            verdict = "timeout"
        else:
            verdict = "crash"
        print(json.dumps({"verdict": verdict}), flush=True)


if __name__ == "__main__":
    main()

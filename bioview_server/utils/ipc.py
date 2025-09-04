def emit_signal(func, *args, **kwargs):
    if func is None:
        return

    try:
        func(*args, **kwargs)
    except Exception:
        print(f"Unable to emit signal: {repr(func)}")

def emit_signal(func, *args, **kwargs): 
    if func is None: 
        return 
    
    try: 
        func(*args, **kwargs)
    except Exception as e: 
        print(f'Unable to emit signal: {repr(func)}')
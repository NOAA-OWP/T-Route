import os

# NOTE:
# ngen sets some env vars from C++ after the Python interpreter has started.
# In embedded Python, os.environ may not reflect those changes.
# getenv_any() falls back to libc getenv() and syncs os.environ.
def getenv_any(key: str, default: str = "") -> str:
    """
    Get an environment variable reliably even when it is set from C/C++
    after the Python interpreter has started (embedded Python).
    Prefers os.environ/os.getenv, falls back to libc getenv.
    """
    # First try Python's mapping
    v = os.environ.get(key)
    if v is not None:
        return v

    # Fallback: direct libc getenv (sees process env even if Python mapping is stale)
    try:
        import ctypes, ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        libc.getenv.restype = ctypes.c_char_p
        b = libc.getenv(key.encode("utf-8"))
        if not b:
            return default
        s = b.decode("utf-8")

        # Sync back into os.environ so future lookups work normally
        os.environ[key] = s
        return s
    except Exception:
        return default

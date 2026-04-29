import functools
import logging
import time
from typing import Any, Callable, TypeVar

# Type variable for generic function typing
F = TypeVar("F", bound=Callable[..., Any])


def log_function_debug(
    logger: logging.Logger = None,
    level: int = logging.DEBUG,
    max_arg_length: int = 100,
    max_return_length: int = 100,
) -> Callable[[F], F]:
    """Debug decorator that logs function entry and exit with full details"""

    def decorator(func: F) -> F:
        # Get logger - use provided logger or create one based on function's module
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__

            # Helper function to truncate long representations
            def truncate_repr(obj, max_length=100):
                repr_str = repr(obj)
                if len(repr_str) <= max_length:
                    return repr_str
                return repr_str[: max_length - 3] + "..."

            # Prepare entry message with arguments
            entry_msg = f"→ Entering {func_name}"
            if args or kwargs:
                args_parts = []

                # Add positional arguments
                if args:
                    args_str = [truncate_repr(arg, max_arg_length) for arg in args]
                    args_parts.extend(args_str)

                # Add keyword arguments
                if kwargs:
                    kwargs_str = [f"{k}={truncate_repr(v, max_arg_length)}" for k, v in kwargs.items()]
                    args_parts.extend(kwargs_str)

                entry_msg += f"({', '.join(args_parts)})"
            else:
                entry_msg += "()"

            logger.log(level, entry_msg)

            start_time = time.time()

            try:
                # Execute the function
                result = func(*args, **kwargs)

                # Calculate execution time
                execution_time = time.time() - start_time

                # Prepare exit message with return value and timing
                exit_msg = f"← Exiting {func_name} (took {execution_time:.4f}s)"

                # Add return value if not None
                if result is not None:
                    return_repr = truncate_repr(result, max_return_length)
                    exit_msg += f" → {return_repr}"
                else:
                    exit_msg += " → None"

                logger.log(level, exit_msg)
                return result

            except Exception as e:
                # Calculate execution time even for exceptions
                execution_time = time.time() - start_time

                # Log exception with details
                error_msg = (
                    f"✗ Exception in {func_name} (took {execution_time:.4f}s): {type(e).__name__}: {str(e)}"
                )

                logger.log(logging.ERROR, error_msg)
                raise

        return wrapper

    return decorator

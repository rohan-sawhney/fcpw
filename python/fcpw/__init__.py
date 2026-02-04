# Import the core C++ extension module
# The module is built by nanobind and contains all FCPW functionality
try:
    from ._fcpw import *

    # Build __all__ from the imported module
    import sys
    _module = sys.modules.get('fcpw._fcpw')
    if _module:
        __all__ = [name for name in dir(_module) if not name.startswith('_')]
    else:
        __all__ = []

except ImportError as e:
    # Provide helpful error message if the extension module fails to load
    import warnings
    warnings.warn(
        f"Failed to import FCPW extension module: {e}\n"
        "This usually means:\n"
        "  1. The package was not built with bindings enabled (FCPW_BUILD_BINDINGS=ON)\n"
        "  2. The extension module is missing from the installation\n"
        "  3. There are missing dependencies (e.g., Slang library for GPU support)",
        ImportWarning
    )
    __all__ = []
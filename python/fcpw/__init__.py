# import the core C++ extension module;
# the module is built by nanobind and contains all FCPW functionality
try:
    from ._fcpw import *

    # build __all__ from the imported module
    import sys
    _module = sys.modules.get('fcpw._fcpw')
    if _module:
        __all__ = [name for name in dir(_module) if not name.startswith('_')]
    else:
        __all__ = []

    # directory containing the packaged slang shaders (present only in GPU-enabled
    # installations), laid out with the same relative structure as a source checkout;
    # pass this as the fcpw_directory_path argument of GPUScene when using an
    # installed package. When running from a source checkout, pass the repository
    # root instead so that shader edits take effect without reinstalling.
    import os
    _slang_directory_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slang")
    if os.path.isdir(_slang_directory_path):
        slang_directory_path = _slang_directory_path
        __all__.append("slang_directory_path")

except ImportError as e:
    # provide helpful error message if the extension module fails to load
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
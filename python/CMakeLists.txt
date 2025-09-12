cmake_minimum_required(VERSION 3.15...3.27)
project(fcpw-python)

nanobind_add_module(py NOMINSIZE STABLE_ABI NB_STATIC fcpw_py.cpp)
target_link_libraries(py PRIVATE fcpw)

# ensure the extension can find adjacent shared libs at runtime
if(UNIX AND NOT APPLE)
    # use $ORIGIN so the module can load libs placed next to it after install
    set_target_properties(py PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
elseif(APPLE)
    # use @loader_path on macOS
    set_target_properties(py PROPERTIES BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
endif()

install(TARGETS py
    LIBRARY DESTINATION fcpw
    RUNTIME DESTINATION fcpw
)
if(FCPW_ENABLE_GPU_SUPPORT)
    if(TARGET slang)
        install(FILES $<TARGET_FILE:slang> DESTINATION fcpw)
    endif()
endif()

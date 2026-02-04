cmake_minimum_required(VERSION 3.15...3.27)
project(fcpw-python)

nanobind_add_module(_fcpw NOMINSIZE STABLE_ABI NB_STATIC fcpw_py.cpp)
target_link_libraries(_fcpw PRIVATE fcpw)

# ensure the extension can find adjacent shared libs at runtime
if(UNIX AND NOT APPLE)
    # use $ORIGIN so the module can load libs placed next to it after install
    set_target_properties(_fcpw PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
elseif(APPLE)
    # use @loader_path on macOS
    set_target_properties(_fcpw PROPERTIES BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
endif()

install(TARGETS _fcpw
    LIBRARY DESTINATION fcpw
    RUNTIME DESTINATION fcpw
)

# Install GPU support libraries if enabled
if(FCPW_ENABLE_GPU_SUPPORT)
    # Slang libraries are fetched via FetchContent and located in _deps/slang-src/lib
    # We need to install all required Slang shared libraries to the fcpw package

    # Try to find the Slang library directory
    set(SLANG_LIB_SEARCH_PATHS
        "${CMAKE_BINARY_DIR}/_deps/slang-src/lib"
        "${CMAKE_CURRENT_BINARY_DIR}/_deps/slang-src/lib"
    )

    set(SLANG_LIB_DIR "")
    foreach(SEARCH_PATH ${SLANG_LIB_SEARCH_PATHS})
        if(EXISTS "${SEARCH_PATH}")
            set(SLANG_LIB_DIR "${SEARCH_PATH}")
            message(STATUS "Found Slang libraries at: ${SLANG_LIB_DIR}")
            break()
        endif()
    endforeach()

    if(SLANG_LIB_DIR)
        # Install all Slang shared libraries using install(CODE) to handle glob at install time
        install(CODE "
            file(GLOB SLANG_LIBS \"${SLANG_LIB_DIR}/libslang*.so*\" \"${SLANG_LIB_DIR}/libgfx*.so*\")
            foreach(LIB_FILE \${SLANG_LIBS})
                message(STATUS \"Installing Slang library: \${LIB_FILE}\")
                file(INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}/fcpw\" TYPE SHARED_LIBRARY FILES \"\${LIB_FILE}\")
            endforeach()
        ")
        message(STATUS "Configured Slang library installation from: ${SLANG_LIB_DIR}")
    else()
        message(WARNING "FCPW_ENABLE_GPU_SUPPORT is ON but Slang library directory not found. "
                        "Searched paths: ${SLANG_LIB_SEARCH_PATHS}\n"
                        "GPU functionality may not work in Python bindings.")
    endif()
endif()

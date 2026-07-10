cmake_minimum_required(VERSION 3.15...3.27)
project(fcpw-python)

nanobind_add_module(_fcpw NOMINSIZE STABLE_ABI NB_STATIC fcpw_bindings.cpp)
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

# install GPU support files if enabled
if(FCPW_ENABLE_GPU_SUPPORT)
    # Package the slang shaders, which GPUScene compiles at runtime. The
    # source-tree relative layout is preserved under fcpw/slang so that
    # GPUScene resolves modules from an installed package without any changes;
    # pass fcpw.slang_directory_path as the fcpw_directory_path argument of
    # GPUScene when using an installed package rather than a source checkout.
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/fcpw/gpu
        DESTINATION fcpw/slang/include/fcpw
        FILES_MATCHING PATTERN "*.slang"
    )

    # slang-rhi is static by default, but it depends on fetched Slang
    # compiler/runtime libraries and optional backend DLLs. Bundle those
    # next to _fcpw so GPU bindings do not depend on build-tree paths.
    if(CMAKE_CONFIGURATION_TYPES)
        set(FCPW_SLANG_RHI_RUNTIME_DIR "${CMAKE_BINARY_DIR}/\${CMAKE_INSTALL_CONFIG_NAME}")
    else()
        set(FCPW_SLANG_RHI_RUNTIME_DIR "${CMAKE_BINARY_DIR}")
    endif()

    if(WIN32)
        # slang-rhi copies Slang, DXC, WinPix, Dawn, and similar runtime DLLs
        # into its runtime output directory via the slang-rhi-copy-files target.
        install(CODE "
            set(_fcpw_slang_rhi_runtime_dir \"${FCPW_SLANG_RHI_RUNTIME_DIR}\")
            file(GLOB _fcpw_slang_rhi_libs \"\${_fcpw_slang_rhi_runtime_dir}/*.dll\")
            foreach(_lib \${_fcpw_slang_rhi_libs})
                message(STATUS \"Installing Slang RHI runtime library: \${_lib}\")
                file(INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}/fcpw\" TYPE SHARED_LIBRARY FILES \"\${_lib}\")
            endforeach()
            if(EXISTS \"\${_fcpw_slang_rhi_runtime_dir}/D3D12\")
                message(STATUS \"Installing Slang RHI D3D12 runtime directory: \${_fcpw_slang_rhi_runtime_dir}/D3D12\")
                file(INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}/fcpw\" TYPE DIRECTORY FILES \"\${_fcpw_slang_rhi_runtime_dir}/D3D12\")
            endif()
        ")
    else()
        if(DEFINED SLANG_RHI_SLANG_BINARY_DIR)
            if(APPLE)
                install(CODE "
                    file(GLOB _fcpw_slang_libs
                        \"${SLANG_RHI_SLANG_BINARY_DIR}/lib/libslang*.dylib\"
                        \"${SLANG_RHI_SLANG_BINARY_DIR}/lib/libgfx*.dylib\")
                    foreach(_lib \${_fcpw_slang_libs})
                        message(STATUS \"Installing Slang library: \${_lib}\")
                        file(INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}/fcpw\" TYPE SHARED_LIBRARY FILES \"\${_lib}\")
                    endforeach()
                ")
            else()
                install(CODE "
                    file(GLOB _fcpw_slang_libs
                        \"${SLANG_RHI_SLANG_BINARY_DIR}/lib/libslang*.so*\"
                        \"${SLANG_RHI_SLANG_BINARY_DIR}/lib/libgfx*.so*\")
                    foreach(_lib \${_fcpw_slang_libs})
                        message(STATUS \"Installing Slang library: \${_lib}\")
                        file(INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}/fcpw\" TYPE SHARED_LIBRARY FILES \"\${_lib}\")
                    endforeach()
                ")
            endif()
        else()
            message(WARNING "FCPW_ENABLE_GPU_SUPPORT is ON but SLANG_RHI_SLANG_BINARY_DIR is not defined. "
                            "GPU functionality may not work in Python bindings.")
        endif()

        # install optional copied backend runtimes such as Dawn if present.
        install(CODE "
            set(_fcpw_slang_rhi_runtime_dir \"${FCPW_SLANG_RHI_RUNTIME_DIR}\")
            file(GLOB _fcpw_slang_rhi_libs
                \"\${_fcpw_slang_rhi_runtime_dir}/*.so*\"
                \"\${_fcpw_slang_rhi_runtime_dir}/*.dylib\")
            foreach(_lib \${_fcpw_slang_rhi_libs})
                message(STATUS \"Installing Slang RHI runtime library: \${_lib}\")
                file(INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}/fcpw\" TYPE SHARED_LIBRARY FILES \"\${_lib}\")
            endforeach()
        ")
    endif()
endif()

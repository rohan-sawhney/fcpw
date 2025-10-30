#pragma once

#include <slang-rhi.h>
#include <slang-rhi/shader-cursor.h>

namespace fcpw {

using namespace rhi;

class GPUContext {
public:
    // members
    DeviceDesc deviceDesc = {};
    ComPtr<IDevice> device;
    ComPtr<ICommandQueue> queue;
    std::vector<std::string> searchPaths;
    std::vector<std::pair<std::string, std::string>> macros;

    // initialize device with the given search paths and macros
    void initDevice(DeviceType deviceType=DeviceType::Default,
                    bool enableDebugLayer=false) {
        // convert search paths and macros to C strings
        cStrSearchPaths.clear();
        cStrSearchPaths.resize(searchPaths.size());
        for (size_t i = 0; i < searchPaths.size(); i++) {
            cStrSearchPaths[i] = searchPaths[i].c_str();
        }
        macrosDesc.clear();
        macrosDesc.resize(macros.size());
        for (size_t i = 0; i < macros.size(); i++) {
            macrosDesc[i].name = macros[i].first.c_str();
            macrosDesc[i].value = macros[i].second.c_str();
        }

        // create device and queue
        deviceDesc.slang.searchPaths = cStrSearchPaths.data();
        deviceDesc.slang.searchPathCount = cStrSearchPaths.size();
        deviceDesc.slang.preprocessorMacros = macrosDesc.data();
        deviceDesc.slang.preprocessorMacroCount = macrosDesc.size();
        deviceDesc.slang.optimizationLevel = SlangOptimizationLevel::SLANG_OPTIMIZATION_LEVEL_HIGH;
        deviceDesc.deviceType = deviceType;
        if (enableDebugLayer) {
            deviceDesc.enableValidation = true;
            deviceDesc.debugCallback = &debugCallback;
        }

        device = getRHI()->createDevice(deviceDesc);
        if (!device) {
            std::cerr << "failed to create device" << std::endl;
            exit(EXIT_FAILURE);
        }
        queue = device->getQueue(QueueType::Graphics);
        if (!queue) {
            std::cerr << "failed to get graphics queue" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::cout << "device: " << device->getInfo().apiName << std::endl;
    }

private:
    // debug callback implementation
    class DebugCallback : public IDebugCallback {
    public:
        virtual SLANG_NO_THROW void SLANG_MCALL handleMessage(DebugMessageType type,
                                                              DebugMessageSource source,
                                                              const char* message) override {
            const char* typeStr = "";
            switch (type) {
                case DebugMessageType::Info:
                    typeStr = "INFO: ";
                    break;
                case DebugMessageType::Warning:
                    typeStr = "WARNING: ";
                    break;
                case DebugMessageType::Error:
                    typeStr = "ERROR: ";
                    break;
                default:
                    break;
            }
            const char* sourceStr = "[GraphicsLayer]: ";
            switch (source) {
                case DebugMessageSource::Slang:
                    sourceStr = "[Slang]: ";
                    break;
                case DebugMessageSource::Driver:
                    sourceStr = "[Driver]: ";
                    break;
            }
            printf("%s%s%s\n", sourceStr, typeStr, message);
        }
    };
    DebugCallback debugCallback;

    // members
    std::vector<const char*> cStrSearchPaths;
    std::vector<slang::PreprocessorMacroDesc> macrosDesc;
};

class GPUModule {
public:
    // member
    slang::IModule *module;

    // load a module with the given name
    Slang::Result load(ComPtr<IDevice>& device, const std::string& name) {
        ComPtr<slang::ISession> slangSession;
        SLANG_RETURN_ON_FAIL(device->getSlangSession(slangSession.writeRef()));
        ComPtr<slang::IBlob> diagnosticsBlob;
        module = slangSession->loadModule(name.c_str(), diagnosticsBlob.writeRef());
        diagnoseIfNeeded(diagnosticsBlob);
        if (!module) return SLANG_FAIL;

        return SLANG_OK;
    }

    // load a module with the given name
    void load(GPUContext& context, const std::string& name) {
        Slang::Result loadResult = load(context.device, name);
        if (loadResult != SLANG_OK) {
            std::cerr << "failed to load " << name << " module" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::cout << "loaded " << name << " module" << std::endl;
    }

private:
    // diagnose issues if any
    void diagnoseIfNeeded(slang::IBlob* diagnosticsBlob) {
        if (diagnosticsBlob != nullptr) {
            std::cerr << (const char*)diagnosticsBlob->getBufferPointer() << std::endl;
        }
    }
};

class ComputeShader {
public:
    // members
    ComPtr<IShaderProgram> program;
    slang::ProgramLayout* reflection = nullptr;
    ComputePipelineDesc pipelineDesc = {};
    ComPtr<IComputePipeline> pipeline;

    // checks if the shader is initialized
    bool isInitialized() const {
        return reflection != nullptr;
    }

    // load a compute program with the given module and entry point names
    Slang::Result loadProgram(ComPtr<IDevice>& device,
                              const GPUModule& mainModule,
                              const std::vector<GPUModule>& libraryModules,
                              const std::string& entryPointName) {
        // create composite program
        std::vector<slang::IComponentType *> componentTypes;
        for (size_t i = 0; i < libraryModules.size(); i++) {
            componentTypes.emplace_back(libraryModules[i].module);
        }
        componentTypes.emplace_back(mainModule.module);
        ComPtr<slang::IEntryPoint> entryPoint;
        SLANG_RETURN_ON_FAIL(mainModule.module->findEntryPointByName(entryPointName.c_str(),
                                                                     entryPoint.writeRef()));
        componentTypes.emplace_back(entryPoint.get());

        ComPtr<slang::ISession> slangSession;
        SLANG_RETURN_ON_FAIL(device->getSlangSession(slangSession.writeRef()));
        ComPtr<slang::IComponentType> composedProgram;
        ComPtr<slang::IBlob> diagnosticsBlob;
        SlangResult result = slangSession->createCompositeComponentType(componentTypes.data(),
                                                                        componentTypes.size(),
                                                                        composedProgram.writeRef(),
                                                                        diagnosticsBlob.writeRef());
        diagnoseIfNeeded(diagnosticsBlob);
        SLANG_RETURN_ON_FAIL(result);

        // link program and get reflection
        ComPtr<slang::IComponentType> linkedProgram;
        result = composedProgram->link(linkedProgram.writeRef(), diagnosticsBlob.writeRef());
        diagnoseIfNeeded(diagnosticsBlob);
        SLANG_RETURN_ON_FAIL(result);

        composedProgram = linkedProgram;
        reflection = composedProgram->getLayout();

        // create pipeline state
        ShaderProgramDesc programDesc = {};
        programDesc.slangGlobalScope = composedProgram.get();
        program = device->createShaderProgram(programDesc);
        pipelineDesc.program = program.get();
        SLANG_RETURN_ON_FAIL(device->createComputePipeline(pipelineDesc, pipeline.writeRef()));

        return SLANG_OK;
    }

    // load a compute program with the given module and entry point names
    void loadProgram(GPUContext& context,
                     const GPUModule& mainModule,
                     const std::vector<GPUModule>& libraryModules,
                     const std::string& entryPointName) {
        Slang::Result loadProgramResult = loadProgram(context.device, mainModule,
                                                      libraryModules, entryPointName);
        if (loadProgramResult != SLANG_OK) {
            std::cerr << "failed to load " << entryPointName << " compute program" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::cout << "loaded " << entryPointName << " compute program" << std::endl;
    }

    // create a shader object for the given reflection type name
    ComPtr<IShaderObject> createShaderObject(GPUContext& context,
                                             const std::string& reflectionType) const {
        ComPtr<IShaderObject> shaderObject;
        Slang::Result createShaderObjectResult = context.device->createShaderObject(
            reflection->findTypeByName(reflectionType.c_str()),
            ShaderObjectContainerType::None, shaderObject.writeRef());
        if (createShaderObjectResult != SLANG_OK) {
            std::cerr << "failed to create shader object" << std::endl;
            exit(EXIT_FAILURE);
        }

        return shaderObject;
    }

private:
    // diagnose issues if any
    void diagnoseIfNeeded(slang::IBlob* diagnosticsBlob) {
        if (diagnosticsBlob != nullptr) {
            std::cerr << (const char*)diagnosticsBlob->getBufferPointer() << std::endl;
        }
    }
};

class GPUBuffer {
public:
    // members
    BufferDesc desc;
    ComPtr<IBuffer> buffer;

    // allocate a buffer with the given data and element count
    template<typename T>
    Slang::Result allocate(ComPtr<IDevice>& device, bool unorderedAccess,
                           const T* initialData, size_t elementCount) {
        const T *data = nullptr;
        if (elementCount == 0) {
            elementCount = 1; // Slang requires buffers to be non-empty

        } else {
            data = initialData;
        }

        desc.size = elementCount*sizeof(T);
        desc.elementSize = sizeof(T);
        desc.format = Format::Undefined;
        desc.memoryType = MemoryType::DeviceLocal;
        if (unorderedAccess) {
            desc.usage = BufferUsage::ShaderResource |
                         BufferUsage::UnorderedAccess |
                         BufferUsage::CopySource |
                         BufferUsage::CopyDestination;
            desc.defaultState = ResourceState::UnorderedAccess;

        } else {
            desc.usage = BufferUsage::ShaderResource |
                         BufferUsage::CopySource |
                         BufferUsage::CopyDestination;
            desc.defaultState = ResourceState::ShaderResource;
        }
        SLANG_RETURN_ON_FAIL(device->createBuffer(desc, (void *)data, buffer.writeRef()));

        return SLANG_OK;
    }

    // allocate a buffer with the given data
    template<typename T>
    void allocate(GPUContext& context, bool unorderedAccess,
                  const std::vector<T>& initialData) {
        Slang::Result createBufferResult = allocate<T>(context.device, unorderedAccess,
                                                       initialData.data(), initialData.size());
        if (createBufferResult != SLANG_OK) {
            std::cerr << "failed to create buffer" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // read the buffer data into a vector
    template<typename T>
    Slang::Result read(ComPtr<IDevice>& device, std::vector<T>& result) const {
        result.clear();
        size_t elementCount = desc.size/sizeof(T);
        if (elementCount > 0) {
            ComPtr<ISlangBlob> resultBlob;
            SLANG_RETURN_ON_FAIL(device->readBuffer(buffer, 0, desc.size,
                                                    resultBlob.writeRef()));
            if (resultBlob->getBufferSize() != desc.size) {
                std::cerr << "incorrect GPU buffer size on read" << std::endl;
                return SLANG_FAIL;
            }

            auto resultPtr = (T *)resultBlob->getBufferPointer();
            result.assign(resultPtr, resultPtr + elementCount);
        }

        return SLANG_OK;
    }

    // read the buffer data into a vector
    template<typename T>
    void read(GPUContext& context, std::vector<T>& result) const {
        Slang::Result readBufferResult = read<T>(context.device, result);
        if (readBufferResult != SLANG_OK) {
            std::cerr << "failed to read buffer from GPU" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
};

template <size_t DIM>
class GPUTexture {
public:
    // members
    TextureDesc desc;
    ComPtr<ITexture> texture;
    ComPtr<ITextureView> view;

    // allocate a texture with the given dimensions and data
    template <typename T, size_t CHANNELS>
    Slang::Result allocate(ComPtr<IDevice>& device, bool unorderedAccess,
                           size_t width, size_t height, size_t depth,
                           const T* initialData, size_t elementCount) {
        desc.type = getTextureType();
        desc.size.width = width;
        desc.size.height = height;
        desc.size.depth = DIM == 3 ? depth : 1;
        desc.arrayLength = 1;
        desc.mipCount = 1;
        Format format = getTextureFormat<T, CHANNELS>();
        desc.format = format;
        desc.memoryType = MemoryType::DeviceLocal;
        if (unorderedAccess) {
            desc.usage = TextureUsage::ShaderResource |
                         TextureUsage::UnorderedAccess |
                         TextureUsage::CopySource |
                         TextureUsage::CopyDestination;
            desc.defaultState = ResourceState::UnorderedAccess;

        } else {
            desc.usage = TextureUsage::ShaderResource |
                         TextureUsage::CopySource |
                         TextureUsage::CopyDestination;
            desc.defaultState = ResourceState::ShaderResource;
        }

        size_t texelSize = getTexelSize(format);
        SubresourceData subresourceData = {};
        subresourceData.data = elementCount == 0 ? nullptr : initialData;
        subresourceData.rowPitch = width*texelSize;
        subresourceData.slicePitch = height*width*texelSize;
        SLANG_RETURN_ON_FAIL(device->createTexture(desc, &subresourceData, texture.writeRef()));

        TextureViewDesc viewDesc = {};
        viewDesc.format = format;
        SLANG_RETURN_ON_FAIL(device->createTextureView(texture, viewDesc, view.writeRef()));

        return SLANG_OK;
    }

    // allocate a texture with the given dimensions and data
    template <typename T, size_t CHANNELS>
    void allocate(GPUContext& context, bool unorderedAccess,
                  size_t width, size_t height, size_t depth,
                  const std::vector<T>& initialData) {
        Slang::Result createTextureResult = allocate<T>(
            context.device, unorderedAccess, width, height, depth,
            initialData.data(), initialData.size());
        if (createTextureResult != SLANG_OK) {
            std::cerr << "failed to create texture" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // read the texture data into a vector
    template <typename T, size_t CHANNELS>
    Slang::Result read(ComPtr<IDevice>& device,
                       std::vector<T>& result) const {
        result.clear();
        size_t elementCount = desc.size.width*desc.size.height*desc.size.depth;
        if (elementCount > 0) {
            ComPtr<ISlangBlob> resultBlob;
            SubresourceLayout layout;
            SLANG_RETURN_ON_FAIL(device->readTexture(texture, 0, 0, resultBlob.writeRef(), &layout));

            Format format = getTextureFormat<T, CHANNELS>();
            size_t expectedBufferSize = elementCount*getTexelSize(format);
            if (resultBlob->getBufferSize() != expectedBufferSize) {
                std::cerr << "incorrect GPU texture size on read" << std::endl;
                return SLANG_FAIL;
            }

            auto resultPtr = (T *)resultBlob->getBufferPointer();
            result.assign(resultPtr, resultPtr + elementCount);
        }

        return SLANG_OK;
    }

    // read the texture data into a vector
    template <typename T, size_t CHANNELS>
    void read(GPUContext& context,
              std::vector<T>& result) const {
        Slang::Result readTextureResult = read<T>(context.device, result);
        if (readTextureResult != SLANG_OK) {
            std::cerr << "failed to read texture from GPU" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

private:
    // helper functions to get texture type
    TextureType getTextureType() const {
        if (DIM != 2 && DIM != 3) {
            std::cerr << "Error: Unsupported texture dimension " << DIM << std::endl;
            exit(EXIT_FAILURE);
        }

        return DIM == 2 ? TextureType::Texture2D : TextureType::Texture3D;
    }

    // helper functions to get texture format
    template <typename T, size_t CHANNELS>
    Format getTextureFormat() const {
        if (std::is_same<T, float>::value) {
            if (CHANNELS == 1) {
                return Format::R32Float;

            } else if (CHANNELS == 4) {
                return Format::RGBA32Float;
            }

        } else if (std::is_same<T, int>::value) {
            if (CHANNELS == 1) {
                return Format::R32Sint;

            } else if (CHANNELS == 4) {
                return Format::RGBA32Sint;
            }

        } else if (std::is_same<T, uint32_t>::value) {
            if (CHANNELS == 1) {
                return Format::R32Uint;

            } else if (CHANNELS == 4) {
                return Format::RGBA32Uint;
            }
        }

        std::cerr << "Error: Unsupported texture format for type T with "
                  << CHANNELS << " channel(s)" << std::endl;
        exit(EXIT_FAILURE);

        return Format::Undefined;
    }

    // helper functions to get texture size
    size_t getTexelSize(Format format) const {
        const FormatInfo& info = getFormatInfo(format);
        return info.blockSizeInBytes/info.pixelsPerBlock;
    }
};

class GPUSampler {
public:
    // members
    SamplerDesc desc;
    ComPtr<ISampler> sampler;

    // allocate a sampler with the given filtering and addressing modes
    Slang::Result allocate(ComPtr<IDevice>& device,
                           TextureFilteringMode filter,
                           TextureAddressingMode address) {
        desc.minFilter = filter;
        desc.magFilter = filter;
        desc.addressU = address;
        desc.addressV = address;
        desc.addressW = address;
        SLANG_RETURN_ON_FAIL(device->createSampler(desc, sampler.writeRef()));

        return SLANG_OK;
    }

    // allocate a sampler with the given filtering and addressing modes
    void allocate(GPUContext& context,
                  TextureFilteringMode filter,
                  TextureAddressingMode address) {
        Slang::Result createSamplerResult = allocate(context.device, filter, address);
        if (createSamplerResult != SLANG_OK) {
            std::cerr << "failed to create sampler" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
};

class GPUShaderObject {
public:
    virtual ~GPUShaderObject() = default;
    virtual void setResources(const ShaderCursor& cursor, bool printLogs) const = 0;
    virtual std::string getReflectionType() const = 0;
};

class GPUShaderEntryPoint {
public:
    virtual ~GPUShaderEntryPoint() = default;
    virtual void setResources(const ShaderCursor& cursor, bool printLogs) const = 0;
    virtual void setDispatchResources(const ShaderCursor& cursor, uint32_t dispatchIndex) const { /* do nothing */ }
};

void printReflectionInfo(const ShaderCursor& cursor, int nFields,
                         const std::string& reflectionType) {
    std::cout << "Reflection: " << reflectionType << std::endl;
    for (int i = 0; i < nFields; i++) {
        std::cout << "\targument[" << i << "]: " << cursor.getTypeLayout()->getFieldByIndex(i)->getName() << std::endl;
    }
}

template <typename GPUEntryPoint>
void runShader(GPUContext& context,
               const ComputeShader& shader,
               const GPUEntryPoint& entryPoint,
               std::function<void(const ComputeShader&, const ShaderCursor&)> bindShaderResources,
               uint32_t nThreadGroups,
               uint32_t nDispatchCalls,
               bool printLogs=false)
{
    // setup command encoder
    auto commandEncoder = context.queue->createCommandEncoder();
    auto computePassEncoder = commandEncoder->beginComputePass();

    // bind shader resources
    auto rootShaderObject = computePassEncoder->bindPipeline(shader.pipeline);
    ShaderCursor rootCursor(rootShaderObject);
    if (bindShaderResources) bindShaderResources(shader, rootCursor);

    // bind entry point arguments
    ShaderCursor entryPointCursor(rootShaderObject->getEntryPoint(0));
    entryPoint.setResources(entryPointCursor, printLogs);

    // dispatch compute shader
    ComPtr<IQueryPool> queryPool;
    QueryPoolDesc queryDesc = {};
    queryDesc.type = QueryType::Timestamp;
    queryDesc.count = 2;
    Slang::Result createQueryPoolResult = context.device->createQueryPool(queryDesc, queryPool.writeRef());
    if (createQueryPoolResult != SLANG_OK) {
        std::cerr << "failed to create query pool" << std::endl;
        exit(EXIT_FAILURE);
    }

    computePassEncoder->writeTimestamp(queryPool, 0);
    for (uint32_t i = 0; i < nDispatchCalls; i++) {
        entryPoint.setDispatchResources(entryPointCursor, i);
        computePassEncoder->dispatchCompute(nThreadGroups, 1, 1);
    }
    computePassEncoder->writeTimestamp(queryPool, 1);
    computePassEncoder->end();

    context.queue->submit(commandEncoder->finish());
    context.queue->waitOnHost();

    // read timestamps
    const DeviceInfo& deviceInfo = context.device->getInfo();
    double timestampFrequency = (double)deviceInfo.timestampFrequency;
    uint64_t timestampData[2] = { 0, 0 };
    Slang::Result getQueryPoolResult = queryPool->getResult(0, 2, timestampData);
    if (getQueryPoolResult != SLANG_OK) {
        std::cerr << "failed to get query pool result" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (printLogs) {
        double timeSpan = (timestampData[1] - timestampData[0])*1000/timestampFrequency;
        std::cout << "Compute shader took " << timeSpan << " ms" << std::endl;
    }
}

} // namespace fcpw

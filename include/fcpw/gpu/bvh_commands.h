#pragma once

#include <fcpw/gpu/bvh_interop_structures.h>

namespace fcpw {

template <typename T, typename S>
void runBvhTraversal(GPUContext& gpuContext,
                     const ComputeShader& shader,
                     const T& gpuBvhBuffers,
                     const S& gpuRunQuery,
                     uint32_t nThreadGroups=4096,
                     bool printLogs=false)
{
    // setup command buffer
    auto commandBuffer = gpuContext.transientHeap->createCommandBuffer();
    auto encoder = commandBuffer->encodeComputeCommands();

    // create bvh shader object
    auto rootShaderObject = encoder->bindPipeline(shader.pipelineState);
    ComPtr<IShaderObject> bvhShaderObject = shader.createShaderObject(
        gpuContext, gpuBvhBuffers.getReflectionType());

    // bind shader resources
    ShaderCursor bvhCursor(bvhShaderObject);
    gpuBvhBuffers.setResources(bvhCursor, printLogs);
    ShaderCursor rootCursor(rootShaderObject);
    rootCursor.getPath("gBvh").setObject(bvhShaderObject);

    // bind entry point arguments
    ShaderCursor entryPointCursor(rootShaderObject->getEntryPoint(0));
    gpuRunQuery.setResources(entryPointCursor, printLogs);

    // perform query
    ComPtr<IQueryPool> queryPool;
    IQueryPool::Desc queryDesc = {};
    queryDesc.type = QueryType::Timestamp;
    queryDesc.count = 2;
    Slang::Result createQueryPoolResult = gpuContext.device->createQueryPool(queryDesc, queryPool.writeRef());
    if (createQueryPoolResult != SLANG_OK) {
        std::cout << "failed to create query pool" << std::endl;
        exit(EXIT_FAILURE);
    }

    encoder->writeTimestamp(queryPool, 0);
    encoder->dispatchCompute(nThreadGroups, 1, 1);
    encoder->writeTimestamp(queryPool, 1);

    // execute command buffer
    encoder->endEncoding();
    commandBuffer->close();
    gpuContext.queue->executeCommandBuffer(commandBuffer);
    gpuContext.queue->waitOnHost();

    // read query timestamps
    const DeviceInfo& deviceInfo = gpuContext.device->getDeviceInfo();
    double timestampFrequency = (double)deviceInfo.timestampFrequency;
    uint64_t timestampData[2] = { 0, 0 };
    Slang::Result getQueryPoolResult = queryPool->getResult(0, 2, timestampData);
    if (getQueryPoolResult != SLANG_OK) {
        std::cout << "failed to get query pool result" << std::endl;
        exit(EXIT_FAILURE);
    }

    // synchronize and reset transient heap
    gpuContext.transientHeap->finish();
    gpuContext.transientHeap->synchronizeAndReset();

    if (printLogs) {
        double timeSpan = (timestampData[1] - timestampData[0])*1000/timestampFrequency;
        std::cout << "Queries took " << timeSpan << " ms" << std::endl;
    }
}

template <typename T>
void runBvhUpdate(GPUContext& gpuContext,
                  const ComputeShader& shader,
                  const T& gpuBvhBuffers,
                  bool printLogs=false)
{
    // setup command buffer
    auto commandBuffer = gpuContext.transientHeap->createCommandBuffer();
    auto encoder = commandBuffer->encodeComputeCommands();

    // create bvh shader object
    auto rootShaderObject = encoder->bindPipeline(shader.pipelineState);
    ComPtr<IShaderObject> bvhShaderObject = shader.createShaderObject(
        gpuContext, gpuBvhBuffers.getReflectionType());

    // bind shader resources
    ShaderCursor bvhCursor(bvhShaderObject);
    gpuBvhBuffers.setResources(bvhCursor, printLogs);
    ShaderCursor rootCursor(rootShaderObject);
    rootCursor.getPath("gBvh").setObject(bvhShaderObject);

    // bind entry point arguments
    ShaderCursor entryPointCursor(rootShaderObject->getEntryPoint(0));
    entryPointCursor.getPath("nodeIndices").setResource(gpuBvhBuffers.nodeIndices.view);

    for (int depth = gpuBvhBuffers.maxUpdateDepth; depth >= 0; --depth) {
        uint32_t firstNodeOffset = gpuBvhBuffers.updateEntryData[depth].first;
        uint32_t nodeCount = gpuBvhBuffers.updateEntryData[depth].second;
        entryPointCursor.getPath("firstNodeOffset").setData(firstNodeOffset);
        entryPointCursor.getPath("nodeCount").setData(nodeCount);

        encoder->dispatchCompute(nodeCount, 1, 1);
        encoder->bufferBarrier(gpuBvhBuffers.nodes.buffer.get(), ResourceState::UnorderedAccess, ResourceState::UnorderedAccess);
    }

    if (printLogs) {
        int entryPointFieldCount = 4;
        printReflectionInfo(entryPointCursor, entryPointFieldCount, "runBvhUpdate");
    }

    // execute command buffer
    encoder->endEncoding();
    commandBuffer->close();
    gpuContext.queue->executeCommandBuffer(commandBuffer);
    gpuContext.queue->waitOnHost();

    // synchronize and reset transient heap
    gpuContext.transientHeap->finish();
    gpuContext.transientHeap->synchronizeAndReset();
}

} // namespace fcpw

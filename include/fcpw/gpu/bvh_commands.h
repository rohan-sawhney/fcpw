#pragma once

#include <fcpw/gpu/bvh_interop_structures.h>

namespace fcpw {

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
        gpuBvhBuffers.nodes.applyBarrier(encoder);
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


#ifndef HELPER_DEVICE_CHECK_H
#define HELPER_DEVICE_CHECK_H

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>


void cudaOutputDeviceProperties () {
  // Get the number of devices.
  int device_cnt = 0;
  cudaGetDeviceCount(&device_cnt);
  printf("Total number of devices: %d\n", device_cnt);

  // Print the information for each device.

  for (int i = 0; i < device_cnt; ++i) {
    // Get the device property.
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, i);

    // Print all device properties.
    printf("=============== Device %d ================\n", i);
    printf("name: %s\n", props.name);
    printf("totalGlobalMem: %lu\n", props.totalGlobalMem);
    printf("sharedMemPerBlock: %lu\n", props.sharedMemPerBlock);
    printf("regsPerBlock: %d\n", props.regsPerBlock);
    printf("warpSize: %d\n", props.warpSize);
    printf("memPitch: %lu\n", props.memPitch);
    printf("maxThreadsPerBlock: %d\n", props.maxThreadsPerBlock);
    printf("maxThreadsDim: %d, %d, %d\n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
    printf("maxGridSize: %d, %d, %d\n", props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
    printf("clockRate: %d\n", props.clockRate);
    printf("totalConstMem: %lu\n", props.totalConstMem);
    printf("major: %d\n", props.major);
    printf("minor: %d\n", props.minor);
    printf("textureAlignment: %lu\n", props.textureAlignment);
    printf("texturePitchAlignment: %lu\n", props.texturePitchAlignment);
    printf("deviceOverlap: %d\n", props.deviceOverlap);
    printf("multiProcessorCount: %d\n", props.multiProcessorCount);
    printf("kernelExecTimeoutEnabled: %d\n", props.kernelExecTimeoutEnabled);
    printf("integrated: %d\n", props.integrated);
    printf("canMapHostMemory: %d\n", props.canMapHostMemory);
    printf("computeMode: %d\n", props.computeMode);
    printf("maxTexture1D: %d\n", props.maxTexture1D);
    printf("maxTexture1DMipmap: %d\n", props.maxTexture1DMipmap);
    printf("maxTexture1DLinear: %d\n", props.maxTexture1DLinear);
    printf("maxTexture2D: %d, %d\n", props.maxTexture2D[0], props.maxTexture2D[1]);
    printf("maxTexture2DMipmap: %d, %d\n", props.maxTexture2DMipmap[0], props.maxTexture2DMipmap[1]);
    printf("maxTexture2DLinear: %d, %d, %d\n", props.maxTexture2DLinear[0], props.maxTexture2DLinear[1], props.maxTexture2DLinear[2]);
    printf("maxTexture2DGather: %d, %d\n", props.maxTexture2DGather[0], props.maxTexture2DGather[1]);
    printf("maxTexture3D: %d, %d, %d\n", props.maxTexture3D[0], props.maxTexture3D[1], props.maxTexture3D[2]);
    printf("maxTexture3DAlt: %d, %d, %d\n", props.maxTexture3DAlt[0], props.maxTexture3DAlt[1], props.maxTexture3DAlt[2]);
    printf("maxTextureCubemap: %d\n", props.maxTextureCubemap);
    printf("maxTexture1DLayered: %d, %d\n", props.maxTexture1DLayered[0], props.maxTexture1DLayered[1]);
    printf("maxTexture2DLayered: %d, %d, %d\n", props.maxTexture2DLayered[0], props.maxTexture2DLayered[1], props.maxTexture2DLayered[2]);
    printf("maxTextureCubemapLayered: %d, %d\n", props.maxTextureCubemapLayered[0], props.maxTextureCubemapLayered[1]);
    printf("maxSurface1D: %d\n", props.maxSurface1D);
    printf("maxSurface2D: %d, %d\n", props.maxSurface2D[0], props.maxSurface2D[1]);
    printf("maxSurface3D: %d, %d, %d\n", props.maxSurface3D[0], props.maxSurface3D[1], props.maxSurface3D[2]);
    printf("maxSurface1DLayered: %d, %d\n", props.maxSurface1DLayered[0], props.maxSurface1DLayered[1]);
    printf("maxSurface2DLayered: %d, %d, %d\n", props.maxSurface2DLayered[0], props.maxSurface2DLayered[1], props.maxSurface2DLayered[2]);
    printf("maxSurfaceCubemap: %d\n", props.maxSurfaceCubemap);
    printf("maxSurfaceCubemapLayered: %d, %d\n", props.maxSurfaceCubemapLayered[0], props.maxSurfaceCubemapLayered[1]);
    printf("surfaceAlignment: %lu\n", props.surfaceAlignment);
    printf("concurrentKernels: %d\n", props.concurrentKernels);
    printf("ECCEnabled: %d\n", props.ECCEnabled);
    printf("pciBusID: %d\n", props.pciBusID);
    printf("pciDeviceID: %d\n", props.pciDeviceID);
    printf("pciDomainID: %d\n", props.pciDomainID);
    printf("tccDriver: %d\n", props.tccDriver);
    printf("asyncEngineCount: %d\n", props.asyncEngineCount);
    printf("unifiedAddressing: %d\n", props.unifiedAddressing);
    printf("memoryClockRate: %d\n", props.memoryClockRate);
    printf("memoryBusWidth: %d\n", props.memoryBusWidth);
    printf("l2CacheSize: %d\n", props.l2CacheSize);
    printf("maxThreadsPerMultiProcessor: %d\n", props.maxThreadsPerMultiProcessor);
    printf("streamPrioritiesSupported: %d\n", props.streamPrioritiesSupported);
    printf("globalL1CacheSupported: %d\n", props.globalL1CacheSupported);
    printf("localL1CacheSupported: %d\n", props.localL1CacheSupported);
    printf("sharedMemPerMultiprocessor: %lu\n", props.sharedMemPerMultiprocessor);
    printf("regsPerMultiprocessor: %d\n", props.regsPerMultiprocessor);
    //printf("managedMemSupported: %d\n", props.managedMemSupported);
    printf("isMultiGpuBoard: %d\n", props.isMultiGpuBoard);
    printf("multiGpuBoardGroupID: %d\n", props.multiGpuBoardGroupID);
    printf("singleToDoublePrecisionPerfRatio: %d\n", props.singleToDoublePrecisionPerfRatio);
    printf("pageableMemoryAccess: %d\n", props.pageableMemoryAccess);
    printf("concurrentManagedAccess: %d\n", props.concurrentManagedAccess);
    //printf("computePreemptionSupported: %d\n", props.computePreemptionSupported);
    //printf("canUseHostPointerForRegisteredMem: %d\n", props.canUseHostPointerForRegisteredMem);
    //printf("cooperativeLaunch: %d\n", props.cooperativeLaunch);
    //printf("cooperativeMultiDeviceLaunch: %d\n", props.cooperativeMultiDeviceLaunch);
  }
  printf("==========================================\n");

  return;
}


#endif

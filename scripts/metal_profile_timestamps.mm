// Diagnostic only. Native GPU command-buffer times, with actual pipeline names.
// Build: clang++ -std=c++17 -fobjc-arc -dynamiclib -framework Foundation
//        -framework Metal metal_profile_timestamps.mm -o metal_profile.dylib
// Use with MLX_MAX_OPS_PER_BUFFER=0 for primitive-isolated attribution.
// This changes submission granularity and is never a performance baseline.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <atomic>
#include <unistd.h>

static char pipelineKey, encoderBufferKey, bufferNamesKey;
static FILE* output;
static std::mutex outputMutex;
static std::atomic<bool> enabled{false};
extern "C" void qwen_metal_profile_enable(int value) { enabled.store(value != 0); }
static id (*originalPipeline)(id, SEL, id, NSError**);
static id (*originalDescriptorPipeline)(id, SEL, id, MTLPipelineOption, id*, NSError**);
static id (*originalEncoder)(id, SEL, MTLDispatchType);
static void (*originalSetPipeline)(id, SEL, id);
static void (*originalCommit)(id, SEL);

static IMP replace(Class cls, SEL selector, IMP replacement) {
  Method method = class_getInstanceMethod(cls, selector);
  if (!method) { std::fprintf(stderr, "metal profile missing %s\n", sel_getName(selector)); std::abort(); }
  IMP original = method_getImplementation(method);
  if (!class_addMethod(cls, selector, replacement, method_getTypeEncoding(method)))
    method_setImplementation(class_getInstanceMethod(cls, selector), replacement);
  return original;
}

static id pipeline(id self, SEL cmd, id<MTLFunction> function, NSError** error) {
  id result = originalPipeline(self, cmd, function, error);
  if (result) objc_setAssociatedObject(result, &pipelineKey, function.name, OBJC_ASSOCIATION_COPY_NONATOMIC);
  return result;
}

static id descriptorPipeline(id self, SEL cmd, MTLComputePipelineDescriptor* descriptor,
                             MTLPipelineOption options, id* reflection, NSError** error) {
  id result = originalDescriptorPipeline(self, cmd, descriptor, options, reflection, error);
  if (result) objc_setAssociatedObject(result, &pipelineKey, descriptor.computeFunction.name,
                                      OBJC_ASSOCIATION_COPY_NONATOMIC);
  return result;
}

static id encoder(id self, SEL cmd, MTLDispatchType type) {
  id result = originalEncoder(self, cmd, type);
  if (result) objc_setAssociatedObject(result, &encoderBufferKey, self, OBJC_ASSOCIATION_ASSIGN);
  return result;
}

static void setPipeline(id self, SEL cmd, id<MTLComputePipelineState> state) {
  id buffer = objc_getAssociatedObject(self, &encoderBufferKey);
  if (buffer) {
    NSMutableArray* names = objc_getAssociatedObject(buffer, &bufferNamesKey);
    if (!names) {
      names = [NSMutableArray array];
      objc_setAssociatedObject(buffer, &bufferNamesKey, names, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }
    NSString* name = objc_getAssociatedObject(state, &pipelineKey) ?: state.label ?: @"unknown";
    [names addObject:name];
    [buffer setLabel:[names componentsJoinedByString:@";"]];
  }
  originalSetPipeline(self, cmd, state);
}

static void commit(id<MTLCommandBuffer> self, SEL cmd) {
  if (!enabled.load()) { originalCommit(self, cmd); return; }
  NSArray* names = [objc_getAssociatedObject(self, &bufferNamesKey) copy] ?: @[];
  [self addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
    @autoreleasepool {
      NSDictionary* record = @{@"pid": @(getpid()), @"kernels": names, @"gpu_start": @(buffer.GPUStartTime),
        @"gpu_end": @(buffer.GPUEndTime), @"status": @(buffer.status),
        @"gpu_seconds": @(buffer.GPUEndTime - buffer.GPUStartTime)};
      NSData* json = [NSJSONSerialization dataWithJSONObject:record options:0 error:nil];
      std::lock_guard<std::mutex> lock(outputMutex);
      std::fwrite(json.bytes, 1, json.length, output);
      std::fputc('\n', output);
      std::fflush(output);
    }
  }];
  originalCommit(self, cmd);
}

__attribute__((constructor)) static void install() {
  const char* path = std::getenv("QWEN_METAL_TIMESTAMPS");
  if (!path) return;
  enabled.store(std::getenv("QWEN_METAL_PROFILE_INITIAL") != nullptr);
  output = std::fopen(path, "a");
  if (!output) std::abort();
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> compute = [buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    Class deviceClass = object_getClass(device);
    originalPipeline = (decltype(originalPipeline))replace(deviceClass,
      @selector(newComputePipelineStateWithFunction:error:), (IMP)pipeline);
    originalDescriptorPipeline = (decltype(originalDescriptorPipeline))replace(deviceClass,
      @selector(newComputePipelineStateWithDescriptor:options:reflection:error:), (IMP)descriptorPipeline);
    originalEncoder = (decltype(originalEncoder))replace(object_getClass(buffer),
      @selector(computeCommandEncoderWithDispatchType:), (IMP)encoder);
    originalSetPipeline = (decltype(originalSetPipeline))replace(object_getClass(compute),
      @selector(setComputePipelineState:), (IMP)setPipeline);
    originalCommit = (decltype(originalCommit))replace(object_getClass(buffer), @selector(commit), (IMP)commit);
    [compute endEncoding]; // The discovery buffer is deliberately never submitted.
    std::fprintf(stderr, "metal profile installed (diagnostic timestamps only)\n");
  }
}

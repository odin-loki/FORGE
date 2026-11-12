// Declarations of the part of Foundation / Metal that gpu/src/metal.mm uses, so that a clang without the Apple SDK
// can parse that file as Objective-C++ (ctest gpu_compile_metal_host_syntax, Linux). It is NOT the Metal API: the
// selectors and types below were written from the documentation, so the check catches C++ and Objective-C syntax and
// type errors in metal.mm, not a misuse of Metal. The real compile is the macOS / iOS CI build.
#pragma once
#include <cstddef>
#include <cstdint>
#define TARGET_OS_OSX 1
typedef unsigned long NSUInteger;
typedef bool BOOL;
#define YES true
#define NO false
#define nil nullptr

__attribute__((objc_root_class))
@interface NSObject
+ (instancetype)alloc;
+ (instancetype)new;
- (instancetype)init;
@end
enum { NSUTF8StringEncoding = 4 };
@interface NSString : NSObject
+ (instancetype)stringWithUTF8String:(const char*)s;
- (instancetype)initWithBytes:(const void*)bytes length:(NSUInteger)len encoding:(NSUInteger)encoding;
@property(readonly) const char* UTF8String;
@end
@interface NSError : NSObject
@property(readonly) NSString* localizedDescription;
@end
@interface NSArray<ObjectType> : NSObject
+ (instancetype)arrayWithObjects:(const ObjectType[])objects count:(NSUInteger)cnt;
- (ObjectType)objectAtIndexedSubscript:(NSUInteger)idx;
@property(readonly) NSUInteger count;
@end

typedef struct { NSUInteger width, height, depth; } MTLSize;
static inline MTLSize MTLSizeMake(NSUInteger w, NSUInteger h, NSUInteger d) { MTLSize s = {w, h, d}; return s; }
typedef NSUInteger MTLResourceOptions;
enum { MTLResourceStorageModeShared = 0 };
enum MTLCommandBufferStatus { MTLCommandBufferStatusCompleted = 4, MTLCommandBufferStatusError = 5 };

@protocol MTLBuffer
@property(readonly) void* contents;
@end
@protocol MTLFunction
@end
@protocol MTLLibrary
- (id<MTLFunction>)newFunctionWithName:(NSString*)name;
@end
@protocol MTLComputePipelineState
@property(readonly) NSUInteger maxTotalThreadsPerThreadgroup;
@end
@protocol MTLComputeCommandEncoder
- (void)setComputePipelineState:(id<MTLComputePipelineState>)state;
- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)setBytes:(const void*)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)dispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)perGroup;
- (void)endEncoding;
@end
@protocol MTLCommandBuffer
- (id<MTLComputeCommandEncoder>)computeCommandEncoder;
- (void)commit;
- (void)waitUntilCompleted;
@property(readonly) MTLCommandBufferStatus status;
@end
@protocol MTLCommandQueue
- (id<MTLCommandBuffer>)commandBuffer;
@end
@interface MTLCompileOptions : NSObject
@property BOOL fastMathEnabled;
@end
@protocol MTLDevice
@property(readonly) NSString* name;
- (id<MTLCommandQueue>)newCommandQueue;
- (id<MTLLibrary>)newLibraryWithSource:(NSString*)source options:(MTLCompileOptions*)options error:(NSError**)error;
- (id<MTLComputePipelineState>)newComputePipelineStateWithFunction:(id<MTLFunction>)function error:(NSError**)error;
- (id<MTLBuffer>)newBufferWithBytes:(const void*)pointer length:(NSUInteger)length options:(MTLResourceOptions)options;
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options;
@end
NSArray<id<MTLDevice>>* MTLCopyAllDevices(void);
id<MTLDevice> MTLCreateSystemDefaultDevice(void);

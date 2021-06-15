// Generated from "C:/w/b/windows/pytorch/torch/csrc/jit/codegen/cuda/runtime/fp16_support.cu"
// 2021-03-24 03:08:11

namespace nvfuser_resources {

constexpr const char* fp16_support_cu = R"(
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short*>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short*>(&(var)))

struct __align__(2) __half {
  __host__ __device__ __half() {}

 protected:
  unsigned short __x;
};

__device__ __half __float2half(const float f) {
  __half val;
  asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(__HALF_TO_US(val)) : "f"(f));
  return val;
}

__device__ float __half2float(const __half h) {
  float val;
  asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(__HALF_TO_CUS(h)));
  return val;
}
)";

} // namespace nvfuser_resources

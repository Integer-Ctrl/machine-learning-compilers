#include "../../main/kernels/matmul_16_6_k.h"
#include "../../main/Brgemm.h"
#include "matmul.bench.h"
#include <benchmark/benchmark.h>

template <uint32_t TMdim, uint32_t TNdim, uint32_t TKdim> class GemmMxNxKFixture : public benchmark::Fixture
{
public:
  float matrix_a[TMdim * TKdim];
  float matrix_b[TKdim * TNdim];
  float matrix_c[TMdim * TNdim];
  double flops;

  void SetUp(::benchmark::State &) override
  {
    flops = 0;

    fill_random_matrix(matrix_a);
    fill_random_matrix(matrix_b);
    fill_random_matrix(matrix_c);
  }

  void TearDown(::benchmark::State &state) override
  {
    state.counters["FLOPS"] = benchmark::Counter(flops, benchmark::Counter::kIsRate);
  }
};

BENCHMARK_TEMPLATE_DEFINE_F(GemmMxNxKFixture, BM_matmul_16_6_k, 16, 6, 128)(benchmark::State &state)
{
  // Generate kernel
  mini_jit::Kernel native_kernel;
  mini_jit::kernels::matmul_16_6_k(native_kernel, 128);
  native_kernel.set_kernel();
  mini_jit::Brgemm::kernel_t kernel = reinterpret_cast<mini_jit::Brgemm::kernel_t>(
    const_cast<void *>(native_kernel.get_kernel()));  // Properly cast from const void* to kernel_t

  for (auto _ : state)
  {
    // Run kernel
    kernel(matrix_a, matrix_b, matrix_c, 16, 128, 16, 1, 1);
  }

  flops = (16 * 6 * 128) * 2;  // M * N * K * 2 instructions (add & mul)
  flops *= state.iterations();
};

BENCHMARK_REGISTER_F(GemmMxNxKFixture, BM_matmul_16_6_k)->MinWarmUpTime(1.0);  // WarmUp in seconds
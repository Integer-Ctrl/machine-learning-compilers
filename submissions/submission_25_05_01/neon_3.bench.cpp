#include <benchmark/benchmark.h>
#include "neon_3.h"

template <uint32_t TMdim, uint32_t TNdim, uint32_t TKdim>
class GemmMxNxKFixture : public benchmark::Fixture
{
public:
    float matrix_a[TMdim * TKdim];
    float matrix_b[TKdim * TNdim];
    float matrix_c[TMdim * TNdim];
    double flops;

    void SetUp(::benchmark::State &_) override
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

BENCHMARK_TEMPLATE_DEFINE_F(GemmMxNxKFixture, BM_matmul_16_6_64, 16, 6, 64)(benchmark::State &state)
{
    for (auto _ : state)
    {
        matmul_16_6_64(matrix_a, matrix_b, matrix_c, 16, 64, 16);
        flops += (4 * 6 * 4 * 2) * 64; // (4 fmla * 4 floats each * 2 instructions (add & mul) * 6 columns) * 64 K-Loop
    }
};

BENCHMARK_REGISTER_F(GemmMxNxKFixture, BM_matmul_16_6_64)->MinWarmUpTime(1.0); // WarmUp in seconds

BENCHMARK_TEMPLATE_DEFINE_F(GemmMxNxKFixture, BM_matmul_64_6_64, 64, 6, 64)(benchmark::State &state)
{
    for (auto _ : state)
    {
        matmul_64_6_64(matrix_a, matrix_b, matrix_c, 64, 64, 64);
        // ((4 fmla * 4 floats each * 2 instructions (add & mul) * 6 columns) * 64 K-Loop) * 4 M-Loop
        flops += ((4 * 6 * 4 * 2) * 64) * 4;
    }
}

BENCHMARK_REGISTER_F(GemmMxNxKFixture, BM_matmul_64_6_64)->MinWarmUpTime(1.0); // WarmUp in seconds

BENCHMARK_TEMPLATE_DEFINE_F(GemmMxNxKFixture, BM_matmul_64_48_64, 64, 48, 64)(benchmark::State &state)
{
    for (auto _ : state)
    {
        matmul_64_48_64(matrix_a, matrix_b, matrix_c, 64, 64, 64);
        // (((4 fmla * 4 floats each * 2 instructions (add & mul) * 6 columns) * 64 K-Loop) * 4 M-Loop) * 6 N-Loop
        flops += (((4 * 6 * 4 * 2) * 64) * 4) * 6;
    }
}

BENCHMARK_REGISTER_F(GemmMxNxKFixture, BM_matmul_64_48_64)->MinWarmUpTime(1.0); // WarmUp in seconds
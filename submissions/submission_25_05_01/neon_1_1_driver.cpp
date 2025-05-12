#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>

extern "C"
{
  /// @brief Execute the fmla 4s instruction for throughput benchmark.
  /// @param iterations The number of iterations the instructions are run.
  /// @return The number of processed instructions in a single loop.
  uint64_t throughput_fmla_4s(uint64_t iterations);

  /// @brief Execute the fmla 2s instruction for throughput benchmark.
  /// @param iterations The number of iterations the instructions are run.
  /// @return The number of processed instructions in a single loop.
  uint64_t throughput_fmla_2s(uint64_t iterations);

  /// @brief Execute the fmadd instruction for throughput benchmarks.
  /// @param iterations The number of iterations the instructions are run.
  /// @return The number of processed instructions in a single loop.
  uint64_t throughput_fmadd(uint64_t iterations);
}

int main()
{
  const uint64_t repetitions = 10'000'000;

  std::cout << "Running the Throughput \"FMLA 4s\" Benchmark:" << std::endl;
  const auto start_throughput_fmla_4s = std::chrono::high_resolution_clock::now();
  const uint64_t run_instructions_throughput_fmla_4s = throughput_fmla_4s(repetitions) * repetitions;
  const auto end_throughput_fmla_4s = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> diff_throughput_fmla_4s = end_throughput_fmla_4s - start_throughput_fmla_4s;
  std::cout << "Executed " << run_instructions_throughput_fmla_4s << " \"FMLA 4s\" Instructions in " << diff_throughput_fmla_4s.count()
            << " milliseconds." << std::endl
            << "Resulting in a Throughput of " << run_instructions_throughput_fmla_4s / diff_throughput_fmla_4s.count() * 1000
            << " Instructions per Second!" << std::endl
            << std::endl;

  std::cout << "Running the Throughput \"FMLA 2s\" Benchmark:" << std::endl;
  const auto start_throughput_fmla_2s = std::chrono::high_resolution_clock::now();
  const uint64_t run_instructions_throughput_fmla_2s = throughput_fmla_2s(repetitions) * repetitions;
  const auto end_throughput_fmla_2s = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> diff_throughput_fmla_2s = end_throughput_fmla_2s - start_throughput_fmla_2s;
  std::cout << "Executed " << run_instructions_throughput_fmla_2s << " \"FMLA 2s\" Instructions in " << diff_throughput_fmla_2s.count()
            << " milliseconds." << std::endl
            << "Resulting in a Throughput of " << run_instructions_throughput_fmla_2s / diff_throughput_fmla_2s.count() * 1000
            << " Instructions per Second!" << std::endl
            << std::endl;

  std::cout << "Running the Throughput \"FMADD\" Benchmark:" << std::endl;
  const auto start_throughput_fmadd = std::chrono::high_resolution_clock::now();
  const uint64_t run_instructions_throughput_fmadd = throughput_fmadd(repetitions) * repetitions;
  const auto end_throughput_fmadd = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> diff_throughput_fmadd = end_throughput_fmadd - start_throughput_fmadd;
  std::cout << "Executed " << run_instructions_throughput_fmadd << " \"FMADD\" Instructions in " << diff_throughput_fmadd.count()
            << " milliseconds." << std::endl
            << "Resulting in a Throughput of " << run_instructions_throughput_fmadd / diff_throughput_fmadd.count() * 1000
            << " Instructions per Second!" << std::endl
            << std::endl;

  return 0;
}
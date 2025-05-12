#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>

extern "C"
{
  /// @brief Execute the add instruction for throughput benchmark.
  /// @param iterations The number of iterations the instructions are run.
  /// @return The number of processed instructions in a single loop.
  uint64_t throughput_add(uint64_t iterations);

  /// @brief Execute the mul instruction for throughput benchmark.
  /// @param iterations The number of iterations the instructions are run.
  /// @return The number of processed instructions in a single loop.
  uint64_t throughput_mul(uint64_t iterations);

  /// @brief Execute the add instruction for latency benchmarks.
  /// @param iterations The number of iterations the instructions are run.
  /// @return The number of processed instructions in a single loop.
  uint64_t latency_add(uint64_t iterations);

  /// @brief Execute the mul instruction for latency benchmarks.
  /// @param iterations The number of iterations the instructions are run.
  /// @return The number of processed instructions in a single loop.
  uint64_t latency_mul(uint64_t iterations);
}

int main()
{
  const uint64_t repetitions = 1'000'000;

  std::cout << "Running the Throughput \"Add\" Benchmark:" << std::endl;
  const auto start_throughput_add = std::chrono::high_resolution_clock::now();
  const uint64_t run_instructions_throughput_add = throughput_add(repetitions) * repetitions;
  const auto end_throughput_add = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> diff_throughput_add = end_throughput_add - start_throughput_add;
  std::cout << "Executed " << run_instructions_throughput_add << " \"Add\" Instructions in " << diff_throughput_add.count()
            << " milliseconds." << std::endl
            << "Resulting in a Throughput of " << run_instructions_throughput_add / diff_throughput_add.count() * 1000
            << " Instructions per Second!" << std::endl
            << std::endl;

  std::cout << "Running the Throughput \"Mul\" Benchmark:" << std::endl;
  const auto start_throughput_mul = std::chrono::high_resolution_clock::now();
  const uint64_t run_instructions_throughput_mul = throughput_mul(repetitions) * repetitions;
  const auto end_throughput_mul = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> diff_throughput_mul = end_throughput_mul - start_throughput_mul;
  std::cout << "Executed " << run_instructions_throughput_mul << " \"Mul\" Instructions in " << diff_throughput_mul.count()
            << " milliseconds." << std::endl
            << "Resulting in a Throughput of " << run_instructions_throughput_mul / diff_throughput_mul.count() * 1000
            << " Instructions per Second!" << std::endl
            << std::endl;

  std::cout << "Running the Latency \"Add\" Benchmark:" << std::endl;
  const auto start_latency_add = std::chrono::high_resolution_clock::now();
  const uint64_t run_instructions_latency_add = latency_add(repetitions) * repetitions;
  const auto end_latency_add = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> diff_latency_add = end_latency_add - start_latency_add;
  std::cout << "Executed " << run_instructions_latency_add << " \"Add\" Instructions in " << diff_latency_add.count()
            << " milliseconds on a single Unit." << std::endl
            << "Resulting in a Throughput of " << run_instructions_latency_add / diff_latency_add.count() * 1000
            << " Instructions per Second!" << std::endl
            << std::endl;

  std::cout << "Running the Latency \"Mul\" Benchmark:" << std::endl;
  const auto start_latency_mul = std::chrono::high_resolution_clock::now();
  const uint64_t run_instructions_latency_mul = latency_mul(repetitions) * repetitions;
  const auto end_latency_mul = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> diff_latency_mul = end_latency_mul - start_latency_mul;
  std::cout << "Executed " << run_instructions_latency_mul << " \"Mul\" Instructions in " << diff_latency_mul.count()
            << " milliseconds on a single Unit." << std::endl
            << "Resulting in a Throughput of " << run_instructions_latency_mul / diff_latency_mul.count() * 1000
            << " Instructions per Second!" << std::endl
            << std::endl;

  return 0;
}

#include <cstdint>
 #include <iomanip>
 #include <iostream>
 #include <chrono>
 
 extern "C"
 {
     /// @brief Execute the fmla 4s instruction for latency benchmarks with dependency on one of the source registers
     /// @param iterations The number of iterations the instructions are run.
     /// @return The number of processed instructions in a single loop.
     uint64_t latency_fmla_4s_source(uint64_t iterations);

     /// @brief Execute the fmla 4s instruction for latency benchmarks with dependency on the destination register
     /// @param iterations The number of iterations the instructions are run.
     /// @return The number of processed instructions in a single loop.
     uint64_t latency_fmla_4s_destination(uint64_t iterations);
 }
 
 int main()
 {
     const uint64_t repetitions = 1'000'000;
 
     std::cout << "Running the Latency \"FMLA 4s\" Benchmark with dependency on one of the source registers:" << std::endl;
     const auto start_latency_fmla_4s_source = std::chrono::high_resolution_clock::now();
     const uint64_t run_instructions_latency_fmla_4s_source = latency_fmla_4s_source(repetitions) * repetitions;
     const auto end_latency_fmla_4s_source = std::chrono::high_resolution_clock::now();
     const std::chrono::duration<double, std::milli> diff_latency_fmla_4s_source = end_latency_fmla_4s_source - start_latency_fmla_4s_source;
     std::cout << "Executed " << run_instructions_latency_fmla_4s_source << " \"FMLA 4s\" Instructions in " << diff_latency_fmla_4s_source.count() << " milliseconds on a single Unit." << std::endl
                 << "Resulting in a Throughput of " << run_instructions_latency_fmla_4s_source / diff_latency_fmla_4s_source.count() * 1000 << " Instructions per Second!"
                 << std::endl
                 << std::endl;
    
    std::cout << "Running the Latency \"FMLA 4s\" Benchmark with dependency on one of the destination registers:" << std::endl;
    const auto start_latency_fmla_4s_destination = std::chrono::high_resolution_clock::now();
    const uint64_t run_instructions_latency_fmla_4s_destination = latency_fmla_4s_destination(repetitions) * repetitions;
    const auto end_latency_fmla_4s_destination = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> diff_latency_fmla_4s_destination = end_latency_fmla_4s_destination - start_latency_fmla_4s_destination;
    std::cout << "Executed " << run_instructions_latency_fmla_4s_destination << " \"FMLA 4s\" Instructions in " << diff_latency_fmla_4s_destination.count() << " milliseconds on a single Unit." << std::endl
                << "Resulting in a Throughput of " << run_instructions_latency_fmla_4s_destination / diff_latency_fmla_4s_destination.count() * 1000 << " Instructions per Second!"
                << std::endl
                << std::endl;

     return 0;
 }
import json
import csv
import os

# Input and output file paths
project_dir = os.path.dirname(os.path.dirname(__file__))
build_dir = os.path.join(project_dir, "build")
input_file = os.path.join(build_dir, "GEMM_benchmarks.json")
output_file = os.path.join(build_dir, "GEMM_benchmarks.csv")


# Read the JSON file
with open(input_file, "r") as json_file:
    data = json.load(json_file)

# Extract the benchmark results
benchmarks = data.get("benchmarks", [])

# Write to the CSV file
with open(output_file, "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write the header
    header = [
        "m", "n", "k", "br_size", "trans_a", "trans_b", "trans_c", "ld_a", "ld_b", "ld_c", "br_stride_a", "br_stride_b", "num_reps", "time",
    ]
    csv_writer.writerow(header)
    
    # Write each benchmark result
    for benchmark in benchmarks:

        _, name, m_str, n_str, k_str, _ = benchmark["name"].split("/")

        if(name != "BM_matmul"):
            continue

        m = int(m_str.split(":")[-1])
        n = int(n_str.split(":")[-1])
        k = int(k_str.split(":")[-1])

        nano_seconds_to_seconds = 1e-9
        total_time = benchmark["real_time"] * benchmark["iterations"] * nano_seconds_to_seconds

        row = [
            m,  # m
            n,  # n
            k,  # k
            1,  # br_size
            0,  # trans_a
            0,  # trans_b
            0,  # trans_c
            m,  # ld_a
            n,  # ld_b
            m,  # ld_c
            0,  # br_stride_a
            0,  # br_stride_b
            benchmark["iterations"],  # num_reps
            total_time,  # time in seconds of all iterations
        ]
        csv_writer.writerow(row)

print(f"Converted JSON: {input_file}\n        to CSV: {output_file}")
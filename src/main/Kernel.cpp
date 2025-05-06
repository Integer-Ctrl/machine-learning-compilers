#include "Kernel.h"
#include <sys/mman.h>
#include <fstream>
#include <stdexcept>
#include <cerrno>
#include <cstring>
#include <string>
#include "release_assert.h"

void* mini_jit::Kernel::allocate_mmap(std::size_t size_bytes) const
{
    release_assert(size_bytes == 0, "Mmap allocation should not be size 0!");

    void* memory = mmap(0, size_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (memory == MAP_FAILED) {
        throw std::runtime_error("Failed to allocate memory: " + std::string(std::strerror(errno)));
    }

    return memory;
}

void mini_jit::Kernel::release_mmap(std::size_t size_bytes, void* memory) const
{
    int release = munmap(memory, size_bytes);

    if (release == -1) {
        throw std::runtime_error("Failed to release memory");
    }
}

void mini_jit::Kernel::set_executable(std::size_t size_bytes, void* memory) const
{
    int result = mprotect(memory, size_bytes, PROT_READ | PROT_EXEC);

    if (result == -1) {
        throw std::runtime_error("Failed to set memory executable: " + std::string(std::strerror(errno)));
    }
}

void mini_jit::Kernel::release_memory()
{
    if (kernel != nullptr) {
        release_mmap(size_alloc, kernel);
    }
    size_alloc = 0;

    kernel = nullptr;
}

mini_jit::Kernel::~Kernel() noexcept
{
    release_memory();
}

void mini_jit::Kernel::add(uint32_t instruction)
{
    buffer.push_back(instruction);
}

void mini_jit::Kernel::add(std::vector<uint32_t> instructions)
{
    // reserve improves performance
    buffer.reserve(buffer.size() + std::distance(instructions.begin(), instructions.end()));
    buffer.insert(buffer.end(), instructions.begin(), instructions.end());
}

std::size_t mini_jit::Kernel::get_size() const
{
    return buffer.size() * sizeof(uint32_t);
}

void mini_jit::Kernel::set_kernel()
{
    release_memory();

    if (buffer.empty()) {
        return;
    }

    // alloc kernel memory
    size_alloc = get_size();
    try {
        kernel = (void*)allocate_mmap(size_alloc);
    }
    catch (std::runtime_error& e) {
        throw std::runtime_error("Failed to allocate memory for kernel: " + std::string(e.what()));
    }

    // copy machine code from buffer to kernel memory
    std::copy(std::begin(buffer), std::end(buffer), reinterpret_cast<uint32_t*>(kernel));

    // clear cache
    char* kernel_ptr = reinterpret_cast<char*>(kernel);
    __builtin___clear_cache(kernel_ptr, kernel_ptr + get_size());

    // set executable
    set_executable(size_alloc, kernel);
}

void const* mini_jit::Kernel::get_kernel() const
{
    return kernel;
}

void mini_jit::Kernel::write(char const* path) const
{
    std::ofstream out(path, std::ios::out | std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file: " + std::string(path));
    }

    out.write(reinterpret_cast<char const*>(buffer.data()), get_size());
}

#ifndef MINI_JIT_KERNEL_H
#define MINI_JIT_KERNEL_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mini_jit
{

  class Kernel
  {
  private:
    //! high-level code buffer
    std::vector<uint32_t> buffer;

    //! size of the kernel
    std::size_t size_allocate = 0;

    //! executable kernel
    void *kernel = nullptr;

    /**
     * Allocates memory through POSIX mmap.
     *
     * @param size_bytes size in bytes.
     **/
    void *allocate_mmap(std::size_t size_bytes) const;

    /**
     * Release POSIX mmap allocated memory.
     *
     * @param size_bytes size in bytes.
     * @param memory pointer to memory which is released.
     **/
    void release_mmap(std::size_t size_bytes, void *memory) const;

    /**
     * Sets the given memory region executable.
     *
     * @param size_bytes number of bytes.
     * @param memory point to memory.
     **/
    void set_executable(std::size_t size_bytes, void *memory) const;

    /**
     * Release memory of the kernel if allocated.
     **/
    void release_memory();

  public:
    /**
     * Constructor
     **/
    Kernel(){};

    /**
     * Destructor
     **/
    ~Kernel() noexcept;

    Kernel(Kernel const &) = delete;
    Kernel &operator=(Kernel const &) = delete;
    Kernel(Kernel &&) noexcept = delete;
    Kernel &operator=(Kernel &&) noexcept = delete;

    /**
     * Adds an instruction to the code buffer.
     *
     * @param instruction instruction which is added.
     **/
    void add(uint32_t instruction);

    /**
     * Adds an instruction to the code buffer.
     *
     * @param instructions instructions which are added.
     **/
    void add(std::vector<uint32_t> instructions);

    /**
     * Gets the size of the code buffer.
     *
     * @return size of the code buffer in bytes.
     **/
    std::size_t get_size() const;

    /**
     * Sets the kernel based on the code buffer.
     **/
    void set_kernel();

    /**
     * Gets a pointer to the executable kernel.
     **/
    void const *get_kernel() const;

    /**
     * Writes the code buffer to the given file.
     *
     * @param path path to the file.
     **/
    void write(char const *path) const;
  };

}  // namespace mini_jit
#endif
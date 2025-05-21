#include <cstdint>
#include <cstdlib>
#include <ctime>

/**
 * @brief Fill the given matrix with random values.
 *
 * @tparam TSize The total size of the matrix.
 * @param matrix The matrix to write to.
 */
template <uint32_t TSize> void fill_random_matrix(float (&matrix)[TSize])
{
  std::srand(std::time(0));
  for (size_t i = 0; i < TSize; i++)
  {
    matrix[i] = (static_cast<float>(std::rand())) / (static_cast<float>(std::rand()));
  }
}

/**
 * @brief Fill the given matrix with random values.
 *
 * @param matrix The matrix to write to.
 * @param matrix_size The number of elements in the matrix.
 */
inline void fill_random_matrix_args(float *matrix, size_t matrix_size)
{
  static bool initialized = false;
  if (!initialized)
  {
    std::srand(std::time(0));  // Seed RNG once
    initialized = true;
  }

  for (size_t i = 0; i < matrix_size; i++)
  {
    matrix[i] = (static_cast<float>(std::rand())) / (static_cast<float>(std::rand()));
  }
}
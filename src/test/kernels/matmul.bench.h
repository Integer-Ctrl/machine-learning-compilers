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
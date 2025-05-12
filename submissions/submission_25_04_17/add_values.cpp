#include <iostream>

extern "C"
{
  void add_values(int32_t *a, int32_t *b, int32_t *c);
}

int main()
{
  int32_t a = 123;
  int32_t b = 1214;
  int32_t c;

  int32_t *a_ptr = &a;
  int32_t *b_ptr = &b;
  int32_t *c_ptr = &c;

  add_values(a_ptr, b_ptr, c_ptr);

  std::cout << "a / b / return value c: " << a << "/" << b << "/" << c << std::endl;

  return 0;
}
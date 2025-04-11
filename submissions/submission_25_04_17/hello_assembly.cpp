#include <iostream>

extern "C"
{
    void hello_assembly();
}

int main()
{
    hello_assembly();
    return 0;
}
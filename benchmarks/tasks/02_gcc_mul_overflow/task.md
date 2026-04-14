You are debugging a wrong-code bug in GCC's match.pd pattern matching.

The bug: `__builtin_mul_overflow_p` with unsigned operands gives incorrect results. The simplification of `__imag__ .MUL_OVERFLOW` in match.pd fails to check whether the type is unsigned, leading to wrong overflow detection for unsigned multiplication.

The code is at /scratch

Reproducer:
```c
// gcc -O2 test.c -o test && ./test
#include <stdio.h>
int main() {
    unsigned a = __INT_MAX__;
    unsigned b = 2;
    printf("overflow: %d\n", __builtin_mul_overflow_p(a, b, (unsigned)0));
    // Should print 1 (overflow), but prints 0 (no overflow) — WRONG
    return 0;
}
```

The fix should be in gcc/match.pd — the pattern that simplifies `__imag__ .MUL_OVERFLOW` needs a `!TYPE_UNSIGNED` guard. Look for the MUL_OVERFLOW simplification patterns.

Build and test:
```bash
cd /scratch
mkdir -p build && cd build
../configure --disable-multilib --disable-bootstrap --enable-languages=c,c++
make -j$(nproc)
./gcc/xgcc -B./gcc -O2 ../test.c -o test && ./test
```

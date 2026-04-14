You are debugging a wrong-code bug in GCC's value numbering pass.

The bug: A struct copy in a loop produces the wrong value at -O2. The root cause is in tree-ssa-sccvn.cc — an unsigned/signed mismatch in an offset comparison causes incorrect aggregate copy translation.

The code is at /scratch

Reproducer:
```c
// gcc -O2 test.c -o test && ./test
struct S { int a, b; };
struct S s = { 0, 5 };
int main() {
    struct S t;
    for (int i = 0; i < 1; i++)
        t = s;
    if (t.b != 5) __builtin_abort();
    return 0;
}
```
Expected: exits normally. At -O2: calls abort (t.b is 0 instead of 5).

The bug is somewhere in the SCC-based value numbering (tree-ssa-sccvn.cc). Look at how aggregate copies are translated and how offsets are compared.

Build and test:
```bash
cd /scratch
mkdir -p build && cd build
../configure --disable-multilib --disable-bootstrap --enable-languages=c
make -j$(nproc)
./gcc/xgcc -B./gcc -O2 ../test.c -o test && ./test
```

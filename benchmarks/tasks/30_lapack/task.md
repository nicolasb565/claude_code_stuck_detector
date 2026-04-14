You are debugging a numerical bug in LAPACK's divide-and-conquer SVD.

The bug: DGELSD (least-squares via SVD) fails with "SVD fails to converge" on certain matrices with clustered singular values. The root cause is deep in the divide-and-conquer tree — an ordering invariant violation in DLASD7 causes deflated singular values to be incorrectly sorted, which cascades through the algorithm.

The code is at /scratch

The call chain is: DGELSD -> DLALSD -> DLASD0 -> DLASD1 -> DLASD7. The bug is in SRC/dlasd7.f (and its single-precision counterpart slasd7.f).

Hint: Look at how deflated singular values are ordered in DLASD7. The documentation claims they are in increasing order, but is that actually enforced? Compare with DLAED8 which has similar logic but uses an insertion-sort workaround for the same problem.

Build and test:
```bash
cd /scratch
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
make -j$(nproc)
ctest --test-dir TESTING -R "xeigtstz|xlsdriver" --output-on-failure
```

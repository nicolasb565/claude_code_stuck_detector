You are debugging a miscompilation bug in LLVM's loop vectorizer.

The bug: The loop vectorizer incorrectly replaces a reduction recipe with a SCEV-computed live-in value, causing a simple 2-iteration loop to return i32 2 instead of the correct i32 1.

The code is at /scratch

Reproducer (LLVM IR):
```
define i32 @test() {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %red = phi i8 [ 0, %entry ], [ %red.next, %loop ]
  %red.next = add i8 %red, 1
  %iv.next = add i32 %iv, 1
  %cmp = icmp ult i32 %iv.next, 2
  br i1 %cmp, label %loop, label %exit

exit:
  %res = zext i8 %red.next to i32
  ret i32 %res
}
```
Expected: returns 2. After vectorization: returns wrong value.

The fix should be in llvm/lib/Transforms/Vectorize/VPlanTransforms.cpp — the SCEV simplification should only apply to live-ins, not to header-phi recipes like reductions.

Build and test:
```bash
cmake -G Ninja -S llvm -B build -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=X86
ninja -C build opt
./build/bin/opt -p loop-vectorize test.ll -S
```

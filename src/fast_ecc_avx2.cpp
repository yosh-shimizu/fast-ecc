// The AVX2 + FMA instance of fast-ecc: src/fast_ecc.cpp compiled a second
// time with /arch:AVX2 (or -mavx2 -mfma) and OpenCV's 256-bit universal
// intrinsics.  CMake adds this file to the library on x86 (FAST_ECC_DISPATCH,
// on by default) with those flags, and the first instance picks this one at
// run time when the CPU has AVX2 and FMA.  If you build fast-ecc by hand
// this file is optional: without it the library runs on 128-bit vectors;
// with FASTECC_AVX2=1 on the single file it runs on 256-bit vectors only.
#define FASTECC_AVX2 1
#define FASTECC_AVX2_TU 1
#include "fast_ecc.cpp"

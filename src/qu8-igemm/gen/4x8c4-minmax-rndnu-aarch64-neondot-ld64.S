// Auto-generated file. Do not edit!
//   Template: src/qu8-igemm/4x8c4-aarch64-neondot-ld64.S.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <xnnpack/assembly.h>

# void xnn_qu8_igemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_ld64(
#     size_t mr,                 x0
#     size_t nc,                 x1
#     size_t kc,                 x2 / x0
#     size_t ks,                 x3 / x9
#     const int8_t**restrict a,  x4
#     const int8_t* restrict w,  x5
#     int8_t* restrict c,        x6
#     size_t cm_stride,          x7
#     size_t cn_stride,          [sp] -> x10
#     size_t a_offset,           [sp + 8] -> x8
#     const int8_t* zero,        [sp + 16] -> x12
#     const union xnn_qu8_conv_minmax_params params) [sp + 24] -> (x11)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

# Register usage
# A0  x13  v0
# A1  x14  v1
# A2  x15  v2
# A3  x11  v3
# B    x5  v4  v5  v6
# C0   x6 v16 v20
# C1  x16 v17 v21
# C2  x17 v18 v22
# C3   x7 v19 v23
# zero_point v7 v24 v25 v26 v27
# unused  v8 v9 v10 v11 v13 v14 v15 v28 v29 v30 v31

BEGIN_FUNCTION xnn_qu8_igemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_ld64

        # Clamp C pointers
        CMP     x0, 2                   // if mr < 2
        LDP     x10, x8, [sp]           // Load cn_stride, a_offset
        ADD     x16, x6, x7             // c1 = c0 + cm_stride
        CSEL    x16, x6,  x16, LO       //   c1 = c0
        ADD     x2, x2, 3               // kc = (kc + 3) & ~3

        ADD     x17, x16, x7            // c2 = c1 + cm_stride
        LDP     x12, x11, [sp, 16]      // Load zero pointer, params
                                        // if mr <= 2
        CSEL    x17, x16, x17, LS       //   c2 = c1
        BIC     x2, x2, 3

        CMP     x0, 4                   // if mr < 4
        ADD     x7,  x17, x7            // c3 = c2 + cm_stride
        CSEL    x7,  x17, x7, LO        //   c3 = c2

        LD1R    {v7.4s}, [x11], 4       // kernel_zero_point

        .p2align 3
0:
        # Load initial bias from w into accumulators
        LDP     q16, q20, [x5], 32
        MOV     v17.16b, v16.16b
        MOV     v18.16b, v16.16b
        MOV     v19.16b, v16.16b
        MOV     v21.16b, v20.16b
        MOV     v22.16b, v20.16b
        MOV     v23.16b, v20.16b
        MOVI    v24.16b, 0
        MOVI    v25.16b, 0
        MOVI    v26.16b, 0
        MOVI    v27.16b, 0

        MOV     x9, x3                  // p = ks

        .p2align 3
1:
        # Load next 4 A pointers
        LDP     x13, x14, [x4], 16
        LDP     x15, x11, [x4], 16

        CMP     x13, x12                // if a0 == zero
        ADD     x13, x13, x8            // a0 += a_offset
        CSEL    x13, x12, x13, EQ       //   a0 = zero, else += a0 + a_offset
        CMP     x14, x12                // if a1 == zero
        ADD     x14, x14, x8            // a1 += a_offset
        CSEL    x14, x12, x14, EQ       //   a1 = zero, else += a1 + a_offset
        CMP     x15, x12                // if a2 == zero
        ADD     x15, x15, x8            // a2 += a_offset
        CSEL    x15, x12, x15, EQ       //   a2 = zero, else += a2 + a_offset
        CMP     x11, x12                // if a3 == zero
        ADD     x11, x11, x8            // a3 += a_offset
        CSEL    x11, x12, x11, EQ       //   a3 = zero, else += a3 + a_offset

        # Is there at least 8 bytes for main loop?
        SUBS    x0, x2, 8               // k = kc - 8
        B.LO    4f

        # Main loop - 8 bytes of A
        .p2align 3
2:
        LDR     d0, [x13], 8
        LDR     q4,  [x5]
        LDR     d1, [x14], 8
        LDR     d2, [x15], 8
        LDR     d3, [x11], 8
        LDR     q5,  [x5, 16]
        UDOT    v24.2s, v7.8b, v0.8b
        UDOT    v25.2s, v7.8b, v1.8b
        UDOT    v26.2s, v7.8b, v2.8b
        UDOT    v27.2s, v7.8b, v3.8b
        UDOT    v16.4s, v4.16b, v0.4b[0]
        UDOT    v17.4s, v4.16b, v1.4b[0]
        LDR     q6,  [x5, 32]
        UDOT    v18.4s, v4.16b, v2.4b[0]
        UDOT    v19.4s, v4.16b, v3.4b[0]
        UDOT    v20.4s, v5.16b, v0.4b[0]
        UDOT    v21.4s, v5.16b, v1.4b[0]
        LDR     q4,  [x5, 48]
        UDOT    v22.4s, v5.16b, v2.4b[0]
        UDOT    v23.4s, v5.16b, v3.4b[0]
        UDOT    v16.4s, v6.16b, v0.4b[1]
        UDOT    v17.4s, v6.16b, v1.4b[1]
        UDOT    v18.4s, v6.16b, v2.4b[1]
        UDOT    v19.4s, v6.16b, v3.4b[1]
        ADD     x5, x5, 64
        UDOT    v20.4s, v4.16b, v0.4b[1]
        UDOT    v21.4s, v4.16b, v1.4b[1]
        SUBS    x0, x0, 8
        UDOT    v22.4s, v4.16b, v2.4b[1]
        UDOT    v23.4s, v4.16b, v3.4b[1]
        B.HS    2b

        # Is there a remainder?- 4 bytes of A
        TBNZ    x0, 2, 4f

        # ks loop
        SUBS    x9, x9, 32              // ks -= MR * sizeof(int8_t*)
        B.HI    1b

3:
        ADDP    v0.2s, v24.2s, v25.2s
        ADDP    v1.2s, v26.2s, v27.2s
        LDR     x11, [sp, 24]           // reload params pointer
        DUP     v24.4s, v0.s[0]
        DUP     v25.4s, v0.s[1]
        DUP     v26.4s, v1.s[0]
        DUP     v27.4s, v1.s[1]
        ADD     x11, x11, 4

        # Subtract zero point from accumulators
        SUB     v16.4s, v16.4s, v24.4s
        SUB     v17.4s, v17.4s, v25.4s
        SUB     v18.4s, v18.4s, v26.4s
        SUB     v19.4s, v19.4s, v27.4s
        SUB     v20.4s, v20.4s, v24.4s
        SUB     v21.4s, v21.4s, v25.4s
        SUB     v22.4s, v22.4s, v26.4s
        SUB     v23.4s, v23.4s, v27.4s

        # Apply params - preshift, scale, postshift, bias and clamp
        LD1R    {v4.4s}, [x11], 4
        SSHL    v16.4s, v16.4s, v4.4s   // shift to upper bits
        SSHL    v17.4s, v17.4s, v4.4s
        SSHL    v18.4s, v18.4s, v4.4s
        SSHL    v19.4s, v19.4s, v4.4s
        LD1R    {v5.4s}, [x11], 4
        SSHL    v20.4s, v20.4s, v4.4s
        SSHL    v21.4s, v21.4s, v4.4s
        SSHL    v22.4s, v22.4s, v4.4s
        SSHL    v23.4s, v23.4s, v4.4s
        LD1R    {v6.4s}, [x11], 4
        SQDMULH v16.4s, v16.4s, v5.4s   // scale without rounding
        SQDMULH v17.4s, v17.4s, v5.4s
        SQDMULH v18.4s, v18.4s, v5.4s
        SQDMULH v19.4s, v19.4s, v5.4s
        SQDMULH v20.4s, v20.4s, v5.4s
        SQDMULH v21.4s, v21.4s, v5.4s
        SQDMULH v22.4s, v22.4s, v5.4s
        SQDMULH v23.4s, v23.4s, v5.4s
        SRSHL   v16.4s, v16.4s, v6.4s   // signed rounding shift left
        SRSHL   v17.4s, v17.4s, v6.4s
        SRSHL   v18.4s, v18.4s, v6.4s
        SRSHL   v19.4s, v19.4s, v6.4s
        SRSHL   v20.4s, v20.4s, v6.4s
        SRSHL   v21.4s, v21.4s, v6.4s
        SRSHL   v22.4s, v22.4s, v6.4s
        SRSHL   v23.4s, v23.4s, v6.4s

        SQXTN   v16.4h, v16.4s
        SQXTN   v17.4h, v17.4s
        SQXTN   v18.4h, v18.4s
        SQXTN   v19.4h, v19.4s
        LD1R    {v6.8h}, [x11], 2        // add bias

        SQXTN2  v16.8h, v20.4s
        SQXTN2  v17.8h, v21.4s
        SQXTN2  v18.8h, v22.4s
        SQXTN2  v19.8h, v23.4s

        SQADD   v16.8h, v16.8h, v6.8h
        SQADD   v17.8h, v17.8h, v6.8h
        SQADD   v18.8h, v18.8h, v6.8h
        SQADD   v19.8h, v19.8h, v6.8h
        LD1R    {v4.16b}, [x11], 1       // clamp min value

        SQXTUN  v0.8b, v16.8h
        SQXTUN  v1.8b, v18.8h
        LD1R    {v5.16b}, [x11]          // clamp max value
        SQXTUN2 v0.16b, v17.8h
        SQXTUN2 v1.16b, v19.8h

        UMAX    v0.16b, v0.16b, v4.16b
        UMAX    v1.16b, v1.16b, v4.16b
        SUBS    x1, x1, 8
        UMIN    v0.16b, v0.16b, v5.16b
        UMIN    v1.16b, v1.16b, v5.16b
        B.LO    5f

        # Store full 4 x 8
        ST1     {v1.d}[1],  [x7], x10
        ST1     {v1.8b}, [x17], x10
        ST1     {v0.d}[1], [x16], x10
        ST1     {v0.8b},  [x6], x10
        SUB     x4, x4, x3              // a -= ks

        # nc loop
        B.HI    0b
        RET

        # Remainder- 4 bytes of A
        .p2align 3
4:
        LDR     s0, [x13], 4
        LDR     q4, [x5], 16
        LDR     s1, [x14], 4
        LDR     s2, [x15], 4
        LDR     s3, [x11], 4
        LDR     q5, [x5], 16
        UDOT    v24.2s, v7.8b, v0.8b
        UDOT    v25.2s, v7.8b, v1.8b
        UDOT    v26.2s, v7.8b, v2.8b
        UDOT    v27.2s, v7.8b, v3.8b
        SUBS    x9, x9, 32              // ks -= MR * sizeof(int8_t*)
        UDOT    v16.4s, v4.16b, v0.4b[0]
        UDOT    v17.4s, v4.16b, v1.4b[0]
        UDOT    v18.4s, v4.16b, v2.4b[0]
        UDOT    v19.4s, v4.16b, v3.4b[0]
        UDOT    v20.4s, v5.16b, v0.4b[0]
        UDOT    v21.4s, v5.16b, v1.4b[0]
        UDOT    v22.4s, v5.16b, v2.4b[0]
        UDOT    v23.4s, v5.16b, v3.4b[0]

        # ks loop
        B.HI    1b
        B       3b

        # Store odd width
        .p2align 3
5:
        TBZ     x1, 2, 6f
        ST1     {v1.s}[2], [x7], 4
        STR     s1, [x17], 4
        ST1     {v0.s}[2], [x16], 4
        STR     s0, [x6], 4
        EXT     v0.16b, v0.16b, v0.16b, 4
        EXT     v1.16b, v1.16b, v1.16b, 4
6:
        TBZ     x1, 1, 7f
        ST1     {v1.h}[4], [x7], 2
        STR     h1, [x17], 2
        ST1     {v0.h}[4], [x16], 2
        STR     h0, [x6], 2
        EXT     v0.16b, v0.16b, v0.16b, 2
        EXT     v1.16b, v1.16b, v1.16b, 2
7:
        TBZ     x1, 0, 8f
        ST1     {v1.b}[8], [x7]
        STR     b1, [x17]
        ST1     {v0.b}[8], [x16]
        STR     b0, [x6]
8:
        RET

END_FUNCTION xnn_qu8_igemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_ld64

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif

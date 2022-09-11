/* Copyright (C) 1995-2011, 2016 Mark Adler
 * Copyright (C) 2017 ARM Holdings Inc.
 * Authors:
 *   Adenilson Cavalcanti <adenilson.cavalcanti@arm.com>
 *   Adam Stylinski <kungfujesus06@gmail.com>
 * For conditions of distribution and use, see copyright notice in zlib.h
 */
#ifdef ARM_NEON_ADLER32
#ifdef _M_ARM64
#  include <arm64_neon.h>
#else
#  include <arm_neon.h>
#endif
#include "../../zbuild.h"
#include "../../adler32_p.h"
#include "../../fallback_builtins.h"

static void NEON_accum32(uint32_t *s, const uint8_t *buf, uint64_t len) {
    static const uint8_t taps[32] = {
        32, 31, 30, 29, 28, 27, 26, 25,
        24, 23, 22, 21, 20, 19, 18, 17,
        16, 15, 14, 13, 12, 11, 10, 9,
        8, 7, 6, 5, 4, 3, 2, 1 };

    uint32x2_t adacc2, s2acc2, as;
    uint8x16_t t0 = vld1q_u8(taps), t1 = vld1q_u8(taps + 16);

    uint32x4_t adacc = vdupq_n_u32(0), s2acc = vdupq_n_u32(0);
    adacc = vsetq_lane_u32(s[0], adacc, 0);
    s2acc = vsetq_lane_u32(s[1], s2acc, 0);

    while (len >= 2) {
        uint8x16_t d0 = vld1q_u8(buf), d1 = vld1q_u8(buf + 16);
        uint16x8_t adler, sum2;
        s2acc = vaddq_u32(s2acc, vshlq_n_u32(adacc, 5));
        adler = vpaddlq_u8(       d0);
        adler = vpadalq_u8(adler, d1);
        sum2 = vmull_u8(      vget_low_u8(t0), vget_low_u8(d0));
        sum2 = vmlal_u8(sum2, vget_high_u8(t0), vget_high_u8(d0));
        sum2 = vmlal_u8(sum2, vget_low_u8(t1), vget_low_u8(d1));
        sum2 = vmlal_u8(sum2, vget_high_u8(t1), vget_high_u8(d1));
        adacc = vpadalq_u16(adacc, adler);
        s2acc = vpadalq_u16(s2acc, sum2);
        len -= 2;
        buf += 32;
    }

    while (len > 0) {
        uint8x16_t d0 = vld1q_u8(buf);
        uint16x8_t adler, sum2;
        s2acc = vaddq_u32(s2acc, vshlq_n_u32(adacc, 4));
        adler = vpaddlq_u8(d0);
        sum2 = vmull_u8(      vget_low_u8(t1), vget_low_u8(d0));
        sum2 = vmlal_u8(sum2, vget_high_u8(t1), vget_high_u8(d0));
        adacc = vpadalq_u16(adacc, adler);
        s2acc = vpadalq_u16(s2acc, sum2);
        buf += 16;
        len--;
    }
    adacc2 = vpadd_u32(vget_low_u32(adacc), vget_high_u32(adacc));
    s2acc2 = vpadd_u32(vget_low_u32(s2acc), vget_high_u32(s2acc));
    as = vpadd_u32(adacc2, s2acc2);
    s[0] = vget_lane_u32(as, 0);
    s[1] = vget_lane_u32(as, 1);
}

static void NEON_handle_tail(uint32_t *pair, const uint8_t *buf, uint64_t len) {
    unsigned int i;
    for (i = 0; i < len; ++i) {
        pair[0] += buf[i];
        pair[1] += pair[0];
    }
}

uint32_t adler32_neon(uint32_t adler, const uint8_t *buf, uint64_t len) {
    /* split Adler-32 into component sums */
    uint32_t sum2 = (adler >> 16) & 0xffff;
    adler &= 0xffff;

    /* in case user likes doing a byte at a time, keep it fast */
    if (len == 1)
        return adler32_len_1(adler, buf, sum2);

    /* initial Adler-32 value (deferred check for len == 1 speed) */
    if (buf == NULL)
        return 1L;

    /* in case short lengths are provided, keep it somewhat fast */
    if (len < 16)
        return adler32_len_16(adler, buf, len, sum2);

    uint32_t pair[2];
    int n = NMAX;
    unsigned int done = 0;

    /* Split Adler-32 into component sums, it can be supplied by
     * the caller sites (e.g. in a PNG file).
     */
    pair[0] = adler;
    pair[1] = sum2;

    /* If memory is not SIMD aligned, do scalar sums to an aligned
     * offset, provided that doing so doesn't completely eliminate
     * SIMD operation. Aligned loads are still faster on ARM, even
     * though there's no explicit aligned load instruction */
    unsigned int align_offset = ((uintptr_t)buf & 15);
    unsigned int align_adj = (align_offset) ? 16 - align_offset : 0;

    if (align_offset && len >= (16 + align_adj)) {
        NEON_handle_tail(pair, buf, align_adj);
        n -= align_adj;
        done += align_adj;

    } else {
        /* If here, we failed the len criteria test, it wouldn't be
         * worthwhile to do scalar aligning sums */
        align_adj = 0;
    }

    while (done < len) {
        int remaining = (int)(len - done);
        n = MIN(remaining, (done == align_adj) ? n : NMAX);

        if (n < 16)
            break;

        NEON_accum32(pair, buf + done, n >> 4);
        pair[0] %= BASE;
        pair[1] %= BASE;

        int actual_nsums = (n >> 4) << 4;
        done += actual_nsums;
    }

    /* Handle the tail elements. */
    if (done < len) {
        NEON_handle_tail(pair, (buf + done), len - done);
        pair[0] %= BASE;
        pair[1] %= BASE;
    }

    /* D = B * 65536 + A, see: https://en.wikipedia.org/wiki/Adler-32. */
    return (pair[1] << 16) | pair[0];
}

#endif

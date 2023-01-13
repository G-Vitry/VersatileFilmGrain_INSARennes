/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2022, InterDigital
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted (subject to the limitations in the disclaimer below) provided that
 * the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of InterDigital nor the names of its contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS
 * LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "vfgs_hw.h"
#include <string.h> // memcpy
#include <assert.h>
#include <stdio.h>

#include <emmintrin.h>
#include <immintrin.h>

#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))
#define round(a,s) (((a)+(1<<((s)-1)))>>(s))

#define PATTERN_INTERPOLATION 0

#define Y_index 0
#define U_index 1
#define V_index 2

//Define some missing intrinsics
// https://github.com/samyvilar/dyn_perf/blob/master/sse2.h
#ifndef intrsc_attrs
#ifdef __INTEL_COMPILER
#   define intrsc_attrs __attribute__((__gnu_inline__, __always_inline__, __artificial__)) //, __artificial__
#else
#   define intrsc_attrs __attribute__((__gnu_inline__, __always_inline__)) //, __artificial__
#endif
#endif

#ifndef static_inline
#   define static_inline static __inline__ intrsc_attrs
#endif


static_inline __m128i intrsc_attrs _mm_srl_epi8(__m128i a, __m128i b) { // 5-7 cycles ...
    return _mm_and_si128(_mm_srl_epi16(a, b), _mm_set1_epi8(0xFFU >> _mm_cvtsi128_si64(b)));
}

static_inline __m128i _mm_sll_epi8(__m128i a, __m128i b) { // (4-6 cycles)
    return _mm_and_si128(_mm_sll_epi16(a, b), _mm_set1_epi8(0xFFU << _mm_cvtsi128_si64(b)));
}

static_inline __m128i _mm_mullo_epi8(__m128i a, __m128i b) {
    return
#       ifdef __SSSE3__
            _mm_unpacklo_epi8( // ~11 cycles, assuming both *set* get translated to a single fast load (which it does at least with 'clang -Ofast')
                _mm_shuffle_epi8(
                    _mm_mullo_epi16(a, b),
                    _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0)
                ),
                _mm_shuffle_epi8(
                    _mm_mullo_epi16(_mm_srli_epi16(a, 8), _mm_srli_epi16(b, 8)),
                    _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0)
                )
            )
#       else
            _mm_or_si128( // ~12 cycles
                _mm_srli_epi16(_mm_slli_epi16(_mm_mullo_epi16(a, b), 8), 8),
                _mm_slli_epi16(_mm_mullo_epi16(_mm_srli_epi16(a, 8), _mm_srli_epi16(b, 8)), 8)
            )
#       endif
    ;
}

//#define _round_simd(_a,_s) {_mm_srl_epi8(_mm_add_epi16(_a,_mm_sll_epi8(_mm_set1_epi8(1),(_mm_sub_epi16(_s,_mm_set1_epi8(1))))),_s)}

static_inline __m128i _round_simd(__m128i a, __m128i b) { // (4-6 cycles)
    return _mm_srl_epi8(_mm_add_epi16(a,_mm_sll_epi8(_mm_set1_epi8(1),(_mm_sub_epi16(b,_mm_set1_epi8(1))))),b);

}

void (*ptr_add_grain_stripe)(void* Y, void* U, void* V, unsigned y, unsigned width, unsigned height, unsigned stride, unsigned cstride);

// Note: declarations optimized for code readability; e.g. pattern storage in
//       actual hardware implementation would differ significantly
static int8 pattern[2][VFGS_MAX_PATTERNS+1][64][64] = {0, }; // +1 to simplify interpolation code
static uint8 sLUT[3][256] = {0, };
static uint8 pLUT[3][256] = {0, };
static uint32 rnd = 0xdeadbeef;
static uint32 rnd_up = 0xdeadbeef;
static uint32 line_rnd = 0xdeadbeef;
static uint32 line_rnd_up = 0xdeadbeef;
static uint8 scale_shift = 5+6;
static uint8 bs = 0; // bitshift = bitdepth - 8
static uint8 Y_min = 0;
static uint8 Y_max = 255;
static uint8 C_min = 0;
static uint8 C_max = 255;
static int csubx = 2;
static int csuby = 2;

// Processing pipeline (needs only 2 registers for each color actually, for horizontal deblocking)
static int16 grain[3][32]; // 9 bit needed because of overlap (has norm > 1)
static uint8 scale[3][32];

// Line buffers (software implementation)
static uint8 offset_x[3][256]; //
static uint8 offset_y[3][256]; // max. 4K image width
static int8  sign[3][256];     //
static int8  grain_buf[18][4096]; // TODO: cache-aligned alloc
static int8  over_buf[2][4096];   // TODO: cache-aligned alloc
static uint8 scale_buf[18][4096]; // last 2 lines never read

/** Pseudo-random number generator */
static uint32 prng(uint32 x)
{
	uint32 s = ((x << 30) ^ (x << 2)) & 0x80000000;
	x = s | (x >> 1);
	return x;
}

/** Derive Y x/y offsets from (random) number
 *
 * Bit fields are designed to minimize overlaps across color channels, to
 * decorrelate them as much as possible.
 *
 * 10-bit for 12 or 13 bins makes a reasonably uniform distribution (1.2%
 * probability error).
 *
 * If 8-bit is requested to further simplify the multiplier, at the cost of less
 * uniform probability, the following bitfields can be considered:
 *
 * Y: sign = rnd[31], x = (rnd[7:0]*13 >> 8)*4,   y = (rnd[21:14]*12 >> 8)*4
 * U: sign = rnd[0],  x = (rnd[17:10]*13 >> 8)*2, y = (rnd[31:24]*12 >> 8)*2
 * V: sign = rnd[13], x = (rnd[27:20]*13 >> 8)*2, y = (rnd[11:4]*12 >> 8)*2
 *
 * Note: to fully support cross-component correlation within patterns, we would
 * need to align luma/chroma offsets.
 */
static void get_offset_y(uint32 rnd, int *s, uint8 *x, uint8 *y)
{
	uint32 bf; // bit field

	*s = ((rnd >> 31) & 1) ? -1 : 1;

	bf = (rnd >> 0) & 0x3ff;
	*x = ((bf * 13) >> 10) * 4; // 13 = 8 + 4 + 1 (two adders)

	bf = (rnd >> 14) & 0x3ff;
	*y = ((bf * 12) >> 10) * 4; // 12 = 8 + 4 (one adder)
	// Note: could shift 9 and * 2, to make a multiple of 2 and make use of all
	// pattern samples (when using overlap).
}

static void get_offset_u(uint32 rnd, int *s, uint8 *x, uint8 *y)
{
	uint32 bf; // bit field

	*s = ((rnd >> 2) & 1) ? -1 : 1;

	bf = (rnd >> 10) & 0x3ff;
	*x = ((bf * 13) >> 10) * (4/csubx);

	bf = ((rnd >> 24) & 0x0ff) | ((rnd << 8) & 0x300);
	*y = ((bf * 12) >> 10) * (4/csuby);
}

static void get_offset_v(uint32 rnd, int *s, uint8 *x, uint8 *y)
{
	uint32 bf; // bit field

	*s = ((rnd >> 15) & 1) ? -1 : 1;

	bf = (rnd >> 20) & 0x3ff;
	*x = ((bf * 13) >> 10) * (4/csubx);

	bf = (rnd >> 4) & 0x3ff;
	*y = ((bf * 12) >> 10) * (4/csuby);
}

/* Public interface ***********************************************************/
/*
void vfgs_add_grain_line(void* Y, void* U, void* V, int y, int width)
{
	// Generate / backup / restore per-line random seeds (needed to make multi-line blocks)
	if (y && (y & 0x0f) == 0)
	{
		// new line of blocks --> backup + copy current to upper
		line_rnd_up = line_rnd;
		line_rnd = rnd;
	}
	rnd_up = line_rnd_up;
	rnd = line_rnd;

	// Process line
	for (int x=0; x<width; x+=16)
	{
		
		// Process pixels for each color component
		ptr_add_grain_block_Y(Y, 0, x, y, width);
		ptr_add_grain_block_U(U, 1, x, y, width);
		ptr_add_grain_block_V(V, 2, x, y, width);
		

		// Crank random generator
		rnd = prng(rnd);
		rnd_up = prng(rnd_up); // upper block (overlapping)
		
	}

}*/

void vfgs_add_grain_stripe_420_8bits(void* Y, void* U, void* V, unsigned y, unsigned width, unsigned height, unsigned stride, unsigned cstride)
{
	unsigned x, i;
	uint8 *I8;
	uint16 *I16;
	int overlap=0;
	int y_base = 0;
	unsigned height_u = height;
	unsigned height_v = height;

	// TODO could assert(height%16) if YUV memory is padded properly
	assert(width>128 && width<=4096 && width<=stride);
	assert((stride & 0x0f) == 0 && stride<=4096);
	assert((y & 0x0f) == 0);
	assert(bs == 0 || bs == 2);
	assert(scale_shift + bs >= 8 && scale_shift + bs <= 13);
	// TODO: assert subx, suby, Y/C min/max, max pLUT values, etc

	// Generate random offsets
	for (x=0; x<width; x+=16)
	{
		int s[3];
		get_offset_y(rnd, &s[Y_index], &offset_x[Y_index][x/16], &offset_y[Y_index][x/16]);
		get_offset_u(rnd, &s[U_index], &offset_x[U_index][x/16], &offset_y[U_index][x/16]);
		get_offset_v(rnd, &s[V_index], &offset_x[V_index][x/16], &offset_y[V_index][x/16]);
		rnd = prng(rnd);
		sign[Y_index][x/16] = s[Y_index];
		sign[U_index][x/16] = s[U_index];
		sign[V_index][x/16] = s[V_index];
	}

	// Compute stripe height (including overlap for next stripe)
	overlap = (y > 0);
	height = min(18, height-y);
	y_base = y;

	// Y: get grain & scale
	I8 = (uint8*)Y;
	for (y=0; y<height; y++)
	{
		for (x=0; x<width; x+=16)
		{
			//int    s = sign[Y_index][x/16];
			uint8 ox = offset_x[Y_index][x/16];
			uint8 oy = offset_y[Y_index][x/16];

			__m128i _intensity, _pi, _P, _piLUT_inter, _shift, _s;
            _shift = _mm_set1_epi8(4);
            _s = _mm_set1_epi8(sign[0][x/16]);
            _intensity = _mm_loadu_si128((__m128i*)&I8[x]);

			_piLUT_inter = _mm_set_epi8(pLUT[0][_mm_extract_epi8(_intensity, 15)],
										pLUT[0][_mm_extract_epi8(_intensity, 14)],
										pLUT[0][_mm_extract_epi8(_intensity, 13)],
										pLUT[0][_mm_extract_epi8(_intensity, 12)],
										pLUT[0][_mm_extract_epi8(_intensity, 11)],
										pLUT[0][_mm_extract_epi8(_intensity, 10)],
										pLUT[0][_mm_extract_epi8(_intensity, 9)],
										pLUT[0][_mm_extract_epi8(_intensity, 8)],
										pLUT[0][_mm_extract_epi8(_intensity, 7)],
										pLUT[0][_mm_extract_epi8(_intensity, 6)],
										pLUT[0][_mm_extract_epi8(_intensity, 5)],
										pLUT[0][_mm_extract_epi8(_intensity, 4)],
										pLUT[0][_mm_extract_epi8(_intensity, 3)],
										pLUT[0][_mm_extract_epi8(_intensity, 2)],
										pLUT[0][_mm_extract_epi8(_intensity, 1)],
										pLUT[0][_mm_extract_epi8(_intensity, 0)]);


            _pi = (__m128i)_mm_srl_epi8(_piLUT_inter, _shift);
			_P = _mm_set_epi8(pattern[0][_mm_extract_epi8(_pi, 15)][oy + y][ox + 15],
							  pattern[0][_mm_extract_epi8(_pi, 14)][oy + y][ox + 14],
							  pattern[0][_mm_extract_epi8(_pi, 13)][oy + y][ox + 13],
							  pattern[0][_mm_extract_epi8(_pi, 12)][oy + y][ox + 12],
							  pattern[0][_mm_extract_epi8(_pi, 11)][oy + y][ox + 11],
							  pattern[0][_mm_extract_epi8(_pi, 10)][oy + y][ox + 10],
							  pattern[0][_mm_extract_epi8(_pi, 9)][oy + y][ox + 9],
							  pattern[0][_mm_extract_epi8(_pi, 8)][oy + y][ox + 8],
							  pattern[0][_mm_extract_epi8(_pi, 7)][oy + y][ox + 7],
							  pattern[0][_mm_extract_epi8(_pi, 6)][oy + y][ox + 6],
							  pattern[0][_mm_extract_epi8(_pi, 5)][oy + y][ox + 5],
							  pattern[0][_mm_extract_epi8(_pi, 4)][oy + y][ox + 4],
							  pattern[0][_mm_extract_epi8(_pi, 3)][oy + y][ox + 3],
							  pattern[0][_mm_extract_epi8(_pi, 2)][oy + y][ox + 2],
							  pattern[0][_mm_extract_epi8(_pi, 1)][oy + y][ox + 1],
							  pattern[0][_mm_extract_epi8(_pi, 0)][oy + y][ox + 0]);
            _P = _mm_mullo_epi8(_P, _s);

            _mm_store_si128((__m128i*)&grain_buf[y], _P);
			scale_buf[y][x] = sLUT[0][_mm_extract_epi8(_intensity, 0)];
			scale_buf[y][x+1] = sLUT[0][_mm_extract_epi8(_intensity, 1)];
			scale_buf[y][x+2] = sLUT[0][_mm_extract_epi8(_intensity, 2)];
			scale_buf[y][x+3] = sLUT[0][_mm_extract_epi8(_intensity, 3)];
			scale_buf[y][x+4] = sLUT[0][_mm_extract_epi8(_intensity, 4)];
			scale_buf[y][x+5] = sLUT[0][_mm_extract_epi8(_intensity, 5)];
			scale_buf[y][x+6] = sLUT[0][_mm_extract_epi8(_intensity, 6)];
			scale_buf[y][x+7] = sLUT[0][_mm_extract_epi8(_intensity, 7)];
			scale_buf[y][x+8] = sLUT[0][_mm_extract_epi8(_intensity, 8)];
			scale_buf[y][x+9] = sLUT[0][_mm_extract_epi8(_intensity, 9)];
			scale_buf[y][x+10] = sLUT[0][_mm_extract_epi8(_intensity, 10)];
			scale_buf[y][x+11] = sLUT[0][_mm_extract_epi8(_intensity, 11)];
			scale_buf[y][x+12] = sLUT[0][_mm_extract_epi8(_intensity, 12)];
			scale_buf[y][x+13] = sLUT[0][_mm_extract_epi8(_intensity, 13)];
			scale_buf[y][x+14] = sLUT[0][_mm_extract_epi8(_intensity, 14)];
			scale_buf[y][x+15] = sLUT[0][_mm_extract_epi8(_intensity, 15)];

			/*for (i=0; i<16; i++) // may overflow past right image border but no problem: allocated space is multiple of 16
			{
				uint8 intensity = I8[x+i];
				uint8 pi = pLUT[Y_index][intensity] >> 4; // pattern index (integer part) / TODO: try also with zero-shift
				// TODO: assert(pi < VFGS_MAX_PATTERNS); // out of loop ?
				uint8 P  = pattern[0][pi][oy + y][ox + i] * s; // We could consider just XORing the sign bit
				grain_buf[y][x+i] = P;
				scale_buf[y][x+i] = sLUT[Y_index][intensity];
			}*/
		}
		I8  += stride;
	}

	// Y: vertical overlap (merge lines over_buf with 0 & 1, then copy 16 & 17 to over_buf)
	// problem: need to store 9-bits now ? or just clip ?
	for (y=0; y<2 && overlap; y++)
	{
		uint8 oc1 = y ? 24 : 12; // current
		uint8 oc2 = y ? 12 : 24; // previous
		for (x=0; x<width; x++)
		{
			int16 g = round(oc1*grain_buf[y][x+i] + oc2*over_buf[y][x+i], 5);
			grain_buf[y][x+i] = max(-127, min(+127, g));
			over_buf[y][x+i] = grain_buf[y+16][x+i];
		}
	}

	// Y: horizontal deblock
	// problem: need to store 9-bits now ? or just clip ?
	// TODO: set grain_buf[y][width] to zero if width == K*16 +1 (to avoid filtering garbage)
	for (y=0; y<16; y++)
		for (x=16; x<width; x+=16)
		{
			int16 l1, l0, r0, r1;
			l1 = grain_buf[y][x -2];
			l0 = grain_buf[y][x -1];
			r0 = grain_buf[y][x +0];
			r1 = grain_buf[y][x +1];
			l1 = round(l1 + 3*l0 + r0, 2); // left
			r1 = round(l0 + 3*r0 + r1, 2); // right
			grain_buf[y][x -1] = max(-127, min(+127, l1));
			grain_buf[y][x +0] = max(-127, min(+127, r1));
		}

	// Y: scale & merge
	height = min(16, height);
	I8 = (uint8*)Y;
	for (y=0; y<height; y++)
	{
		for (x=0; x<width; x++)
		{
			int32 g = round(scale_buf[y][x] * (int16)grain_buf[y][x], scale_shift);
			I8[x] = max(Y_min, min(Y_max, I8[x] + g));
		}
		I8  += stride;
	}

	
	// U
	height_u = min(18, (height_u-y_base));
	const int stepy = 2;
	const int stepx = 2;
	// U: get grain & scale
	I8 = (uint8*)U;
	for (y=0; y<height_u; y+=stepy)
	{
		for (x=0; x<width; x+=16)
		{
			int    s = sign[U_index][x/16];
			uint8 ox = offset_x[U_index][x/16];
			uint8 oy = offset_y[U_index][x/16];
			for (i=0; i<16; i+=stepx) // may overflow past right image border but no problem: allocated space is multiple of 16
			{
				uint8 intensity = I8[x+i];
				uint8 pi = pLUT[U_index][intensity] >> 4; // pattern index (integer part) / TODO: try also with zero-shift
				// TODO: assert(pi < VFGS_MAX_PATTERNS); // out of loop ?
				uint8 P  = pattern[0][pi][oy + y][ox + i] * s; // We could consider just XORing the sign bit
				grain_buf[y][x+i] = P;
				scale_buf[y][x+i] = sLUT[U_index][intensity];
			}
		}
		I8  += cstride;
	}
	
	//Vertical overlap
	for (y=0; y<2 && overlap; y+=stepy)
	{
		uint8 oc1 = y ? 24 : 12; // current
		uint8 oc2 = y ? 12 : 24; // previous
		for (x=0; x<width; x+=stepx)
		{
			int16 g = round(oc1*grain_buf[y][x+i] + oc2*over_buf[y][x+i], 5);
			grain_buf[y][x+i] = max(-127, min(+127, g));
			over_buf[y][x+i] = grain_buf[y+16][x+i];
		}
	}
	//Horizontal deblocking
	for (y=0; y<16; y+=stepy)
		for (x=16; x<width; x+=16)
		{
			int16 l1, l0, r0, r1;
			l1 = grain_buf[y][x -2];
			l0 = grain_buf[y][x -1];
			r0 = grain_buf[y][x +0];
			r1 = grain_buf[y][x +1];
			l1 = round(l1 + 3*l0 + r0, 2); // left
			r1 = round(l0 + 3*r0 + r1, 2); // right
			grain_buf[y][x -1] = max(-127, min(+127, l1));
			grain_buf[y][x +0] = max(-127, min(+127, r1));
		}

	height_u = min(16, height_u);
	I8 = (uint8*)U;
	for (y=0; y<height_u; y+=stepy)
	{
		for (x=0; x<width; x+=stepx)
		{
			int32 g = round(scale_buf[y][x] * (int16)grain_buf[y][x], scale_shift);
			I8[x] = max(C_min, min(C_max, I8[x] + g));
		}
		I8  += cstride;
	}


	// V
	height_v = min(18, (height_v-y_base));
	// V: get grain & scale
	I8 = (uint8*)V;
	for (y=0; y<height_v; y+=stepy)
	{
		for (x=0; x<width; x+=16)
		{
			int    s = sign[V_index][x/16];
			uint8 ox = offset_x[V_index][x/16];
			uint8 oy = offset_y[V_index][x/16];
			for (i=0; i<16; i+=stepx) // may overflow past right image border but no problem: allocated space is multiple of 16
			{
				uint8 intensity = I8[x+i];
				uint8 pi = pLUT[V_index][intensity] >> 4; // pattern index (integer part) / TODO: try also with zero-shift
				// TODO: assert(pi < VFGS_MAX_PATTERNS); // out of loop ?
				uint8 P  = pattern[0][pi][oy + y][ox + i] * s; // We could consider just XORing the sign bit
				grain_buf[y][x+i] = P;
				scale_buf[y][x+i] = sLUT[V_index][intensity];
			}
		}
		I8  += cstride;
	}
	
	//Vertical overlap
	for (y=0; y<2 && overlap; y+=stepy)
	{
		uint8 oc1 = y ? 24 : 12; // current
		uint8 oc2 = y ? 12 : 24; // previous
		for (x=0; x<width; x++)
		{
			int16 g = round(oc1*grain_buf[y][x+i] + oc2*over_buf[y][x+i], 5);
			grain_buf[y][x+i] = max(-127, min(+127, g));
			over_buf[y][x+i] = grain_buf[y+16][x+i];
		}
	}
	//Horizontal deblocking
	for (y=0; y<16; y+=stepy)
		for (x=16; x<width; x+=16)
		{
			int16 l1, l0, r0, r1;
			l1 = grain_buf[y][x -2];
			l0 = grain_buf[y][x -1];
			r0 = grain_buf[y][x +0];
			r1 = grain_buf[y][x +1];
			l1 = round(l1 + 3*l0 + r0, 2); // left
			r1 = round(l0 + 3*r0 + r1, 2); // right
			grain_buf[y][x -1] = max(-127, min(+127, l1));
			grain_buf[y][x +0] = max(-127, min(+127, r1));
		}

	height_v = min(16, height_v/csuby);
	I8 = (uint8*)V;
	for (y=0; y<height_v; y+=stepy)
	{
		for (x=0; x<width; x+=stepx)
		{
			int32 g = round(scale_buf[y][x] * (int16)grain_buf[y][x], scale_shift);
			I8[x] = max(C_min, min(C_max, I8[x] + g));
		}
		I8  += cstride;
	}
}

void vfgs_add_grain_stripe_420_10bits(void* Y, void* U, void* V, unsigned y, unsigned width, unsigned height, unsigned stride, unsigned cstride)
{
	unsigned x, i;
	uint8 *I8;
	uint16 *I16;
	int overlap=0;
	int y_base = 0;
	unsigned height_u = height;
	unsigned height_v = height;

	__m128i _intensity, _pi, _P, _piLUT_inter, _shift_4, _s, _shift_2;
	__m256i _intensity_inter;

	_shift_4 = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4);
	_shift_2 = _mm_set_epi8(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2);

	// TODO could assert(height%16) if YUV memory is padded properly
	assert(width>128 && width<=4096 && width<=stride);
	assert((stride & 0x0f) == 0 && stride<=4096);
	assert((y & 0x0f) == 0);
	assert(bs == 0 || bs == 2);
	assert(scale_shift + bs >= 8 && scale_shift + bs <= 13);
	// TODO: assert subx, suby, Y/C min/max, max pLUT values, etc

	// Generate random offsets
	for (x=0; x<width; x+=16)
	{
		int s[3];
		get_offset_y(rnd, &s[Y_index], &offset_x[Y_index][x/16], &offset_y[Y_index][x/16]);
		get_offset_u(rnd, &s[U_index], &offset_x[U_index][x/16], &offset_y[U_index][x/16]);
		get_offset_v(rnd, &s[V_index], &offset_x[V_index][x/16], &offset_y[V_index][x/16]);
		rnd = prng(rnd);
		sign[Y_index][x/16] = s[Y_index];
		sign[U_index][x/16] = s[U_index];
		sign[V_index][x/16] = s[V_index];
	}

	// Compute stripe height (including overlap for next stripe)
	overlap = (y > 0);
	height = min(18, height-y);
	y_base = y;

	// Y: get grain & scale
	I16 = (uint16*)Y;
	for (y=0; y<height; y++)
	{
		for (x=0; x<width; x+=16)
		{
			int    s = sign[Y_index][x/16];
			uint8 ox = offset_x[Y_index][x/16];
			uint8 oy = offset_y[Y_index][x/16];

            _s = _mm_set1_epi8(sign[0][x/16]);

            _intensity_inter = _mm256_loadu_si256((__m256i*)&I16[x]);
			_intensity_inter = (__m256i)_mm256_srl_epi16(_intensity_inter, _shift_2);

			// We need to retrieve the LSB of each 16 bits integer
			_intensity = _mm_set_epi8(_mm256_extract_epi8(_intensity_inter, 30),
									  _mm256_extract_epi8(_intensity_inter, 28),
									  _mm256_extract_epi8(_intensity_inter, 26),
									  _mm256_extract_epi8(_intensity_inter, 24),
									  _mm256_extract_epi8(_intensity_inter, 22),
									  _mm256_extract_epi8(_intensity_inter, 20),
									  _mm256_extract_epi8(_intensity_inter, 18),
									  _mm256_extract_epi8(_intensity_inter, 16),
									  _mm256_extract_epi8(_intensity_inter, 14),
									  _mm256_extract_epi8(_intensity_inter, 12),
									  _mm256_extract_epi8(_intensity_inter, 10),
									  _mm256_extract_epi8(_intensity_inter, 8),
									  _mm256_extract_epi8(_intensity_inter, 6),
									  _mm256_extract_epi8(_intensity_inter, 4),
									  _mm256_extract_epi8(_intensity_inter, 2),
									  _mm256_extract_epi8(_intensity_inter, 0));

			_piLUT_inter = _mm_set_epi8(pLUT[0][_mm_extract_epi8(_intensity, 15)],
										pLUT[0][_mm_extract_epi8(_intensity, 14)],
										pLUT[0][_mm_extract_epi8(_intensity, 13)],
										pLUT[0][_mm_extract_epi8(_intensity, 12)],
										pLUT[0][_mm_extract_epi8(_intensity, 11)],
										pLUT[0][_mm_extract_epi8(_intensity, 10)],
										pLUT[0][_mm_extract_epi8(_intensity, 9)],
										pLUT[0][_mm_extract_epi8(_intensity, 8)],
										pLUT[0][_mm_extract_epi8(_intensity, 7)],
										pLUT[0][_mm_extract_epi8(_intensity, 6)],
										pLUT[0][_mm_extract_epi8(_intensity, 5)],
										pLUT[0][_mm_extract_epi8(_intensity, 4)],
										pLUT[0][_mm_extract_epi8(_intensity, 3)],
										pLUT[0][_mm_extract_epi8(_intensity, 2)],
										pLUT[0][_mm_extract_epi8(_intensity, 1)],
										pLUT[0][_mm_extract_epi8(_intensity, 0)]);


            _pi = (__m128i)_mm_srl_epi8(_piLUT_inter, _shift_4);
			_P = _mm_set_epi8(pattern[0][_mm_extract_epi8(_pi, 15)][oy + y][ox + 15],
							  pattern[0][_mm_extract_epi8(_pi, 14)][oy + y][ox + 14],
							  pattern[0][_mm_extract_epi8(_pi, 13)][oy + y][ox + 13],
							  pattern[0][_mm_extract_epi8(_pi, 12)][oy + y][ox + 12],
							  pattern[0][_mm_extract_epi8(_pi, 11)][oy + y][ox + 11],
							  pattern[0][_mm_extract_epi8(_pi, 10)][oy + y][ox + 10],
							  pattern[0][_mm_extract_epi8(_pi, 9)][oy + y][ox + 9],
							  pattern[0][_mm_extract_epi8(_pi, 8)][oy + y][ox + 8],
							  pattern[0][_mm_extract_epi8(_pi, 7)][oy + y][ox + 7],
							  pattern[0][_mm_extract_epi8(_pi, 6)][oy + y][ox + 6],
							  pattern[0][_mm_extract_epi8(_pi, 5)][oy + y][ox + 5],
							  pattern[0][_mm_extract_epi8(_pi, 4)][oy + y][ox + 4],
							  pattern[0][_mm_extract_epi8(_pi, 3)][oy + y][ox + 3],
							  pattern[0][_mm_extract_epi8(_pi, 2)][oy + y][ox + 2],
							  pattern[0][_mm_extract_epi8(_pi, 1)][oy + y][ox + 1],
							  pattern[0][_mm_extract_epi8(_pi, 0)][oy + y][ox + 0]);
            _P = _mm_mullo_epi8(_P, _s);

            _mm_store_si128((__m128i*)&grain_buf[y][x], _P);
			scale_buf[y][x] = sLUT[0][_mm_extract_epi8(_intensity, 0)];
			scale_buf[y][x+1] = sLUT[0][_mm_extract_epi8(_intensity, 1)];
			scale_buf[y][x+2] = sLUT[0][_mm_extract_epi8(_intensity, 2)];
			scale_buf[y][x+3] = sLUT[0][_mm_extract_epi8(_intensity, 3)];
			scale_buf[y][x+4] = sLUT[0][_mm_extract_epi8(_intensity, 4)];
			scale_buf[y][x+5] = sLUT[0][_mm_extract_epi8(_intensity, 5)];
			scale_buf[y][x+6] = sLUT[0][_mm_extract_epi8(_intensity, 6)];
			scale_buf[y][x+7] = sLUT[0][_mm_extract_epi8(_intensity, 7)];
			scale_buf[y][x+8] = sLUT[0][_mm_extract_epi8(_intensity, 8)];
			scale_buf[y][x+9] = sLUT[0][_mm_extract_epi8(_intensity, 9)];
			scale_buf[y][x+10] = sLUT[0][_mm_extract_epi8(_intensity, 10)];
			scale_buf[y][x+11] = sLUT[0][_mm_extract_epi8(_intensity, 11)];
			scale_buf[y][x+12] = sLUT[0][_mm_extract_epi8(_intensity, 12)];
			scale_buf[y][x+13] = sLUT[0][_mm_extract_epi8(_intensity, 13)];
			scale_buf[y][x+14] = sLUT[0][_mm_extract_epi8(_intensity, 14)];
			scale_buf[y][x+15] = sLUT[0][_mm_extract_epi8(_intensity, 15)];
			/*
			for (i=0; i<16; i++) // may overflow past right image border but no problem: allocated space is multiple of 16
			{
				uint8 intensity = I16[x+i] >> 2;
				uint8 pi = pLUT[Y_index][intensity] >> 4; // pattern index (integer part) / TODO: try also with zero-shift
				// TODO: assert(pi < VFGS_MAX_PATTERNS); // out of loop ?
				uint8 P  = pattern[0][pi][oy + y][ox + i] * s; // We could consider just XORing the sign bit
				grain_buf[y][x+i] = P;
				scale_buf[y][x+i] = sLUT[Y_index][intensity];
			}*/
		}
		I16 += stride;
	}

	// Y: vertical overlap (merge lines over_buf with 0 & 1, then copy 16 & 17 to over_buf)
	// problem: need to store 9-bits now ? or just clip ?

	__m128i v_127 = _mm_set1_epi8(127);
    __m128i v_neg_127 = _mm_set1_epi8(-127);
	
	for (y=0; y<2 && overlap; y++)
	{
		__m128i _oc1 = _mm_set1_epi8(y ? 24 : 12); // current
    	__m128i _oc2 = _mm_set1_epi8(y ? 12 : 24); // previous

		for (x=0; x<width; x+=16)
		{
			//perfom the round operation 
			__m128i _grain_buf = _mm_load_si128((__m128i *)&grain_buf[y][x]);
            __m128i _over_buf = _mm_load_si128((__m128i *)&over_buf[y][x]);

			__m128i _g1 = _mm_mullo_epi16(_oc1, _grain_buf);
			__m128i _g2 = _mm_mullo_epi16(_oc2, _over_buf);
			__m128i _g = _mm_add_epi16(_g1, _g2);
			
			__m128i _g_round = _round_simd(_g, _mm_set1_epi8(5));

			// perform the clamp operation
            __m128i _g_clamp = _mm_min_epi8(_g_round, v_127);
            _g_clamp = _mm_max_epi8(_g_clamp, v_neg_127);

			// store the result
            _mm_store_si128((__m128i *)&grain_buf[y][x], _g_clamp);
            _mm_store_si128((__m128i *)&over_buf[y][x], _g_clamp);
		}
	}
	/*	
	for (y=0; y<2 && overlap; y++)
	{
		uint8 oc1 = y ? 24 : 12; // current
		uint8 oc2 = y ? 12 : 24; // previous
		
		for (x=0; x<width; x++)
		{
			int16 g = round(oc1*grain_buf[y][x] + oc2*over_buf[y][x], 5);
			grain_buf[y][x] = max(-127, min(+127, g));
			over_buf[y][x] = grain_buf[y][x];
		}
	}*/

	// Y: horizontal deblock
	// problem: need to store 9-bits now ? or just clip ?
	// TODO: set grain_buf[y][width] to zero if width == K*16 +1 (to avoid filtering garbage)
	for (y=0; y<16; y++)
		for (x=16; x<width; x+=16)
		{
			int16 l1, l0, r0, r1;
			l1 = grain_buf[y][x -2];
			l0 = grain_buf[y][x -1];
			r0 = grain_buf[y][x +0];
			r1 = grain_buf[y][x +1];
			l1 = round(l1 + 3*l0 + r0, 2); // left
			r1 = round(l0 + 3*r0 + r1, 2); // right
			grain_buf[y][x -1] = max(-127, min(+127, l1));
			grain_buf[y][x +0] = max(-127, min(+127, r1));
		}

	// Y: scale & merge
	height = min(16, height);
	I16 = (uint16*)Y;
	for (y=0; y<height; y++)
	{
		for (x=0; x<width; x++)
		{
			int32 g = round(scale_buf[y][x] * (int16)grain_buf[y][x], scale_shift);
			I16[x] = max(Y_min<<2, min(Y_max<<2, I16[x] + g));
			
		}
		I16 += stride;
	}

	// U
	height_u = min(18, (height_u-y_base));
	const int stepy = 2;
	const int stepx = 2;

	// U: get grain & scale
	I16 = (uint16*)U;
	for (y=0; y<height_u; y+=stepy)
	{
		for (x=0; x<width; x+=16)
		{
			int    s = sign[U_index][x/16];
			uint8 ox = offset_x[U_index][x/16];
			uint8 oy = offset_y[U_index][x/16];

            _s = _mm_set1_epi8(sign[0][x/16]);

            _intensity_inter = _mm256_loadu_si256((__m256i*)&I16[x]);
			_intensity_inter = (__m256i)_mm256_srl_epi16(_intensity_inter, _shift_2);

			//_intensity = _mm256_castsi256_si128(_intensity_inter);
			_intensity = _mm_set_epi8(_mm256_extract_epi8(_intensity_inter, 30),
									  _mm256_extract_epi8(_intensity_inter, 28),
									  _mm256_extract_epi8(_intensity_inter, 26),
									  _mm256_extract_epi8(_intensity_inter, 24),
									  _mm256_extract_epi8(_intensity_inter, 22),
									  _mm256_extract_epi8(_intensity_inter, 20),
									  _mm256_extract_epi8(_intensity_inter, 18),
									  _mm256_extract_epi8(_intensity_inter, 16),
									  _mm256_extract_epi8(_intensity_inter, 14),
									  _mm256_extract_epi8(_intensity_inter, 12),
									  _mm256_extract_epi8(_intensity_inter, 10),
									  _mm256_extract_epi8(_intensity_inter, 8),
									  _mm256_extract_epi8(_intensity_inter, 6),
									  _mm256_extract_epi8(_intensity_inter, 4),
									  _mm256_extract_epi8(_intensity_inter, 2),
									  _mm256_extract_epi8(_intensity_inter, 0));

			_piLUT_inter = _mm_set_epi8(pLUT[U_index][_mm_extract_epi8(_intensity, 15)],
										pLUT[U_index][_mm_extract_epi8(_intensity, 14)],
										pLUT[U_index][_mm_extract_epi8(_intensity, 13)],
										pLUT[U_index][_mm_extract_epi8(_intensity, 12)],
										pLUT[U_index][_mm_extract_epi8(_intensity, 11)],
										pLUT[U_index][_mm_extract_epi8(_intensity, 10)],
										pLUT[U_index][_mm_extract_epi8(_intensity, 9)],
										pLUT[U_index][_mm_extract_epi8(_intensity, 8)],
										pLUT[U_index][_mm_extract_epi8(_intensity, 7)],
										pLUT[U_index][_mm_extract_epi8(_intensity, 6)],
										pLUT[U_index][_mm_extract_epi8(_intensity, 5)],
										pLUT[U_index][_mm_extract_epi8(_intensity, 4)],
										pLUT[U_index][_mm_extract_epi8(_intensity, 3)],
										pLUT[U_index][_mm_extract_epi8(_intensity, 2)],
										pLUT[U_index][_mm_extract_epi8(_intensity, 1)],
										pLUT[U_index][_mm_extract_epi8(_intensity, 0)]);


            _pi = (__m128i)_mm_srl_epi8(_piLUT_inter, _shift_4);
			_P = _mm_set_epi8(pattern[1][_mm_extract_epi8(_pi, 15)][oy + y][ox + 15],
							  pattern[1][_mm_extract_epi8(_pi, 14)][oy + y][ox + 14],
							  pattern[1][_mm_extract_epi8(_pi, 13)][oy + y][ox + 13],
							  pattern[1][_mm_extract_epi8(_pi, 12)][oy + y][ox + 12],
							  pattern[1][_mm_extract_epi8(_pi, 11)][oy + y][ox + 11],
							  pattern[1][_mm_extract_epi8(_pi, 10)][oy + y][ox + 10],
							  pattern[1][_mm_extract_epi8(_pi, 9)][oy + y][ox + 9],
							  pattern[1][_mm_extract_epi8(_pi, 8)][oy + y][ox + 8],
							  pattern[1][_mm_extract_epi8(_pi, 7)][oy + y][ox + 7],
							  pattern[1][_mm_extract_epi8(_pi, 6)][oy + y][ox + 6],
							  pattern[1][_mm_extract_epi8(_pi, 5)][oy + y][ox + 5],
							  pattern[1][_mm_extract_epi8(_pi, 4)][oy + y][ox + 4],
							  pattern[1][_mm_extract_epi8(_pi, 3)][oy + y][ox + 3],
							  pattern[1][_mm_extract_epi8(_pi, 2)][oy + y][ox + 2],
							  pattern[1][_mm_extract_epi8(_pi, 1)][oy + y][ox + 1],
							  pattern[1][_mm_extract_epi8(_pi, 0)][oy + y][ox + 0]);
            _P = _mm_mullo_epi8(_P, _s);

            _mm_store_si128((__m128i*)&grain_buf[y][x], _P);
			scale_buf[y][x] = sLUT[U_index][_mm_extract_epi8(_intensity, 0)];
			scale_buf[y][x+1] = sLUT[U_index][_mm_extract_epi8(_intensity, 1)];
			scale_buf[y][x+2] = sLUT[U_index][_mm_extract_epi8(_intensity, 2)];
			scale_buf[y][x+3] = sLUT[U_index][_mm_extract_epi8(_intensity, 3)];
			scale_buf[y][x+4] = sLUT[U_index][_mm_extract_epi8(_intensity, 4)];
			scale_buf[y][x+5] = sLUT[U_index][_mm_extract_epi8(_intensity, 5)];
			scale_buf[y][x+6] = sLUT[U_index][_mm_extract_epi8(_intensity, 6)];
			scale_buf[y][x+7] = sLUT[U_index][_mm_extract_epi8(_intensity, 7)];
			scale_buf[y][x+8] = sLUT[U_index][_mm_extract_epi8(_intensity, 8)];
			scale_buf[y][x+9] = sLUT[U_index][_mm_extract_epi8(_intensity, 9)];
			scale_buf[y][x+10] = sLUT[U_index][_mm_extract_epi8(_intensity, 10)];
			scale_buf[y][x+11] = sLUT[U_index][_mm_extract_epi8(_intensity, 11)];
			scale_buf[y][x+12] = sLUT[U_index][_mm_extract_epi8(_intensity, 12)];
			scale_buf[y][x+13] = sLUT[U_index][_mm_extract_epi8(_intensity, 13)];
			scale_buf[y][x+14] = sLUT[U_index][_mm_extract_epi8(_intensity, 14)];
			scale_buf[y][x+15] = sLUT[U_index][_mm_extract_epi8(_intensity, 15)];
			/*
			for (i=0; i<16; i+=stepx) // may overflow past right image border but no problem: allocated space is multiple of 16
			{
				uint8 intensity = I16[x+i] >> 2;
				uint8 pi = pLUT[U_index][intensity] >> 4; // pattern index (integer part) / TODO: try also with zero-shift
				// TODO: assert(pi < VFGS_MAX_PATTERNS); // out of loop ?
				uint8 P  = pattern[1][pi][oy + y][ox + i] * s; // We could consider just XORing the sign bit
				grain_buf[y][x+i] = P;
				scale_buf[y][x+i] = sLUT[U_index][intensity];
			}
			*/
		}
		I16 += cstride;
	}
	
	//Vertical overlap
	for (y=0; y<2 && overlap; y+=stepy)
	{
		uint8 oc1 = y ? 24 : 12; // current
		uint8 oc2 = y ? 12 : 24; // previous
		for (x=0; x<width; x+=stepx)
		{
			int16 g = round(oc1*grain_buf[y][x] + oc2*over_buf[y][x], 5);
			grain_buf[y][x] = max(-127, min(+127, g));
			over_buf[y][x] = grain_buf[y+16][x];
		}
	}
	//Horizontal deblocking
	for (y=0; y<16; y+=stepy)
		for (x=16; x<width; x+=16)
		{
			int16 l1, l0, r0, r1;
			l1 = grain_buf[y][x -2];
			l0 = grain_buf[y][x -1];
			r0 = grain_buf[y][x +0];
			r1 = grain_buf[y][x +1];
			l1 = round(l1 + 3*l0 + r0, 2); // left
			r1 = round(l0 + 3*r0 + r1, 2); // right
			grain_buf[y][x -1] = max(-127, min(+127, l1));
			grain_buf[y][x +0] = max(-127, min(+127, r1));
		}

	height_u = min(16, height_u);
	I16 = (uint16*)U;
	for (y=0; y<height_u; y+=stepy)
	{
		for (x=0; x<width; x+=stepx)
		{
			int32 g = round(scale_buf[y][x] * (int16)grain_buf[y][x], scale_shift);
			I16[x] = max(C_min<<2, min(C_max<<2, I16[x] + g));
		}
		I16 += cstride;
	}


	// V
	height_v = min(18, (height_v-y_base));
	// V: get grain & scale
	I16 = (uint16*)V;
	for (y=0; y<height_v; y+=stepy)
	{
		for (x=0; x<width; x+=16)
		{
			int    s = sign[V_index][x/16];
			uint8 ox = offset_x[V_index][x/16];
			uint8 oy = offset_y[V_index][x/16];
			
			 _s = _mm_set1_epi8(sign[0][x/16]);

            _intensity_inter = _mm256_loadu_si256((__m256i*)&I16[x]);
			_intensity_inter = (__m256i)_mm256_srl_epi16(_intensity_inter, _shift_2);

			_intensity = _mm_set_epi8(_mm256_extract_epi8(_intensity_inter, 30),
									  _mm256_extract_epi8(_intensity_inter, 28),
									  _mm256_extract_epi8(_intensity_inter, 26),
									  _mm256_extract_epi8(_intensity_inter, 24),
									  _mm256_extract_epi8(_intensity_inter, 22),
									  _mm256_extract_epi8(_intensity_inter, 20),
									  _mm256_extract_epi8(_intensity_inter, 18),
									  _mm256_extract_epi8(_intensity_inter, 16),
									  _mm256_extract_epi8(_intensity_inter, 14),
									  _mm256_extract_epi8(_intensity_inter, 12),
									  _mm256_extract_epi8(_intensity_inter, 10),
									  _mm256_extract_epi8(_intensity_inter, 8),
									  _mm256_extract_epi8(_intensity_inter, 6),
									  _mm256_extract_epi8(_intensity_inter, 4),
									  _mm256_extract_epi8(_intensity_inter, 2),
									  _mm256_extract_epi8(_intensity_inter, 0));

			_piLUT_inter = _mm_set_epi8(pLUT[Y_index][_mm_extract_epi8(_intensity, 15)],
										pLUT[Y_index][_mm_extract_epi8(_intensity, 14)],
										pLUT[Y_index][_mm_extract_epi8(_intensity, 13)],
										pLUT[Y_index][_mm_extract_epi8(_intensity, 12)],
										pLUT[Y_index][_mm_extract_epi8(_intensity, 11)],
										pLUT[Y_index][_mm_extract_epi8(_intensity, 10)],
										pLUT[Y_index][_mm_extract_epi8(_intensity, 9)],
										pLUT[Y_index][_mm_extract_epi8(_intensity, 8)],
										pLUT[Y_index][_mm_extract_epi8(_intensity, 7)],
										pLUT[Y_index][_mm_extract_epi8(_intensity, 6)],
										pLUT[Y_index][_mm_extract_epi8(_intensity, 5)],
										pLUT[Y_index][_mm_extract_epi8(_intensity, 4)],
										pLUT[Y_index][_mm_extract_epi8(_intensity, 3)],
										pLUT[Y_index][_mm_extract_epi8(_intensity, 2)],
										pLUT[Y_index][_mm_extract_epi8(_intensity, 1)],
										pLUT[Y_index][_mm_extract_epi8(_intensity, 0)]);


            _pi = (__m128i)_mm_srl_epi8(_piLUT_inter, _shift_4);
			_P = _mm_set_epi8(pattern[1][_mm_extract_epi8(_pi, 15)][oy + y][ox + 15],
							  pattern[1][_mm_extract_epi8(_pi, 14)][oy + y][ox + 14],
							  pattern[1][_mm_extract_epi8(_pi, 13)][oy + y][ox + 13],
							  pattern[1][_mm_extract_epi8(_pi, 12)][oy + y][ox + 12],
							  pattern[1][_mm_extract_epi8(_pi, 11)][oy + y][ox + 11],
							  pattern[1][_mm_extract_epi8(_pi, 10)][oy + y][ox + 10],
							  pattern[1][_mm_extract_epi8(_pi, 9)][oy + y][ox + 9],
							  pattern[1][_mm_extract_epi8(_pi, 8)][oy + y][ox + 8],
							  pattern[1][_mm_extract_epi8(_pi, 7)][oy + y][ox + 7],
							  pattern[1][_mm_extract_epi8(_pi, 6)][oy + y][ox + 6],
							  pattern[1][_mm_extract_epi8(_pi, 5)][oy + y][ox + 5],
							  pattern[1][_mm_extract_epi8(_pi, 4)][oy + y][ox + 4],
							  pattern[1][_mm_extract_epi8(_pi, 3)][oy + y][ox + 3],
							  pattern[1][_mm_extract_epi8(_pi, 2)][oy + y][ox + 2],
							  pattern[1][_mm_extract_epi8(_pi, 1)][oy + y][ox + 1],
							  pattern[1][_mm_extract_epi8(_pi, 0)][oy + y][ox + 0]);
            _P = _mm_mullo_epi8(_P, _s);

            _mm_store_si128((__m128i*)&grain_buf[y][x], _P);
			scale_buf[y][x] = sLUT[Y_index][_mm_extract_epi8(_intensity, 0)];
			scale_buf[y][x+1] = sLUT[Y_index][_mm_extract_epi8(_intensity, 1)];
			scale_buf[y][x+2] = sLUT[Y_index][_mm_extract_epi8(_intensity, 2)];
			scale_buf[y][x+3] = sLUT[Y_index][_mm_extract_epi8(_intensity, 3)];
			scale_buf[y][x+4] = sLUT[Y_index][_mm_extract_epi8(_intensity, 4)];
			scale_buf[y][x+5] = sLUT[Y_index][_mm_extract_epi8(_intensity, 5)];
			scale_buf[y][x+6] = sLUT[Y_index][_mm_extract_epi8(_intensity, 6)];
			scale_buf[y][x+7] = sLUT[Y_index][_mm_extract_epi8(_intensity, 7)];
			scale_buf[y][x+8] = sLUT[Y_index][_mm_extract_epi8(_intensity, 8)];
			scale_buf[y][x+9] = sLUT[Y_index][_mm_extract_epi8(_intensity, 9)];
			scale_buf[y][x+10] = sLUT[Y_index][_mm_extract_epi8(_intensity, 10)];
			scale_buf[y][x+11] = sLUT[Y_index][_mm_extract_epi8(_intensity, 11)];
			scale_buf[y][x+12] = sLUT[Y_index][_mm_extract_epi8(_intensity, 12)];
			scale_buf[y][x+13] = sLUT[Y_index][_mm_extract_epi8(_intensity, 13)];
			scale_buf[y][x+14] = sLUT[Y_index][_mm_extract_epi8(_intensity, 14)];
			scale_buf[y][x+15] = sLUT[Y_index][_mm_extract_epi8(_intensity, 15)];
			/*
			for (i=0; i<16; i+=stepx) // may overflow past right image border but no problem: allocated space is multiple of 16
			{
				uint8 intensity = I16[x+i] >> 2;
				uint8 pi = pLUT[V_index][intensity] >> 4; // pattern index (integer part) / TODO: try also with zero-shift
				// TODO: assert(pi < VFGS_MAX_PATTERNS); // out of loop ?
				uint8 P  = pattern[1][pi][oy + y][ox + i] * s; // We could consider just XORing the sign bit
				grain_buf[y][x+i] = P;
				scale_buf[y][x+i] = sLUT[V_index][intensity];
			}*/
		}
		I16 += cstride;
	}
	
	//Vertical overlap
	for (y=0; y<2 && overlap; y+=stepy)
	{
		uint8 oc1 = y ? 24 : 12; // current
		uint8 oc2 = y ? 12 : 24; // previous
		for (x=0; x<width; x++)
		{
			int16 g = round(oc1*grain_buf[y][x] + oc2*over_buf[y][x], 5);
			grain_buf[y][x] = max(-127, min(+127, g));
			over_buf[y][x] = grain_buf[y+16][x];
		}
	}
	//Horizontal deblocking
	for (y=0; y<16; y+=stepy)
		for (x=16; x<width; x+=16)
		{
			int16 l1, l0, r0, r1;
			l1 = grain_buf[y][x -2];
			l0 = grain_buf[y][x -1];
			r0 = grain_buf[y][x +0];
			r1 = grain_buf[y][x +1];
			l1 = round(l1 + 3*l0 + r0, 2); // left
			r1 = round(l0 + 3*r0 + r1, 2); // right
			grain_buf[y][x -1] = max(-127, min(+127, l1));
			grain_buf[y][x +0] = max(-127, min(+127, r1));
		}

	height_v = min(16, height_v/csuby);
	I16 = (uint16*)V;
	for (y=0; y<height_v; y+=stepy)
	{
		for (x=0; x<width; x+=stepx)
		{
			int32 g = round(scale_buf[y][x] * (int16)grain_buf[y][x], scale_shift);

			I16[x] = max(C_min<<2, min(C_max<<2, I16[x] + g));
		}
		I16 += cstride;
	}
}

void vfgs_add_grain_stripe_422_8bits(void* Y, void* U, void* V, unsigned y, unsigned width, unsigned height, unsigned stride, unsigned cstride)
{
	unsigned x, i;
	uint8 *I8;
	uint16 *I16;
	int overlap=0;
	int y_base = 0;
	unsigned height_u = height;
	unsigned height_v = height;

	// TODO could assert(height%16) if YUV memory is padded properly
	assert(width>128 && width<=4096 && width<=stride);
	assert((stride & 0x0f) == 0 && stride<=4096);
	assert((y & 0x0f) == 0);
	assert(bs == 0 || bs == 2);
	assert(scale_shift + bs >= 8 && scale_shift + bs <= 13);
	// TODO: assert subx, suby, Y/C min/max, max pLUT values, etc

	// Generate random offsets
	for (x=0; x<width; x+=16)
	{
		int s[3];
		get_offset_y(rnd, &s[Y_index], &offset_x[Y_index][x/16], &offset_y[Y_index][x/16]);
		get_offset_u(rnd, &s[U_index], &offset_x[U_index][x/16], &offset_y[U_index][x/16]);
		get_offset_v(rnd, &s[V_index], &offset_x[V_index][x/16], &offset_y[V_index][x/16]);
		rnd = prng(rnd);
		sign[Y_index][x/16] = s[Y_index];
		sign[U_index][x/16] = s[U_index];
		sign[V_index][x/16] = s[V_index];
	}

	// Compute stripe height (including overlap for next stripe)
	overlap = (y > 0);
	height = min(18, height-y);
	y_base = y;

	// Y: get grain & scale
	I8 = (uint8*)Y;
	for (y=0; y<height; y++)
	{
		for (x=0; x<width; x+=16)
		{
			int    s = sign[Y_index][x/16];
			uint8 ox = offset_x[Y_index][x/16];
			uint8 oy = offset_y[Y_index][x/16];
			for (i=0; i<16; i++) // may overflow past right image border but no problem: allocated space is multiple of 16
			{
				uint8 intensity = I8[x+i];
				uint8 pi = pLUT[Y_index][intensity] >> 4; // pattern index (integer part) / TODO: try also with zero-shift
				// TODO: assert(pi < VFGS_MAX_PATTERNS); // out of loop ?
				uint8 P  = pattern[0][pi][oy + y][ox + i] * s; // We could consider just XORing the sign bit
				grain_buf[y][x+i] = P;
				scale_buf[y][x+i] = sLUT[Y_index][intensity];
			}
		}
		I8  += stride;
	}

	// Y: vertical overlap (merge lines over_buf with 0 & 1, then copy 16 & 17 to over_buf)
	// problem: need to store 9-bits now ? or just clip ?
	for (y=0; y<2 && overlap; y++)
	{
		uint8 oc1 = y ? 24 : 12; // current
		uint8 oc2 = y ? 12 : 24; // previous
		for (x=0; x<width; x++)
		{
			int16 g = round(oc1*grain_buf[y][x+i] + oc2*over_buf[y][x+i], 5);
			grain_buf[y][x+i] = max(-127, min(+127, g));
			over_buf[y][x+i] = grain_buf[y+16][x+i];
		}
	}

	// Y: horizontal deblock
	// problem: need to store 9-bits now ? or just clip ?
	// TODO: set grain_buf[y][width] to zero if width == K*16 +1 (to avoid filtering garbage)
	for (y=0; y<16; y++)
		for (x=16; x<width; x+=16)
		{
			int16 l1, l0, r0, r1;
			l1 = grain_buf[y][x -2];
			l0 = grain_buf[y][x -1];
			r0 = grain_buf[y][x +0];
			r1 = grain_buf[y][x +1];
			l1 = round(l1 + 3*l0 + r0, 2); // left
			r1 = round(l0 + 3*r0 + r1, 2); // right
			grain_buf[y][x -1] = max(-127, min(+127, l1));
			grain_buf[y][x +0] = max(-127, min(+127, r1));
		}

	// Y: scale & merge
	height = min(16, height);
	I8 = (uint8*)Y;
	for (y=0; y<height; y++)
	{
		for (x=0; x<width; x++)
		{
			int32 g = round(scale_buf[y][x] * (int16)grain_buf[y][x], scale_shift);
			I8[x] = max(Y_min, min(Y_max, I8[x] + g));
		}
		I8  += stride;
	}

	
	// U
	height_u = min(18, (height_u-y_base));
	const int stepy = 1;
	const int stepx = 2;
	// U: get grain & scale
	I8 = (uint8*)U;
	for (y=0; y<height_u; y+=stepy)
	{
		for (x=0; x<width; x+=16)
		{
			int    s = sign[U_index][x/16];
			uint8 ox = offset_x[U_index][x/16];
			uint8 oy = offset_y[U_index][x/16];
			for (i=0; i<16; i+=stepx) // may overflow past right image border but no problem: allocated space is multiple of 16
			{
				uint8 intensity = I8[x+i];
				uint8 pi = pLUT[U_index][intensity] >> 4; // pattern index (integer part) / TODO: try also with zero-shift
				// TODO: assert(pi < VFGS_MAX_PATTERNS); // out of loop ?
				uint8 P  = pattern[0][pi][oy + y][ox + i] * s; // We could consider just XORing the sign bit
				grain_buf[y][x+i] = P;
				scale_buf[y][x+i] = sLUT[U_index][intensity];
			}
		}
		I8  += cstride;
	}
	
	//Vertical overlap
	for (y=0; y<2 && overlap; y+=stepy)
	{
		uint8 oc1 = y ? 24 : 12; // current
		uint8 oc2 = y ? 12 : 24; // previous
		for (x=0; x<width; x+=stepx)
		{
			int16 g = round(oc1*grain_buf[y][x+i] + oc2*over_buf[y][x+i], 5);
			grain_buf[y][x+i] = max(-127, min(+127, g));
			over_buf[y][x+i] = grain_buf[y+16][x+i];
		}
	}
	//Horizontal deblocking
	for (y=0; y<16; y+=stepy)
		for (x=16; x<width; x+=16)
		{
			int16 l1, l0, r0, r1;
			l1 = grain_buf[y][x -2];
			l0 = grain_buf[y][x -1];
			r0 = grain_buf[y][x +0];
			r1 = grain_buf[y][x +1];
			l1 = round(l1 + 3*l0 + r0, 2); // left
			r1 = round(l0 + 3*r0 + r1, 2); // right
			grain_buf[y][x -1] = max(-127, min(+127, l1));
			grain_buf[y][x +0] = max(-127, min(+127, r1));
		}

	height_u = min(16, height_u);
	I8 = (uint8*)U;
	for (y=0; y<height_u; y+=stepy)
	{
		for (x=0; x<width; x+=stepx)
		{
			int32 g = round(scale_buf[y][x] * (int16)grain_buf[y][x], scale_shift);
			I8[x] = max(C_min, min(C_max, I8[x] + g));
		}
		I8  += cstride;
	}


	// V
	height_v = min(18, (height_v-y_base));
	// V: get grain & scale
	I8 = (uint8*)V;
	for (y=0; y<height_v; y+=stepy)
	{
		for (x=0; x<width; x+=16)
		{
			int    s = sign[V_index][x/16];
			uint8 ox = offset_x[V_index][x/16];
			uint8 oy = offset_y[V_index][x/16];
			for (i=0; i<16; i+=stepx) // may overflow past right image border but no problem: allocated space is multiple of 16
			{
				uint8 intensity = I8[x+i];
				uint8 pi = pLUT[V_index][intensity] >> 4; // pattern index (integer part) / TODO: try also with zero-shift
				// TODO: assert(pi < VFGS_MAX_PATTERNS); // out of loop ?
				uint8 P  = pattern[0][pi][oy + y][ox + i] * s; // We could consider just XORing the sign bit
				grain_buf[y][x+i] = P;
				scale_buf[y][x+i] = sLUT[V_index][intensity];
			}
		}
		I8  += cstride;
	}
	
	//Vertical overlap
	for (y=0; y<2 && overlap; y+=stepy)
	{
		uint8 oc1 = y ? 24 : 12; // current
		uint8 oc2 = y ? 12 : 24; // previous
		for (x=0; x<width; x++)
		{
			int16 g = round(oc1*grain_buf[y][x+i] + oc2*over_buf[y][x+i], 5);
			grain_buf[y][x+i] = max(-127, min(+127, g));
			over_buf[y][x+i] = grain_buf[y+16][x+i];
		}
	}
	//Horizontal deblocking
	for (y=0; y<16; y+=stepy)
		for (x=16; x<width; x+=16)
		{
			int16 l1, l0, r0, r1;
			l1 = grain_buf[y][x -2];
			l0 = grain_buf[y][x -1];
			r0 = grain_buf[y][x +0];
			r1 = grain_buf[y][x +1];
			l1 = round(l1 + 3*l0 + r0, 2); // left
			r1 = round(l0 + 3*r0 + r1, 2); // right
			grain_buf[y][x -1] = max(-127, min(+127, l1));
			grain_buf[y][x +0] = max(-127, min(+127, r1));
		}

	height_v = min(16, height_v/csuby);
	I8 = (uint8*)V;
	for (y=0; y<height_v; y+=stepy)
	{
		for (x=0; x<width; x+=stepx)
		{
			int32 g = round(scale_buf[y][x] * (int16)grain_buf[y][x], scale_shift);
			I8[x] = max(C_min, min(C_max, I8[x] + g));
		}
		I8  += cstride;
	}
}

void vfgs_add_grain_stripe_422_10bits(void* Y, void* U, void* V, unsigned y, unsigned width, unsigned height, unsigned stride, unsigned cstride)
{
	unsigned x, i;
	uint8 *I8;
	uint16 *I16;
	int overlap=0;
	int y_base = 0;
	unsigned height_u = height;
	unsigned height_v = height;

	// TODO could assert(height%16) if YUV memory is padded properly
	assert(width>128 && width<=4096 && width<=stride);
	assert((stride & 0x0f) == 0 && stride<=4096);
	assert((y & 0x0f) == 0);
	assert(bs == 0 || bs == 2);
	assert(scale_shift + bs >= 8 && scale_shift + bs <= 13);
	// TODO: assert subx, suby, Y/C min/max, max pLUT values, etc

	// Generate random offsets
	for (x=0; x<width; x+=16)
	{
		int s[3];
		get_offset_y(rnd, &s[Y_index], &offset_x[Y_index][x/16], &offset_y[Y_index][x/16]);
		get_offset_u(rnd, &s[U_index], &offset_x[U_index][x/16], &offset_y[U_index][x/16]);
		get_offset_v(rnd, &s[V_index], &offset_x[V_index][x/16], &offset_y[V_index][x/16]);
		rnd = prng(rnd);
		sign[Y_index][x/16] = s[Y_index];
		sign[U_index][x/16] = s[U_index];
		sign[V_index][x/16] = s[V_index];
	}

	// Compute stripe height (including overlap for next stripe)
	overlap = (y > 0);
	height = min(18, height-y);
	y_base = y;

	// Y: get grain & scale
	I16 = (uint16*)Y;
	for (y=0; y<height; y++)
	{
		for (x=0; x<width; x+=16)
		{
			int    s = sign[Y_index][x/16];
			uint8 ox = offset_x[Y_index][x/16];
			uint8 oy = offset_y[Y_index][x/16];
			for (i=0; i<16; i++) // may overflow past right image border but no problem: allocated space is multiple of 16
			{
				uint8 intensity = I16[x+i] >> 2;
				uint8 pi = pLUT[Y_index][intensity] >> 4; // pattern index (integer part) / TODO: try also with zero-shift
				// TODO: assert(pi < VFGS_MAX_PATTERNS); // out of loop ?
				uint8 P  = pattern[0][pi][oy + y][ox + i] * s; // We could consider just XORing the sign bit
				grain_buf[y][x+i] = P;
				scale_buf[y][x+i] = sLUT[Y_index][intensity];
			}
		}
		I16 += stride;
	}

	// Y: vertical overlap (merge lines over_buf with 0 & 1, then copy 16 & 17 to over_buf)
	// problem: need to store 9-bits now ? or just clip ?
	for (y=0; y<2 && overlap; y++)
	{
		uint8 oc1 = y ? 24 : 12; // current
		uint8 oc2 = y ? 12 : 24; // previous
		for (x=0; x<width; x++)
		{
			int16 g = round(oc1*grain_buf[y][x+i] + oc2*over_buf[y][x+i], 5);
			grain_buf[y][x+i] = max(-127, min(+127, g));
			over_buf[y][x+i] = grain_buf[y+16][x+i];
		}
	}

	// Y: horizontal deblock
	// problem: need to store 9-bits now ? or just clip ?
	// TODO: set grain_buf[y][width] to zero if width == K*16 +1 (to avoid filtering garbage)
	for (y=0; y<16; y++)
		for (x=16; x<width; x+=16)
		{
			int16 l1, l0, r0, r1;
			l1 = grain_buf[y][x -2];
			l0 = grain_buf[y][x -1];
			r0 = grain_buf[y][x +0];
			r1 = grain_buf[y][x +1];
			l1 = round(l1 + 3*l0 + r0, 2); // left
			r1 = round(l0 + 3*r0 + r1, 2); // right
			grain_buf[y][x -1] = max(-127, min(+127, l1));
			grain_buf[y][x +0] = max(-127, min(+127, r1));
		}

	// Y: scale & merge
	height = min(16, height);
	I16 = (uint16*)Y;
	for (y=0; y<height; y++)
	{
		for (x=0; x<width; x++)
		{
			int32 g = round(scale_buf[y][x] * (int16)grain_buf[y][x], scale_shift);
			I16[x] = max(Y_min<<2, min(Y_max<<2, I16[x] + g));
			
		}
		I16 += stride;
	}

	// U
	height_u = min(18, (height_u-y_base));
	const int stepy = 1;
	const int stepx = 2;
	// U: get grain & scale
	I16 = (uint16*)U;
	for (y=0; y<height_u; y+=stepy)
	{
		for (x=0; x<width; x+=16)
		{
			int    s = sign[U_index][x/16];
			uint8 ox = offset_x[U_index][x/16];
			uint8 oy = offset_y[U_index][x/16];
			for (i=0; i<16; i+=stepx) // may overflow past right image border but no problem: allocated space is multiple of 16
			{
				uint8 intensity = I16[x+i] >> 2;
				uint8 pi = pLUT[U_index][intensity] >> 4; // pattern index (integer part) / TODO: try also with zero-shift
				// TODO: assert(pi < VFGS_MAX_PATTERNS); // out of loop ?
				uint8 P  = pattern[0][pi][oy + y][ox + i] * s; // We could consider just XORing the sign bit
				grain_buf[y][x+i] = P;
				scale_buf[y][x+i] = sLUT[U_index][intensity];
			}
		}
		I16 += cstride;
	}
	
	//Vertical overlap
	for (y=0; y<2 && overlap; y+=stepy)
	{
		uint8 oc1 = y ? 24 : 12; // current
		uint8 oc2 = y ? 12 : 24; // previous
		for (x=0; x<width; x+=stepx)
		{
			int16 g = round(oc1*grain_buf[y][x+i] + oc2*over_buf[y][x+i], 5);
			grain_buf[y][x+i] = max(-127, min(+127, g));
			over_buf[y][x+i] = grain_buf[y+16][x+i];
		}
	}
	//Horizontal deblocking
	for (y=0; y<16; y+=stepy)
		for (x=16; x<width; x+=16)
		{
			int16 l1, l0, r0, r1;
			l1 = grain_buf[y][x -2];
			l0 = grain_buf[y][x -1];
			r0 = grain_buf[y][x +0];
			r1 = grain_buf[y][x +1];
			l1 = round(l1 + 3*l0 + r0, 2); // left
			r1 = round(l0 + 3*r0 + r1, 2); // right
			grain_buf[y][x -1] = max(-127, min(+127, l1));
			grain_buf[y][x +0] = max(-127, min(+127, r1));
		}

	height_u = min(16, height_u);
	I16 = (uint16*)U;
	for (y=0; y<height_u; y+=stepy)
	{
		for (x=0; x<width; x+=stepx)
		{
			int32 g = round(scale_buf[y][x] * (int16)grain_buf[y][x], scale_shift);
			I16[x] = max(C_min<<2, min(C_max<<2, I16[x] + g));
		}
		I16 += cstride;
	}


	// V
	height_v = min(18, (height_v-y_base));
	// V: get grain & scale
	I16 = (uint16*)V;
	for (y=0; y<height_v; y+=stepy)
	{
		for (x=0; x<width; x+=16)
		{
			int    s = sign[V_index][x/16];
			uint8 ox = offset_x[V_index][x/16];
			uint8 oy = offset_y[V_index][x/16];
			for (i=0; i<16; i+=stepx) // may overflow past right image border but no problem: allocated space is multiple of 16
			{
				uint8 intensity = I16[x+i] >> 2;
				uint8 pi = pLUT[V_index][intensity] >> 4; // pattern index (integer part) / TODO: try also with zero-shift
				// TODO: assert(pi < VFGS_MAX_PATTERNS); // out of loop ?
				uint8 P  = pattern[0][pi][oy + y][ox + i] * s; // We could consider just XORing the sign bit
				grain_buf[y][x+i] = P;
				scale_buf[y][x+i] = sLUT[V_index][intensity];
			}
		}
		I16 += cstride;
	}
	
	//Vertical overlap
	for (y=0; y<2 && overlap; y+=stepy)
	{
		uint8 oc1 = y ? 24 : 12; // current
		uint8 oc2 = y ? 12 : 24; // previous
		for (x=0; x<width; x++)
		{
			int16 g = round(oc1*grain_buf[y][x+i] + oc2*over_buf[y][x+i], 5);
			grain_buf[y][x+i] = max(-127, min(+127, g));
			over_buf[y][x+i] = grain_buf[y+16][x+i];
		}
	}
	//Horizontal deblocking
	for (y=0; y<16; y+=stepy)
		for (x=16; x<width; x+=16)
		{
			int16 l1, l0, r0, r1;
			l1 = grain_buf[y][x -2];
			l0 = grain_buf[y][x -1];
			r0 = grain_buf[y][x +0];
			r1 = grain_buf[y][x +1];
			l1 = round(l1 + 3*l0 + r0, 2); // left
			r1 = round(l0 + 3*r0 + r1, 2); // right
			grain_buf[y][x -1] = max(-127, min(+127, l1));
			grain_buf[y][x +0] = max(-127, min(+127, r1));
		}

	height_v = min(16, height_v/csuby);
	I16 = (uint16*)V;
	for (y=0; y<height_v; y+=stepy)
	{
		for (x=0; x<width; x+=stepx)
		{
			int32 g = round(scale_buf[y][x] * (int16)grain_buf[y][x], scale_shift);

			I16[x] = max(C_min<<2, min(C_max<<2, I16[x] + g));
		}
		I16 += cstride;
	}
}

void vfgs_add_grain_stripe_444_8bits(void* Y, void* U, void* V, unsigned y, unsigned width, unsigned height, unsigned stride, unsigned cstride)
{
	unsigned x, i;
	uint8 *I8;
	uint16 *I16;
	int overlap=0;
	int y_base = 0;
	unsigned height_u = height;
	unsigned height_v = height;

	// TODO could assert(height%16) if YUV memory is padded properly
	assert(width>128 && width<=4096 && width<=stride);
	assert((stride & 0x0f) == 0 && stride<=4096);
	assert((y & 0x0f) == 0);
	assert(bs == 0 || bs == 2);
	assert(scale_shift + bs >= 8 && scale_shift + bs <= 13);
	// TODO: assert subx, suby, Y/C min/max, max pLUT values, etc

	// Generate random offsets
	for (x=0; x<width; x+=16)
	{
		int s[3];
		get_offset_y(rnd, &s[Y_index], &offset_x[Y_index][x/16], &offset_y[Y_index][x/16]);
		get_offset_u(rnd, &s[U_index], &offset_x[U_index][x/16], &offset_y[U_index][x/16]);
		get_offset_v(rnd, &s[V_index], &offset_x[V_index][x/16], &offset_y[V_index][x/16]);
		rnd = prng(rnd);
		sign[Y_index][x/16] = s[Y_index];
		sign[U_index][x/16] = s[U_index];
		sign[V_index][x/16] = s[V_index];
	}

	// Compute stripe height (including overlap for next stripe)
	overlap = (y > 0);
	height = min(18, height-y);
	y_base = y;

	// Y: get grain & scale
	I8 = (uint8*)Y;
	for (y=0; y<height; y++)
	{
		for (x=0; x<width; x+=16)
		{
			int    s = sign[Y_index][x/16];
			uint8 ox = offset_x[Y_index][x/16];
			uint8 oy = offset_y[Y_index][x/16];
			for (i=0; i<16; i++) // may overflow past right image border but no problem: allocated space is multiple of 16
			{
				uint8 intensity = I8[x+i];
				uint8 pi = pLUT[Y_index][intensity] >> 4; // pattern index (integer part) / TODO: try also with zero-shift
				// TODO: assert(pi < VFGS_MAX_PATTERNS); // out of loop ?
				uint8 P  = pattern[0][pi][oy + y][ox + i] * s; // We could consider just XORing the sign bit
				grain_buf[y][x+i] = P;
				scale_buf[y][x+i] = sLUT[Y_index][intensity];
			}
		}
		I8  += stride;
	}

	// Y: vertical overlap (merge lines over_buf with 0 & 1, then copy 16 & 17 to over_buf)
	// problem: need to store 9-bits now ? or just clip ?
	for (y=0; y<2 && overlap; y++)
	{
		uint8 oc1 = y ? 24 : 12; // current
		uint8 oc2 = y ? 12 : 24; // previous
		for (x=0; x<width; x++)
		{
			int16 g = round(oc1*grain_buf[y][x+i] + oc2*over_buf[y][x+i], 5);
			grain_buf[y][x+i] = max(-127, min(+127, g));
			over_buf[y][x+i] = grain_buf[y+16][x+i];
		}
	}

	// Y: horizontal deblock
	// problem: need to store 9-bits now ? or just clip ?
	// TODO: set grain_buf[y][width] to zero if width == K*16 +1 (to avoid filtering garbage)
	for (y=0; y<16; y++)
		for (x=16; x<width; x+=16)
		{
			int16 l1, l0, r0, r1;
			l1 = grain_buf[y][x -2];
			l0 = grain_buf[y][x -1];
			r0 = grain_buf[y][x +0];
			r1 = grain_buf[y][x +1];
			l1 = round(l1 + 3*l0 + r0, 2); // left
			r1 = round(l0 + 3*r0 + r1, 2); // right
			grain_buf[y][x -1] = max(-127, min(+127, l1));
			grain_buf[y][x +0] = max(-127, min(+127, r1));
		}

	// Y: scale & merge
	height = min(16, height);
	I8 = (uint8*)Y;
	for (y=0; y<height; y++)
	{
		for (x=0; x<width; x++)
		{
			int32 g = round(scale_buf[y][x] * (int16)grain_buf[y][x], scale_shift);
			I8[x] = max(Y_min, min(Y_max, I8[x] + g));
		}
		I8  += stride;
	}

	
	// U
	height_u = min(18, (height_u-y_base));
	const int stepy = 1;
	const int stepx = 1;
	// U: get grain & scale
	I8 = (uint8*)U;
	for (y=0; y<height_u; y+=stepy)
	{
		for (x=0; x<width; x+=16)
		{
			int    s = sign[U_index][x/16];
			uint8 ox = offset_x[U_index][x/16];
			uint8 oy = offset_y[U_index][x/16];
			for (i=0; i<16; i+=stepx) // may overflow past right image border but no problem: allocated space is multiple of 16
			{
				uint8 intensity = I8[x+i];
				uint8 pi = pLUT[U_index][intensity] >> 4; // pattern index (integer part) / TODO: try also with zero-shift
				// TODO: assert(pi < VFGS_MAX_PATTERNS); // out of loop ?
				uint8 P  = pattern[0][pi][oy + y][ox + i] * s; // We could consider just XORing the sign bit
				grain_buf[y][x+i] = P;
				scale_buf[y][x+i] = sLUT[U_index][intensity];
			}
		}
		I8  += cstride;
	}
	
	//Vertical overlap
	for (y=0; y<2 && overlap; y+=stepy)
	{
		uint8 oc1 = y ? 24 : 12; // current
		uint8 oc2 = y ? 12 : 24; // previous
		for (x=0; x<width; x+=stepx)
		{
			int16 g = round(oc1*grain_buf[y][x+i] + oc2*over_buf[y][x+i], 5);
			grain_buf[y][x+i] = max(-127, min(+127, g));
			over_buf[y][x+i] = grain_buf[y+16][x+i];
		}
	}
	//Horizontal deblocking
	for (y=0; y<16; y+=stepy)
		for (x=16; x<width; x+=16)
		{
			int16 l1, l0, r0, r1;
			l1 = grain_buf[y][x -2];
			l0 = grain_buf[y][x -1];
			r0 = grain_buf[y][x +0];
			r1 = grain_buf[y][x +1];
			l1 = round(l1 + 3*l0 + r0, 2); // left
			r1 = round(l0 + 3*r0 + r1, 2); // right
			grain_buf[y][x -1] = max(-127, min(+127, l1));
			grain_buf[y][x +0] = max(-127, min(+127, r1));
		}

	height_u = min(16, height_u);
	I8 = (uint8*)U;
	for (y=0; y<height_u; y+=stepy)
	{
		for (x=0; x<width; x+=stepx)
		{
			int32 g = round(scale_buf[y][x] * (int16)grain_buf[y][x], scale_shift);
			I8[x] = max(C_min, min(C_max, I8[x] + g));
		}
		I8  += cstride;
	}


	// V
	height_v = min(18, (height_v-y_base));
	// V: get grain & scale
	I8 = (uint8*)V;
	for (y=0; y<height_v; y+=stepy)
	{
		for (x=0; x<width; x+=16)
		{
			int    s = sign[V_index][x/16];
			uint8 ox = offset_x[V_index][x/16];
			uint8 oy = offset_y[V_index][x/16];
			for (i=0; i<16; i+=stepx) // may overflow past right image border but no problem: allocated space is multiple of 16
			{
				uint8 intensity = I8[x+i];
				uint8 pi = pLUT[V_index][intensity] >> 4; // pattern index (integer part) / TODO: try also with zero-shift
				// TODO: assert(pi < VFGS_MAX_PATTERNS); // out of loop ?
				uint8 P  = pattern[0][pi][oy + y][ox + i] * s; // We could consider just XORing the sign bit
				grain_buf[y][x+i] = P;
				scale_buf[y][x+i] = sLUT[V_index][intensity];
			}
		}
		I8  += cstride;
	}
	
	//Vertical overlap
	for (y=0; y<2 && overlap; y+=stepy)
	{
		uint8 oc1 = y ? 24 : 12; // current
		uint8 oc2 = y ? 12 : 24; // previous
		for (x=0; x<width; x++)
		{
			int16 g = round(oc1*grain_buf[y][x+i] + oc2*over_buf[y][x+i], 5);
			grain_buf[y][x+i] = max(-127, min(+127, g));
			over_buf[y][x+i] = grain_buf[y+16][x+i];
		}
	}
	//Horizontal deblocking
	for (y=0; y<16; y+=stepy)
		for (x=16; x<width; x+=16)
		{
			int16 l1, l0, r0, r1;
			l1 = grain_buf[y][x -2];
			l0 = grain_buf[y][x -1];
			r0 = grain_buf[y][x +0];
			r1 = grain_buf[y][x +1];
			l1 = round(l1 + 3*l0 + r0, 2); // left
			r1 = round(l0 + 3*r0 + r1, 2); // right
			grain_buf[y][x -1] = max(-127, min(+127, l1));
			grain_buf[y][x +0] = max(-127, min(+127, r1));
		}

	height_v = min(16, height_v/csuby);
	I8 = (uint8*)V;
	for (y=0; y<height_v; y+=stepy)
	{
		for (x=0; x<width; x+=stepx)
		{
			int32 g = round(scale_buf[y][x] * (int16)grain_buf[y][x], scale_shift);
			I8[x] = max(C_min, min(C_max, I8[x] + g));
		}
		I8  += cstride;
	}
}

void vfgs_add_grain_stripe_444_10bits(void* Y, void* U, void* V, unsigned y, unsigned width, unsigned height, unsigned stride, unsigned cstride)
{
	unsigned x, i;
	uint8 *I8;
	uint16 *I16;
	int overlap=0;
	int y_base = 0;
	unsigned height_u = height;
	unsigned height_v = height;

	// TODO could assert(height%16) if YUV memory is padded properly
	assert(width>128 && width<=4096 && width<=stride);
	assert((stride & 0x0f) == 0 && stride<=4096);
	assert((y & 0x0f) == 0);
	assert(bs == 0 || bs == 2);
	assert(scale_shift + bs >= 8 && scale_shift + bs <= 13);
	// TODO: assert subx, suby, Y/C min/max, max pLUT values, etc

	// Generate random offsets
	for (x=0; x<width; x+=16)
	{
		int s[3];
		get_offset_y(rnd, &s[Y_index], &offset_x[Y_index][x/16], &offset_y[Y_index][x/16]);
		get_offset_u(rnd, &s[U_index], &offset_x[U_index][x/16], &offset_y[U_index][x/16]);
		get_offset_v(rnd, &s[V_index], &offset_x[V_index][x/16], &offset_y[V_index][x/16]);
		rnd = prng(rnd);
		sign[Y_index][x/16] = s[Y_index];
		sign[U_index][x/16] = s[U_index];
		sign[V_index][x/16] = s[V_index];
	}

	// Compute stripe height (including overlap for next stripe)
	overlap = (y > 0);
	height = min(18, height-y);
	y_base = y;

	// Y: get grain & scale
	I16 = (uint16*)Y;
	for (y=0; y<height; y++)
	{
		for (x=0; x<width; x+=16)
		{
			int    s = sign[Y_index][x/16];
			uint8 ox = offset_x[Y_index][x/16];
			uint8 oy = offset_y[Y_index][x/16];
			for (i=0; i<16; i++) // may overflow past right image border but no problem: allocated space is multiple of 16
			{
				uint8 intensity = I16[x+i] >> 2;
				uint8 pi = pLUT[Y_index][intensity] >> 4; // pattern index (integer part) / TODO: try also with zero-shift
				// TODO: assert(pi < VFGS_MAX_PATTERNS); // out of loop ?
				uint8 P  = pattern[0][pi][oy + y][ox + i] * s; // We could consider just XORing the sign bit
				grain_buf[y][x+i] = P;
				scale_buf[y][x+i] = sLUT[Y_index][intensity];
			}
		}
		I16 += stride;
	}

	// Y: vertical overlap (merge lines over_buf with 0 & 1, then copy 16 & 17 to over_buf)
	// problem: need to store 9-bits now ? or just clip ?
	for (y=0; y<2 && overlap; y++)
	{
		uint8 oc1 = y ? 24 : 12; // current
		uint8 oc2 = y ? 12 : 24; // previous
		for (x=0; x<width; x++)
		{
			int16 g = round(oc1*grain_buf[y][x+i] + oc2*over_buf[y][x+i], 5);
			grain_buf[y][x+i] = max(-127, min(+127, g));
			over_buf[y][x+i] = grain_buf[y+16][x+i];
		}
	}

	// Y: horizontal deblock
	// problem: need to store 9-bits now ? or just clip ?
	// TODO: set grain_buf[y][width] to zero if width == K*16 +1 (to avoid filtering garbage)
	for (y=0; y<16; y++)
		for (x=16; x<width; x+=16)
		{
			int16 l1, l0, r0, r1;
			l1 = grain_buf[y][x -2];
			l0 = grain_buf[y][x -1];
			r0 = grain_buf[y][x +0];
			r1 = grain_buf[y][x +1];
			l1 = round(l1 + 3*l0 + r0, 2); // left
			r1 = round(l0 + 3*r0 + r1, 2); // right
			grain_buf[y][x -1] = max(-127, min(+127, l1));
			grain_buf[y][x +0] = max(-127, min(+127, r1));
		}

	// Y: scale & merge
	height = min(16, height);
	I16 = (uint16*)Y;
	for (y=0; y<height; y++)
	{
		for (x=0; x<width; x++)
		{
			int32 g = round(scale_buf[y][x] * (int16)grain_buf[y][x], scale_shift);
			I16[x] = max(Y_min<<2, min(Y_max<<2, I16[x] + g));
			
		}
		I16 += stride;
	}

	// U
	height_u = min(18, (height_u-y_base));
	const int stepy = 1;
	const int stepx = 1;
	// U: get grain & scale
	I16 = (uint16*)U;
	for (y=0; y<height_u; y+=stepy)
	{
		for (x=0; x<width; x+=16)
		{
			int    s = sign[U_index][x/16];
			uint8 ox = offset_x[U_index][x/16];
			uint8 oy = offset_y[U_index][x/16];
			for (i=0; i<16; i+=stepx) // may overflow past right image border but no problem: allocated space is multiple of 16
			{
				uint8 intensity = I16[x+i] >> 2;
				uint8 pi = pLUT[U_index][intensity] >> 4; // pattern index (integer part) / TODO: try also with zero-shift
				// TODO: assert(pi < VFGS_MAX_PATTERNS); // out of loop ?
				uint8 P  = pattern[0][pi][oy + y][ox + i] * s; // We could consider just XORing the sign bit
				grain_buf[y][x+i] = P;
				scale_buf[y][x+i] = sLUT[U_index][intensity];
			}
		}
		I16 += cstride;
	}
	
	//Vertical overlap
	for (y=0; y<2 && overlap; y+=stepy)
	{
		uint8 oc1 = y ? 24 : 12; // current
		uint8 oc2 = y ? 12 : 24; // previous
		for (x=0; x<width; x+=stepx)
		{
			int16 g = round(oc1*grain_buf[y][x+i] + oc2*over_buf[y][x+i], 5);
			grain_buf[y][x+i] = max(-127, min(+127, g));
			over_buf[y][x+i] = grain_buf[y+16][x+i];
		}
	}
	//Horizontal deblocking
	for (y=0; y<16; y+=stepy)
		for (x=16; x<width; x+=16)
		{
			int16 l1, l0, r0, r1;
			l1 = grain_buf[y][x -2];
			l0 = grain_buf[y][x -1];
			r0 = grain_buf[y][x +0];
			r1 = grain_buf[y][x +1];
			l1 = round(l1 + 3*l0 + r0, 2); // left
			r1 = round(l0 + 3*r0 + r1, 2); // right
			grain_buf[y][x -1] = max(-127, min(+127, l1));
			grain_buf[y][x +0] = max(-127, min(+127, r1));
		}

	height_u = min(16, height_u);
	I16 = (uint16*)U;
	for (y=0; y<height_u; y+=stepy)
	{
		for (x=0; x<width; x+=stepx)
		{
			int32 g = round(scale_buf[y][x] * (int16)grain_buf[y][x], scale_shift);
			I16[x] = max(C_min<<2, min(C_max<<2, I16[x] + g));
		}
		I16 += cstride;
	}


	// V
	height_v = min(18, (height_v-y_base));
	// V: get grain & scale
	I16 = (uint16*)V;
	for (y=0; y<height_v; y+=stepy)
	{
		for (x=0; x<width; x+=16)
		{
			int    s = sign[V_index][x/16];
			uint8 ox = offset_x[V_index][x/16];
			uint8 oy = offset_y[V_index][x/16];
			for (i=0; i<16; i+=stepx) // may overflow past right image border but no problem: allocated space is multiple of 16
			{
				uint8 intensity = I16[x+i] >> 2;
				uint8 pi = pLUT[V_index][intensity] >> 4; // pattern index (integer part) / TODO: try also with zero-shift
				// TODO: assert(pi < VFGS_MAX_PATTERNS); // out of loop ?
				uint8 P  = pattern[0][pi][oy + y][ox + i] * s; // We could consider just XORing the sign bit
				grain_buf[y][x+i] = P;
				scale_buf[y][x+i] = sLUT[V_index][intensity];
			}
		}
		I16 += cstride;
	}
	
	//Vertical overlap
	for (y=0; y<2 && overlap; y+=stepy)
	{
		uint8 oc1 = y ? 24 : 12; // current
		uint8 oc2 = y ? 12 : 24; // previous
		for (x=0; x<width; x++)
		{
			int16 g = round(oc1*grain_buf[y][x+i] + oc2*over_buf[y][x+i], 5);
			grain_buf[y][x+i] = max(-127, min(+127, g));
			over_buf[y][x+i] = grain_buf[y+16][x+i];
		}
	}
	//Horizontal deblocking
	for (y=0; y<16; y+=stepy)
		for (x=16; x<width; x+=16)
		{
			int16 l1, l0, r0, r1;
			l1 = grain_buf[y][x -2];
			l0 = grain_buf[y][x -1];
			r0 = grain_buf[y][x +0];
			r1 = grain_buf[y][x +1];
			l1 = round(l1 + 3*l0 + r0, 2); // left
			r1 = round(l0 + 3*r0 + r1, 2); // right
			grain_buf[y][x -1] = max(-127, min(+127, l1));
			grain_buf[y][x +0] = max(-127, min(+127, r1));
		}

	height_v = min(16, height_v/csuby);
	I16 = (uint16*)V;
	for (y=0; y<height_v; y+=stepy)
	{
		for (x=0; x<width; x+=stepx)
		{
			int32 g = round(scale_buf[y][x] * (int16)grain_buf[y][x], scale_shift);

			I16[x] = max(C_min<<2, min(C_max<<2, I16[x] + g));
		}
		I16 += cstride;
	}
}


void wrapper_add_grain_stripe(void* Y, void* U, void* V, unsigned y, unsigned width, unsigned height, unsigned stride, unsigned cstride)
{
	ptr_add_grain_stripe(Y, U, V, y, width, height, stride, cstride);
}

void vfgs_set_luma_pattern(int index, int8* P)
{
	assert(index >= 0 && index < 8);
	memcpy(pattern[0][index], P, 64*64);
}

void vfgs_set_chroma_pattern(int index, int8 *P)
{
	assert(index >= 0 && index < 8);
	for (int i=0; i<64/csuby; i++)
		memcpy(pattern[1][index][i], P + (64/csuby)*i, 64/csubx);
}

void vfgs_set_scale_lut(int c, uint8 lut[])
{
	assert(c>=0 && c<3);
	memcpy(sLUT[c], lut, 256);
}

void vfgs_set_pattern_lut(int c, uint8 lut[])
{
	assert(c>=0 && c<3);
	memcpy(pLUT[c], lut, 256);
}

void vfgs_set_seed(uint32 seed)
{
	rnd = rnd_up = line_rnd = line_rnd_up = seed;
}

void vfgs_set_scale_shift(int shift)
{
	assert(shift >= 2 && shift < 8);
	scale_shift = shift + 6 - bs;
}

void vfgs_set_depth(int depth)
{
	assert(depth==8 || depth==10);

	if (bs==0 && depth>8)
		scale_shift -= 2;
	if (bs==2 && depth==8)
		scale_shift += 2;

	bs = depth - 8;
}

void vfgs_set_legal_range(int legal)
{
	if (legal)
	{
		Y_min = 16;
		Y_max = 235;
		C_min = 16;
		C_max = 240;
	}
	else
	{
		Y_min = 0;
		Y_max = 255;
		C_min = 0;
		C_max = 255;
	}
}

void vfgs_set_chroma_subsampling(int subx, int suby)
{
	assert(subx==1 || subx==2);
	assert(suby==1 || suby==2);
	csubx = subx;
	csuby = suby;
	/*
	if(subx == 2 && suby == 2)
	{
		ptr_add_grain_block_Y = add_grain_block_Y;
		ptr_add_grain_block_U = add_grain_block_U420;
		ptr_add_grain_block_V = add_grain_block_V420;
	}
	else if(subx == 2 && suby ==1)
	{
		ptr_add_grain_block_Y = add_grain_block_Y;
		ptr_add_grain_block_U = add_grain_block_U422;
		ptr_add_grain_block_V = add_grain_block_V422;
	}
	else
	{
		ptr_add_grain_block_Y = add_grain_block_Y;
		ptr_add_grain_block_U = add_grain_block_U444;
		ptr_add_grain_block_V = add_grain_block_V444;
	}*/

	if(subx == 2 && suby == 2)
	{
		if(bs == 2)
			ptr_add_grain_stripe = vfgs_add_grain_stripe_420_10bits;
		else
			ptr_add_grain_stripe = vfgs_add_grain_stripe_420_8bits;
	}
	else if(subx == 2 && suby ==1)
	{
		if(bs == 2)
			ptr_add_grain_stripe = vfgs_add_grain_stripe_422_10bits;
		else
			ptr_add_grain_stripe = vfgs_add_grain_stripe_422_8bits;
	}
	else
	{
		if(bs == 2)
			ptr_add_grain_stripe = vfgs_add_grain_stripe_444_10bits;
		else
			ptr_add_grain_stripe = vfgs_add_grain_stripe_444_8bits;
	}
}


#include <time.h>   /* Test: Generate random integers. */
#include <stdlib.h> /* Test: Generate random integers. */
#include <stdio.h> /* Test: print out results.*/
#include <string.h> /* Test: Use std memcmp() to validate the simd_memcpy(). */

#include <stdint.h> /* MUST include to use const width types. */
#include <immintrin.h> /* MUST include it for SIMD intrinsics. */
#include <assert.h>
typedef unsigned char u8; /* Define the single byte type. */

/**
 * Detect the endian type. 1: Big Endian, 0: Little Endian.
 * Here we used the <stdint.h> to make the type strict. Normally you can 
 * use `int x` without including <stdint.h> to detect the endian type.
 * The Endian type matters. 
 */
#define GET_ENDIAN_TYPE ({ uint32_t x = 0xFFFFFF00; (((u8 *)&x)[0]) ? 1 : 0; })

/**
 * Copy n bytes of memory from src to dest. In this function, we tried the 
 * simplest SIMD approach to improve the performance.
 */
void *simd_memcpy(void *dest, const void *src, size_t n) {
    if(!dest || !src) 
        return NULL; /* Return immediately to avoid risks. */

    /* Cast pointers to SIMD types. */
    const __m128i *src_ptr = (__m128i *)src;
    __m128i *dest_ptr = (__m128i *)dest;

    /* Determine the rounds and residual bytes that cannot be handled by SIMD. */
    size_t rounds = n >> 4, residual_bytes = n & 0x0F;

    /* Copy 128 bits (16 byte) in a loop. */
    for(size_t i = 0; i < rounds; ++ i) {
        /* Load the data. */
        __m128i vec = _mm_loadu_si128(src_ptr + i);
        /* Store the data. */
        _mm_storeu_si128(dest_ptr + i, vec);
    }
    /* Copy the residual bytes byte-by-byte. */
    for(u8 j = 0; j < residual_bytes; ++ j) {
        ((u8 *)(dest_ptr + rounds))[j] = ((u8 *)(src_ptr + rounds))[j];
    }
    return dest; /* Return the final dest ptr.*/
}
/**
 * Compare n bytes of memory between src and dest. In this function, we tried 
 * the simplest SIMD approach to improve the performance.
 */
int simd256_memcmp(const void *dest, const void *src, size_t n) {
    if(!dest || !src) /* Return immediately to avoid risks. */
        return 0x100; /* The final return should be -255 ~ 255. 
                         so we use 256 (0x100) refers to invalid. */
    
    /* Cast pointers to SIMD 256bit types. */
    const __m256i *src_ptr = (__m256i *)src;
    const __m256i *dest_ptr = (__m256i *)dest;
    
    /* Determine the rounds and residual bytes that cannot be handled by SIMD. */
    size_t rounds = n >> 5, residual_bytes = n & 0x1F;

    /* Compare 256bits (32 bytes) in a loop. */
    for(size_t i = 0; i < rounds; ++ i) {
        /* Load 256-bit data to 2 vectors. */
        __m256i vec_src = _mm256_loadu_si256(src_ptr + i);
        __m256i vec_dest = _mm256_loadu_si256(dest_ptr + i);
        /* Compare the vectors. */
        __m256i vec_cmp = _mm256_cmpeq_epi8(vec_src, vec_dest);
        /* Extract a 32-bit mask, each bit represents a raw byte comparison. */
        uint32_t mask = (uint32_t)_mm256_movemask_epi8(vec_cmp);
        if(mask == 0xFFFFFFFF) 
            continue; /* No difference, go ahead. */
        u8 j = 0;
        uint32_t target = 0x00000001;
        /* Find the first 0 bit in the raw mask. */
        while(j < 32 && (mask & target)) {
            target <<= 1; ++ j;
        }
        /* Return the difference between the first different byte. */
        return (int)(((u8 *)(dest_ptr + i))[j] - ((u8 *)(src_ptr + i))[j]);
    }
    /* Compare the residual bytes byte-by-byte. */
    for(u8 j = 0; j < residual_bytes; ++ j) {
        if(((u8 *)(src_ptr + rounds))[j] - ((u8 *)(dest_ptr + rounds))[j])
            return (int)(((u8 *)(dest_ptr + rounds))[j] - ((u8 *)(src_ptr + rounds))[j]);
    }
    return 0; /* Return 0 if no difference detected. */
}

/**
 * Copy n bytes of memory from src to dest. In this function, we tried the 
 * simplest SIMD approach to improve the performance.
 */
int naive_memcmp(const void *dest, const void *src, size_t n) {
    if(!dest || !src) 
        return -1; /* Return immediately to avoid risks. */
    
    for(size_t i = 0; i < n; ++ i) {
        if(((u8 *)src)[i] - ((u8 *)dest)[i]) 
            return (int)(((u8 *)dest)[i] - ((u8 *)src)[i]);
    }
    return 0;
}

/* Now let's do a test. */
#define SIZE 500011
#define DTYPE long

int main(int argc, char **argv) {
    DTYPE src[SIZE], dest[SIZE];
    srand(time(0));
    for(size_t i = 0; i < SIZE; ++ i) 
        src[i] = (DTYPE)rand();

    simd_memcpy(dest, src, SIZE * sizeof(DTYPE));
    printf("TOTAL_SIZE:\t%ld bytes\n", SIZE * sizeof(DTYPE));

    clock_t s = clock();
    printf("\nEQUAL? ( 0 - equal, non-0, not ): %d\n", \
            memcmp(src, dest, SIZE * sizeof(DTYPE)));
    clock_t e = clock();
    printf("time - std memcmp:\t%ld\n", e - s);

    s = clock();
    printf("\nEQUAL? ( 0 - equal, non-0, not ): %d\n", \
            simd256_memcmp(src, dest, SIZE * sizeof(DTYPE)));
    e = clock();
    printf("time - simd memcmp:\t%ld\n", e - s);
    s = clock();
    printf("\nEQUAL? ( 0 - equal, non-0, not ): %d\n", \
            naive_memcmp(src, dest, SIZE * sizeof(DTYPE)));
    e = clock();
    printf("time - naive memcmp:\t%ld\n", e - s);

    /* Flip a random number. */
    size_t diff_index = rand() % SIZE;
    dest[diff_index] = (DTYPE)rand();
    printf("\nThe element at index %ld changed.\n", diff_index);

    s = clock();
    printf("\nEQUAL? ( 0 - equal, non-0, not ): %d\n", \
            memcmp(src, dest, SIZE * sizeof(DTYPE)));
    e = clock();
    printf("time - std memcmp:\t%ld\n", e - s);
    s = clock();
    printf("\nEQUAL? ( 0 - equal, non-0, not ): %d\n", \
            simd256_memcmp(src, dest, SIZE * sizeof(DTYPE)));
    e = clock();
    printf("time - simd memcmp:\t%ld\n", e - s);
    s = clock();
    printf("\nEQUAL? ( 0 - equal, non-0, not ): %d\n", \
            naive_memcmp(src, dest, SIZE * sizeof(DTYPE)));
    e = clock();
    printf("time - naive memcmp:\t%ld\n", e - s);

    simd256_memcmp(NULL, NULL, 1);
}

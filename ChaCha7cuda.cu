//Our code is run to evaluate the backward correlation of 7-round ChaCha based on a 5-round differential-linear distinguisher.
//assignment 100...00 is used for consecutive PNBs
//169 PNBs are used for the evaluation of the backward correlation.


#define NUMBER_OF_DEVICES 1
#define NUMBER_OF_THREADS (1<<7)
#define NUMBER_OF_BLOCKS (1<<7)
#define NUMBER_OF_TEST_PER_THREAD (1<<14)

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <inttypes.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#include "device_launch_parameters.h"
#include <iostream>
#include <windows.h>


long start_time = GetTickCount();
void random_uint32(uint32_t *x)
{
	*x = 0;
	*x |= (rand() & 0xFF);
	*x |= (rand() & 0xFF) << 8;
	*x |= (rand() & 0xFF) << 16;
	*x |= (rand() & 0xFF) << 24;
}

void random_uint64(uint64_t *x)
{
	*x = 0;
	*x |= (rand() & 0xFFFF);
	*x |= (((uint64_t)rand() & 0xFFFF)) << 16;
	*x |= ((uint64_t)(rand() & 0xFFFF)) << 32;
	*x |= ((uint64_t)(rand() & 0xFFFF)) << 48;
}

void random_uint32_array(uint32_t *x, uint32_t size)
{
	for (uint32_t i = 0; i < size; i++)
		random_uint32(&x[i]);
}

void transform_state_to_bits(uint32_t state[16], uint8_t bits[512])
{
	int count = 0;
	for (int i = 0; i < 16; i++)
	{
		for (int b = 0; b < 32; b++)
		{
			bits[count] = (state[i] >> b) & 1;
			count++;
		}
	}
}

typedef struct chacha_ctx chacha_ctx;

#define U32C(v) (v##U)

#define U32V(v) ((uint32_t)(v) &U32C(0xFFFFFFFF))

#define ROTATE(v, c) ((v<<c)^(v>>(32-c)))
#define XOR(v, w) ((v) ^ (w))
#define PLUS(v, w) (((v) + (w)))
#define MINUS(v, w) (((v) - (w)))
#define PLUSONE(v) (PLUS((v), 1))

//QR function
#define QUARTERROUND(a, b, c, d) \
a = PLUS(a, b);              \
d = ROTATE(XOR(d, a), 16);   \
c = PLUS(c, d);              \
b = ROTATE(XOR(b, c), 12);   \
a = PLUS(a, b);              \
d = ROTATE(XOR(d, a), 8);    \
c = PLUS(c, d);              \
b = ROTATE(XOR(b, c), 7);

//decryption of QR function
#define INVERT_QUARTERROUND(a,b,c,d)\
b = XOR(ROTATE(b,25), c); \
c = MINUS(c, d);              \
d = XOR(ROTATE(d,24), a); \
a = MINUS(a, b);              \
b = XOR(ROTATE(b,20), c); \
c = MINUS(c, d);              \
d = XOR(ROTATE(d,16), a); \
a = MINUS(a, b);

#define LOAD32_LE(v) (*((uint32_t *) (v)))
#define STORE32_LE(c,x) (memcpy(c,&x,4))

__host__ __device__ void chacha_init(uint32_t state[16], uint32_t k[8], uint32_t nonce[2], uint32_t ctr[2])
{
	state[0] = U32C(0x61707865);
	state[1] = U32C(0x3320646e);
	state[2] = U32C(0x79622d32);
	state[3] = U32C(0x6b206574);
	state[4] = k[0];
	state[5] = k[1];
	state[6] = k[2];
	state[7] = k[3];
	state[8] = k[4];
	state[9] = k[5];
	state[10] = k[6];
	state[11] = k[7];
	state[12] = ctr[0];
	state[13] = ctr[1];
	state[14] = nonce[0];
	state[15] = nonce[1];
}

//round function of odd round
__host__ __device__ void chacha_odd_round(uint32_t x[16])
{
	QUARTERROUND(x[0], x[4], x[8], x[12])
		QUARTERROUND(x[1], x[5], x[9], x[13])
		QUARTERROUND(x[2], x[6], x[10], x[14])
		QUARTERROUND(x[3], x[7], x[11], x[15])
}


//round function of even round
__host__ __device__ void chacha_even_round(uint32_t x[16])
{
	QUARTERROUND(x[0], x[5], x[10], x[15])
		QUARTERROUND(x[1], x[6], x[11], x[12])
		QUARTERROUND(x[2], x[7], x[8], x[13])
		QUARTERROUND(x[3], x[4], x[9], x[14])
}


//decryption function of odd round
__host__ __device__ void chacha_invert_odd_round(uint32_t x[16])
{
	INVERT_QUARTERROUND(x[3], x[7], x[11], x[15])
		INVERT_QUARTERROUND(x[2], x[6], x[10], x[14])
		INVERT_QUARTERROUND(x[1], x[5], x[9], x[13])
		INVERT_QUARTERROUND(x[0], x[4], x[8], x[12])
}

//decryption function of even round
__host__ __device__ void chacha_invert_even_round(uint32_t x[16])
{
	INVERT_QUARTERROUND(x[3], x[4], x[9], x[14])
		INVERT_QUARTERROUND(x[2], x[7], x[8], x[13])
		INVERT_QUARTERROUND(x[1], x[6], x[11], x[12])
		INVERT_QUARTERROUND(x[0], x[5], x[10], x[15])
}

__host__ __device__ void chacha_rounds(uint32_t state[16], uint32_t rounds, uint32_t lastRound)
{
	uint32_t i;

	for (i = 1; i <= rounds; i++) {
		if ((i + lastRound) % 2)
			chacha_odd_round(state);
		else
			chacha_even_round(state);
	}
}



__host__ __device__ void chacha_invert_rounds(uint32_t state[16], uint32_t rounds, uint32_t lastRound)
{
	uint32_t i;

	lastRound = lastRound % 2;

	if (lastRound)
	{
		for (i = 1; i <= rounds; i++) {
			if (i % 2)
				chacha_invert_odd_round(state);
			else
				chacha_invert_even_round(state);
		}
	}
	else
	{
		for (i = 1; i <= rounds; i++) {
			if (i % 2)
				chacha_invert_even_round(state);
			else
				chacha_invert_odd_round(state);
		}
	}
}

__host__ __device__ void chacha_encrypt(uint32_t output[16], uint32_t input[16], uint32_t rounds)
{
	uint32_t x[16];
	uint32_t i;

	for (i = 0; i < 16; ++i) x[i] = input[i];
	chacha_rounds(x, rounds, 0);
	for (i = 0; i < 16; ++i) x[i] = PLUS(x[i], input[i]);

	memcpy(output, x, 64);
}

__host__ __device__ void chacha_invert(uint32_t output[16], uint32_t input[16], uint32_t intermediate[16], uint32_t rounds, uint32_t lastRound)
{
	for (int i = 0; i < 16; ++i) intermediate[i] = MINUS(output[i], input[i]);
	chacha_invert_rounds(intermediate, rounds, lastRound);
}

#define ALG_TYPE_SALSA 0
#define ALG_TYPE_CHACHA 1

typedef struct {
	uint32_t algType;
	uint32_t key_positions[8];
	uint32_t iv_positions[4];
	void(*init)(uint32_t *, uint32_t *, uint32_t *, uint32_t *);
	void(*encrypt)(uint32_t *, uint32_t *, uint32_t);
	void(*rounds)(uint32_t *, uint32_t, uint32_t);
	void(*halfrounds)(uint32_t *, uint32_t, uint32_t, int);
	void(*invert)(uint32_t *, uint32_t *, uint32_t *, uint32_t, uint32_t);
	char name[20];
} ALGORITHM;

__host__ __device__ void define_alg(ALGORITHM *alg, uint32_t type)
{
	uint32_t chacha_iv_positions[4] = { 12,13,14,15 };
	uint32_t chacha_key_positions[8] = { 4,5,6,7,8,9,10,11 };

	switch (type)
	{
	case ALG_TYPE_CHACHA:
		memcpy(alg->key_positions, chacha_key_positions, 8 * sizeof(uint32_t));
		memcpy(alg->iv_positions, chacha_iv_positions, 4 * sizeof(uint32_t));
		alg->algType = ALG_TYPE_CHACHA;
		alg->init = &chacha_init;
		alg->encrypt = &chacha_encrypt;
		alg->invert = &chacha_invert;
		alg->rounds = &chacha_rounds;
		alg->name[0] = 'C'; alg->name[1] = 'h'; alg->name[2] = 'a'; alg->name[3] = 'c'; alg->name[4] = 'h'; alg->name[5] = 'a'; alg->name[6] = 0;
		break;

	default:
		break;
	}
}

__host__ __device__ void xor_array(uint32_t *z, uint32_t *x, uint32_t *y, int size)
{
	for (int i = 0; i < size; i++)
		z[i] = x[i] ^ y[i];
}

__host__ __device__ void sub_array(uint32_t *z, uint32_t *x, uint32_t *y, int size)
{
	for (int i = 0; i < size; i++)
		z[i] = x[i] - y[i];
}

__host__ __device__ uint8_t get_bit_in_position(uint32_t state[16], uint32_t pos)
{
	int w = pos / 32;
	int bit = pos % 32;

	return((state[w] >> bit) & 1);
}

__host__ __device__ uint8_t get_bit_from_word_and_bit(uint32_t state[16], uint32_t w, uint32_t bit)
{
	return((state[w] >> bit) & 1);
}

__host__ __device__ void set_bit(uint32_t state[16], uint32_t w, uint32_t bit)
{
	state[w] ^= (1 << bit);
}

__host__ __device__ void set_list_of_bits(uint32_t state[16], uint32_t *w, uint32_t *bit, uint32_t numberOfBits)
{
	for (uint32_t i = 0; i < numberOfBits; i++)
		set_bit(state, w[i], bit[i]);
}

__host__ __device__ void and_array(uint32_t *z, uint32_t *x, uint32_t *y, int size)
{
	for (int i = 0; i < size; i++)
		z[i] = x[i] & y[i];
}

__host__ __device__ uint8_t xor_bits_of_state(uint32_t state[16])
{
	uint32_t x = state[0];
	for (int i = 1; i < 16; i++)
		x ^= state[i];

	x = x ^ (x >> 16);
	x = x ^ (x >> 8);
	x = x ^ (x >> 4);
	x = x ^ (x >> 2);
	return ((x ^ (x >> 1)) & 1);
}

__host__ __device__ uint8_t check_parity_of_equation(uint32_t state[16], uint32_t ODmask[16])
{
	uint32_t aux[16];

	and_array(aux, state, ODmask, 16);
	return(xor_bits_of_state(aux));
}

__device__ uint8_t check_parity_of_linear_relation_cuda(uint32_t inputMask[16], uint32_t inputState[16], uint32_t outputMask[16], uint32_t outputState[16])
{
	uint32_t aux[16], aux2[16];

	and_array(aux, inputState, inputMask, 16);
	and_array(aux2, outputState, outputMask, 16);
	xor_array(aux, aux, aux2, 16);

	return(xor_bits_of_state(aux));
}


__global__ void compute_bias_of_g_for_random_key_kernel(

	unsigned long long seed, uint32_t enc_rounds, uint32_t dec_rounds,
	uint32_t *IDmask, uint32_t *ODmask,
	uint32_t *pnb, uint32_t number_of_pnb, int ntestForEachThread,
	int *d_result, int algType
)
{
	ALGORITHM alg;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t K_with_zeros[8] = { 0 }, state[16] = { 0 }, alt_state[16] = { 0 };
	uint32_t final_state[16] = { 0 }, alt_final_state[16] = { 0 }, aux[16];
	uint32_t intermediate_state[16] = { 0 }, alt_intermediate_state[16] = { 0 };
	uint32_t nonce[2] = { 0 }, ctr[2] = { 0 };
	curandState_t rng;
	uint32_t f_parity, g_parity;
	uint32_t sumParity = 0;
	uint32_t mask;

	uint32_t Krand[8];


	uint32_t pnb_order[256] = { 0 };
	uint32_t loc_pnb = 0;
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < number_of_pnb; j++)
		{
			if (i == pnb[j])
			{
				pnb_order[loc_pnb] = i;
				loc_pnb += 1;
				break;
			}
		}
	}

	uint32_t pnb_assign1_loc[256] = { 0 };
	loc_pnb = 0;
	//the PNBs with assignment 1 is stored in "pnb_assign1_loc"
	for (int i = 1; i < number_of_pnb; i++)
	{
		if (((pnb_order[i] % 32) != 0)&((pnb_order[i] - pnb_order[i - 1]) == 1)&((pnb_order[i + 1] - pnb_order[i]) > 1))
		{
			pnb_assign1_loc[loc_pnb] = pnb_order[i];
			loc_pnb += 1;
		}
	}


	define_alg(&alg, algType);
	curand_init(seed, tid, 0, &rng);

	for (int i = 0; i < 8; i++)
		Krand[i] = curand(&rng);

	for (int i = 0; i < 8; i++)
		K_with_zeros[i] = Krand[i];

	
	for (uint32_t j = 0; j < number_of_pnb; j++)
	{
		mask = ~(1 << (pnb[j] % 32));
		K_with_zeros[pnb[j] / 32] = K_with_zeros[pnb[j] / 32] & mask;
	}

	//assignment 100...00 is used for consecutive PNBs
	for (uint32_t j = 0; j < loc_pnb; j++)
	{
		K_with_zeros[pnb_assign1_loc[j] / 32] = K_with_zeros[pnb_assign1_loc[j] / 32] + (1 << (pnb_assign1_loc[j] % 32));
	}

	for (int t = 0; t < ntestForEachThread; t++)
	{
		nonce[0] = curand(&rng); nonce[1] = curand(&rng);
		ctr[0] = curand(&rng); ctr[1] = curand(&rng);

		alg.init(state, Krand, nonce, ctr);
		xor_array(alt_state, state, IDmask, 16);

		//encrypt 7-round ChaCha
		alg.encrypt(final_state, state, enc_rounds);
		alg.encrypt(alt_final_state, alt_state, enc_rounds);

		//decrypt 2-round ChaCha
		alg.invert(final_state, state, intermediate_state, dec_rounds, enc_rounds);
		alg.invert(alt_final_state, alt_state, alt_intermediate_state, dec_rounds, enc_rounds);

		xor_array(aux, intermediate_state, alt_intermediate_state, 16);
		f_parity = check_parity_of_equation(aux, ODmask);


		alg.init(state, K_with_zeros, nonce, ctr);
		xor_array(alt_state, state, IDmask, 16);

		//decrypt 2-round ChaCha with PNBs
		alg.invert(final_state, state, intermediate_state, dec_rounds, enc_rounds);
		alg.invert(alt_final_state, alt_state, alt_intermediate_state, dec_rounds, enc_rounds);

		xor_array(aux, intermediate_state, alt_intermediate_state, 16);
		g_parity = check_parity_of_equation(aux, ODmask);

		if (f_parity == g_parity)
			sumParity++;

	}

	atomicAdd(d_result, (int)sumParity);
}





//compute backward correlation
double compute_mean_bias_of_g_cuda(
	uint64_t N,
	uint32_t ID[16],
	uint32_t ODmask[16],
	uint32_t enc_rounds,
	uint32_t dec_rounds,
	uint32_t *pnb,
	uint32_t number_of_pnb,
	ALGORITHM alg
)
{
	
	int nTestsForEachThread = NUMBER_OF_TEST_PER_THREAD, nThreads = NUMBER_OF_THREADS, nBlocks = NUMBER_OF_BLOCKS;
	int executionsPerKernel = nTestsForEachThread * nThreads*nBlocks;
	uint64_t iterations;
	int *dSumParity;
	uint32_t *dID, *dODmask, *dPNB;
	int localSumParity = 0;
	double prob = 0;

	uint64_t seed = rand();
	random_uint64(&seed);

	iterations = N / (executionsPerKernel);

	cudaMalloc(&dSumParity, sizeof(int));
	cudaMalloc(&dID, 16 * sizeof(uint32_t));
	cudaMalloc(&dODmask, 16 * sizeof(uint32_t));
	cudaMalloc(&dPNB, number_of_pnb * sizeof(uint32_t));

	cudaMemcpy(dID, ID, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(dODmask, ODmask, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(dPNB, pnb, number_of_pnb * sizeof(uint32_t), cudaMemcpyHostToDevice);

	for (int i = 0; i < iterations; i++)
	{
		random_uint64(&seed);
		localSumParity = 0;
		cudaMemcpy(dSumParity, &localSumParity, sizeof(int), cudaMemcpyHostToDevice);

		compute_bias_of_g_for_random_key_kernel << < nBlocks, nThreads >> > ((unsigned long long)seed,
			enc_rounds, dec_rounds, dID, dODmask, dPNB, number_of_pnb, nTestsForEachThread,
			dSumParity, alg.algType);
		

		cudaMemcpy(&localSumParity, dSumParity, sizeof(uint32_t), cudaMemcpyDeviceToHost);

		prob += ((double)(localSumParity)) / executionsPerKernel;
	}
	prob /= iterations;

	cudaFree(dSumParity);
	cudaFree(dID);
	cudaFree(dODmask);

	return(2 * prob - 1);
}

void main()
{
	srand(time(NULL));
	uint32_t ODmask[16] = { 0 };
	uint32_t ID[16] = { 0 };
	uint64_t N;
	ALGORITHM alg;
	double neutrality_list[256];
	uint32_t number_of_pnb;
	
	
	
	define_alg(&alg, ALG_TYPE_CHACHA);

	//Go 7, back 2
	memset(ODmask, 0x00, sizeof(uint32_t) * 16);

	//input difference
	N = 1;
	N <<= 14;
	ID[15] = (1 << 9) + (1 << 29);

	//output linear mask
	uint32_t listOfWords[5] = { 2,6,6,10,14 };
	uint32_t listOfBits[5] = { 0,7,19,12,0 };
	set_list_of_bits(ODmask, listOfWords, listOfBits, 5);

	

	double varepsilon_a = 0;


	N = 1;
	N <<= 36;

    //169 PNBs are used.
	uint32_t pnb[169] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 19, 20, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
		51, 52, 53, 54, 55, 56,
	57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 71, 72, 73, 74, 77, 78, 79, 80, 83, 84, 85, 86, 89, 90, 95, 99, 100, 103, 104,
		105, 106, 107, 108, 109, 123, 124, 125, 126, 127, 128, 129, 140, 141, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162,
		163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
		188, 189, 190, 191, 192, 193, 194, 198, 199, 200, 204, 205, 206, 207, 210, 211, 218, 219, 220, 221, 222, 223, 224, 225, 226,
		227, 244, 245, 246, 247, 255, 248, 9, 130, 142, 21, 91, 212, 110, 231, 22, 143, 232, 111, 228, 10, 201, 249, 115, 147, 14,
		81, 26 };
	number_of_pnb = 169;


	printf("number_of_pnb: %d \n", number_of_pnb);
	for (int i = 0; i < number_of_pnb; i++)
		printf("%d, ", pnb[i]);

	printf("\n\n");



	varepsilon_a = compute_mean_bias_of_g_cuda(N, ID, ODmask,
		7, 2, pnb, number_of_pnb, alg);
	double time_complexity, data_complexity;



	printf("backward correlation = %f\n", varepsilon_a);
}




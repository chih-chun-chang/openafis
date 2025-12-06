#include "MatchManyGPU.h" 
#include <cuda_runtime.h>
#include <math_constants.h>   // for CUDART_PI_F
#include <cstdio>           // for printf
#include "AngleLUT.h"
#include "Field.h"
#include <vector>
#include "FastMath.h"

namespace OpenAFIS {

// constant tables on device
__constant__ float d_cos_table[OpenAFIS::AngleLUT::kAngleTableSize];
__constant__ float d_sin_table[OpenAFIS::AngleLUT::kAngleTableSize];

// These must match CPU's FastMath constants
__device__ __constant__ int ANGLE_MODULO = 256;     // TwoPI8+1 = 255+1
__device__ __constant__ int ANGLE_HALF   = 128;     // half of 256
__device__ __constant__ int TwoPI8       = 255;     // full circle in 8-bit angle units

// handy alias
using AngleType = OpenAFIS::Field::AngleType;

__device__ __forceinline__
float gpu_cos(AngleType theta)
{
    int idx = static_cast<int>(theta - AngleLUT::kAngleMin);
    // optional safety (debug only):
    // if (idx < 0 || idx >= AngleLUT::kAngleTableSize) return 1.0f;
    return d_cos_table[idx];
}

__device__ __forceinline__
float gpu_sin(AngleType theta)
{
    int idx = static_cast<int>(theta - AngleLUT::kAngleMin);
    return d_sin_table[idx];
}

// equivalent to FastMath::rotateAngle for float angles
__device__ inline int rotate_angle_f(int a, int b) {
    // CPU version does something like "b - a", wrapped into range
    if (b > a) {
        return b - a;
    }
    return b - a + TwoPI8;
}

__device__ __forceinline__
int gpu_minimum_angle(int a, int b)
{
    int d = b > a ? b - a : a - b;
    return min(d, TwoPI8 - d);
}

__global__ void global_match_kernel(
    const MinutiaPairGPU* __restrict__ pairs,
    int                    numPairs,
    ParamGPU               param,
    uint8_t*               outScore)
{
    int tid    = threadIdx.x;
    int stride = blockDim.x;

    int localMax = 0;

    // Each thread tries different "seed" pair i (like p1 in CPU)
    for (int i = tid; i < numPairs; i += stride) {
        const auto& p1 = pairs[i];

        // theta = candidateAngle - probeAngle (same as CPU)
        float theta    = p1.cAng - p1.pAng;
        float cosTheta = gpu_cos(theta);
        float sinTheta = gpu_sin(theta);

        //if (i == 10 || i == 15)
        //printf("GPU: [%d] i=%d %.0f %.0f %.0f %.0f %.0f, %.0f, %.0f, %.3f, %.3f\n", tid, i, p1.cx, p1.cy, p1.px, p1.py, p1.cAng, p1.pAng, theta, cosTheta, sinTheta);

        int matched     = 1;                 // include the seed pair
        int minRemaining = numPairs - 1;     // like CPU's "auto min = m_pairs.size() - 1;"

        for (int j = 0; j < numPairs; ++j) {
            //if (i == j) continue;

            const auto& p2 = pairs[j];
            if (&p1 == &p2) {
                continue;
            }

            // ----- 5.2.2 position check (same as CPU "lengths" lambda) -----
            float x = p1.cx + cosTheta * (p2.px - p1.px) - sinTheta * (p2.py - p1.py);
            float y = p1.cy + sinTheta * (p2.px - p1.px) + cosTheta * (p2.py - p1.py);

            float dx = x - p2.cx;
            float dy = y - p2.cy;
            float dist2 = dx * dx + dy * dy;
            float maxGlobal2 = param.maximumGlobalDistance * param.maximumGlobalDistance;

            if (dist2 > maxGlobal2) {
                // no match; but CPU does *not* decrement min in this branch
                //printf("GPU [%d] i=%d j=%d %.0f %.0f %.0f %.0f, %f, %f\n", tid, i, j, p1.cx, p1.cy, p2.px, p2.py, x, y);
                //break;
                continue;
            }

            // ----- Direction check (same as CPU "directions" lambda) -----
            //float sAng    = wrap_angle_f(p2.pAng + theta);
            //float dirDiff = min_angle_f(sAng, p2.cAng);
            int sAng = static_cast<int>(p2.pAng + theta); // wrap to angle domain
            int cAng = static_cast<int>(p2.cAng);
            int diff = gpu_minimum_angle(sAng, cAng);
            //printf("GPU [%d] i=%d j=%d %d, %d, %d\n", tid, i, j, sAng, cAng, diff);
            //break;
            // compare with threshold
            bool directions = (diff <= param.maximumDirectionDifference);
            if (!directions) {
                //printf("GPU [%d] i=%d j=%d %d, %d, %d\n", tid, i, j, sAng, cAng, diff);
                //break;
                continue;
            }

            // ----- Beta angles 5.2.x (same as CPU "anglesBeta" lambda) -----
            // pBeta = rotateAngle(p1.probe()->angle(), p2.probe()->angle());
            // cBeta = rotateAngle(p1.candidate()->angle(), p2.candidate()->angle());
            // minimumAngle(pBeta, cBeta) <= Param::maximumAngleDifference()
            int pBeta    = rotate_angle_f(static_cast<int>(p1.pAng), static_cast<int>(p2.pAng));
            int cBeta    = rotate_angle_f(static_cast<int>(p1.cAng), static_cast<int>(p2.cAng));
            int betaDiff = gpu_minimum_angle(pBeta, cBeta);
            if (betaDiff > param.maximumAngleDifference) {
                continue;
            }

            // All checks passed → count as matched
            matched++;
            // CPU does: if (matched + --min < MinimumMinutiae) break;
            // i.e. decrement only on successful match:
            minRemaining--;
            if (matched + minRemaining < param.minimumMinutiae) {
                break;
            }
        }

        if (matched > localMax)
            localMax = matched;
    }

    // Reduce max across threads in block
    __shared__ int s_max;
    if (threadIdx.x == 0) s_max = 0;
    __syncthreads();

    atomicMax(&s_max, localMax);
    __syncthreads();

    if (threadIdx.x == 0) {
        int maxMatched = s_max;
        int pCount     = param.pCount;
        int cCount     = param.cCount;

        uint8_t score = 0;
        // CPU uses "if (maxMatched > Param::MinimumMinutiae)" (strict >)
        if (maxMatched > param.minimumMinutiae &&
            pCount > 0 && cCount > 0)
        {
            int num = maxMatched * maxMatched * 100;
            int den = pCount * cCount;
            int s   = num / den;
            if (s > 255) s = 255;
            score = static_cast<uint8_t>(s);
        }
        *outScore = score;
    }
}


__global__ void global_match_kernel_multi(
    const MinutiaPairGPU* __restrict__ pairs,
    const int*            __restrict__ offsets,
    const int*            __restrict__ pairCounts,
    const int*            __restrict__ cCounts,
    ParamGPU              paramCommon,
    int                   numCands,
    uint8_t*              __restrict__ scores)
{
    int laneId      = threadIdx.x & 31;      // 0..31
    int warpInBlock = threadIdx.x >> 5;      // 0..(warpsPerBlock-1)
    int globalWarp  = warpInBlock + blockIdx.x * (blockDim.x >> 5);

    if (globalWarp >= numCands) {
        return;
    }

    int candIdx   = globalWarp;
    int start     = offsets[candIdx];
    int count     = pairCounts[candIdx];
    int cCount    = cCounts[candIdx];
    int pCount    = paramCommon.pCount;

    if (count <= 0 || pCount <= 0 || cCount <= 0) {
        if (laneId == 0) {
            scores[candIdx] = 0;
        }
        return;
    }

    int localMax = 0;

    // Each warp does the same algorithm you already have, but:
    //   - indices are shifted by 'start'
    //   - loops are over 'count' instead of 'numPairs'
    //   - i is strided per lane
    for (int i = laneId; i < count; i += 32) {
        const auto& p1 = pairs[start + i];

        float theta    = p1.cAng - p1.pAng;
        float cosTheta = gpu_cos(theta);
        float sinTheta = gpu_sin(theta);

        int matched      = 1;
        int minRemaining = count - 1;

        for (int j = 0; j < count; ++j) {
            if (i == j) continue;

            const auto& p2 = pairs[start + j];

            // ----- position check -----
            float x = p1.cx + cosTheta * (p2.px - p1.px) - sinTheta * (p2.py - p1.py);
            float y = p1.cy + sinTheta * (p2.px - p1.px) + cosTheta * (p2.py - p1.py);

            float dx = x - p2.cx;
            float dy = y - p2.cy;
            float dist2 = dx * dx + dy * dy;
            float maxGlobal2 = paramCommon.maximumGlobalDistance * paramCommon.maximumGlobalDistance;

            if (dist2 > maxGlobal2) {
                continue;
            }

            // ----- direction check -----
            int sAng = static_cast<int>(p2.pAng + theta);
            int cAng = static_cast<int>(p2.cAng);
            int diff = gpu_minimum_angle(sAng, cAng);

            if (diff > paramCommon.maximumDirectionDifference) {
                continue;
            }

            // ----- beta angles -----
            int pBeta    = rotate_angle_f(static_cast<int>(p1.pAng), static_cast<int>(p2.pAng));
            int cBeta    = rotate_angle_f(static_cast<int>(p1.cAng), static_cast<int>(p2.cAng));
            int betaDiff = gpu_minimum_angle(pBeta, cBeta);

            if (betaDiff > paramCommon.maximumAngleDifference) {
                continue;
            }

            matched++;
            minRemaining--;
            if (matched + minRemaining < paramCommon.minimumMinutiae) {
                break;
            }
        }

        if (matched > localMax) {
            localMax = matched;
        }
    }

    // Warp-wide reduction of localMax
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other = __shfl_down_sync(0xffffffff, localMax, offset);
        localMax = max(localMax, other);
    }

    if (laneId == 0) {
        int maxMatched = localMax;
        uint8_t score  = 0;

        if (maxMatched > paramCommon.minimumMinutiae &&
            pCount > 0 && cCount > 0) {

            int num = maxMatched * maxMatched * 100;
            int den = pCount * cCount;
            int s   = num / den;
            if (s > 255) s = 255;
            score = static_cast<uint8_t>(s);
        }

        scores[candIdx] = score;
    }
}



// Host wrapper
uint8_t gpu_global_score(const MinutiaPairGPU* pairs,
                         int numPairs,
                         const ParamGPU& param)
{
    if (numPairs <= 0)
        return 0;

    MinutiaPairGPU* d_pairs = nullptr;
    uint8_t*        d_score = nullptr;

    size_t bytesPairs = numPairs * sizeof(MinutiaPairGPU);

    cudaMalloc(&d_pairs, bytesPairs);
    cudaMalloc(&d_score, sizeof(uint8_t));

    cudaMemcpy(d_pairs, pairs, bytesPairs, cudaMemcpyHostToDevice);

    dim3 block(64);
    dim3 grid(1);
    global_match_kernel<<<grid, block>>>(d_pairs, numPairs, param, d_score);

    uint8_t h_score{};
    cudaMemcpy(&h_score, d_score, sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(d_pairs);
    cudaFree(d_score);

    return h_score;
}

uint8_t gpu_global_score_batched(const MinutiaPairGPU* allPairs,
                                 int                  totalPairs,
                                 const int*           offsets,
                                 const int*           pairCounts,
                                 const int*           cCounts,
                                 int                  numCands,
                                 const ParamGPU&      paramCommon,
                                 std::vector<uint8_t>& scoresOut,
                                 cudaStream_t         stream)
{
    if (numCands <= 0 || totalPairs <= 0) {
        scoresOut.assign(numCands, 0);
        return 0;
    }

    scoresOut.resize(numCands);

    // Device buffers
    MinutiaPairGPU* d_pairs    = nullptr;
    int*            d_offsets  = nullptr;
    int*            d_counts   = nullptr;
    int*            d_cCounts  = nullptr;
    uint8_t*        d_scores   = nullptr;

    cudaMallocAsync(&d_pairs,   totalPairs * sizeof(MinutiaPairGPU), stream);
    cudaMallocAsync(&d_offsets, numCands   * sizeof(int), stream);
    cudaMallocAsync(&d_counts,  numCands   * sizeof(int), stream);
    cudaMallocAsync(&d_cCounts, numCands   * sizeof(int), stream);
    cudaMallocAsync(&d_scores,  numCands   * sizeof(uint8_t), stream);

    cudaMemcpyAsync(d_pairs,   allPairs,        totalPairs * sizeof(MinutiaPairGPU), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_offsets, offsets,         numCands   * sizeof(int),            cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_counts,  pairCounts,      numCands   * sizeof(int),            cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_cCounts, cCounts,         numCands   * sizeof(int),            cudaMemcpyHostToDevice, stream);

    // Launch config: e.g. 128 threads/block = 4 warps/block
    int warpsPerBlock = 4;
    int threadsPerBlock = 32 * warpsPerBlock;
    int numWarps = numCands;
    int grid = (numWarps + warpsPerBlock - 1) / warpsPerBlock;

    global_match_kernel_multi<<<grid, threadsPerBlock, 0, stream>>>(
        d_pairs,
        d_offsets,
        d_counts,
        d_cCounts,
        paramCommon,
        numCands,
        d_scores
    );

    cudaMemcpyAsync(scoresOut.data(), d_scores, numCands * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream);

    cudaFreeAsync(d_pairs, stream);
    cudaFreeAsync(d_offsets, stream);
    cudaFreeAsync(d_counts, stream);
    cudaFreeAsync(d_cCounts, stream);
    cudaFreeAsync(d_scores, stream);

    // Optionally return max score
    uint8_t maxScore = 0;
    for (int i = 0; i < numCands; ++i) {
        if (scoresOut[i] > maxScore) maxScore = scoresOut[i];
    }
    return maxScore;
}


void init_gpu_trig_tables()
{
    using namespace AngleLUT;

    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    std::vector<float> h_cos(kAngleTableSize);
    std::vector<float> h_sin(kAngleTableSize);

    // exactly same index mapping as FastMath::Cosines/Sines
    for (int i = 0; i < kAngleTableSize; ++i) {
        Field::AngleType angle = static_cast<Field::AngleType>(kAngleMin + i);
        h_cos[i] = FastMath::cos(angle);
        h_sin[i] = FastMath::sin(angle);
    }

    cudaMemcpyToSymbol(d_cos_table, h_cos.data(),
                       kAngleTableSize * sizeof(float));
    cudaMemcpyToSymbol(d_sin_table, h_sin.data(),
                       kAngleTableSize * sizeof(float));
}


}
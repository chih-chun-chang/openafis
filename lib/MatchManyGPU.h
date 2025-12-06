// MatchManyGPU.h
#ifndef MATCHMANYGPU_H
#define MATCHMANYGPU_H

#include <cstdint>
#include <vector>
#include "Field.h"
#include <cuda_runtime.h>

namespace OpenAFIS {

struct MinutiaPairGPU {
    float  px, py;   // probe x,y
    float  cx, cy;   // candidate x,y
    float  pAng;     // probe angle in radians
    float  cAng;     // candidate angle in radians
};

struct ParamGPU {
    float maximumGlobalDistance;
    float maximumDirectionDifference;
    float maximumAngleDifference;
    int   minimumMinutiae;
    int   pCount;
    int   cCount;
};

// Host-side entry point to global GPU scoring
uint8_t gpu_global_score(const MinutiaPairGPU* pairs,
                         int numPairs,
                         const ParamGPU& param);

uint8_t gpu_global_score_batched(const MinutiaPairGPU* allPairs,
                                 int                  totalPairs,
                                 const int*           offsets,
                                 const int*           pairCounts,
                                 const int*           cCounts,
                                 int                  numCands,
                                 const ParamGPU&      paramCommon,
                                 std::vector<uint8_t>& scoresOut,
                                 cudaStream_t         stream = 0);

void init_gpu_trig_tables();

} // namespace OpenAFIS

#endif // MATCHMANYGPU_H
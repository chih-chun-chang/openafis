#ifndef MATCHMANY_H
#define MATCHMANY_H


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//#include "ThreadPool.h"

#include <vector>

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <cuda_runtime_api.h>

#include "Match.h"                 // for Match<R,F,P>
#include "MatchManyGPU.h"

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
namespace OpenAFIS
{

template <class TemplateType> 
class MatchMany {
public:
    using Templates = std::vector<TemplateType>;
    using OneManyResult = std::pair<uint8_t, const TemplateType*>;

    using FingerprintType = Fingerprint;                    // adjust if your TemplateType differs
    using PairType        = MinutiaPoint::Pair;             // for R = uint8_t CPU match
    using MatchType       = Match<uint8_t, FingerprintType, PairType>;

    MatchMany();
    ~MatchMany();

    OneManyResult oneMany(const TemplateType& probe, const Templates& candidates) const;
    void manyMany(std::vector<uint8_t>& scores, const Templates& templates) const;

    //[[nodiscard]] size_t concurrency() const { return m_pool.size(); }
    [[nodiscard]] size_t concurrency() const { return m_executor.num_workers(); }

    void setUseGpu(bool v) { m_useGpu = v; }
    bool useGpu() const { return m_useGpu; }

private:
    struct WorkerBuffer {  // one pair-set per candidate that produced non-empty pairs
        std::vector<std::vector<MinutiaPairGPU>> pairs_per_candidate;
        std::vector<int>                         cCounts;   // candidate minutiaeCount
        std::vector<const TemplateType*>         candPtrs;  // pointer to that candidate
    };

    //mutable ThreadPool m_pool;
    mutable tf::Executor m_executor;
    mutable std::vector<MatchType> m_threadMatches;
    mutable std::vector<WorkerBuffer> m_workerBuffers;
    bool m_useGpu = true;

    mutable std::vector<cudaStream_t> m_streams;
};
}

#endif // MATCHMANY_H

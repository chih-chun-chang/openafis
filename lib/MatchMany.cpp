
#include "MatchMany.h"
#include "Match.h"
#include "Param.h"
#include "TemplateISO19794_2_2005.h"
#include "MatchManyGPU.h"
#include <future>


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
namespace OpenAFIS
{


    
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
/*
template <class T>
MatchMany<T>::MatchMany()
    : m_pool(std::min(Param::MaximumConcurrency, std::thread::hardware_concurrency()), ThreadPool::Priority::Critical)
{
}
*/
template <class TemplateType>
MatchMany<TemplateType>::MatchMany()
: m_executor(std::thread::hardware_concurrency()) {

    auto W = m_executor.num_workers();
    m_threadMatches.resize(W);
    m_workerBuffers.resize(W);

    m_streams.resize(W);
    for (size_t w = 0; w < W; ++w) {
        cudaStreamCreate(&m_streams[w]);
    }

}

template <class TemplateType>
MatchMany<TemplateType>::~MatchMany() {
    for (auto s : m_streams) {
        if (s) cudaStreamDestroy(s);
    }
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <class T> typename MatchMany<T>::OneManyResult MatchMany<T>::oneMany(const T& probe, const Templates& candidates) const
{
    // Trivial case
    if (candidates.empty()) {
        return std::make_pair(uint8_t {}, nullptr);
    }

    // Assume 1 fingerprint per template
    const auto& probeFps = probe.fingerprints();
    if (probeFps.empty()) {
        return std::make_pair(uint8_t{0}, nullptr);
    }
    const auto& probeFp = probeFps[0];

    // --- CPU-only fallback (no GPU or too few candidates) -------------------
    constexpr size_t gpuThreshold = 8; // tune this
    if (!m_useGpu || candidates.size() < gpuThreshold) {
        std::cout << "CPU-only oneMany\n";
        MatchType match;  // local CPU-only matcher

        uint8_t bestScore = 0;
        const T* bestTpl = nullptr;

        for (const auto& candT : candidates) {
            const auto& candFps = candT.fingerprints();
            if (candFps.empty()) {
                continue;
            }
            const auto& candFp = candFps[0];

            uint8_t score = 0;
            //match.compute(score, probeFp, candFp);  // CPU-only (local+global)
            match.compute(probeFp, candFp);
            match.compute_global_matching_cpu(score, probeFp, candFp); // CPU-only global stage

            if (score > bestScore) {
                bestScore = score;
                bestTpl   = &candT;
            }
        }
        return std::make_pair(bestScore, bestTpl);
    }


    const size_t N = candidates.size();

    // Ensure per-worker state is sized correctly
    const size_t W = m_executor.num_workers();
    for (auto& buf : m_workerBuffers) {
        buf.pairs_per_candidate.clear();
        buf.cCounts.clear();
        buf.candPtrs.clear();
    }

    // Per-worker best scores & best template pointers
    //std::vector<uint8_t>  bestScorePerWorker(W, 0);
    //std::vector<const T*> bestTplPerWorker(W, nullptr);

    tf::Taskflow tf;

    int begin = 0;
    int end   = static_cast<int>(N);
    
    init_gpu_trig_tables();

    tf.for_each_index(std::ref(begin), std::ref(end), 1, [&](int idx) {
        size_t wid = m_executor.this_worker_id();
        if (wid >= W) return;

        auto& match = m_threadMatches[wid];
        auto& buf   = m_workerBuffers[wid];

        const T& candT = candidates[static_cast<size_t>(idx)];
        const auto& candFps = candT.fingerprints();
        if (candFps.empty()) return;
        const auto& candFp = candFps[0];

        // 1) Local matching (5.1) on CPU – fills m_pairs inside match
        if (probeFps.empty() || candFps.empty()) return;
        match.compute(probeFp, candFp);


        // 2) Build GPU-friendly minutia pairs from match.m_pairs
        auto gpuPairs = match.buildGpuPairs();
        if (gpuPairs.empty()) return;

        // 3) Stash them; each entry corresponds to ONE candidate (for this worker)
        buf.pairs_per_candidate.emplace_back(std::move(gpuPairs));
        buf.cCounts.push_back(static_cast<int>(candFp.minutiaeCount()));
        buf.candPtrs.push_back(&candT);

        /*
        ParamGPU param{};
        param.maximumGlobalDistance      = static_cast<float>(Param::MaximumGlobalDistance);
        param.maximumDirectionDifference = Param::maximumDirectionDifference();
        param.maximumAngleDifference     = Param::maximumAngleDifference();
        param.minimumMinutiae            = Param::MinimumMinutiae;
        param.pCount                     = static_cast<int>(probeFp.minutiaeCount());
        param.cCount                     = static_cast<int>(candFp.minutiaeCount());

        // 4) Global matching (5.2) on GPU
        uint8_t score = gpu_global_score(
            gpuPairs.data(),
            static_cast<int>(gpuPairs.size()),
            param
        );

        // 5) Update this worker’s best result
        if (score > bestScorePerWorker[wid] && score <= 100) {
            std::cout << "thread[" << wid << "] new best score: " << static_cast<int>(score) << " (idx=" << idx << ")\n";
            bestScorePerWorker[wid] = score;
            bestTplPerWorker[wid]   = &candT;
        }
        */
    });

    // Run the Taskflow graph
    m_executor.run(tf).wait();

    // ---- Flatten all workers into single arrays ----
    std::vector<MinutiaPairGPU> allPairs;
    std::vector<int>            offsets;   // candidate -> starting index in allPairs
    std::vector<int>            pairCounts;
    std::vector<int>            cCounts;   // candidate minutiae counts
    std::vector<const T*>       candPtrs;  // candidate pointers in same order

    int totalPairs = 0;
    int totalCands = 0;

    // First pass: count totals
    for (size_t w = 0; w < W; ++w) {
        const auto& buf = m_workerBuffers[w];
        totalCands += static_cast<int>(buf.pairs_per_candidate.size());
        for (const auto& v : buf.pairs_per_candidate) {
            totalPairs += static_cast<int>(v.size());
        }
    }

    if (totalCands == 0 || totalPairs == 0) {
        // No matches
        return {0u, nullptr};
    }

    allPairs.reserve(totalPairs);
    offsets.reserve(totalCands + 1);
    pairCounts.reserve(totalCands);
    cCounts.reserve(totalCands);
    candPtrs.reserve(totalCands);

    int curOffset = 0;
    for (size_t w = 0; w < W; ++w) {
        auto& buf = m_workerBuffers[w];
        for (size_t k = 0; k < buf.pairs_per_candidate.size(); ++k) {
            const auto& v = buf.pairs_per_candidate[k];

            offsets.push_back(curOffset);
            pairCounts.push_back(static_cast<int>(v.size()));
            cCounts.push_back(buf.cCounts[k]);
            candPtrs.push_back(buf.candPtrs[k]);

            allPairs.insert(allPairs.end(), v.begin(), v.end());
            curOffset += static_cast<int>(v.size());
        }
    }
    offsets.push_back(curOffset);  // sentinel at the end

    const int numPackedCands = static_cast<int>(candPtrs.size());
    // sanity
    // assert(numPackedCands == totalCands);

    // Reduce over workers to find global best
    /*
    uint8_t bestScore = 0;
    const T* bestTpl = nullptr;

    for (size_t w = 0; w < W; ++w) {
        if (bestScorePerWorker[w] > bestScore) {
            bestScore = bestScorePerWorker[w];
            bestTpl   = bestTplPerWorker[w];
        }
    }

    return std::make_pair(bestScore, bestTpl);
    */

    ParamGPU param{};
    param.maximumGlobalDistance      = static_cast<float>(Param::MaximumGlobalDistance);
    param.maximumDirectionDifference = Param::maximumDirectionDifference();
    param.maximumAngleDifference     = Param::maximumAngleDifference();
    param.minimumMinutiae            = Param::MinimumMinutiae;
    param.pCount                     = static_cast<int>(probeFp.minutiaeCount());
    // cCount is per candidate; we use cCounts[] array

    std::vector<uint8_t> scores;
    gpu_global_score_batched(
        allPairs.data(),
        static_cast<int>(allPairs.size()),
        offsets.data(),
        pairCounts.data(),
        cCounts.data(),
        numPackedCands,
        param,
        scores
    );

    // final max over packed candidates
    uint8_t bestScore = 0;
    const T* bestTpl  = nullptr;

    for (int i = 0; i < numPackedCands; ++i) {
        uint8_t s = scores[i];
        if (s > bestScore && s <= 100) {
            bestScore = s;
            bestTpl   = candPtrs[i];
        }
    }

    return std::make_pair(bestScore, bestTpl);

    /*
    // GPU Params
    ParamGPU param{};
    param.maximumGlobalDistance      = static_cast<float>(Param::MaximumGlobalDistance);
    param.maximumDirectionDifference = static_cast<float>(Param::maximumDirectionDifference());
    param.maximumAngleDifference     = static_cast<float>(Param::maximumAngleDifference());
    param.minimumMinutiae            = Param::MinimumMinutiae;



    //if (m_pool.size() == 1) {
        MatchSimilarity match;

        uint8_t maxSimilarity {};
        const T* maxCandidate {};
        const auto& probeT = probe.fingerprints()[0];

        auto count = 0;
        for (const auto& t : candidates) {
            uint8_t similarity {};
            //match.compute(similarity, probeT, t.fingerprints()[0]);
            match.compute(probeT, t.fingerprints()[0]);
            //std::exit(1);

            param.pCount = static_cast<int>(probeT.minutiaeCount());
            param.cCount = static_cast<int>(t.fingerprints()[0].minutiaeCount());
            auto gpuPairs = match.buildGpuPairs(); 
            uint8_t gpuScore = 0;
            if (!gpuPairs.empty()) {
                gpuScore = gpu_global_score(gpuPairs.data(),
                                            static_cast<int>(gpuPairs.size()),
                                            param);
            }
            similarity = gpuScore;
            count++;

            if (gpuPairs.size() > 32) std::cout << "[DEBUG] numPairs: " << gpuPairs.size() << "\n";

            if (similarity > maxSimilarity && similarity <= 100) {
                maxSimilarity = similarity;
                maxCandidate = &t;
                std::cout << count++ << " " << static_cast<size_t>(maxSimilarity) << " numPairs: " << gpuPairs.size() << "\n";
            } 
        }
        return std::make_pair(maxSimilarity, maxCandidate);
    //}
    */



    /*
    std::vector<std::future<OneManyResult>> futures;
    futures.reserve(m_pool.size());

    for (auto fromIt = candidates.begin();;) {
        auto endIt = fromIt + std::min(static_cast<size_t>(candidates.end() - fromIt), static_cast<size_t>(std::ceil(static_cast<float>(candidates.size()) / m_pool.size())));

        futures.emplace_back(m_pool.enqueue([=, &probeT = probe.fingerprints()[0]]() {
            MatchSimilarity match;

            uint8_t maxSimilarity {};
            const T* maxCandidate {};

            for (auto it = fromIt; it < endIt; ++it) {
                uint8_t similarity {};
                match.compute(similarity, probeT, it->fingerprints()[0]);
                if (similarity > maxSimilarity) {
                    maxSimilarity = similarity;
                    maxCandidate = &(*it);
                }
            }
            return std::make_pair(maxSimilarity, maxCandidate);
        }));

        if (endIt == candidates.end()) {
            break;
        }
        fromIt = endIt;
    }

    OneManyResult bestR;
    for (auto& f : futures) {
        const auto& r = f.get();
        if (r.second && r.first > bestR.first) {
            bestR = r;
        }
    }
    return bestR;
    */
}


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <class T> void MatchMany<T>::manyMany(std::vector<uint8_t>& scores, const Templates& templates) const
{
    if (scores.size() != templates.size() * templates.size()) {
        return;
    }
    
    /*
    if (m_pool.size() == 1) {
        MatchSimilarity match;
        auto* scoresPtr = scores.data();

        for (const auto& t1 : templates) {
            auto& t1t = t1.fingerprints()[0];
            for (const auto& t2 : templates) {
                match.compute(*scoresPtr++, t1t, t2.fingerprints()[0]);
            }
        }
        return;
    }
    std::vector<std::future<void>> futures;
    futures.reserve(templates.size());
    size_t i {};
    for (const auto& t1 : templates) {
        futures.emplace_back(std::async(std::launch::async, [=, &t1t = t1.fingerprints()[0], &scores, &templates]() {
            MatchSimilarity match;
            auto* scoresPtr = &scores[i];
            for (const auto& t2 : templates) {
                match.compute(*scoresPtr++, t1t, t2.fingerprints()[0]);
            }
        }));
        i += templates.size();
    }
    for (const auto& f : futures) {
        f.wait();
    }
    */


    const size_t N = templates.size();

    // Ensure per-worker state is sized correctly
    const size_t W = m_executor.num_workers();
    for (auto& buf : m_workerBuffers) {
        buf.pairs_per_candidate.clear();
        buf.cCounts.clear();
        buf.candPtrs.clear();
    }

    tf::Taskflow tf;

    int begin = 0;
    int end   = static_cast<int>(N);
    
    init_gpu_trig_tables();

    std::vector< std::vector<MinutiaPairGPU> > m_allPairs(W);
    std::vector< std::vector<int> >            m_offsets(W);   // candidate -> starting index in allPairs
    std::vector< std::vector<int> >            m_pairCounts(W);
    std::vector< std::vector<int> >            m_cCounts(W);   // candidate minutiae counts
    std::vector< std::vector<const T*> >       m_candPtrs(W);  // candidate pointers in same order
    std::vector< std::vector<uint8_t> >        m_scores(W);

    tf.for_each_index(std::ref(begin), std::ref(end), 1, [&](int idx) {
        size_t wid = m_executor.this_worker_id();
        if (wid >= W) return;

        auto& match = m_threadMatches[wid];
        auto& buf   = m_workerBuffers[wid];

        const auto& probeFps = templates[static_cast<size_t>(idx)].fingerprints();
        const auto& probeFp = probeFps[0];
        
        for (size_t j = 0; j < N; ++j) {

            const T& candT = templates[j];
            const auto& candFps = candT.fingerprints();
            if (candFps.empty()) return;
            const auto& candFp = candFps[0];

            // 1) Local matching (5.1) on CPU – fills m_pairs inside match
            if (probeFps.empty() || candFps.empty()) return;
            match.compute(probeFp, candFp);


            // 2) Build GPU-friendly minutia pairs from match.m_pairs
            auto gpuPairs = match.buildGpuPairs();
            if (gpuPairs.empty()) return;

            // 3) Stash them; each entry corresponds to ONE candidate (for this worker)
            buf.pairs_per_candidate.emplace_back(std::move(gpuPairs));
            buf.cCounts.push_back(static_cast<int>(candFp.minutiaeCount()));
            buf.candPtrs.push_back(&candT);
        }

        int totalPairs = 0;
        int totalCands = static_cast<int>(buf.pairs_per_candidate.size());
        for (const auto& v : buf.pairs_per_candidate) {
            totalPairs += static_cast<int>(v.size());
        }

        auto& allPairs   = m_allPairs[wid];
        auto& offsets    = m_offsets[wid];
        auto& pairCounts = m_pairCounts[wid];
        auto& cCounts    = m_cCounts[wid];
        auto& candPtrs   = m_candPtrs[wid];
        auto& sc         = m_scores[wid];

        allPairs.clear();   allPairs.reserve(totalPairs);
        offsets.clear();    offsets.reserve(totalCands + 1);
        pairCounts.clear(); pairCounts.reserve(totalCands);
        cCounts.clear();    cCounts.reserve(totalCands);
        candPtrs.clear();   candPtrs.reserve(totalCands);

        int curOffset = 0;
        for (size_t k = 0; k < buf.pairs_per_candidate.size(); ++k) {
            const auto& v = buf.pairs_per_candidate[k];

            offsets.push_back(curOffset);
            pairCounts.push_back(static_cast<int>(v.size()));
            cCounts.push_back(buf.cCounts[k]);
            candPtrs.push_back(buf.candPtrs[k]);

            allPairs.insert(allPairs.end(), v.begin(), v.end());
            curOffset += static_cast<int>(v.size());
        }
        offsets.push_back(curOffset);  // sentinel at the end

        const int numPackedCands = static_cast<int>(candPtrs.size());
        

        ParamGPU param{};
        param.maximumGlobalDistance      = static_cast<float>(Param::MaximumGlobalDistance);
        param.maximumDirectionDifference = Param::maximumDirectionDifference();
        param.maximumAngleDifference     = Param::maximumAngleDifference();
        param.minimumMinutiae            = Param::MinimumMinutiae;
        param.pCount                     = static_cast<int>(probeFp.minutiaeCount());

        gpu_global_score_batched(
            allPairs.data(),
            static_cast<int>(allPairs.size()),
            offsets.data(),
            pairCounts.data(),
            cCounts.data(),
            numPackedCands,
            param,
            sc,
            m_streams[wid]
        );

        for (size_t l = 0; l < N; ++l) {
            uint8_t s = sc[l];
            if (s > 100) s = 100;
            scores[idx * N + l] = s;
        }

    });

    // Run the Taskflow graph
    m_executor.run(tf).wait();

    for (auto s : m_streams) {
        if (s) cudaStreamSynchronize(s);
    }


}


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Explicit instantiations...
//
template class MatchMany<TemplateISO19794_2_2005<uint32_t, Fingerprint>>;
template class MatchMany<TemplateISO19794_2_2005<std::string, Fingerprint>>;
}

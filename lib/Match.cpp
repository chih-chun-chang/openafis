
#include "Match.h"
#include "FastMath.h"
#include "Log.h"
#include "Param.h"
#include "MatchManyGPU.h"

#include <algorithm>


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
namespace OpenAFIS
{


template <class R, class F, class P>
std::vector<MinutiaPairGPU>
Match<R,F,P>::buildGpuPairs() const
{
    std::vector<MinutiaPairGPU> gpuPairs;
    gpuPairs.reserve(m_pairs.size());

    for (const auto& pair : m_pairs) {
        const auto* p = pair.probe();
        const auto* c = pair.candidate();

        MinutiaPairGPU g{};
        g.px   = static_cast<float>(p->x());
        g.py   = static_cast<float>(p->y());
        g.cx   = static_cast<float>(c->x());
        g.cy   = static_cast<float>(c->y());
        g.pAng = static_cast<float>(p->angle());
        g.cAng = static_cast<float>(c->angle());

        gpuPairs.push_back(g);
    }
    return gpuPairs;
}


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// https://doi.org/10.3390/s120303418
//
template <class R, class F, class P> void Match<R,F,P>::compute_global_matching_cpu(R& result, const F& probe, const F& candidate) const
{
    // Global matching 5.2...
    auto maxMatched = 0;
    for (std::size_t i = 0; i < m_pairs.size(); ++i) {
        const auto& p1 = m_pairs[i];
        const auto theta = static_cast<Field::AngleType>(p1.candidate()->angle() - p1.probe()->angle());
        const auto cosTheta = FastMath::cos(theta);
        const auto sinTheta = FastMath::sin(theta);

        auto matched = 1;
        auto min = m_pairs.size() - 1;

        //for (const auto& p2 : m_pairs) {
        for (std::size_t j = 0; j < m_pairs.size(); ++j) {
            const auto& p2 = m_pairs[j];

            if (&p1 == &p2) {
                continue;
            }

            // 5.2.2...
            const auto x = p1.candidate()->x() + cosTheta * (p2.probe()->x() - p1.probe()->x()) - sinTheta * (p2.probe()->y() - p1.probe()->y());
            const auto y = p1.candidate()->y() + sinTheta * (p2.probe()->x() - p1.probe()->x()) + cosTheta * (p2.probe()->y() - p1.probe()->y());
            const auto a = x - p2.candidate()->x();
            const auto b = y - p2.candidate()->y();
            const auto c = a * a + b * b;
            bool lengths = c <= Param::MaximumGlobalDistance * Param::MaximumGlobalDistance;
            
            if (!lengths) {
                continue;
            }

            bool directions = false;
            if constexpr (std::is_same_v<Field::AngleType, float>) { // float angle mode
                float s = p2.probe()->angle() + theta; // wrap into [0, 2π)
                if (s > FastMath::TwoPI)
                    s -= FastMath::TwoPI;
                else if (s < 0)
                    s += FastMath::TwoPI;
                auto diff = FastMath::minimumAngle(s, p2.candidate()->angle());
                directions = (diff <= Param::maximumDirectionDifference());
            } else if constexpr (std::is_same_v<Field::AngleType, int16_t>) { // fixed angle mode
                Field::AngleSize s = static_cast<Field::AngleSize>(p2.probe()->angle() + theta);
                auto diff = FastMath::minimumAngle(s, p2.candidate()->angle());
                //std::cout << i << " " << j << " s=" << static_cast<int>(s) << " " << p2.candidate()->angle() << " diff=" << diff <<"\n";
                directions = (diff <= Param::maximumDirectionDifference());
            }

            if (!directions) {
                continue;
            }

            const auto p = FastMath::rotateAngle(p1.probe()->angle(), p2.probe()->angle());
            const auto ca = FastMath::rotateAngle(p1.candidate()->angle(), p2.candidate()->angle());
            bool anglesBeta = FastMath::minimumAngle(p, ca) <= Param::maximumAngleDifference();

            if (!anglesBeta) {
                continue;
            }
            matched++;
            if (matched + --min < Param::MinimumMinutiae) {
                break;
            }

            // When this class is specialized for rendering only - no overhead when computing similarities...
            if constexpr (std::is_same_v<R, MinutiaPoint::PairRenderable::Set>) {
                std::cout << "for rendering only\n";
                result.insert(&p2);
            }
        }
        maxMatched = std::max(maxMatched, matched);
    }
    if constexpr (std::is_same_v<R, uint8_t>) {
        if (maxMatched > Param::MinimumMinutiae) {
            result = static_cast<uint8_t>((maxMatched * maxMatched * 100) / (probe.minutiaeCount() * candidate.minutiaeCount()));
        }
    }

}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//template <class R, class F, class P> void Match<R, F, P>::compute(R& result, const F& probe, const F& candidate) const
template <class R, class F, class P> void Match<R, F, P>::compute(const F& probe, const F& candidate) const
{
    //if constexpr (std::is_same_v<R, uint8_t>) {
    //    //result = 0;
    //    return;
    //}

    const auto& probeT = probe.triplets();
    const auto& candidateT = candidate.triplets();
    m_tripletPairs.clear();

    // Local matching 5.1.1-2...
    for (const auto& p : probeT) {

        auto it = std::lower_bound(candidateT.begin(), candidateT.end(), static_cast<Field::MinutiaDistanceType>(p.maxDistance() - Param::MaximumLocalDistance));
        if (it == candidateT.end()) {
            continue;
        }
        const auto end = std::upper_bound(it, candidateT.end(), static_cast<Field::MinutiaDistanceType>(p.maxDistance() + Param::MaximumLocalDistance));

        for (; it < end; ++it) {
            if (!it->skipPair(p)) {
                it->emplacePair(m_tripletPairs, p);
            }
        }
    }

    if (m_tripletPairs.size() < Param::MinimumMinutiae) {
        return;
    }

    // Local matching 5.1.3-5...
    std::sort(m_tripletPairs.begin(), m_tripletPairs.end());
    m_probeDupes.clear();
    m_candidateDupes.clear();
    m_pairs.clear();

    for (const auto& p : m_tripletPairs) {
        for (decltype(p.probe()->minutiae().size()) i = 0; i < p.probe()->minutiae().size(); ++i) {
            const auto dp = m_probeDupes.emplace(p.probe()->minutiae()[i].key());
            const auto dc = m_candidateDupes.emplace(p.candidate()->minutiae()[i].key());
            if (!dp.second && !dc.second) {
                // duplicate point...
                continue;
            }
            if constexpr (std::is_same_v<R, MinutiaPoint::PairRenderable::Set>) {
                // Similarity values are scaled for integers over [0,1000], for render % is fine...
                m_pairs.emplace_back(&p.probe()->minutiae()[i], &p.candidate()->minutiae()[i], std::lround(static_cast<float>(p.similarity()) / 10.0f));
            }
            if constexpr (std::is_same_v<R, uint8_t>) {
                m_pairs.emplace_back(&p.probe()->minutiae()[i], &p.candidate()->minutiae()[i]);
            }
        }
    }
    //std::cout << m_pairs.size() << "\n";

    /*
    // Global matching 5.2...
    auto maxMatched = 0;
    //for (const auto& p1 : m_pairs) {
    for (std::size_t i = 0; i < m_pairs.size(); ++i) {
        const auto& p1 = m_pairs[i];
        const auto theta = static_cast<Field::AngleType>(p1.candidate()->angle() - p1.probe()->angle());
        const auto cosTheta = FastMath::cos(theta);
        const auto sinTheta = FastMath::sin(theta);

        auto matched = 1;
        auto min = m_pairs.size() - 1;

        //for (const auto& p2 : m_pairs) {
        for (std::size_t j = 0; j < m_pairs.size(); ++j) {
            const auto& p2 = m_pairs[j];

            if (&p1 == &p2) {
                continue;
            }

            // 5.2.2...
            const auto x = p1.candidate()->x() + cosTheta * (p2.probe()->x() - p1.probe()->x()) - sinTheta * (p2.probe()->y() - p1.probe()->y());
            const auto y = p1.candidate()->y() + sinTheta * (p2.probe()->x() - p1.probe()->x()) + cosTheta * (p2.probe()->y() - p1.probe()->y());
            const auto a = x - p2.candidate()->x();
            const auto b = y - p2.candidate()->y();
            const auto c = a * a + b * b;
            bool lengths = c <= Param::MaximumGlobalDistance * Param::MaximumGlobalDistance;
            
            if (!lengths) {
                continue;
            }

            bool directions = false;
            if constexpr (std::is_same_v<Field::AngleType, float>) { // float angle mode
                float s = p2.probe()->angle() + theta; // wrap into [0, 2π)
                if (s > FastMath::TwoPI)
                    s -= FastMath::TwoPI;
                else if (s < 0)
                    s += FastMath::TwoPI;
                auto diff = FastMath::minimumAngle(s, p2.candidate()->angle());
                directions = (diff <= Param::maximumDirectionDifference());
            } else if constexpr (std::is_same_v<Field::AngleType, int16_t>) { // fixed angle mode
                Field::AngleSize s = static_cast<Field::AngleSize>(p2.probe()->angle() + theta);
                auto diff = FastMath::minimumAngle(s, p2.candidate()->angle());
                //std::cout << i << " " << j << " s=" << static_cast<int>(s) << " " << p2.candidate()->angle() << " diff=" << diff <<"\n";
                directions = (diff <= Param::maximumDirectionDifference());
            }

            if (!directions) {
                continue;
            }

            const auto p = FastMath::rotateAngle(p1.probe()->angle(), p2.probe()->angle());
            const auto ca = FastMath::rotateAngle(p1.candidate()->angle(), p2.candidate()->angle());
            bool anglesBeta = FastMath::minimumAngle(p, ca) <= Param::maximumAngleDifference();

            if (!anglesBeta) {
                continue;
            }
            matched++;
            if (matched + --min < Param::MinimumMinutiae) {
                break;
            }

            // When this class is specialized for rendering only - no overhead when computing similarities...
            if constexpr (std::is_same_v<R, MinutiaPoint::PairRenderable::Set>) {
                std::cout << "for rendering only\n";
                result.insert(&p2);
            }
        }
        maxMatched = std::max(maxMatched, matched);
    }
    */

    /*
    if constexpr (std::is_same_v<R, uint8_t>) {
        uint8_t cpuScore = 0;

        if (maxMatched > Param::MinimumMinutiae) {
            const auto pCount = probe.minutiaeCount();
            const auto cCount = candidate.minutiaeCount();

            if (pCount > 0 && cCount > 0) {
                const int num = static_cast<int>(maxMatched) * static_cast<int>(maxMatched) * 100;
                const int den = static_cast<int>(pCount * cCount);
                int sc        = den ? (num / den) : 0;
                if (sc > 255) sc = 255;
                cpuScore = static_cast<uint8_t>(sc);
            }
        }
        
        
        // ----- GPU score (global stage only) -----
        auto gpuPairs = buildGpuPairs();  // packs m_pairs into MinutiaPairGPU[]

        ParamGPU param{};
        param.maximumGlobalDistance      = static_cast<float>(Param::MaximumGlobalDistance);
        param.maximumDirectionDifference = static_cast<float>(Param::maximumDirectionDifference());
        param.maximumAngleDifference     = static_cast<float>(Param::maximumAngleDifference());
        param.minimumMinutiae            = Param::MinimumMinutiae;
        param.pCount                     = static_cast<int>(probe.minutiaeCount());
        param.cCount                     = static_cast<int>(candidate.minutiaeCount());

        uint8_t gpuScore = 0;
        if (!gpuPairs.empty()) {
            gpuScore = gpu_global_score(gpuPairs.data(),
                                        static_cast<int>(gpuPairs.size()),
                                        param);
        }

        // Debug: compare CPU vs GPU
        //std::cout << "[DEBUG] CPU score = " << static_cast<int>(cpuScore) << "\n";
        //std::cout << "[DEBUG] GPU score = " << static_cast<int>(gpuScore) << "\n";
        

        // For now, trust CPU as ground truth; GPU is for debugging
        //result = cpuScore;
        result = gpuScore;
    }
    */

}


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Explicit instantiations...
//
template class Match<uint8_t, Fingerprint, MinutiaPoint::Pair>;
template class Match<MinutiaPoint::PairRenderable::Set, FingerprintRenderable, MinutiaPoint::PairRenderable>;
}

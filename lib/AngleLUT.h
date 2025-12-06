#pragma once
#include "Field.h"
#include <limits>

namespace OpenAFIS::AngleLUT {

constexpr Field::AngleType kAngleMin =
    -std::numeric_limits<Field::AngleSize>::max();
constexpr Field::AngleType kAngleMax =
     std::numeric_limits<Field::AngleSize>::max();

constexpr int kAngleTableSize = static_cast<int>(kAngleMax - kAngleMin + 1);

} // namespace OpenAFIS::AngleLUT
#pragma once

#include "rkvs/types.hpp"
#include <string>

namespace rkvs {
std::string json_escape(const std::string& value);
std::string to_json(const ResultBatch& result);
}  // namespace rkvs

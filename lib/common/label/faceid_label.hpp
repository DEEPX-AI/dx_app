#pragma once

#include <map>

namespace dxapp
{
namespace faceid
{
    static int faceid_numClasses = 4;
    static std::map<uint16_t, std::string> faceid_labels = {
        {0	,	"background"},
        {1	,	"person"},
        {2	,	"no_mask"},
        {3	,	"mask"},
    };
    
}
} // namespace dxapp
#pragma once

#include <map>

namespace diagnosis
{
    static std::map<uint16_t, std::string> diagnosis_labels = {
            {0,   "sinus_rhythm"},
            {1,   "atrial_fibrillation"},
            {2,   "atrial_flutter"},
            {3,   "sinus_arrhythmia"},
            {4,   "supraventricular_tachycardia"},
            {5,   "premature_ventricular_contractions"}};
}
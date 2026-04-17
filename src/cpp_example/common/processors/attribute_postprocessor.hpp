/**
 * @file attribute_postprocessor.hpp
 * @brief Person/Face attribute recognition postprocessor (sigmoid multi-label)
 *
 * Supports:
 *   - DeepMAR: [1, 35] logits → sigmoid → threshold (PETA dataset)
 *   - CelebA:  [1, 40, 2] logits → softmax per attr → positive class prob
 */

#ifndef ATTRIBUTE_POSTPROCESSOR_HPP
#define ATTRIBUTE_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace dxapp {

// PETA dataset 35 person attributes (used by DeepMAR models)
inline const std::vector<std::string>& getPETA35Labels() {
    static const std::vector<std::string> labels = {
        "Hat",                    //  0
        "Muffler",                //  1
        "Nothing on head",        //  2
        "Sunglasses",             //  3
        "Long Hair",              //  4
        "Casual Upper",           //  5
        "Formal Upper",           //  6
        "Jacket",                 //  7
        "Logo on Upper",          //  8
        "Plaid on Upper",         //  9
        "Short Sleeve",           // 10
        "Thin Stripes on Upper",  // 11
        "T-shirt",                // 12
        "Other Upper",            // 13
        "V-Neck",                 // 14
        "Casual Lower",           // 15
        "Formal Lower",           // 16
        "Jeans",                  // 17
        "Shorts",                 // 18
        "Long Lower",             // 19
        "Skirt",                  // 20
        "Thin Stripes on Lower",  // 21
        "Female",                 // 22
        "Age 17-30",              // 23
        "Age 31-45",              // 24
        "Age 46-60",              // 25
        "Age Over 60",            // 26
        "Body Fat",               // 27
        "Body Normal",            // 28
        "Body Thin",              // 29
        "Customer",               // 30
        "Employee",               // 31
        "Backpack",               // 32
        "Carrying Other",         // 33
        "Messenger Bag",          // 34
    };
    return labels;
}

// CelebA 40 face attributes
inline const std::vector<std::string>& getCelebA40Labels() {
    static const std::vector<std::string> labels = {
        "5 o'Clock Shadow", "Arched Eyebrows", "Attractive", "Bags Under Eyes",
        "Bald", "Bangs", "Big Lips", "Big Nose", "Black Hair", "Blond Hair",
        "Blurry", "Brown Hair", "Bushy Eyebrows", "Chubby", "Double Chin",
        "Eyeglasses", "Goatee", "Gray Hair", "Heavy Makeup", "High Cheekbones",
        "Male", "Mouth Slightly Open", "Mustache", "Narrow Eyes", "No Beard",
        "Oval Face", "Pale Skin", "Pointy Nose", "Receding Hairline",
        "Rosy Cheeks", "Sideburns", "Smiling", "Straight Hair", "Wavy Hair",
        "Wearing Earrings", "Wearing Hat", "Wearing Lipstick",
        "Wearing Necklace", "Wearing Necktie", "Young",
    };
    return labels;
}

class AttributePostprocessor : public IPostprocessor<ClassificationResult> {
public:
    enum class LabelSet { PETA_35, CELEBA_40 };

    AttributePostprocessor(float threshold = 0.5f,
                           LabelSet label_set = LabelSet::PETA_35)
        : threshold_(threshold), label_set_(label_set) {}

    std::vector<ClassificationResult> process(
        const dxrt::TensorPtrs& outputs,
        const PreprocessContext& ctx) override {
        (void)ctx;
        std::vector<ClassificationResult> results;
        if (outputs.empty()) return results;

        const auto& tensor = outputs.front();
        if (tensor->type() != dxrt::DataType::FLOAT) return results;

        const float* data = static_cast<const float*>(tensor->data());
        auto shape = tensor->shape();

        const auto& labels = (label_set_ == LabelSet::CELEBA_40)
                                 ? getCelebA40Labels()
                                 : getPETA35Labels();

        // Detect CelebA [1, N, 2] vs DeepMAR [1, N]
        bool is_celeba = (shape.size() == 3 && shape.back() == 2);

        int num_attrs = is_celeba ? static_cast<int>(shape[1])
                                  : static_cast<int>(shape.back());

        for (int i = 0; i < num_attrs; ++i) {
            float prob;
            if (is_celeba) {
                float neg = data[i * 2 + 0];
                float pos = data[i * 2 + 1];
                float max_val = std::max(neg, pos);
                float e_neg = std::exp(neg - max_val);
                float e_pos = std::exp(pos - max_val);
                prob = e_pos / (e_neg + e_pos);
            } else {
                prob = 1.0f / (1.0f + std::exp(-data[i]));
            }

            if (prob > threshold_) {
                ClassificationResult cr;
                cr.class_id = i;
                cr.confidence = prob;
                cr.class_name = (i < static_cast<int>(labels.size()))
                                    ? labels[i]
                                    : "attr_" + std::to_string(i);
                results.push_back(cr);
            }
        }

        // Sort by confidence descending
        std::sort(results.begin(), results.end(),
                  [](const ClassificationResult& a, const ClassificationResult& b) {
                      return a.confidence > b.confidence;
                  });

        return results;
    }

    std::string getModelName() const override { return "AttributeRecognition"; }

private:
    float threshold_;
    LabelSet label_set_;
};

}  // namespace dxapp

#endif  // ATTRIBUTE_POSTPROCESSOR_HPP

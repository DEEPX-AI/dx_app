/**
 * @file model_config.hpp
 * @brief Lightweight JSON configuration loader for model parameters
 *
 * Reads a flat JSON config file (config.json) so that Factory parameters
 * (thresholds, class counts, etc.) can be changed at runtime without
 * recompiling.  Only flat key-value pairs are supported — nested objects
 * are silently skipped.
 *
 * Usage:
 *   ModelConfig cfg("config.json");
 *   float score = cfg.get<float>("score_threshold", 0.3f);
 */

#ifndef DXAPP_MODEL_CONFIG_HPP
#define DXAPP_MODEL_CONFIG_HPP

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

namespace dxapp {

class ModelConfig {
public:
    /// Construct from a JSON file path.  Prints a warning if the file
    /// cannot be opened; the caller can check isLoaded().
    explicit ModelConfig(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "[WARN] Config file not found: " << path << std::endl;
            return;
        }
        loaded_ = true;
        std::string content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
        parse(content);
        std::cout << "[INFO] Config loaded: " << path
                  << " (" << values_.size() << " keys)" << std::endl;
    }

    /// Retrieve a typed value.  Returns @p default_value when the key
    /// is absent or the file was not loaded.
    template <typename T>
    T get(const std::string& key, T default_value) const {
        auto it = values_.find(key);
        if (it == values_.end()) return default_value;
        try {
            return convert<T>(it->second);
        } catch (...) {
            std::cerr << "[WARN] Config: failed to convert key '"
                      << key << "' value '" << it->second << "'" << std::endl;
            return default_value;
        }
    }

    bool isLoaded() const { return loaded_; }

    /// Print all loaded key-value pairs (for debugging).
    void dump() const {
        for (const auto& kv : values_) {
            std::cout << "  " << kv.first << " = " << kv.second << std::endl;
        }
    }

private:
    std::map<std::string, std::string> values_;
    bool loaded_ = false;

    // ----------------------------------------------------------------
    // Minimal flat-JSON parser
    // ----------------------------------------------------------------
    void parse(const std::string& content) {
        size_t pos = 0;
        const size_t len = content.size();
        while (pos < len) {
            // ---- find opening quote of key ----
            auto key_start = content.find('"', pos);
            if (key_start == std::string::npos) break;
            auto key_end = content.find('"', key_start + 1);
            if (key_end == std::string::npos) break;
            std::string key = content.substr(key_start + 1, key_end - key_start - 1);

            // ---- find colon ----
            auto colon = content.find(':', key_end + 1);
            if (colon == std::string::npos) break;
            pos = colon + 1;

            // skip whitespace
            while (pos < len && std::isspace(static_cast<unsigned char>(content[pos]))) pos++;
            if (pos >= len) break;

            if (content[pos] == '"') {
                values_[key] = readStringValue_(content, pos);
                if (pos >= len) break;
            } else if (content[pos] == '{') {
                skipBlock_(content, pos, '{', '}');
            } else if (content[pos] == '[') {
                skipBlock_(content, pos, '[', ']');
            } else {
                values_[key] = readScalarValue_(content, pos);
            }
        }
    }

    // Skip a nested block delimited by opener/closer (e.g. '{'/'}' or '['/']').
    // On entry pos points at the opener; on return pos is past the closer.
    void skipBlock_(const std::string& content, size_t& pos,
                    char opener, char closer) const {
        int depth = 1;
        ++pos;
        while (pos < content.size() && depth > 0) {
            if      (content[pos] == opener) ++depth;
            else if (content[pos] == closer) --depth;
            ++pos;
        }
    }

    // Read a JSON string value. pos must point at the opening '"'.
    // On return pos is past the closing '"'.
    std::string readStringValue_(const std::string& content, size_t& pos) const {
        auto val_end = content.find('"', pos + 1);
        if (val_end == std::string::npos) { pos = content.size(); return {}; }
        std::string val = content.substr(pos + 1, val_end - pos - 1);
        pos = val_end + 1;
        return val;
    }

    // Read a JSON scalar (number / boolean / null) token starting at pos.
    // On return pos is past the token.
    std::string readScalarValue_(const std::string& content, size_t& pos) const {
        auto val_start = pos;
        while (pos < content.size() && content[pos] != ',' &&
               content[pos] != '}' && content[pos] != '\n' &&
               content[pos] != '\r') {
            ++pos;
        }
        std::string val = content.substr(val_start, pos - val_start);
        auto last = val.find_last_not_of(" \t\r\n");
        if (last != std::string::npos) val.erase(last + 1);
        return val;
    }

    // ----------------------------------------------------------------
    // Type conversion helpers
    // ----------------------------------------------------------------
    template <typename T>
    T convert(const std::string& val) const;
};

template <> inline float ModelConfig::convert<float>(const std::string& val) const {
    return std::stof(val);
}
template <> inline double ModelConfig::convert<double>(const std::string& val) const {
    return std::stod(val);
}
template <> inline int ModelConfig::convert<int>(const std::string& val) const {
    return std::stoi(val);
}
template <> inline bool ModelConfig::convert<bool>(const std::string& val) const {
    return val == "true" || val == "1";
}
template <> inline std::string ModelConfig::convert<std::string>(const std::string& val) const {
    return val;
}

}  // namespace dxapp

#endif  // DXAPP_MODEL_CONFIG_HPP

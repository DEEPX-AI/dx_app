/**
 * @file model_config.hpp
 * @brief Lightweight JSON configuration loader for model parameters
 *
 * Reads a JSON config file (config.json) so that Factory parameters
 * (thresholds, class counts, etc.) can be changed at runtime without
 * recompiling.  It reads the top-level string, number, boolean/null and
 * array values.  The members of a top-level `config` object overlay the top
 * level (the per-variant config.json layout).  Other nested objects,
 * including "registry_config", are skipped.  Keys, string values and list entries are JSON-decoded (escapes
 * included), and a bracket or brace inside a string is text, not structure.
 *
 * Usage:
 *   ModelConfig cfg("config.json");
 *   float score = cfg.get<float>("score_threshold", 0.3f);
 */

#ifndef DXAPP_MODEL_CONFIG_HPP
#define DXAPP_MODEL_CONFIG_HPP

#include <algorithm>
#include <cerrno>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace dxapp {

/// Says whether a ModelConfig's string argument is a path or the JSON
/// itself. Only the TWO-argument constructor takes one, so the
/// single-argument `ModelConfig("config.json")` spelling used by all
/// existing examples is unchanged and unambiguous.
enum class ConfigSource { kFile, kText };

class ModelConfig {
public:
    /**
     * @brief Construct from a path, or from JSON text already in memory.
     *
     * The kText form was added for the graph engine: a graph node's
     * "params" are an in-memory overlay applied AFTER config.json (factory
     * defaults < config.json < node params), and a factory can only be
     * reconfigured through loadConfig(const ModelConfig&). Without it the
     * overlay would have to be spilled to a temporary file just to be read
     * straight back.
     *
     * Both forms end in the same parse(), so the two cannot interpret the
     * same text differently. isLoaded() is true when the file opened, or
     * when the text was non-empty.
     */
    ModelConfig(const std::string& source_text, ConfigSource source) {
        if (source == ConfigSource::kText) {
            loadText(source_text);
        } else {
            loadFile(source_text);
        }
    }

    /// Construct from a JSON file path.  Prints a warning if the file
    /// cannot be opened; the caller can check isLoaded().
    explicit ModelConfig(const std::string& path) { loadFile(path); }

    /// Retrieve a typed value.  Returns @p default_value when the key
    /// is absent or the file was not loaded.
    template <typename T>
    T get(const std::string& key, T default_value) const {
        auto it = values_.find(key);
        if (it == values_.end()) return default_value;
        try {
            return convert<T>(it->second);
        } catch (const std::exception&) {
            std::cerr << "[DXAPP] [WARN] Config: failed to convert key '"
                      << key << "' value '" << it->second << "'" << std::endl;
            return default_value;
        }
    }

    /// Retrieve a list of strings from a JSON array value.
    /// Returns empty vector when the key is absent.
    std::vector<std::string> get_string_list(const std::string& key) const {
        auto it = arrays_.find(key);
        if (it == arrays_.end()) return {};
        return parse_string_array_(it->second);
    }

    /// Every array value as its raw JSON text (brackets included), by key.
    /// The graph overlay copies these forward so a node's params cannot
    /// wipe a config.json list they do not name.
    const std::map<std::string, std::string>& rawArrays() const { return arrays_; }

    bool isLoaded() const { return loaded_; }

    /// Print all loaded key-value pairs (for debugging).
    void dump() const {
        for (const auto& kv : values_) {
            std::cout << "  " << kv.first << " = " << kv.second << std::endl;
        }
    }

private:
    void loadFile(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "[DXAPP] [WARN] Config file not found: " << path << std::endl;
            return;
        }
        loaded_ = true;
        std::string content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
        parse(content);
        std::cout << "[DXAPP] [INFO] Config loaded: " << path
                  << " (" << values_.size() << " keys)" << std::endl;
    }

    void loadText(const std::string& json_text) {
        if (json_text.empty()) return;
        loaded_ = true;
        parse(json_text);
    }

    std::map<std::string, std::string> values_;
    std::map<std::string, std::string> arrays_;  // raw JSON array content
    bool loaded_ = false;

    // ----------------------------------------------------------------
    // Top-level JSON reader: string, number, boolean/null and array
    // values by key. Nested objects are skipped, except a top-level
    // "config" object, whose members overlay the top level (the
    // per-variant config.json layout): they are read after the top level,
    // by the same routine, so they win whatever the key order, and an
    // object inside "config" is skipped like any other. Keys, string
    // values and the string elements of a list are JSON-decoded; a
    // bracket or brace inside a string is text (U-77).
    // ----------------------------------------------------------------
    void parse(const std::string& content) {
        std::string nested_config;
        parseMembers_(content, &nested_config);
        if (!nested_config.empty()) parseMembers_(nested_config, nullptr);
    }

    // Read every key of the object text in `content` into values_ and
    // arrays_. A later key replaces an earlier one, in whichever map held
    // it. With `nested_config` set, the object value of a key "config" is
    // returned there instead of being skipped.
    void parseMembers_(const std::string& content, std::string* nested_config) {
        size_t pos = 0;
        const size_t len = content.size();
        while (pos < len) {
            // ---- the next key: a JSON string ----
            pos = content.find('"', pos);
            if (pos == std::string::npos) break;
            const std::string key = readJsonString_(content, pos);

            // ---- find colon ----
            const size_t colon = content.find(':', pos);
            if (colon == std::string::npos) break;
            pos = colon + 1;

            // skip whitespace
            while (pos < len && std::isspace(static_cast<unsigned char>(content[pos]))) pos++;
            if (pos >= len) break;

            // The last "config" wins, as for any key: a later non-object
            // "config" cancels an earlier object's overlay.
            if (nested_config != nullptr && key == "config") nested_config->clear();
            if (content[pos] == '"') {
                arrays_.erase(key);
                values_[key] = readJsonString_(content, pos);
            } else if (content[pos] == '{') {
                std::string block = readBlock_(content, pos, '{', '}');
                if (nested_config != nullptr && key == "config") {
                    *nested_config = std::move(block);
                }  // any other nested object is not read
            } else if (content[pos] == '[') {
                values_.erase(key);
                arrays_[key] = readBlock_(content, pos, '[', ']');
            } else {
                arrays_.erase(key);
                values_[key] = readScalarValue_(content, pos);
            }
        }
    }

    // Advance pos past the JSON string that starts at content[pos] == '"'.
    static void skipJsonString_(const std::string& content, size_t& pos) {
        ++pos;
        while (pos < content.size()) {
            const char c = content[pos++];
            if (c == '\\') {
                if (pos < content.size()) ++pos;
            } else if (c == '"') {
                return;
            }
        }
    }

    // A nested block delimited by opener/closer, returned with its
    // delimiters. Strings are skipped whole, so a bracket or brace inside
    // a string value is text, not structure (U-77). On entry pos points at
    // the opener; on return it is past the closer.
    static std::string readBlock_(const std::string& content, size_t& pos,
                                  char opener, char closer) {
        const size_t start = pos;
        int depth = 1;
        ++pos;
        while (pos < content.size() && depth > 0) {
            const char c = content[pos];
            if (c == '"') {
                skipJsonString_(content, pos);
                continue;
            }
            if (c == opener) ++depth;
            else if (c == closer) --depth;
            ++pos;
        }
        return content.substr(start, pos - start);
    }

    // Every JSON string anywhere in the array's raw text, decoded, in order:
    // strings inside nested arrays and objects (keys too) are included, and
    // non-string elements are skipped. Meant for flat lists of strings
    // (the lists read here are class names).
    std::vector<std::string> parse_string_array_(const std::string& raw) const {
        std::vector<std::string> result;
        size_t pos = 0;
        while (true) {
            pos = raw.find('"', pos);
            if (pos == std::string::npos) break;
            result.push_back(readJsonString_(raw, pos));
        }
        return result;
    }

    static bool readHex4_(const std::string& s, size_t& pos, unsigned* value) {
        if (pos + 4 > s.size()) return false;
        unsigned v = 0;
        for (size_t i = 0; i < 4; ++i) {
            const char c = s[pos + i];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<unsigned>(c - 'A' + 10);
            else return false;
        }
        pos += 4;
        *value = v;
        return true;
    }

    static void appendUtf8_(unsigned code, std::string* out) {
        if (code < 0x80) {
            *out += static_cast<char>(code);
        } else if (code < 0x800) {
            *out += static_cast<char>(0xC0 | (code >> 6));
            *out += static_cast<char>(0x80 | (code & 0x3F));
        } else if (code < 0x10000) {
            *out += static_cast<char>(0xE0 | (code >> 12));
            *out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
            *out += static_cast<char>(0x80 | (code & 0x3F));
        } else {
            *out += static_cast<char>(0xF0 | (code >> 18));
            *out += static_cast<char>(0x80 | ((code >> 12) & 0x3F));
            *out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
            *out += static_cast<char>(0x80 | (code & 0x3F));
        }
    }

    // The JSON string at content[pos] == '"', decoded: \" \\ \/ \b \f \n \r
    // \t and \uXXXX (a surrogate pair is one code point, a lone surrogate
    // U+FFFD; UTF-8 out). An
    // unknown escape keeps its backslash and character, as the old reader
    // kept everything. On return pos is past the closing quote, or at the
    // end of an unterminated string.
    static std::string readJsonString_(const std::string& content, size_t& pos) {
        std::string out;
        ++pos;
        while (pos < content.size()) {
            const char c = content[pos++];
            if (c == '"') return out;
            if (c != '\\' || pos >= content.size()) {
                out += c;
                continue;
            }
            const char e = content[pos++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned code = 0;
                    if (!readHex4_(content, pos, &code)) {
                        out += "\\u";
                        break;
                    }
                    if (code >= 0xD800 && code <= 0xDBFF && pos + 1 < content.size() &&
                        content[pos] == '\\' && content[pos + 1] == 'u') {
                        size_t low_pos = pos + 2;
                        unsigned low = 0;
                        if (readHex4_(content, low_pos, &low) && low >= 0xDC00 && low <= 0xDFFF) {
                            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                            pos = low_pos;
                        }
                    }
                    // A lone surrogate is no code point: U+FFFD, as JSON
                    // decoders do (its 3-byte form is not valid UTF-8).
                    if (code >= 0xD800 && code <= 0xDFFF) code = 0xFFFD;
                    appendUtf8_(code, &out);
                    break;
                }
                default:
                    out += '\\';
                    out += e;
                    break;
            }
        }
        return out;
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

/// The config at `path` when that file opens; an empty, not-loaded config,
/// without the "Config file not found" warning, when there is no such file;
/// and ModelConfig(path)'s warning when the file exists but cannot be read
/// (a permission error must not pass for an absent file). For a caller to
/// whom a config.json is optional (a graph stage: the factory's defaults
/// apply); a path a user passed goes through ModelConfig(path), which warns.
inline ModelConfig LoadOptionalConfig(const std::string& path) {
    errno = 0;
    std::ifstream probe(path.c_str());
    if (!probe.is_open()) {
        if (errno == ENOENT || errno == ENOTDIR) {
            return ModelConfig(std::string(), ConfigSource::kText);
        }
        return ModelConfig(path);  // present but unreadable: ModelConfig warns
    }
    probe.close();
    return ModelConfig(path);
}

}  // namespace dxapp

#endif  // DXAPP_MODEL_CONFIG_HPP

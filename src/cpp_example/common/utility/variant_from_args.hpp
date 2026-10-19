/**
 * @file variant_from_args.hpp
 * @brief Derive the variant key from the model path already on the command line.
 *
 * Under the dx-modelzoo family/variant layout one example serves every variant of its
 * family, so something has to say WHICH variant to build for. A variant key IS the
 * ``.dxnn`` stem, so the ``-m`` / ``--model_path`` argument already carries it -- no new
 * flag, and none of the 26 shared runner headers need to change. main() calls this
 * before constructing the factory, because it is the only place that sees argv.
 *
 * Returns an empty string when no model path is present, which the family factory reads
 * as "use the family default".
 */

#ifndef VARIANT_FROM_ARGS_HPP
#define VARIANT_FROM_ARGS_HPP

#include <string>

namespace dxapp {

inline std::string variantFromModelPath(const std::string& path) {
    if (path.empty()) {
        return std::string();
    }
    const auto slash = path.find_last_of("/\\");
    std::string stem = (slash == std::string::npos) ? path : path.substr(slash + 1);
    const std::string ext = ".dxnn";
    if (stem.size() > ext.size() && stem.compare(stem.size() - ext.size(), ext.size(), ext) == 0) {
        stem.erase(stem.size() - ext.size());
    }
    return stem;
}

inline std::string variantFromArgs(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        // Long form with an inline value.
        if (arg.rfind("--model_path=", 0) == 0) {
            return variantFromModelPath(arg.substr(std::string("--model_path=").size()));
        }
        if (arg.rfind("--model=", 0) == 0) {
            return variantFromModelPath(arg.substr(std::string("--model=").size()));
        }
        // Separate value forms.
        if ((arg == "-m" || arg == "--model_path" || arg == "--model") && i + 1 < argc) {
            return variantFromModelPath(argv[i + 1]);
        }
    }
    return std::string();
}

}  // namespace dxapp

#endif  // VARIANT_FROM_ARGS_HPP

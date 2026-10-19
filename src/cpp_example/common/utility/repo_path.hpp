/**
 * @file repo_path.hpp
 * @brief One rule for a path that is relative to the repository, and where
 *        that repository is at run time.
 *
 * Sample inputs, galleries and the images a gallery names ship as
 * "sample/..." paths, and a graph stage reads its config.json from
 * "src/cpp_example/<task>/<family>/<variant>/". All are relative to one
 * root, not to the caller's working directory, so the graph CLI's --check,
 * a graph stage's config overlay and the retrieval visualizer each find the
 * file the others find, from any directory.
 *
 * The root is decided at run time (RuntimeRoot), so a program built outside
 * this repository - an exported graph project
 * (scripts/export_graph_project.py), whose CMake defines no
 * PROJECT_ROOT_DIR - finds its own files wherever it is moved.
 *
 * An absolute path is kept as given; an empty one stays empty.
 */
#ifndef DXAPP_REPO_PATH_HPP
#define DXAPP_REPO_PATH_HPP

#include <cstdlib>
#include <fstream>
#include <string>

#if defined(_WIN32)
#include <direct.h>
#else
#include <unistd.h>
#endif

namespace dxapp {

/// "/a", "\\a" and "C:..." are absolute; "" and "sample/x" are not.
inline bool IsAbsolutePath(const std::string& path) {
    if (path.empty()) return false;
    if (path[0] == '/' || path[0] == '\\') return true;
    return path.size() > 1 && path[1] == ':';
}

/// `path` read against `root`: a relative path is joined, an absolute or
/// empty one is returned as given.
inline std::string ResolveAgainstRoot(const std::string& root, const std::string& path) {
    if (path.empty() || IsAbsolutePath(path)) return path;
    return root + "/" + path;
}

/// The file at the root of an exported graph project, next to its graph.json.
inline const char* RootMarkerName() { return ".dxapp-root"; }

/// `dir`, then each of its parents up to "/": the first that holds
/// RootMarkerName(), or "" when none does.
inline std::string FindMarkedRoot(std::string dir) {
    while (dir.size() > 1 && (dir[dir.size() - 1] == '/' || dir[dir.size() - 1] == '\\')) {
        dir.erase(dir.size() - 1);
    }
    while (!dir.empty()) {
        std::ifstream probe(((dir == "/" ? std::string() : dir) + "/" + RootMarkerName()).c_str());
        if (probe.good()) return dir;
        const std::string::size_type slash = dir.find_last_of("/\\");
        if (slash == std::string::npos || dir == "/") return std::string();
        dir = slash == 0 ? std::string("/") : dir.substr(0, slash);
    }
    return std::string();
}

/// The running program's directory; "" where it cannot be told (not Linux).
inline std::string ExecutableDir() {
#if defined(__linux__)
    char buffer[4096];
    const ssize_t length = ::readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (length <= 0) return std::string();
    const std::string path(buffer, static_cast<std::size_t>(length));
    const std::string::size_type slash = path.find_last_of('/');
    if (slash == std::string::npos) return std::string();
    return slash == 0 ? std::string("/") : path.substr(0, slash);
#else
    return std::string();
#endif
}

/// The working directory; "." when it cannot be read.
inline std::string WorkingDir() {
    char buffer[4096];
#if defined(_WIN32)
    const char* got = ::_getcwd(buffer, sizeof(buffer));
#else
    const char* got = ::getcwd(buffer, sizeof(buffer));
#endif
    return got != NULL ? std::string(got) : std::string(".");
}

/**
 * @brief The root repository-relative paths are read against.
 *
 *   1. $DXAPP_ROOT, when it is set and not empty (an exported project's
 *      run.sh sets it);
 *   2. the repository this build was configured from, PROJECT_ROOT_DIR,
 *      when the build defines it (dx_app's own CMake does);
 *   3. the nearest folder holding .dxapp-root, from the program's directory
 *      up, then from the working directory up (decided once per process);
 *   4. the working directory.
 */
inline std::string RuntimeRoot() {
    const char* given = std::getenv("DXAPP_ROOT");
    if (given != NULL && given[0] != '\0') {
        std::string root(given);  // "/p/" and "/p" name one root
        while (root.size() > 1 && (root[root.size() - 1] == '/' || root[root.size() - 1] == '\\')) {
            root.erase(root.size() - 1);
        }
        return root;
    }
#if defined(PROJECT_ROOT_DIR)
    return std::string(PROJECT_ROOT_DIR);
#else
    static const std::string found = [] {
        std::string root = FindMarkedRoot(ExecutableDir());
        if (root.empty()) root = FindMarkedRoot(WorkingDir());
        return root.empty() ? WorkingDir() : root;
    }();
    return found;
#endif
}

/// `path` read against RuntimeRoot().
inline std::string ResolveRepoRelative(const std::string& path) {
    return ResolveAgainstRoot(RuntimeRoot(), path);
}

}  // namespace dxapp

#endif  // DXAPP_REPO_PATH_HPP

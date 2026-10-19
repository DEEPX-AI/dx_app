#include "common/graph/user_logic.hpp"

#include <map>
#include <mutex>

namespace dxapp {
namespace graph {

namespace {

std::mutex& RegistryMutex() {
    static std::mutex mutex;
    return mutex;
}

std::map<std::string, UserLogicMaker>& Registry() {
    static std::map<std::string, UserLogicMaker> registry;
    return registry;
}

}  // namespace

void UserContext::Raise(const std::string& event, int track_id, const RecordItem& data) {
    GraphEvent raised;
    raised.node = node_;
    raised.event = event;
    raised.track_id = track_id;
    raised.time_s = time_s_;
    raised.data = data;
    events_.push_back(raised);
}

void UserContext::Label(const std::string& text, bool alert) {
    OverlayText label;  // at (-1, -1): the top-left panel
    label.text = text;
    label.alert = alert;
    labels_.push_back(label);
}

void UserContext::LabelAt(const std::string& text, const cv::Point2f& at, bool alert) {
    OverlayText label;
    label.at = at;
    label.text = text;
    label.alert = alert;
    labels_.push_back(label);
}

void RegisterUserLogic(const std::string& name, const UserLogicMaker& make) {
    std::lock_guard<std::mutex> lock(RegistryMutex());
    Registry()[name] = make;
}

std::unique_ptr<UserLogic> MakeUserLogic(const std::string& name) {
    UserLogicMaker make;
    {
        std::lock_guard<std::mutex> lock(RegistryMutex());
        std::map<std::string, UserLogicMaker>::const_iterator found = Registry().find(name);
        if (found == Registry().end()) return std::unique_ptr<UserLogic>();
        make = found->second;
    }
    return make();
}

std::vector<std::string> RegisteredUserLogicNames() {
    std::lock_guard<std::mutex> lock(RegistryMutex());
    std::vector<std::string> names;
    for (std::map<std::string, UserLogicMaker>::const_iterator it = Registry().begin();
         it != Registry().end(); ++it) {
        names.push_back(it->first);
    }
    return names;
}

void ClearUserLogic() {
    std::lock_guard<std::mutex> lock(RegistryMutex());
    Registry().clear();
}

}  // namespace graph
}  // namespace dxapp

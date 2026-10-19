/**
 * @file user_logic.hpp
 * @brief The code behind a "user" logic node, supplied by the program that
 *        runs the graph.
 *
 * A node {"type": "logic", "op": "user", "params": {"name": N, "input": K,
 * "output": K}} passes its input through unless the program registered a
 * class under N before the graph runs. The multi_model_graph CLI and dx_graph
 * register none, so Studio's runs and the sample graphs pass through. An
 * exported graph project (scripts/export_graph_project.py) registers one class
 * per name from its user_logic/ folder (user_logic/register.cpp).
 *
 * One object per (node, stream): made on the node's first frame in that
 * stream, made again when the stream restarts (--loop), because it lives in
 * the node's LogicState. Its members are the node's memory. Run is called
 * once per frame, in frame order, never from two threads at once for one
 * object: both executors run a stream's logic nodes in frame order, the async
 * one on its driver thread.
 */
#ifndef DXAPP_GRAPH_USER_LOGIC_HPP
#define DXAPP_GRAPH_USER_LOGIC_HPP

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "common/graph/logic_ops.hpp"
#include "common/graph/shape.hpp"

namespace dxapp {
namespace graph {

/// What a user node's code sees of its frame, and how it raises events and
/// draws text.
class UserContext {
 public:
    UserContext(const std::string& node, double time_s, const cv::Size& frame_size)
        : node_(node), time_s_(time_s), frame_size_(frame_size) {}

    /// The node's id in the graph.
    const std::string& node() const { return node_; }
    /// Seconds since the stream started: frame / fps for a file, wall
    /// seconds for a camera (FrameReport::time_s).
    double time_s() const { return time_s_; }
    /// The source frame's size in pixels.
    const cv::Size& frame_size() const { return frame_size_; }

    /// Raises an event: an [EVENT] line on stdout, a line in --events, an
    /// entry in --report. `event`: 1 to 64 letters, digits, _ . : - (another
    /// name is a frame error and the event is dropped). `track_id`: -1 when
    /// it is about no one track. `data`: numbers and text, written flat.
    void Raise(const std::string& event, int track_id = -1, const RecordItem& data = RecordItem());
    /// A line of text in the result's top-left panel.
    void Label(const std::string& text, bool alert = false);
    /// A label next to `at`, in source-frame pixels.
    void LabelAt(const std::string& text, const cv::Point2f& at, bool alert = false);

    const std::vector<GraphEvent>& events() const { return events_; }
    const std::vector<OverlayText>& labels() const { return labels_; }

 private:
    std::string node_;
    double time_s_;
    cv::Size frame_size_;
    std::vector<GraphEvent> events_;
    std::vector<OverlayText> labels_;
};

/// The engine's view of one user node's object.
class UserLogic {
 public:
    virtual ~UserLogic() {}
    /// `input`: what the node receives (boxes in source-frame pixels; other
    /// kinds as the node before it produced them). Returns the node's output,
    /// of the same kind. May throw: the frame then records the error and the
    /// input passes through.
    virtual StageDataPtr Run(const StageDataPtr& input, UserContext& ctx) = 0;
};

/// A class with `Data Run(const Data& input, UserContext& ctx)` as a UserLogic.
template <class Data, class Logic>
class TypedUserLogic : public UserLogic {
 public:
    StageDataPtr Run(const StageDataPtr& input, UserContext& ctx) override {
        const Data* in = dynamic_cast<const Data*>(input.get());
        if (in == NULL) {
            throw std::runtime_error(std::string("it was given ") + ToString(input->shape()) +
                                     ", a kind its Run does not take");
        }
        return std::make_shared<Data>(logic_.Run(*in, ctx));
    }

 private:
    Logic logic_;
};

/// Makes one object for one (node, stream).
typedef std::function<std::unique_ptr<UserLogic>()> UserLogicMaker;

/// Registers `make` under a "user" node's "name". A second registration of a
/// name replaces the first.
void RegisterUserLogic(const std::string& name, const UserLogicMaker& make);

/// RegisterUserLogic for a class `Logic` whose Run takes and returns `Data`
/// (BoxesData, ScoresData, RecordsData, ...).
template <class Data, class Logic>
void RegisterUserLogic(const std::string& name) {
    RegisterUserLogic(name, []() {
        return std::unique_ptr<UserLogic>(new TypedUserLogic<Data, Logic>());
    });
}

/// A new object for `name`; null when nothing is registered under it (the
/// node then passes its input through).
std::unique_ptr<UserLogic> MakeUserLogic(const std::string& name);

/// The registered names, sorted.
std::vector<std::string> RegisteredUserLogicNames();

/// Forgets every registration (tests).
void ClearUserLogic();

}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_GRAPH_USER_LOGIC_HPP

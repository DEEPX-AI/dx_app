"""Run the common/ plain-assert test binary (common_unit_test).

Same pattern as test_graph_engine.py: the repository has no gtest, so C++
behaviour is exercised by pytest driving built binaries. The build tree is
preferred over bin/ so a fresh `ninja -C build_x86_64` is what gets tested
without an install step.
"""
import os
import re
import subprocess

import pytest

from conftest import PROJECT_ROOT, resolve_bin_dir

BINARY = "common_unit_test"


def _find_binary():
    name = BINARY + (".exe" if os.name == "nt" else "")
    candidates = [
        PROJECT_ROOT / "build_x86_64" / "src" / "cpp_example" / name,
        resolve_bin_dir() / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def test_common_unit_tests_pass():
    binary = _find_binary()
    if binary is None:
        pytest.skip("{} not built (configure with -DDXAPP_BUILD_COMMON_TESTS=ON)".format(BINARY))

    completed = subprocess.run([str(binary)], capture_output=True, text=True, timeout=300)
    print(completed.stdout)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert ", 0 failures," in completed.stdout


# The 14 async runner IMPLEMENTATIONS (keypoint/object_pose alias the pose
# runner, panoptic aliases the detection runner). Each shares one
# postprocessor across every completion callback, so each must build it
# through Serialize() - the decorator the binary above tests.
ASYNC_RUNNERS = [
    "3d_object_detection", "anomaly", "classification", "depth", "detection", "embedding",
    "face_alignment", "face", "hand_landmark", "obb", "pose", "restoration",
    "segmentation", "semantic_seg",
]


@pytest.mark.parametrize("runner", ASYNC_RUNNERS)
def test_async_runner_serializes_its_shared_postprocessor(runner):
    path = (PROJECT_ROOT / "src" / "cpp_example" / "common" / "runner"
            / "async_{}_runner.hpp".format(runner))
    text = path.read_text()
    assert '#include "common/processors/serialized_postprocessor.hpp"' in text
    problem = _serialize_problem(text)
    assert problem is None, problem


# Loops on running_ that may ignore g_interrupted() because something else
# bounds them. Anything else that waits on running_ must also check
# g_interrupted(), or one SIGINT/SIGTERM cannot stop it: the image-mode
# "keep the window open" loop was exactly such a loop, and with a window
# nobody can close (QT_QPA_PLATFORM=offscreen) the runner ignored SIGTERM.
BOUNDED_RUNNING_LOOPS = {
    # the display thread; ends once the main thread clears running_
    "running_ || !display_queue_.empty()",
    # ends when the last submitted job completes
    "!inference_done.load(std::memory_order_acquire) && running_",
    # ends when the display thread has drained its queue
    "!display_queue_.empty() && running_",
    # ends at a deadline
    "running_ && std::chrono::steady_clock::now() < deadline",
}


def test_loop_scanner_reads_multi_line_and_do_while_conditions_and_skips_comments():
    """U-40: the old scan read one line at a time and needed it to end in
    '{', so it missed a condition over two lines and a do/while, and it read
    loop heads inside comments."""
    text = ("while (running_ &&\n"
            "       !queue_.empty()) {\n"
            "}\n"
            "do {\n"
            "} while (running_);\n"
            "// while (running_) {\n"
            "/* for (;running_;) {\n"
            "   } */\n"
            "for (int i = 0; running_ && i < n; ++i) {\n"
            "}\n")
    assert list(_running_loop_conditions(text)) == [
        ("running_ && !queue_.empty()", "line 1: running_ && !queue_.empty()"),
        ("running_", "line 5: running_"),
        ("running_ && i < n", "line 9: running_ && i < n"),
    ]


def test_serialize_check_reads_code_not_comments():
    """U-49: the old check counted the text "Serialize(", so a comment
    naming it failed the check and a decoder used past Serialize() passed."""
    good = ("auto uptr = factory_->createPostprocessor(w, h);\n"
            "auto pp = Serialize(std::move(uptr));  // see Serialize( and createPostprocessor(\n")
    assert _serialize_problem(good) is None
    assert _serialize_problem("x = Serialize(Ops::createPostprocessor(*f, w, h));\n") is None
    assert "used 3 times" in _serialize_problem(good + "uptr->process(outputs, ctx);\n")
    other = ("auto uptr = factory_->createPostprocessor(w, h);\n"
             "auto pp = Serialize(std::move(spare));\n")
    assert "not the createPostprocessor() result" in _serialize_problem(other)
    assert "expected one Serialize( call" in _serialize_problem(
        "auto uptr = factory_->createPostprocessor(w, h);\n")


def test_comment_stripper_reads_comments_and_literals_in_one_pass():
    """Task 1 review: block comments were stripped before line comments, so a
    '/*' inside a // comment opened a block that swallowed real code up to a
    later '*/', and a "//" inside a string literal cut the rest of its line."""
    assert list(_running_loop_conditions(
        "// glob src/*/x\nwhile (running_) {\n}\n// end */\n")) == [
        ("running_", "line 2: running_")]
    assert list(_running_loop_conditions(
        'log("http://x"); while (running_) {\n}\n')) == [
        ("running_", "line 1: running_")]
    assert list(_running_loop_conditions(
        'const char* g = "src/*.cpp";\nwhile (running_) {\n}\nx = "*/";\n')) == [
        ("running_", "line 2: running_")]
    assert list(_running_loop_conditions(
        "char c = '\"'; // \"\nwhile (running_) {\n}\n")) == [
        ("running_", "line 2: running_")]
    decoder = ("// a/*b\n"
               "auto uptr = factory_->createPostprocessor(w, h);\n"
               "auto pp = Serialize(std::move(uptr));\n")
    assert _serialize_problem(decoder + "// */\n") is None
    assert "used 3 times" in _serialize_problem(
        decoder + "uptr->process(outputs, ctx);\n// */\n")


# One left-to-right pass: whichever of a string literal, a char literal, a
# // comment or a /* */ comment starts first wins, so a "/*" inside a //
# comment or a "//" inside a string is just text. (No raw strings in the
# runner headers.)
_LITERAL_OR_COMMENT = re.compile(
    r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|//[^\n]*|/\*.*?\*/', re.S)


def _strip_comments(text):
    """`text` without // and /* */ comments; string and char literals are
    kept as they are. A comment leaves only its newlines, so a line number in
    the result is the line number in the file."""
    def keep_literal(match):
        token = match.group(0)
        return "\n" * token.count("\n") if token.startswith("/") else token
    return _LITERAL_OR_COMMENT.sub(keep_literal, text)


def _closing_paren(code, open_index, opening="(", closing=")"):
    """The index just past the `closing` that matches the `opening` at
    code[open_index] ('(' and ')' unless told otherwise)."""
    depth = 0
    for index in range(open_index, len(code)):
        if code[index] == opening:
            depth += 1
        elif code[index] == closing:
            depth -= 1
            if depth == 0:
                return index + 1
    raise AssertionError("unbalanced {!r} at offset {}".format(opening, open_index))


_LOOP_KEYWORD = re.compile(r"\b(while|for)\s*\(")


def _running_loop_conditions(text):
    """(condition, where) for every while / for / do-while condition that
    names running_, whitespace-collapsed however many lines it spans. A
    for-loop's condition is its middle clause."""
    code = _strip_comments(text)
    for match in _LOOP_KEYWORD.finditer(code):
        open_index = match.end() - 1
        head = code[open_index + 1:_closing_paren(code, open_index) - 1]
        if match.group(1) == "for":
            parts = head.split(";")
            if len(parts) == 3:
                head = parts[1]
        condition = " ".join(head.split())
        if re.search(r"\brunning_\b", condition):
            line = code.count("\n", 0, match.start()) + 1
            yield condition, "line {}: {}".format(line, condition)


def _serialize_problem(text):
    """None when the one decoder a runner builds reaches its callbacks only
    through Serialize(); else what is wrong. Reads comment-stripped code, so
    a comment naming either call changes nothing (U-49)."""
    code = _strip_comments(text)
    if len(re.findall(r"\bcreatePostprocessor\s*\(", code)) != 1:
        return "expected one createPostprocessor( call"
    calls = [m.end() - 1 for m in re.finditer(r"\bSerialize\s*\(", code)]
    if len(calls) != 1:
        return "expected one Serialize( call, found {}".format(len(calls))
    argument = " ".join(code[calls[0] + 1:_closing_paren(code, calls[0]) - 1].split())
    moved = re.fullmatch(r"std::move\(\s*(\w+)\s*\)", argument)
    if moved is None:
        if re.search(r"\bcreatePostprocessor\s*\(", argument):
            return None
        return "Serialize() wraps {!r}, not the decoder".format(argument)
    name = re.escape(moved.group(1))
    if not re.search(r"\b{}\s*=[^;]*\bcreatePostprocessor\s*\(".format(name), code):
        return "{} is not the createPostprocessor() result".format(moved.group(1))
    uses = len(re.findall(r"\b{}\b".format(name), code))
    if uses != 2:
        return "{} is used {} times; only its declaration and the move may name it".format(
            moved.group(1), uses)
    return None


@pytest.mark.parametrize("runner", ASYNC_RUNNERS)
def test_async_runner_waits_on_running_also_honour_the_interrupt_flag(runner):
    path = (PROJECT_ROOT / "src" / "cpp_example" / "common" / "runner"
            / "async_{}_runner.hpp".format(runner))
    unguarded = [line for condition, line in _running_loop_conditions(path.read_text())
                 if "g_interrupted()" not in condition
                 and condition not in BOUNDED_RUNNING_LOOPS]
    assert not unguarded, (
        "loops that wait on running_ but ignore g_interrupted() "
        "(SIGINT/SIGTERM cannot end them): {}".format(unguarded))


def _runner_text(runner):
    return (PROJECT_ROOT / "src" / "cpp_example" / "common" / "runner"
            / "async_{}_runner.hpp".format(runner)).read_text()


_REORDER_PUSH = re.compile(r"\breorder\.push\(std::move\(\w+\), (\w+)\);")


def _reorder_emit_body(code):
    """The body of the lambda the display thread hands to reorder.push(): the
    code that runs once FrameReorderBuffer releases a frame in submit order."""
    push = _REORDER_PUSH.search(code)
    assert push, "no reorder.push(std::move(args), emit) call"
    decl = re.search(r"\bauto {}\s*=\s*\[[^\]]*\]\s*\([^)]*\)\s*\{{".format(
        re.escape(push.group(1))), code)
    assert decl, "no lambda named {}".format(push.group(1))
    open_index = decl.end() - 1
    return code[open_index:_closing_paren(code, open_index, "{", "}")]


def test_reorder_emit_body_is_the_lambda_passed_to_push():
    code = ("auto other = [&](A& a) { dump(a); };\n"
            "auto renderArgs = [&](A& args) { if (x) { draw(args); } };\n"
            "while (running_) { reorder.push(std::move(args), renderArgs); }\n")
    assert _reorder_emit_body(code) == "{ if (x) { draw(args); } }"


@pytest.mark.parametrize("runner", ASYNC_RUNNERS)
def test_async_runner_delivers_through_the_reorder_buffer(runner):
    """dxrt completes jobs out of order; TARGET's FrameReorderBuffer
    (frame_reorder.hpp) puts them back in submit order before a frame is
    rendered, written or dumped."""
    code = _strip_comments(_runner_text(runner))
    assert re.search(r"\bFrameReorderBuffer<\w+> reorder\(", code), "the display thread owns a FrameReorderBuffer"
    push = _REORDER_PUSH.search(code)
    assert push, "every popped item goes through reorder.push()"
    assert "reorder.drain({})".format(push.group(1)) in code, (
        "what the buffer still holds at shutdown is emitted, in order, by the same "
        "lambda reorder.push() emits through")
    assert code.count("RunAsync(") <= len(re.findall(r"frame_index = static_cast<uint64_t>\(", code)), (
        "every submitted frame carries its submit index")


@pytest.mark.parametrize("runner", ASYNC_RUNNERS)
def test_async_runner_dumps_verify_where_the_reorder_buffer_releases_the_frame(runner):
    code = _strip_comments(_runner_text(runner))
    assert code.count("verify::dumpVerifyJson(") == 1, "one dump per runner, for every delivered frame"
    assert "verify::dumpVerifyJson(" in _reorder_emit_body(code), (
        "the dump runs where FrameReorderBuffer releases the frame (input order, "
        "one thread), not in the dxrt callback")


@pytest.mark.parametrize("runner", ASYNC_RUNNERS)
def test_async_runner_waits_for_every_frame_before_the_display_thread_stops(runner):
    """Callbacks complete out of order and ie.Wait(last_job_id) waits for the
    last job only: the end-of-run wait must also hold until the display
    thread has taken every submitted frame, or an earlier frame still in its
    callback is never rendered or dumped (seen as 59 of 60 frames in the
    parity sweep)."""
    code = " ".join(_strip_comments(_runner_text(runner)).split())
    pop = re.search(r"display_queue_\.try_pop\(args, [^;]*\) (?:continue;|\{ continue; \})"
                    r" \{ std::lock_guard<std::mutex> lock\(metrics_\.metrics_mutex\);"
                    r" metrics_\.display_received\+\+; \}", code)
    assert pop, "the display thread counts every item it takes off display_queue_"
    assert "received = metrics_.display_received;" in code
    assert re.search(r"const bool pending = !display_queue_\.empty\(\) \|\| received < processCount \|\|",
                     code), "the end-of-run wait holds until every submitted frame was received"


@pytest.mark.parametrize("runner", ASYNC_RUNNERS)
def test_async_runner_tail_bail_warning_names_what_may_be_lost(runner):
    """The end-of-run wait also covers frames still in a callback without
    --save, so its 5 s warning names the saved video only when saving, and
    otherwise what a non-saving run can lose: the last frames' display and
    DXAPP_VERIFY records."""
    code = " ".join(_strip_comments(_runner_text(runner)).split())
    warn = re.search(r'std::cerr << "\[DXAPP\] \[WARN\] Output frames stopped draining; " '
                     r'<< \(args\.saveMode \? "saved video may be truncated\." '
                     r': "the last frames may be missing from the display and " '
                     r'"DXAPP_VERIFY records\."\) << std::endl;', code)
    assert warn, "the warning is chosen by args.saveMode"
    assert code.count("Output frames stopped draining") == 1


def test_common_unit_tests_leave_no_temp_directories(tmp_path):
    binary = _find_binary()
    if binary is None:
        pytest.skip("{} not built (configure with -DDXAPP_BUILD_COMMON_TESTS=ON)".format(BINARY))
    env = dict(os.environ, TMPDIR=str(tmp_path))  # fs::temp_directory_path() reads TMPDIR
    completed = subprocess.run([str(binary)], capture_output=True, text=True, timeout=300, env=env)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == []

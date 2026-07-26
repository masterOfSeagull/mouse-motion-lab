#include <mousemotionlab/mouse_io.h>

#include <cassert>

int main() {
    static_assert(sizeof(mml_mouse_event) == 44);
    mml_capture_stats stats{};
    assert(mml_capture_get_stats(&stats) == MML_CAPTURE_OK);
    assert(stats.active == 0);
    assert(mml_capture_qpc_now() > 0);
    int32_t screen_x = 0;
    int32_t screen_y = 0;
    assert(mml_capture_cursor_position(&screen_x, &screen_y) == MML_CAPTURE_OK);
    assert(mml_capture_stop() == MML_CAPTURE_NOT_ACTIVE);
    assert(mml_capture_drain(nullptr, 0, nullptr) == MML_CAPTURE_INVALID_ARGUMENT);
    return 0;
}

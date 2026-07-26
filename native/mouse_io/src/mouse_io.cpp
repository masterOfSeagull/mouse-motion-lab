#include <mousemotionlab/mouse_io.h>

#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

namespace {
struct CaptureState {
    std::mutex mutex;
    HWND collection_window = nullptr;
    std::deque<mml_mouse_event> events;
    uint32_t capacity = 0;
    uint64_t observed_events = 0;
    uint64_t overflow_events = 0;
    uint64_t qpc_frequency_hz = 0;
    uint32_t last_status = MML_CAPTURE_NOT_ACTIVE;
    bool active = false;
};

CaptureState state;

uint64_t qpc_now() {
    LARGE_INTEGER value{};
    QueryPerformanceCounter(&value);
    return static_cast<uint64_t>(value.QuadPart);
}

uint64_t qpc_frequency() {
    LARGE_INTEGER value{};
    QueryPerformanceFrequency(&value);
    return static_cast<uint64_t>(value.QuadPart);
}

bool register_mouse(HWND window) {
    RAWINPUTDEVICE device{};
    device.usUsagePage = 0x01;
    device.usUsage = 0x02;
    device.dwFlags = 0; // Focused collection window only; no background input sink.
    device.hwndTarget = window;
    return RegisterRawInputDevices(&device, 1, sizeof(device)) == TRUE;
}

void unregister_mouse() {
    RAWINPUTDEVICE device{};
    device.usUsagePage = 0x01;
    device.usUsage = 0x02;
    device.dwFlags = RIDEV_REMOVE;
    device.hwndTarget = nullptr;
    RegisterRawInputDevices(&device, 1, sizeof(device));
}
} // namespace

static_assert(sizeof(mml_mouse_event) == 44, "Mouse event ABI changed unexpectedly");

uint32_t mml_capture_start(uintptr_t collection_window, uint32_t capacity) {
    if (collection_window == 0 || capacity == 0) {
        return MML_CAPTURE_INVALID_ARGUMENT;
    }
    std::scoped_lock lock(state.mutex);
    if (state.active) {
        return MML_CAPTURE_INVALID_ARGUMENT;
    }
    const auto window = reinterpret_cast<HWND>(collection_window);
    if (!register_mouse(window)) {
        state.last_status = MML_CAPTURE_REGISTER_FAILED;
        return state.last_status;
    }
    state.collection_window = window;
    state.events.clear();
    state.capacity = capacity;
    state.observed_events = 0;
    state.overflow_events = 0;
    state.qpc_frequency_hz = qpc_frequency();
    state.last_status = MML_CAPTURE_OK;
    state.active = true;
    return MML_CAPTURE_OK;
}

uint32_t mml_capture_stop() {
    std::scoped_lock lock(state.mutex);
    if (!state.active) {
        return MML_CAPTURE_NOT_ACTIVE;
    }
    unregister_mouse();
    state.active = false;
    state.collection_window = nullptr;
    state.last_status = MML_CAPTURE_OK;
    return MML_CAPTURE_OK;
}

uint32_t mml_capture_handle_raw_input(intptr_t raw_input_handle) {
    std::scoped_lock lock(state.mutex);
    if (!state.active) {
        return MML_CAPTURE_NOT_ACTIVE;
    }
    UINT bytes = 0;
    const auto handle = reinterpret_cast<HRAWINPUT>(raw_input_handle);
    if (GetRawInputData(handle, RID_INPUT, nullptr, &bytes, sizeof(RAWINPUTHEADER)) != 0 || bytes == 0) {
        state.last_status = MML_CAPTURE_RAW_INPUT_FAILED;
        return state.last_status;
    }
    std::vector<BYTE> data(bytes);
    if (GetRawInputData(handle, RID_INPUT, data.data(), &bytes, sizeof(RAWINPUTHEADER)) != bytes) {
        state.last_status = MML_CAPTURE_RAW_INPUT_FAILED;
        return state.last_status;
    }
    const auto* input = reinterpret_cast<const RAWINPUT*>(data.data());
    if (input->header.dwType != RIM_TYPEMOUSE) {
        return MML_CAPTURE_OK;
    }
    ++state.observed_events;
    if (state.events.size() >= state.capacity) {
        ++state.overflow_events;
        state.last_status = MML_CAPTURE_BUFFER_OVERFLOW;
        return state.last_status;
    }
    POINT point{};
    GetCursorPos(&point);
    const auto& mouse = input->data.mouse;
    state.events.push_back({
        qpc_now(),
        mouse.lLastX,
        mouse.lLastY,
        point.x,
        point.y,
        mouse.usButtonFlags,
        mouse.usFlags,
        reinterpret_cast<uint64_t>(input->header.hDevice),
        GetForegroundWindow() == state.collection_window ? 1u : 0u,
        0u,
    });
    state.last_status = MML_CAPTURE_OK;
    return MML_CAPTURE_OK;
}

uint32_t mml_capture_drain(mml_mouse_event* destination, uint32_t destination_capacity, uint32_t* drained) {
    if (drained == nullptr || (destination_capacity > 0 && destination == nullptr)) {
        return MML_CAPTURE_INVALID_ARGUMENT;
    }
    std::scoped_lock lock(state.mutex);
    const auto count = std::min<uint32_t>(destination_capacity, static_cast<uint32_t>(state.events.size()));
    for (uint32_t index = 0; index < count; ++index) {
        destination[index] = state.events.front();
        state.events.pop_front();
    }
    *drained = count;
    return state.last_status;
}

uint32_t mml_capture_get_stats(mml_capture_stats* stats) {
    if (stats == nullptr) {
        return MML_CAPTURE_INVALID_ARGUMENT;
    }
    std::scoped_lock lock(state.mutex);
    *stats = {
        state.active ? 1u : 0u,
        state.capacity,
        static_cast<uint32_t>(state.events.size()),
        0u,
        state.observed_events,
        state.overflow_events,
        state.qpc_frequency_hz,
        state.last_status,
        0u,
    };
    return MML_CAPTURE_OK;
}

uint64_t mml_capture_qpc_now() {
    return qpc_now();
}

uint32_t mml_capture_cursor_position(int32_t* screen_x, int32_t* screen_y) {
    if (screen_x == nullptr || screen_y == nullptr) {
        return MML_CAPTURE_INVALID_ARGUMENT;
    }
    POINT point{};
    if (!GetCursorPos(&point)) {
        return MML_CAPTURE_RAW_INPUT_FAILED;
    }
    *screen_x = point.x;
    *screen_y = point.y;
    return MML_CAPTURE_OK;
}

uint32_t mml_capture_client_to_screen(uintptr_t collection_window, int32_t client_x, int32_t client_y, int32_t* screen_x, int32_t* screen_y) {
    if (collection_window == 0 || screen_x == nullptr || screen_y == nullptr) {
        return MML_CAPTURE_INVALID_ARGUMENT;
    }
    POINT point{client_x, client_y};
    if (!ClientToScreen(reinterpret_cast<HWND>(collection_window), &point)) {
        return MML_CAPTURE_RAW_INPUT_FAILED;
    }
    *screen_x = point.x;
    *screen_y = point.y;
    return MML_CAPTURE_OK;
}

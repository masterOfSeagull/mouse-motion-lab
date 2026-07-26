#include <mousemotionlab/mouse_io.h>

#include <windows.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>

namespace {
struct CaptureState {
    // Lifecycle only. The WM_INPUT producer never takes this mutex.
    std::mutex lifecycle_mutex;
    std::unique_ptr<mml_mouse_event[]> ring;
    std::array<BYTE, sizeof(RAWINPUT)> raw_input_storage{};
    std::atomic<uint64_t> head{0};
    std::atomic<uint64_t> tail{0};
    std::atomic<uintptr_t> collection_window{0};
    std::atomic<uint32_t> capacity{0};
    std::atomic<uint64_t> observed_events{0};
    std::atomic<uint64_t> overflow_events{0};
    std::atomic<uint64_t> qpc_frequency_hz{0};
    std::atomic<uint32_t> last_status{MML_CAPTURE_NOT_ACTIVE};
    std::atomic<bool> active{false};
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
    std::scoped_lock lock(state.lifecycle_mutex);
    if (state.active.load(std::memory_order_acquire)) {
        return MML_CAPTURE_INVALID_ARGUMENT;
    }
    const auto window = reinterpret_cast<HWND>(collection_window);
    if (!register_mouse(window)) {
        state.last_status.store(MML_CAPTURE_REGISTER_FAILED, std::memory_order_release);
        return MML_CAPTURE_REGISTER_FAILED;
    }
    try {
        state.ring = std::make_unique<mml_mouse_event[]>(capacity);
    } catch (...) {
        unregister_mouse();
        state.last_status.store(MML_CAPTURE_INVALID_ARGUMENT, std::memory_order_release);
        return MML_CAPTURE_INVALID_ARGUMENT;
    }
    state.head.store(0, std::memory_order_release);
    state.tail.store(0, std::memory_order_release);
    state.collection_window.store(collection_window, std::memory_order_release);
    state.capacity.store(capacity, std::memory_order_release);
    state.observed_events.store(0, std::memory_order_release);
    state.overflow_events.store(0, std::memory_order_release);
    state.qpc_frequency_hz.store(qpc_frequency(), std::memory_order_release);
    state.last_status.store(MML_CAPTURE_OK, std::memory_order_release);
    state.active.store(true, std::memory_order_release);
    return MML_CAPTURE_OK;
}

uint32_t mml_capture_stop() {
    std::scoped_lock lock(state.lifecycle_mutex);
    if (!state.active.exchange(false, std::memory_order_acq_rel)) {
        return MML_CAPTURE_NOT_ACTIVE;
    }
    unregister_mouse();
    // Keep the ring allocated and readable until the caller has drained its final tail events.
    state.collection_window.store(0, std::memory_order_release);
    state.last_status.store(MML_CAPTURE_OK, std::memory_order_release);
    return MML_CAPTURE_OK;
}

uint32_t mml_capture_handle_raw_input(intptr_t raw_input_handle) {
    if (!state.active.load(std::memory_order_acquire)) {
        return MML_CAPTURE_NOT_ACTIVE;
    }
    UINT bytes = static_cast<UINT>(state.raw_input_storage.size());
    const auto handle = reinterpret_cast<HRAWINPUT>(raw_input_handle);
    const auto copied = GetRawInputData(handle, RID_INPUT, state.raw_input_storage.data(), &bytes, sizeof(RAWINPUTHEADER));
    if (copied == static_cast<UINT>(-1) || copied == 0 || copied > state.raw_input_storage.size()) {
        state.last_status.store(MML_CAPTURE_RAW_INPUT_FAILED, std::memory_order_release);
        return MML_CAPTURE_RAW_INPUT_FAILED;
    }
    const auto* input = reinterpret_cast<const RAWINPUT*>(state.raw_input_storage.data());
    if (input->header.dwType != RIM_TYPEMOUSE) {
        return MML_CAPTURE_OK;
    }
    const auto& mouse = input->data.mouse;
    if ((mouse.usFlags & MOUSE_MOVE_ABSOLUTE) != 0) {
        state.last_status.store(MML_CAPTURE_ABSOLUTE_MOUSE_UNSUPPORTED, std::memory_order_release);
        return MML_CAPTURE_ABSOLUTE_MOUSE_UNSUPPORTED;
    }
    POINT point{};
    if (!GetCursorPos(&point)) {
        state.last_status.store(MML_CAPTURE_RAW_INPUT_FAILED, std::memory_order_release);
        return MML_CAPTURE_RAW_INPUT_FAILED;
    }
    const auto capacity = state.capacity.load(std::memory_order_acquire);
    const auto write_index = state.head.load(std::memory_order_relaxed);
    const auto read_index = state.tail.load(std::memory_order_acquire);
    state.observed_events.fetch_add(1, std::memory_order_relaxed);
    if (capacity == 0 || write_index - read_index >= capacity) {
        state.overflow_events.fetch_add(1, std::memory_order_relaxed);
        state.last_status.store(MML_CAPTURE_BUFFER_OVERFLOW, std::memory_order_release);
        return MML_CAPTURE_BUFFER_OVERFLOW;
    }
    state.ring[write_index % capacity] = {
        qpc_now(), mouse.lLastX, mouse.lLastY, point.x, point.y, mouse.usButtonFlags, mouse.usFlags,
        reinterpret_cast<uint64_t>(input->header.hDevice),
        GetForegroundWindow() == reinterpret_cast<HWND>(state.collection_window.load(std::memory_order_acquire)) ? 1u : 0u,
        0u,
    };
    state.head.store(write_index + 1, std::memory_order_release);
    state.last_status.store(MML_CAPTURE_OK, std::memory_order_release);
    return MML_CAPTURE_OK;
}

uint32_t mml_capture_drain(mml_mouse_event* destination, uint32_t destination_capacity, uint32_t* drained) {
    if (drained == nullptr || (destination_capacity > 0 && destination == nullptr)) {
        return MML_CAPTURE_INVALID_ARGUMENT;
    }
    const auto capacity = state.capacity.load(std::memory_order_acquire);
    const auto read_index = state.tail.load(std::memory_order_relaxed);
    const auto write_index = state.head.load(std::memory_order_acquire);
    const auto count = capacity == 0 ? 0u : std::min<uint64_t>(destination_capacity, write_index - read_index);
    for (uint32_t index = 0; index < count; ++index) {
        destination[index] = state.ring[(read_index + index) % capacity];
    }
    state.tail.store(read_index + count, std::memory_order_release);
    *drained = count;
    return state.last_status.load(std::memory_order_acquire);
}

uint32_t mml_capture_get_stats(mml_capture_stats* stats) {
    if (stats == nullptr) {
        return MML_CAPTURE_INVALID_ARGUMENT;
    }
    const auto write_index = state.head.load(std::memory_order_acquire);
    const auto read_index = state.tail.load(std::memory_order_acquire);
    *stats = {
        state.active.load(std::memory_order_acquire) ? 1u : 0u,
        state.capacity.load(std::memory_order_acquire),
        static_cast<uint32_t>(write_index - read_index), 0u,
        state.observed_events.load(std::memory_order_acquire), state.overflow_events.load(std::memory_order_acquire),
        state.qpc_frequency_hz.load(std::memory_order_acquire), state.last_status.load(std::memory_order_acquire), 0u,
    };
    return MML_CAPTURE_OK;
}

uint64_t mml_capture_qpc_now() { return qpc_now(); }

uint32_t mml_capture_cursor_position(int32_t* screen_x, int32_t* screen_y) {
    if (screen_x == nullptr || screen_y == nullptr) return MML_CAPTURE_INVALID_ARGUMENT;
    POINT point{};
    if (!GetCursorPos(&point)) return MML_CAPTURE_RAW_INPUT_FAILED;
    *screen_x = point.x;
    *screen_y = point.y;
    return MML_CAPTURE_OK;
}

uint32_t mml_capture_client_to_screen(uintptr_t collection_window, int32_t client_x, int32_t client_y, int32_t* screen_x, int32_t* screen_y) {
    if (collection_window == 0 || screen_x == nullptr || screen_y == nullptr) return MML_CAPTURE_INVALID_ARGUMENT;
    POINT point{client_x, client_y};
    if (!ClientToScreen(reinterpret_cast<HWND>(collection_window), &point)) return MML_CAPTURE_RAW_INPUT_FAILED;
    *screen_x = point.x;
    *screen_y = point.y;
    return MML_CAPTURE_OK;
}

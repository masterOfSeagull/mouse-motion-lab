#pragma once

#include <stdint.h>

#if defined(_WIN32)
  #if defined(MML_MOUSE_IO_EXPORTS)
    #define MML_MOUSE_IO_API extern "C" __declspec(dllexport)
  #else
    #define MML_MOUSE_IO_API extern "C" __declspec(dllimport)
  #endif
#else
  #define MML_MOUSE_IO_API extern "C"
#endif

// Keep this ABI independent from Qt and Python. All timestamps are QPC ticks.
enum mml_capture_status : uint32_t {
    MML_CAPTURE_OK = 0,
    MML_CAPTURE_NOT_ACTIVE = 1,
    MML_CAPTURE_INVALID_ARGUMENT = 2,
    MML_CAPTURE_REGISTER_FAILED = 3,
    MML_CAPTURE_RAW_INPUT_FAILED = 4,
    MML_CAPTURE_BUFFER_OVERFLOW = 5,
};

#pragma pack(push, 1)
struct mml_mouse_event {
    uint64_t timestamp_ticks;
    int32_t raw_dx;
    int32_t raw_dy;
    int32_t screen_x;
    int32_t screen_y;
    uint16_t button_flags;
    uint16_t event_flags;
    uint64_t device_handle;
    uint32_t foreground_collection_window;
    uint32_t reserved;
};
#pragma pack(pop)

struct mml_capture_stats {
    uint32_t active;
    uint32_t capacity;
    uint32_t buffered_events;
    uint32_t reserved;
    uint64_t observed_events;
    uint64_t overflow_events;
    uint64_t qpc_frequency_hz;
    uint32_t last_status;
    uint32_t reserved2;
};

MML_MOUSE_IO_API uint32_t mml_capture_start(uintptr_t collection_window, uint32_t capacity);
MML_MOUSE_IO_API uint32_t mml_capture_stop();
// Call only from the owning window's WM_INPUT handling path.
MML_MOUSE_IO_API uint32_t mml_capture_handle_raw_input(intptr_t raw_input_handle);
MML_MOUSE_IO_API uint32_t mml_capture_drain(mml_mouse_event* destination, uint32_t destination_capacity, uint32_t* drained);
MML_MOUSE_IO_API uint32_t mml_capture_get_stats(mml_capture_stats* stats);
MML_MOUSE_IO_API uint64_t mml_capture_qpc_now();
MML_MOUSE_IO_API uint32_t mml_capture_cursor_position(int32_t* screen_x, int32_t* screen_y);
// Converts physical client coordinates to physical desktop coordinates for the active Qt window.
MML_MOUSE_IO_API uint32_t mml_capture_client_to_screen(uintptr_t collection_window, int32_t client_x, int32_t client_y, int32_t* screen_x, int32_t* screen_y);

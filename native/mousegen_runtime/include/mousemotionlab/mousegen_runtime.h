#pragma once
#include <cstddef>
#include <cstdint>

#if defined(MML_MOUSEGEN_STATIC)
#define MML_MOUSEGEN_API
#elif defined(_WIN32)
#ifdef MML_MOUSEGEN_EXPORTS
#define MML_MOUSEGEN_API __declspec(dllexport)
#else
#define MML_MOUSEGEN_API __declspec(dllimport)
#endif
#else
#define MML_MOUSEGEN_API
#endif

extern "C" {
typedef void* MGHandle;
enum MGResult { MG_OK = 0, MG_INVALID_ARGUMENT = 1, MG_IO_ERROR = 2, MG_RUNTIME_ERROR = 3 };
enum MGEndpointMode : std::uint8_t { MG_ENDPOINT_WITHIN_RADIUS = 0, MG_ENDPOINT_EXACT_CENTER = 1 };
struct MGGenerationRequest {
    double start_x, start_y, target_center_x, target_center_y, target_radius;
    double previous_velocity_x, previous_velocity_y;
    double virtual_desktop_left, virtual_desktop_top, virtual_desktop_width, virtual_desktop_height;
    double dpi_scale;
    std::uint8_t click_requested;
    std::uint64_t random_seed;
    std::uint32_t output_rate_hz;
    std::uint32_t solver_steps;
    std::uint8_t endpoint_mode;
};
struct MGTrajectorySample { std::int64_t relative_time_ns; double x, y; };
struct MGTrajectory {
    MGTrajectorySample* samples;
    std::size_t sample_count;
    std::int64_t movement_duration_ns;
    double endpoint_x, endpoint_y;
    std::uint8_t click_requested, out_of_distribution, endpoint_projected;
    double condition_distance_score;
    std::uint64_t seed;
    std::uint32_t desktop_clipped_point_count;
};
MML_MOUSEGEN_API MGResult mg_create(const char* model_directory, MGHandle* out_handle);
MML_MOUSEGEN_API MGResult mg_generate(MGHandle handle, const MGGenerationRequest* request, MGTrajectory* out_trajectory);
MML_MOUSEGEN_API void mg_free_trajectory(MGTrajectory* trajectory);
MML_MOUSEGEN_API void mg_destroy(MGHandle handle);
MML_MOUSEGEN_API const char* mg_last_error(MGHandle handle);
MML_MOUSEGEN_API const char* mg_model_id(MGHandle handle);
MML_MOUSEGEN_API std::size_t mg_position_count(MGHandle handle);
}

#ifdef __cplusplus
#include <stdexcept>
#include <string>
namespace mousemotionlab {
class Generator {
public:
    explicit Generator(const std::string& directory) { if (mg_create(directory.c_str(), &handle_) != MG_OK) throw std::runtime_error(mg_last_error(nullptr)); }
    ~Generator() { mg_destroy(handle_); }
    Generator(const Generator&) = delete; Generator& operator=(const Generator&) = delete;
    MGHandle get() const noexcept { return handle_; }
private: MGHandle handle_{};
};
}
#endif

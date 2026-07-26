#include "mousemotionlab/mousegen_runtime.h"
#include <cassert>
#include <cstring>
int main() {
    MGHandle handle{};
    assert(mg_create(nullptr, &handle) == MG_INVALID_ARGUMENT);
    assert(std::strlen(mg_last_error(nullptr)) > 0);
    MGGenerationRequest request{}; MGTrajectory trajectory{};
    assert(mg_generate(nullptr, &request, &trajectory) == MG_INVALID_ARGUMENT);
    mg_free_trajectory(nullptr); mg_destroy(nullptr);
    return 0;
}

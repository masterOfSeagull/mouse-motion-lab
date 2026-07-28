#include "mousemotionlab/mousegen_runtime.h"
#include <windows.h>
#include <bcrypt.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numbers>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string global_error;
struct Runtime {
    std::string model_id, error;
    std::size_t condition_size{}, output_size{}, training_count{}, latent_size{}, mixture_count{}, position_count{};
    double temperature{}, maximum_neighbor_distance{};
    std::vector<double> condition_mean, condition_scale, training_conditions, output_mean, output_scale;
    std::vector<double> components, condition_centers, latent_means, latent_scales, component_priors;
};
template<class T> void read_exact(std::ifstream& in, T* data, std::size_t count) {
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(sizeof(T) * count));
    if (!in) throw std::runtime_error("PCA payload is truncated");
}
std::string read_text(const std::filesystem::path& path) {
    std::ifstream in(path); if (!in) throw std::runtime_error("manifest is missing");
    return {std::istreambuf_iterator<char>(in), {}};
}
std::string capture(const std::string& text, const std::string& key) {
    std::smatch match;
    if (!std::regex_search(text, match, std::regex("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"")))
        throw std::runtime_error("manifest is missing " + key);
    return match[1];
}
std::string sha256(const std::filesystem::path& path) {
    BCRYPT_ALG_HANDLE algorithm{}; BCRYPT_HASH_HANDLE hash{}; DWORD object_size{}, digest_size{}, bytes{};
    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0) throw std::runtime_error("could not initialize SHA-256");
    auto close_algorithm = [&] { if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0); };
    if (BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&object_size), sizeof(object_size), &bytes, 0) < 0 ||
        BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH, reinterpret_cast<PUCHAR>(&digest_size), sizeof(digest_size), &bytes, 0) < 0) {
        close_algorithm(); throw std::runtime_error("could not query SHA-256");
    }
    std::vector<unsigned char> object(object_size), digest(digest_size);
    if (BCryptCreateHash(algorithm, &hash, object.data(), object_size, nullptr, 0, 0) < 0) { close_algorithm(); throw std::runtime_error("could not create SHA-256"); }
    std::ifstream in(path, std::ios::binary);
    if (!in) { BCryptDestroyHash(hash); close_algorithm(); throw std::runtime_error("export artifact is missing"); }
    std::array<char, 65536> buffer{};
    while (in) {
        in.read(buffer.data(), buffer.size()); const auto count = in.gcount();
        if (count > 0 && BCryptHashData(hash, reinterpret_cast<PUCHAR>(buffer.data()), static_cast<ULONG>(count), 0) < 0) {
            BCryptDestroyHash(hash); close_algorithm(); throw std::runtime_error("could not hash export artifact");
        }
    }
    if (BCryptFinishHash(hash, digest.data(), digest_size, 0) < 0) { BCryptDestroyHash(hash); close_algorithm(); throw std::runtime_error("could not finish SHA-256"); }
    BCryptDestroyHash(hash); close_algorithm();
    static constexpr char digits[] = "0123456789abcdef"; std::string result; result.reserve(digest.size() * 2);
    for (auto value : digest) { result.push_back(digits[value >> 4]); result.push_back(digits[value & 15]); }
    return result;
}
void set_error(Runtime* runtime, const std::string& value) { global_error = value; if (runtime) runtime->error = value; }
bool finite_request(const MGGenerationRequest& q) {
    const double values[]{q.start_x,q.start_y,q.target_center_x,q.target_center_y,q.target_radius,q.previous_velocity_x,q.previous_velocity_y,
      q.virtual_desktop_left,q.virtual_desktop_top,q.virtual_desktop_width,q.virtual_desktop_height,q.dpi_scale};
    return std::all_of(std::begin(values), std::end(values), [](double value){ return std::isfinite(value); });
}
class PortableRandom {
public:
    explicit PortableRandom(std::uint64_t seed): state_(seed) {}
    double uniform() { return (static_cast<double>(splitmix() >> 11) + 0.5) / 9007199254740992.0; }
    double normal() {
        if (has_spare_) { has_spare_ = false; return spare_; }
        const auto radius = std::sqrt(-2.0 * std::log(uniform())); const auto angle = 2.0 * std::numbers::pi * uniform();
        spare_ = radius * std::sin(angle); has_spare_ = true; return radius * std::cos(angle);
    }
private:
    std::uint64_t splitmix() {
        state_ += 0x9E3779B97F4A7C15ULL; auto value = state_;
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL; return value ^ (value >> 31);
    }
    std::uint64_t state_{}; bool has_spare_{}; double spare_{};
};
std::vector<double> condition(const MGGenerationRequest& q) {
    const auto dx=q.target_center_x-q.start_x, dy=q.target_center_y-q.start_y, distance=std::hypot(dx,dy), safe=std::max(distance,1e-9);
    const auto angle=distance>0?std::atan2(dy,dx):0.0, difficulty=std::log2(distance/(2*q.target_radius)+1), left=q.virtual_desktop_left, top=q.virtual_desktop_top;
    const auto width=q.virtual_desktop_width,height=q.virtual_desktop_height,right=left+width,bottom=top+height;
    return {std::log(safe),std::log(q.target_radius),std::log1p(difficulty),q.target_radius/safe,std::sin(angle),std::cos(angle),
      (q.start_x-left)/width,(q.start_y-top)/height,(q.target_center_x-left)/width,(q.target_center_y-top)/height,
      (q.start_x-left)/width,(right-q.start_x)/width,(q.start_y-top)/height,(bottom-q.start_y)/height,
      (q.target_center_x-left)/width,(right-q.target_center_x)/width,(q.target_center_y-top)/height,(bottom-q.target_center_y)/height,
      q.previous_velocity_x/width,q.previous_velocity_y/height,q.dpi_scale};
}
}

extern "C" MGResult mg_create(const char* directory, MGHandle* output) {
    if (!directory || !output) { set_error(nullptr, "model directory and output handle are required"); return MG_INVALID_ARGUMENT; }
    try {
        const std::filesystem::path root=std::filesystem::u8path(directory); const auto manifest=read_text(root/"manifest.json");
        if (capture(manifest,"format") != "mousemotionlab-pca-mixture") throw std::runtime_error("unsupported PCA export manifest");
        if (sha256(root/"pca.bin") != capture(manifest,"pca.bin")) throw std::runtime_error("export artifact hash changed: pca.bin");
        auto r=std::make_unique<Runtime>(); r->model_id=capture(manifest,"model_id");
        std::ifstream in(root/"pca.bin",std::ios::binary); char magic[8]; std::uint32_t c,o,n,k,m,p;
        read_exact(in,magic,8);read_exact(in,&c,1);read_exact(in,&o,1);read_exact(in,&n,1);read_exact(in,&k,1);read_exact(in,&m,1);read_exact(in,&p,1);read_exact(in,&r->temperature,1);read_exact(in,&r->maximum_neighbor_distance,1);
        if(std::memcmp(magic,"MMLPCA1\0",8)!=0||c!=21||o!=p*2+1||!n||!k||!m||!p||!std::isfinite(r->temperature)||r->temperature<=0||!std::isfinite(r->maximum_neighbor_distance)||r->maximum_neighbor_distance<=0)
            throw std::runtime_error("unsupported PCA payload dimensions or configuration");
        r->condition_size=c;r->output_size=o;r->training_count=n;r->latent_size=k;r->mixture_count=m;r->position_count=p;
        auto read=[&](std::vector<double>& values,std::size_t count){values.resize(count);read_exact(in,values.data(),count);if(!std::all_of(values.begin(),values.end(),[](double v){return std::isfinite(v);}))throw std::runtime_error("PCA payload contains non-finite values");};
        read(r->condition_mean,c);read(r->condition_scale,c);read(r->training_conditions,static_cast<std::size_t>(n)*c);read(r->output_mean,o);read(r->output_scale,o);read(r->components,static_cast<std::size_t>(k)*o);read(r->condition_centers,static_cast<std::size_t>(m)*c);read(r->latent_means,static_cast<std::size_t>(m)*k);read(r->latent_scales,static_cast<std::size_t>(m)*k);read(r->component_priors,m);
        if(std::any_of(r->condition_scale.begin(),r->condition_scale.end(),[](double v){return v<=0;})||std::any_of(r->output_scale.begin(),r->output_scale.end(),[](double v){return v<=0;})||std::any_of(r->latent_scales.begin(),r->latent_scales.end(),[](double v){return v<0;}))throw std::runtime_error("PCA payload contains invalid scales");
        char extra; if(in.read(&extra,1))throw std::runtime_error("PCA payload has trailing data");
        *output=r.release();global_error.clear();return MG_OK;
    } catch(const std::exception& e) { set_error(nullptr,e.what()); return MG_RUNTIME_ERROR; }
}

extern "C" MGResult mg_generate(MGHandle handle,const MGGenerationRequest* q,MGTrajectory* out) {
    auto* r=static_cast<Runtime*>(handle); if(!r||!q||!out){set_error(r,"handle, request, and trajectory are required");return MG_INVALID_ARGUMENT;}
    try {
        *out={};
        if(!finite_request(*q)||!(q->target_radius>0&&q->virtual_desktop_width>=1&&q->virtual_desktop_height>=1&&q->dpi_scale>0))throw std::invalid_argument("request values must be finite and dimensions positive");
        const auto right=q->virtual_desktop_left+q->virtual_desktop_width-1,bottom=q->virtual_desktop_top+q->virtual_desktop_height-1;
        if(q->target_center_x<q->virtual_desktop_left||q->target_center_x>right||q->target_center_y<q->virtual_desktop_top||q->target_center_y>bottom)throw std::invalid_argument("target must be clamped to the virtual desktop before generation");
        const auto distance=std::hypot(q->target_center_x-q->start_x,q->target_center_y-q->start_y);
        if(distance<=1e-9){out->endpoint_x=q->target_center_x;out->endpoint_y=q->target_center_y;out->seed=q->random_seed;return MG_OK;}
        const auto raw=condition(*q);std::vector<double> normalized(r->condition_size);
        for(std::size_t i=0;i<normalized.size();++i)normalized[i]=(raw[i]-r->condition_mean[i])/r->condition_scale[i];
        double nearest=std::numeric_limits<double>::infinity();
        for(std::size_t row=0;row<r->training_count;++row){double sum=0;for(std::size_t i=0;i<r->condition_size;++i){const auto d=normalized[i]-r->training_conditions[row*r->condition_size+i];sum+=d*d;}nearest=std::min(nearest,std::sqrt(sum));}
        std::vector<double> weights(r->mixture_count);double min_distance=std::numeric_limits<double>::infinity();
        for(std::size_t component=0;component<r->mixture_count;++component){double sum=0;for(std::size_t i=0;i<r->condition_size;++i){const auto d=normalized[i]-r->condition_centers[component*r->condition_size+i];sum+=d*d;}weights[component]=std::sqrt(sum);min_distance=std::min(min_distance,weights[component]);}
        double total=0;for(std::size_t i=0;i<weights.size();++i){weights[i]=r->component_priors[i]*std::exp(-(weights[i]-min_distance)/r->temperature);total+=weights[i];}if(!std::isfinite(total)||total<=0)throw std::runtime_error("PCA mixture weights are invalid");
        PortableRandom random(q->random_seed);const auto pick=random.uniform()*total;double cumulative=0;std::size_t selected=weights.size()-1;for(std::size_t i=0;i<weights.size();++i){cumulative+=weights[i];if(pick<cumulative){selected=i;break;}}
        std::vector<double> latent(r->latent_size);for(std::size_t i=0;i<latent.size();++i)latent[i]=r->latent_means[selected*r->latent_size+i]+r->latent_scales[selected*r->latent_size+i]*random.normal();
        std::vector<double> values(r->output_size);for(std::size_t j=0;j<values.size();++j){double standardized=0;for(std::size_t i=0;i<latent.size();++i)standardized+=latent[i]*r->components[i*r->output_size+j];values[j]=standardized*r->output_scale[j]+r->output_mean[j];}
        values[0]=values[1]=0;bool normalized_endpoint=false;
        if(q->endpoint_mode==MG_ENDPOINT_EXACT_CENTER){const auto ex=values[(r->position_count-1)*2],ey=values[(r->position_count-1)*2+1],norm2=ex*ex+ey*ey;if(!std::isfinite(norm2)||norm2<1e-18)throw std::runtime_error("raw PCA endpoint is too close to zero to normalize");for(std::size_t i=0;i<r->position_count;++i){const auto x=values[i*2],y=values[i*2+1];values[i*2]=(x*ex+y*ey)/norm2;values[i*2+1]=(-x*ey+y*ex)/norm2;}values[0]=values[1]=0;values[(r->position_count-1)*2]=1;values[(r->position_count-1)*2+1]=0;normalized_endpoint=true;}
        else {const auto ratio=q->target_radius/distance,ox=values[(r->position_count-1)*2]-1,oy=values[(r->position_count-1)*2+1],norm=std::hypot(ox,oy),allowed=ratio*.99;if(norm>allowed){values[(r->position_count-1)*2]=1+ox*allowed/norm;values[(r->position_count-1)*2+1]=oy*allowed/norm;normalized_endpoint=true;}}
        if(!std::all_of(values.begin(),values.end(),[](double v){return std::isfinite(v);}))throw std::runtime_error("PCA generation produced non-finite values");
        const auto duration=static_cast<std::int64_t>(std::llround(std::exp(std::clamp(values.back(),std::log(1e6),std::log(6e10)))));auto* samples=new MGTrajectorySample[r->position_count];
        const auto dx=q->target_center_x-q->start_x,dy=q->target_center_y-q->start_y,angle=std::atan2(dy,dx),cs=std::cos(angle),sn=std::sin(angle);std::uint32_t clipped=0;
        for(std::size_t i=0;i<r->position_count;++i){if(i==0){samples[i]={0,q->start_x,q->start_y};continue;}if(i+1==r->position_count){samples[i]={duration,q->target_center_x,q->target_center_y};continue;}const auto cx=values[i*2]*distance,cy=values[i*2+1]*distance;double x=q->start_x+cs*cx-sn*cy,y=q->start_y+sn*cx+cs*cy;const auto original_x=x,original_y=y;x=std::clamp(x,q->virtual_desktop_left,right);y=std::clamp(y,q->virtual_desktop_top,bottom);if(x!=original_x||y!=original_y)++clipped;samples[i]={static_cast<std::int64_t>(std::llround(static_cast<double>(duration)*i/(r->position_count-1))),x,y};}
        *out={samples,r->position_count,duration,q->target_center_x,q->target_center_y,q->click_requested,static_cast<std::uint8_t>(nearest>r->maximum_neighbor_distance),static_cast<std::uint8_t>(normalized_endpoint),nearest,q->random_seed,clipped};r->error.clear();return MG_OK;
    } catch(const std::exception& e) { mg_free_trajectory(out);set_error(r,e.what());return MG_RUNTIME_ERROR; }
}
extern "C" void mg_free_trajectory(MGTrajectory* value){if(value){delete[] value->samples;*value={};}}
extern "C" void mg_destroy(MGHandle handle){delete static_cast<Runtime*>(handle);}
extern "C" const char* mg_last_error(MGHandle handle){auto* r=static_cast<Runtime*>(handle);return r?r->error.c_str():global_error.c_str();}
extern "C" const char* mg_model_id(MGHandle handle){auto* r=static_cast<Runtime*>(handle);return r?r->model_id.c_str():"";}
extern "C" std::size_t mg_position_count(MGHandle handle){auto* r=static_cast<Runtime*>(handle);return r?r->position_count:0;}

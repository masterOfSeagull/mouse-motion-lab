#include "mousemotionlab/mousegen_runtime.h"
#include <onnxruntime_cxx_api.h>
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
#include <sstream>
#include <string>
#include <vector>

namespace {
thread_local std::string global_error;
struct Runtime {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "MouseMotionLab"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    std::vector<double> condition_mean, condition_scale, training_conditions, output_mean, output_scale;
    std::size_t condition_size{}, output_size{}, training_count{};
    std::uint32_t solver_steps{16}; bool heun{true}; std::string error;
};
template<class T> void read_exact(std::ifstream& in, T* data, std::size_t count) {
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(sizeof(T) * count));
    if (!in) throw std::runtime_error("normalization artifact is truncated");
}
std::string read_text(const std::filesystem::path& path) { std::ifstream in(path); if (!in) throw std::runtime_error("manifest is missing"); return {std::istreambuf_iterator<char>(in), {}}; }
std::string sha256(const std::filesystem::path& path) {
    BCRYPT_ALG_HANDLE algorithm{}; BCRYPT_HASH_HANDLE hash{}; DWORD object_size{}, digest_size{}, bytes{};
    if(BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0) throw std::runtime_error("could not initialize SHA-256");
    if(BCryptGetProperty(algorithm,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&object_size),sizeof(object_size),&bytes,0)<0||BCryptGetProperty(algorithm,BCRYPT_HASH_LENGTH,reinterpret_cast<PUCHAR>(&digest_size),sizeof(digest_size),&bytes,0)<0){BCryptCloseAlgorithmProvider(algorithm,0);throw std::runtime_error("could not query SHA-256");}
    std::vector<unsigned char> object(object_size),digest(digest_size);if(BCryptCreateHash(algorithm,&hash,object.data(),object_size,nullptr,0,0)<0){BCryptCloseAlgorithmProvider(algorithm,0);throw std::runtime_error("could not create SHA-256");}
    std::ifstream in(path,std::ios::binary);if(!in){BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(algorithm,0);throw std::runtime_error("export artifact is missing");}std::array<char,65536> buffer{};
    while(in){in.read(buffer.data(),buffer.size());const auto count=in.gcount();if(count>0&&BCryptHashData(hash,reinterpret_cast<PUCHAR>(buffer.data()),static_cast<ULONG>(count),0)<0){BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(algorithm,0);throw std::runtime_error("could not hash export artifact");}}
    if(BCryptFinishHash(hash,digest.data(),digest_size,0)<0){BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(algorithm,0);throw std::runtime_error("could not finish SHA-256");}BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(algorithm,0);
    static constexpr char digits[]="0123456789abcdef";std::string result;result.reserve(digest.size()*2);for(auto value:digest){result.push_back(digits[value>>4]);result.push_back(digits[value&15]);}return result;
}
void verify_file(const std::filesystem::path& root,const std::string& manifest,const std::string& name){std::smatch match;const std::regex pattern("\\\""+name+"\\\":\\\"([0-9a-f]{64})\\\"");if(!std::regex_search(manifest,match,pattern)||sha256(root/name)!=match[1])throw std::runtime_error("export artifact hash changed: "+name);}
std::uint64_t splitmix(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL; auto z = state; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31);
}
double uniform(std::uint64_t& state) { return (static_cast<double>(splitmix(state) >> 11) + 0.5) / 9007199254740992.0; }
std::vector<float> normal_source(std::uint64_t seed, std::size_t count) {
    std::vector<float> result; result.reserve(count); auto state = seed;
    while (result.size() < count) { const auto r = std::sqrt(-2.0 * std::log(uniform(state))); const auto a = 2.0 * std::numbers::pi * uniform(state); result.push_back(static_cast<float>(r * std::cos(a))); if (result.size() < count) result.push_back(static_cast<float>(r * std::sin(a))); }
    return result;
}
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
std::vector<float> velocity(Runtime& r, const std::vector<float>& state, const std::vector<float>& cond, float time) {
    std::array<std::int64_t,2> state_shape{1,static_cast<std::int64_t>(r.output_size)}, cond_shape{1,static_cast<std::int64_t>(r.condition_size)}, time_shape{1,1};
    auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault); float time_value=time;
    std::array<Ort::Value,3> inputs{
      Ort::Value::CreateTensor<float>(memory,const_cast<float*>(state.data()),state.size(),state_shape.data(),2),
      Ort::Value::CreateTensor<float>(memory,const_cast<float*>(cond.data()),cond.size(),cond_shape.data(),2),
      Ort::Value::CreateTensor<float>(memory,&time_value,1,time_shape.data(),2)};
    const char* names[]{"state","condition","time"}; const char* outputs[]{"velocity"};
    auto values=r.session->Run(Ort::RunOptions{nullptr},names,inputs.data(),inputs.size(),outputs,1); const float* data=values[0].GetTensorData<float>(); return {data,data+r.output_size};
}
void set_error(Runtime* runtime, const std::string& value) { global_error=value; if(runtime) runtime->error=value; }
}

extern "C" MGResult mg_create(const char* directory, MGHandle* output) {
    if(!directory||!output){set_error(nullptr,"model directory and output handle are required");return MG_INVALID_ARGUMENT;}
    try {
        auto r=std::make_unique<Runtime>(); const std::filesystem::path root=std::filesystem::u8path(directory);const auto manifest=read_text(root/"manifest.json");verify_file(root,manifest,"velocity.onnx");verify_file(root,manifest,"normalization.bin");verify_file(root,manifest,"normalization.npz");
        std::ifstream in(root/"normalization.bin",std::ios::binary); if(!in) throw std::runtime_error("normalization.bin is missing");
        char magic[8]; std::uint32_t c,o,n; read_exact(in,magic,8); read_exact(in,&c,1); read_exact(in,&o,1); read_exact(in,&n,1);
        if(std::memcmp(magic,"MMLNORM1",8)!=0||c!=21||o!=129||n==0) throw std::runtime_error("unsupported normalization artifact");
        r->condition_size=c;r->output_size=o;r->training_count=n; r->condition_mean.resize(c);r->condition_scale.resize(c);r->training_conditions.resize(static_cast<std::size_t>(c)*n);r->output_mean.resize(o);r->output_scale.resize(o);
        read_exact(in,r->condition_mean.data(),c);read_exact(in,r->condition_scale.data(),c);read_exact(in,r->training_conditions.data(),static_cast<std::size_t>(c)*n);read_exact(in,r->output_mean.data(),o);read_exact(in,r->output_scale.data(),o);
        std::smatch match; if(std::regex_search(manifest,match,std::regex("\"solver_steps\":([0-9]+)"))) r->solver_steps=static_cast<std::uint32_t>(std::stoul(match[1])); r->heun=manifest.find("\"solver\":\"heun\"")!=std::string::npos;
        r->options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL); r->options.SetIntraOpNumThreads(1); r->session=std::make_unique<Ort::Session>(r->env,(root/"velocity.onnx").c_str(),r->options);
        *output=r.release();global_error.clear();return MG_OK;
    } catch(const std::exception& e){set_error(nullptr,e.what());return MG_RUNTIME_ERROR;}
}
extern "C" MGResult mg_generate(MGHandle handle,const MGGenerationRequest* q,MGTrajectory* out){
    auto* r=static_cast<Runtime*>(handle); if(!r||!q||!out){set_error(r,"handle, request, and trajectory are required");return MG_INVALID_ARGUMENT;}
    try {
        if(!(q->target_radius>0&&q->virtual_desktop_width>0&&q->virtual_desktop_height>0&&q->dpi_scale>0)) throw std::invalid_argument("request dimensions must be positive");
        auto raw=condition(*q);std::vector<float> cond(r->condition_size);for(std::size_t i=0;i<cond.size();++i)cond[i]=static_cast<float>((raw[i]-r->condition_mean[i])/r->condition_scale[i]);
        double nearest=std::numeric_limits<double>::infinity();for(std::size_t row=0;row<r->training_count;++row){double sum=0;for(std::size_t i=0;i<r->condition_size;++i){const auto d=cond[i]-r->training_conditions[row*r->condition_size+i];sum+=d*d;}nearest=std::min(nearest,std::sqrt(sum));}
        auto state=normal_source(q->random_seed,r->output_size);const auto steps=q->solver_steps?q->solver_steps:r->solver_steps;const float dt=1.0f/steps;
        for(std::uint32_t i=0;i<steps;++i){auto v=velocity(*r,state,cond,i*dt);if(r->heun){auto predicted=state;for(std::size_t j=0;j<state.size();++j)predicted[j]+=dt*v[j];auto next=velocity(*r,predicted,cond,(i+1)*dt);for(std::size_t j=0;j<state.size();++j)state[j]+=dt*(v[j]+next[j])/2;}else for(std::size_t j=0;j<state.size();++j)state[j]+=dt*v[j];}
        std::vector<double> values(r->output_size);for(std::size_t i=0;i<values.size();++i)values[i]=state[i]*r->output_scale[i]+r->output_mean[i];values[0]=values[1]=0;
        const auto dx=q->target_center_x-q->start_x,dy=q->target_center_y-q->start_y,distance=std::hypot(dx,dy),ratio=q->target_radius/std::max(distance,1e-9),ox=values[126]-1,oy=values[127],norm=std::hypot(ox,oy),allowed=ratio*.99;bool projected=false;if(norm>allowed){values[126]=1+ox*allowed/norm;values[127]=oy*allowed/norm;projected=true;}
        const auto duration=static_cast<std::int64_t>(std::llround(std::exp(std::clamp(values[128],std::log(1e6),std::log(6e10)))));auto* samples=new MGTrajectorySample[64];const auto angle=distance>1e-9?std::atan2(dy,dx):0.0,cs=std::cos(angle),sn=std::sin(angle);std::uint32_t clipped=0;
        for(std::size_t i=0;i<64;++i){double x=q->start_x,y=q->start_y;if(distance>1e-9){const auto cx=values[i*2]*distance,cy=values[i*2+1]*distance;x+=cs*cx-sn*cy;y+=sn*cx+cs*cy;}const auto original_x=x,original_y=y;x=std::clamp(x,q->virtual_desktop_left,q->virtual_desktop_left+q->virtual_desktop_width);y=std::clamp(y,q->virtual_desktop_top,q->virtual_desktop_top+q->virtual_desktop_height);if(x!=original_x||y!=original_y)++clipped;samples[i]={static_cast<std::int64_t>(std::llround(static_cast<double>(duration)*i/63)),x,y};}
        samples[0].x=q->start_x;samples[0].y=q->start_y;samples[0].relative_time_ns=0;samples[63].relative_time_ns=duration;
        *out={samples,64,duration,samples[63].x,samples[63].y,q->click_requested,static_cast<std::uint8_t>(nearest>6.0),static_cast<std::uint8_t>(projected),nearest,q->random_seed,clipped};r->error.clear();return MG_OK;
    }catch(const std::exception& e){set_error(r,e.what());return MG_RUNTIME_ERROR;}
}
extern "C" void mg_free_trajectory(MGTrajectory* value){if(value){delete[] value->samples;*value={};}}
extern "C" void mg_destroy(MGHandle handle){delete static_cast<Runtime*>(handle);}
extern "C" const char* mg_last_error(MGHandle handle){auto* r=static_cast<Runtime*>(handle);return r?r->error.c_str():global_error.c_str();}

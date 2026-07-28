#include "mousemotionlab/mousegen_runtime.h"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
int main(int argc,char** argv){
    if(argc!=12&&argc!=13){std::cerr<<"usage: mousegen_cli MODEL start_x start_y target_x target_y radius seed left top width height [exact]\n";return 2;}
    try{const auto load_start=std::chrono::steady_clock::now();mousemotionlab::Generator generator(argv[1]);const auto load_end=std::chrono::steady_clock::now();MGGenerationRequest q{};q.start_x=std::stod(argv[2]);q.start_y=std::stod(argv[3]);q.target_center_x=std::stod(argv[4]);q.target_center_y=std::stod(argv[5]);q.target_radius=std::stod(argv[6]);q.random_seed=std::stoull(argv[7]);q.virtual_desktop_left=std::stod(argv[8]);q.virtual_desktop_top=std::stod(argv[9]);q.virtual_desktop_width=std::stod(argv[10]);q.virtual_desktop_height=std::stod(argv[11]);q.dpi_scale=1;q.output_rate_hz=250;
      if(argc>12&&std::string(argv[12])=="exact")q.endpoint_mode=MG_ENDPOINT_EXACT_CENTER;MGTrajectory out{};const auto generate_start=std::chrono::steady_clock::now();if(mg_generate(generator.get(),&q,&out)!=MG_OK){std::cerr<<mg_last_error(generator.get())<<'\n';return 1;}const auto generate_end=std::chrono::steady_clock::now();std::cout<<std::setprecision(17)<<"{\"load_ms\":"<<std::chrono::duration<double,std::milli>(load_end-load_start).count()<<",\"generation_ms\":"<<std::chrono::duration<double,std::milli>(generate_end-generate_start).count()<<",\"duration_ns\":"<<out.movement_duration_ns<<",\"distance\":"<<out.condition_distance_score<<",\"points\":[";for(std::size_t i=0;i<out.sample_count;++i){if(i)std::cout<<',';std::cout<<'['<<out.samples[i].relative_time_ns<<','<<out.samples[i].x<<','<<out.samples[i].y<<']';}std::cout<<"]}\n";mg_free_trajectory(&out);return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

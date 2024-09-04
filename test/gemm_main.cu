#include <map>
#include <functional>
#include <fstream>
#include <inttypes.h>
#include "gemm_test.h"

#define CASE_STRING_FMT() "m%" PRId64 "n%" PRId64 "k%" PRId64 "_case%" PRId64

#define CHECK_RC(call) do { \
    ppl::common::RetCode __status = (call); \
    if (__status != ppl::common::RC_SUCCESS) { \
        return 1; \
    } \
} while (0)


static const std::unordered_map<std::string, std::function<gemm_test_base*()>> gemm_test_map = {
    {"i8i8", []() { return new gemm_test_i8i8(); }},
    {"w4a16", []() { return new gemm_test_w4a16(); }},
    {"i8i8col32", []() { return new gemm_test_i8i8col32(); }},
#ifdef PPLNN_ENABLE_FP8
    {"fp8", []() { return new gemm_test_fp8(); }},
#endif
    {"fp16", []() { return new gemm_test_fp16(); }}
};

struct Argument {
    std::string name;
    std::string value;
};

std::unordered_map<std::string, std::string> args;
static const std::vector<Argument> default_arguments = {
    {"cfg", ""},
    {"gemm_type", ""},
    {"warmup", "4"},
    {"benchmark", "20"}
};


bool loop_check(ppl::common::RetCode status) {
    if (status != ppl::common::RC_SUCCESS) {
        std::cerr << "Error: LOOP_CHECK_RC " << status << std::endl;
        return false;
    }
    return true;
}


bool parse_param(int argc, char **argv) {
    if(argc < 2) {
        return false;
    }
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 < argc) {
            std::string argName = argv[i];
            if (argName.size() >= 2 && argName.substr(0, 2) == "--") {
                argName = argName.substr(2);
            }
            args[argName] = argv[i + 1];
        } else {
            std::cerr << "Error: Argument " << argv[i] << " requires a value." << std::endl;
            return false;
        }
    }

    for (const auto& argument : default_arguments) {
        args[argument.name] = args[argument.name].empty() ? argument.value : args[argument.name];
        if (args[argument.name].empty()) {
            std::cerr << "Error: Missing required argument: " << argument.name << std::endl;
            return false;
        }
    }

    return true; 
}


int main(int argc, char **argv) {
    if(!parse_param(argc, argv)) {
        std::cerr << "Usage: test_gemm --cfg <config_file> --gemm_type <gemm_type> [--warmup 4] [--benchmark 10]" << std::endl;
        std::cerr << "--gemm_type   i8i8  i8i8col32  fp16  fp8  w4a16" << std::endl;
        return 1;
    }


    std::cerr << "==============================================================" << std::endl;
    std::cerr << "Reading the config file..." << std::endl;
    std::ifstream cfgfile(args["cfg"], std::ios_base::in | std::ios_base::binary);
    if (!cfgfile.is_open()) {
        std::cerr << "Error opening the config file." << std::endl;
        return 1;
    }


    auto gemm_it = gemm_test_map.find(args["gemm_type"]);
    if (gemm_it == gemm_test_map.end()) {
        std::cerr << "Invalid test type entered." << std::endl;
        return 1;
    }
    gemm_test_base* gemm_test = gemm_it->second();
    CHECK_RC(gemm_test->init_cuda());
    CHECK_RC(gemm_test->init_handle());


    std::string gemm_type = args["gemm_type"];
    std::string output_file = "test_output_" + gemm_type + ".csv";
    std::ofstream csv_file(output_file, std::ios::out);
    if (!csv_file.is_open()) {
        std::cerr << "Error opening CSV file for writing." << std::endl;
        return 1;
    }
    csv_file << "M,N,K,TotalExecTime(ms),AvgExecTime(ms),AvgGFLOPS,AvgGBPS" << std::endl;
    
 
    std::cerr << "==============================================================" << std::endl;
    int64_t line_no = 0;
    int64_t case_no = 0;
    double all_case_gflops = 0.;
    double all_case_gbps = 0.;
    double all_case_time = 0.;
    std::string line;
    while (std::getline(cfgfile, line)) {
        line_no++;
        int64_t M, N, K, case_name;
        if (4 != sscanf(line.c_str(), CASE_STRING_FMT(), &M, &N, &K, &case_name)) {
            std::cerr << line_no << "," << line << ",invalid format" << std::endl;
            continue;
        }
        std::cerr << "Testing the case: M = " << M << ", N = " << N <<", K = " << K << "." << std::endl;

        gemm_test->init_params(M, N, K);
        gemm_test->generate_data();
        if (!loop_check(gemm_test->init_device_data())) continue;
        if (!loop_check(gemm_test->init_scale())) continue;
        if (!loop_check(gemm_test->warmup(std::stoll(args["warmup"])))) continue;
        if (!loop_check(gemm_test->benchmark(std::stoll(args["benchmark"])))) continue;
        gemm_test->deallocate_host_mem();
        if (!loop_check(gemm_test->deallocate_cuda_mem())) continue;
        if (!loop_check(gemm_test->destroy_scale())) continue;

        Result result = gemm_test->get_result();
        csv_file << M << "," << N << "," << K << ","
        << result.tot_exe_time << "," 
        << result.avg_exe_time << "," 
        << result.avg_gflops << "," 
        << result.avg_gbps << std::endl;

        case_no++;
        all_case_gflops += result.avg_gflops;
        all_case_gbps += result.avg_gbps;
        all_case_time += result.avg_exe_time;  
    }

    std::cerr
        << "tot time(ms): "<< all_case_time << "\t"
        << "avg gflops: " << all_case_gflops / case_no  << "\t"
        << "avg gbps: " << all_case_gbps / case_no << std::endl;

    cfgfile.close();
    csv_file.close();
    CHECK_RC(gemm_test->destroy_cuda());
    CHECK_RC(gemm_test->destroy_handle());
    delete gemm_test;

    return 0;
}
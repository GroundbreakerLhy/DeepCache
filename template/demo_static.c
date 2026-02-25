#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/c_runtime_api.h>

void tvm_runtime_set_input(TVMModuleHandle mod, const char *name, DLTensor *input_tensor)
{
    TVMFunctionHandle f_set_input;
    int ret_type_code;
    TVMValue ret_val;
    TVMValue args[2];
    int type_codes[2];

    if (TVMModGetFunction(mod, "set_input", 0, &f_set_input) != 0)
    {
        fprintf(stderr, "Cannot find set_input function\n");
        exit(1);
    }

    args[0].v_str = (const char *)name;
    type_codes[0] = kTVMStr;
    args[1].v_handle = input_tensor;
    type_codes[1] = kTVMDLTensorHandle;

    if (TVMFuncCall(f_set_input, args, type_codes, 2, &ret_val, &ret_type_code) != 0)
    {
        fprintf(stderr, "set_input failed for %s\n", name);
        exit(1);
    }
}

static char *read_file(const char *filename, size_t *out_size)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(size + 1);
    if (buf)
    {
        fread(buf, 1, size, f);
        buf[size] = '\0';
    }
    fclose(f);
    if (out_size)
        *out_size = size;
    return buf;
}

#define OUTPUT_LEN 1000

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <input_bin>\n", argv[0]);
        return 1;
    }

    int64_t shape[4] = {1, 3, 224, 224};

    TVMModuleHandle mod_syslib;
    if (TVMModLoadFromFile("build/model.so", "so", &mod_syslib) != 0)
    {
        fprintf(stderr, "Failed to load model.so: %s\n", TVMGetLastError());
        return 1;
    }

    size_t graph_size, params_size;
    char *graph_json = read_file("build/graph_c.json", &graph_size);
    char *params_data = read_file("build/model.params", &params_size);

    if (!graph_json || !params_data)
    {
        fprintf(stderr, "Failed to load graph or params\n");
        return 1;
    }

    TVMFunctionHandle f_create;
    TVMFuncGetGlobal("tvm.graph_executor.create", &f_create);
    TVMValue args[4];
    int type_codes[4];
    TVMValue ret_val;
    int ret_type_code;

    args[0].v_str = graph_json;
    type_codes[0] = kTVMStr;
    args[1].v_handle = mod_syslib;
    type_codes[1] = kTVMModuleHandle;
    args[2].v_int64 = kDLCPU;
    type_codes[2] = kTVMArgInt;
    args[3].v_int64 = 0;
    type_codes[3] = kTVMArgInt;

    TVMFuncCall(f_create, args, type_codes, 4, &ret_val, &ret_type_code);
    TVMModuleHandle handle = ret_val.v_handle;

    TVMFunctionHandle f_load_params;
    TVMModGetFunction(handle, "load_params", 0, &f_load_params);
    TVMByteArray params_arr = {params_data, params_size};
    args[0].v_handle = &params_arr;
    type_codes[0] = kTVMBytes;
    TVMFuncCall(f_load_params, args, type_codes, 1, &ret_val, &ret_type_code);

    int64_t total_elements = 1;

    DLTensor input;
    input.data = NULL;
    input.device = (DLDevice){kDLCPU, 0};
#ifdef NLP_MODEL
    input.dtype = (DLDataType){kDLInt, 64, 1};  // int64 for NLP/LLM
#else
    input.dtype = (DLDataType){kDLFloat, 32, 1};  // float32 for Vision
#endif
    input.strides = NULL;
    input.byte_offset = 0;

    // Python 脚本替换目标
    input.ndim = 4;
    input.shape = shape;

    for (int i = 0; i < input.ndim; i++)
        total_elements *= shape[i];

#ifdef NLP_MODEL
    int64_t *input_data = (int64_t *)malloc(total_elements * sizeof(int64_t));
    FILE *f_input = fopen(argv[1], "rb");
    if (f_input)
    {
        fread(input_data, sizeof(int64_t), total_elements, f_input);
        fclose(f_input);
    }
    else
    {
        for (int i = 0; i < total_elements; i++)
            input_data[i] = 1;  // Token ID
    }
#else
    float *input_data = (float *)malloc(total_elements * sizeof(float));
    FILE *f_input = fopen(argv[1], "rb");
    if (f_input)
    {
        fread(input_data, sizeof(float), total_elements, f_input);
        fclose(f_input);
    }
    else
    {
        for (int i = 0; i < total_elements; i++)
            input_data[i] = 0.1f;
    }
#endif
    input.data = input_data;

    tvm_runtime_set_input(handle, "data_0", &input);

    TVMFunctionHandle f_run;
    TVMModGetFunction(handle, "run", 0, &f_run);

#ifndef NUM_RUNS
#define NUM_RUNS 50
#endif
    for (int r = 0; r < NUM_RUNS; r++)
        TVMFuncCall(f_run, NULL, NULL, 0, &ret_val, &ret_type_code);

    free(input_data);
    free(graph_json);
    free(params_data);
    return 0;
}
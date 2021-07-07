
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>

#include <iostream>
#include <stdio.h>
#define DATATYPE float
#define arraySize 5
#define threadnum 16
#define blocknum 8
#define arrayNsize 10
#define arrayMsize 15
#define single 1
cudaError_t addWithCuda(int* c, int* a, int* b, int size);
//示例程序
__global__ void addKernel(int *c, int *a, int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

//单block单thread向量加法
__global__ void vector_add_gpu_1(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

//单block多thread向量加法
__global__ void vector_add_gpu_2(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n) {
    int tid = threadIdx.x;
    const int t_n = blockDim.x;//线程总数
    while (tid < n) {
        c[tid] = a[tid] + b[tid];
        tid += t_n;
    }
}

//多block多thread向量加法
__global__ void vector_add_gpu_3(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n) {
    //全局线程索引：tid=blockIdx.x*blockDim.x+threadIdx.x，跳步大小：gird内所有thread数量(gridDim.x*blockDim.x)
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int t_n = gridDim.x * blockDim.x;
    int tid = bidx * blockDim.x + tidx;
    
    while (tid < n) {
        c[tid] = a[tid] + b[tid];
        tid += t_n;
    }
}
//m×n矩阵串行计算
void vector_add_mn(DATATYPE** a, DATATYPE** b, DATATYPE** c, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
}

//维度为m × n的矩阵加法并行计算
__global__ void vector_add_gpu_4(DATATYPE (* a)[arrayNsize], DATATYPE(* b)[arrayNsize], DATATYPE(* c)[arrayNsize], int m, int n) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    while (tid < m && tidy < m) {
        c[tid][tidy] = a[tid][tidy] + b[tid][tidy];
    }
}
//CPU串行向量内积运算
void vector_dot_product_serial(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n) {
    double temp = 0;
    for (int i = 0; i < n; ++i) {
        temp += a[i] * b[i];
    }
    *c = temp;
}

//GPU分散归约向量内积
__global__ void vector_dot_product_gpu_1(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n) {
    __shared__ DATATYPE tmp[threadnum];
    const int tidx = threadIdx.x;
    const int t_n = blockDim.x;
    int tid = tidx;
    double temp = 0.0;
    while (tid < n) {
        temp += a[tid] * b[tid];
        tid += t_n;
        tmp[tidx] = temp;
        __syncthreads();
        int i = 2, j = 1;
        while (i <= threadnum) {
            if ((tidx % i) == 0) {
                tmp[tidx] += tmp[tidx + j];
            }
            __syncthreads();
            i *= 2;
            j *= 2;
        }
        if (tidx == 0) {
            c[0] = tmp[0];
        }
    }

}

//单block低线程归约向量内积
__global__ void vector_dot_product_gpu_2(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n) {
    __shared__ DATATYPE tmp[threadnum];
    const int tidx = threadIdx.x;
    const int t_n = blockDim.x;
    int tid = tidx;
    double temp = 0.0;
    while (tid < n) {
        temp += a[tid] * b[tid];
        tid += t_n;
    }
    tmp[tidx] = temp;
    __syncthreads();
    int i = threadnum / 2;
    while (i != 0) {
        if (tidx < i) {
            tmp[tidx] += tmp[tidx + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (tidx == 0) {
        c[0] = tmp[0];
    }
}

//多block向量内积（CPU二次归约）
__global__ void vector_dot_product_gpu_3(DATATYPE* a, DATATYPE* b, DATATYPE* c_tmp, int n) {
    __shared__ DATATYPE tmp[threadnum];
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int t_n = blockDim.x * gridDim.x;
    int tid = bidx * blockDim.x + tidx;
    double temp = 0.0;
    while (tid < n) {
        temp += a[tid] * b[tid];
        tid += t_n;
    }
    tmp[tidx] = temp;
    __syncthreads();
    int i = threadnum / 2;
    while (i != 0) {
        if (tidx < i) {
            tmp[tidx] += tmp[tidx + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (tidx == 0) {
        c_tmp[bidx] = tmp[0];
    }
}

//GPU归约
__global__ void vector_dot_product_gpu_4 (float* result_tmp, float* result) {
    __shared__ float temp[blocknum];
    const int tidx = threadIdx.x;
    temp[tidx] = result_tmp[tidx];
    __syncthreads();
    int i = blocknum / 2;
    while (i != 0) {
        if (tidx < i) {
            temp[tidx] += temp[tidx + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (tidx == 0) {
        result[0] = temp[0];
    }
}

//原子操作多block向量内积(两次归约替换一次原子操作）
__global__ void vector_dot_product_gpu_5_0(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n) {
    if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
        c[0] = 0.0;
    }
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int t_n = blockDim.x * gridDim.x;
    int tid = bidx * blockDim.x + tidx;
    double temp = 0.0;
    while (tid < n) {
        temp += a[tid] * b[tid];
        tid += t_n;
    }
    atomicAdd(c, temp);
}
//原子操作多block向量内积（block内归约block间原子操作）
__global__ void vector_dot_product_gpu_5(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n) {
    if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
        c[0] = 0.0;
    }
    __shared__ DATATYPE tmp[threadnum];
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int t_n = blockDim.x * gridDim.x;
    int tid = bidx * blockDim.x + tidx;
    double temp = 0.0;
    while (tid < n) {
        temp += a[tid] * b[tid];
        tid += t_n;
    }
    tmp[tidx] = temp;
    __syncthreads();
    int i = blockDim.x / 2;
    while (i != 0) {
        if (tidx < i) {
            tmp[tidx] += tmp[tidx + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (tidx == 0) {
        atomicAdd(c, tmp[0]);
    }

}

//串行向量加法
void vector_add_serial(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}
int main()
{
    //int arraySize = 5;
    //dim3 blocknum(1);
    //dim3 threadnum(arrayMsize,arrayNsize);
    float a[arraySize] = { 1, 2, 3, 4, 5 };
    float b[arraySize] = { 10, 20, 30, 40, 50 };
    float c[arraySize] = { 0 };
    float c1[arraySize] = { 0 };
    float **aa;
    float **bb;
    float **cc;
    aa = (float**)malloc(sizeof(float*) * arrayMsize);
    bb = (float**)malloc(sizeof(float*) * arrayMsize);
    cc = (float**)malloc(sizeof(float*) * arrayMsize);
    for (int i = 0; i < arrayMsize; ++i) {
        aa[i] = (float*)malloc(sizeof(float*) * arrayNsize);
        bb[i] = (float*)malloc(sizeof(float*) * arrayNsize);
        cc[i] = (float*)malloc(sizeof(float*) * arrayNsize);

    }
    for (int i = 0; i < arrayMsize; ++i) {
        for (int j = 0; j < arrayNsize; ++j) {
            aa[i][j] = j;
            bb[i][j] = j * 10;
            cc[i][j] = 0;
        }
    }
    //串行测试
    vector_add_serial(a, b, c, arraySize);
    //printf("serial :{1,2,3,4,5} + {10,20,30,40,50} = {%f,%f,%f,%f,%f}\n",c[0], c[1], c[2], c[3], c[4]);
    for (int i = 0; i < arraySize; ++i) {
        c[i] = 0;
    }

    //单block单thread加法测试
        //GPU内存分配
    DATATYPE* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_b, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_c, sizeof(DATATYPE) * arraySize);
        //复制数据到GPU
    cudaMemcpy(d_a, a, sizeof(DATATYPE) * arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(DATATYPE) * arraySize, cudaMemcpyHostToDevice);
        //计算
    vector_add_gpu_1<<<single,single>>>(d_a, d_b, d_c, arraySize);
        //复制结果到CPU
    cudaMemcpy(c, d_c, sizeof(DATATYPE) * arraySize, cudaMemcpyDeviceToHost);
        //释放空间
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    //printf("single block single thread :{1,2,3,4,5} + {10,20,30,40,50} = {%f,%f,%f,%f,%f}\n",c[0], c[1], c[2], c[3], c[4]);
    for (int i = 0; i < arraySize; ++i) {
        c[i] = 0;
    }

    //单block多thread加法
    cudaMalloc((void**)&d_a, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_b, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_c, sizeof(DATATYPE) * arraySize);
        //复制数据到GPU
    cudaMemcpy(d_a, a, sizeof(DATATYPE) * arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(DATATYPE) * arraySize, cudaMemcpyHostToDevice);
        //计算
    vector_add_gpu_2 <<<1, threadnum >>> (d_a, d_b, d_c, arraySize);
        //复制结果到CPU
    cudaMemcpy(c, d_c, sizeof(DATATYPE) * arraySize, cudaMemcpyDeviceToHost);
        //释放空间
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    //printf("single block multiple thread :{1,2,3,4,5} + {10,20,30,40,50} = {%f,%f,%f,%f,%f}\n",c[0], c[1], c[2], c[3], c[4]);
    for (int i = 0; i < arraySize; ++i) {
        c[i] = 0;
    }

    //多block多thread加法
    cudaMalloc((void**)&d_a, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_b, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_c, sizeof(DATATYPE) * arraySize);
        //复制数据到GPU
    cudaMemcpy(d_a, a, sizeof(DATATYPE) * arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(DATATYPE) * arraySize, cudaMemcpyHostToDevice);
        //计算
    vector_add_gpu_3 <<<blocknum, threadnum >>> (d_a, d_b, d_c, arraySize);
        //复制结果到CPU
    cudaMemcpy(c, d_c, sizeof(DATATYPE) * arraySize, cudaMemcpyDeviceToHost);
        //释放空间
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    //printf("multiple block multiple thread :{1,2,3,4,5} + {10,20,30,40,50} = {%f,%f,%f,%f,%f}\n",c[0], c[1], c[2], c[3], c[4]);
    for (int i = 0; i < arraySize; ++i) {
        c[i] = 0;
    }
    //cublas库向量加法
    DATATYPE* d_aa, * d_bb;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaMalloc((void**)&d_aa, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_bb, sizeof(DATATYPE) * arraySize);
    float alpha = 1.0;
    cublasSetVector(arraySize, sizeof(DATATYPE), a, 1, d_aa, 1);
    cublasSetVector(arraySize, sizeof(DATATYPE), b, 1, d_bb, 1);
    cublasSaxpy_v2(handle, arraySize, &alpha, d_aa, 1, d_bb, 1);
    cublasGetVector(arraySize, sizeof(DATATYPE), d_bb, 1, c1, 1);
    cudaFree(d_aa);
    cudaFree(d_bb);
    cublasDestroy(handle);
    //printf("cublas :{1,2,3,4,5} + {10,20,30,40,50} = {%f,%f,%f,%f,%f}\n",c1[0], c1[1], c1[2], c1[3], c1[4]);
    for (int i = 0; i < arraySize; ++i) {
        c1[i] = 0;
    }

    //m×n矩阵并行加法
    /*DATATYPE(*d_aaa)[arrayNsize], (*d_bbb)[arrayNsize], (*d_ccc)[arrayNsize];
    cudaMalloc((void**)&d_aaa, sizeof(DATATYPE) * arrayMsize * arrayNsize);
    cudaMalloc((void**)&d_bbb, sizeof(DATATYPE) * arrayMsize * arrayNsize);
    cudaMalloc((void**)&d_ccc, sizeof(DATATYPE) * arrayMsize * arrayNsize);
    cudaMemcpy(d_aaa, aa, sizeof(DATATYPE) * arrayNsize * arrayMsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bbb, bb, sizeof(DATATYPE)* arrayNsize* arrayMsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ccc, cc, sizeof(DATATYPE)* arrayNsize* arrayMsize, cudaMemcpyHostToDevice);
    vector_add_gpu_4 << <blocknum, threadnum >> > (d_aaa, d_bbb, d_ccc, arrayMsize, arrayNsize);
    cudaMemcpy(cc, d_ccc, sizeof(DATATYPE)* arrayNsize*arrayMsize, cudaMemcpyDeviceToHost);
    cudaFree(d_aaa);
    cudaFree(d_bbb);
    cudaFree(d_ccc);
    std::cout << "m×n matrix add\n";
    for (int i = 0; i < arrayMsize; ++i) {
        for (int j = 0; j < arrayNsize; ++j) {
            std::cout << cc[i][j] << " ";
            cc[i][j] = 0;
        }
        std::cout << "\n";
    }
    */
    //m×n加法结果验证
    vector_add_mn(aa, bb, cc, arrayMsize, arrayNsize);
    //std::cout << "m×n matrix valid\n";
    /*for (int i = 0; i < arrayMsize; ++i) {
        for (int j = 0; j < arrayNsize; ++j) {
            std::cout << cc[i][j] << " ";
            cc[i][j] = 0;
        }
        std::cout << "\n";
    }*/

    //单block分散归约向量内积
    DATATYPE* d_cccc,*d_ca;
    DATATYPE ccccd,*cccc;
    cccc = &ccccd;
    cudaMalloc((void**)&d_a, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_b, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_cccc, sizeof(DATATYPE));
    //复制数据到GPU
    cudaMemcpy(d_a, a, sizeof(DATATYPE) * arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(DATATYPE) * arraySize, cudaMemcpyHostToDevice);
    //计算
    vector_dot_product_gpu_1 << <single, threadnum >> > (d_a, d_b, d_c, arraySize);
    //复制结果到CPU
    cudaMemcpy(cccc, d_c, sizeof(DATATYPE), cudaMemcpyDeviceToHost);
    //释放空间
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    printf("single block:{1,2,3,4,5} · {10,20,30,40,50} = {%f}\n",ccccd);
    cccc = 0;

    //单block低线程归约向量内积
    cudaMalloc((void**)&d_a, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_b, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_c, sizeof(DATATYPE) * arraySize);
    //复制数据到GPU
    cudaMemcpy(d_a, a, sizeof(DATATYPE) * arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(DATATYPE) * arraySize, cudaMemcpyHostToDevice);
    //计算
    vector_dot_product_gpu_2 << <single, threadnum >> > (d_a, d_b, d_c, arraySize);
    //复制结果到CPU
    cudaMemcpy(c, d_c, sizeof(DATATYPE) * arraySize, cudaMemcpyDeviceToHost);
    //释放空间
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    printf("single block:{1,2,3,4,5} · {10,20,30,40,50} = {%f}\n",c[0]);
    c[0] = 0;

    //多block向量内积（CPU二次归约）
    cudaMalloc((void**)&d_a, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_b, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_c, sizeof(DATATYPE) * arraySize);
    //复制数据到GPU
    cudaMemcpy(d_a, a, sizeof(DATATYPE) * arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(DATATYPE) * arraySize, cudaMemcpyHostToDevice);
    //计算
    vector_dot_product_gpu_3 << <blocknum, threadnum >> > (d_a, d_b, d_c, arraySize);
    //复制结果到CPU
    cudaMemcpy(c, d_c, sizeof(DATATYPE) * arraySize, cudaMemcpyDeviceToHost);
    //释放空间
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    double temp=0;
    for (int i = 0; i < blocknum&&i<arraySize; i++) {
        if(c[i]!=NULL)
        temp += c[i];
    }
    c[0] = temp;
    printf("multiple block(CPU):{1,2,3,4,5} · {10,20,30,40,50} = {%f}\n", c[0]);
    c[0] = 0;
    for (int i = 0; i < arraySize; ++i) {
        c[i] = 0;
    }
    //多block向量内积（GPU二次归约）
    cudaMalloc((void**)&d_a, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_b, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_c, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_ca, sizeof(DATATYPE) * arraySize);
    //复制数据到GPU
    cudaMemcpy(d_a, a, sizeof(DATATYPE) * arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(DATATYPE) * arraySize, cudaMemcpyHostToDevice);
    //计算
    vector_dot_product_gpu_3 << <blocknum, threadnum >> > (d_a, d_b, d_c, arraySize);
    vector_dot_product_gpu_4 <<< 1, blocknum >> > (d_c, d_ca);
    //复制结果到CPU
    cudaMemcpy(c, d_ca, sizeof(DATATYPE) * arraySize, cudaMemcpyDeviceToHost);
    //释放空间
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_ca);
    printf("multiple block(GPU):{1,2,3,4,5} · {10,20,30,40,50} = {%f}\n", c[0]);
    c[0] = 0;
    for (int i = 0; i < arraySize; ++i) {
        c[i] = 0;
    }

    //原子操作
    cudaMalloc((void**)&d_a, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_b, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_c, sizeof(DATATYPE) * arraySize);
    //复制数据到GPU
    cudaMemcpy(d_a, a, sizeof(DATATYPE) * arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(DATATYPE) * arraySize, cudaMemcpyHostToDevice);
    //计算
    vector_dot_product_gpu_5_0 << <blocknum, threadnum >> > (d_a, d_b, d_c, arraySize);
    //复制结果到CPU
    cudaMemcpy(c, d_c, sizeof(DATATYPE) * arraySize, cudaMemcpyDeviceToHost);
    //释放空间
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    printf("atomic_0 multiple block(GPU):{1,2,3,4,5} · {10,20,30,40,50} = {%f}\n", c[0]);
    c[0] = 0;
    for (int i = 0; i < arraySize; ++i) {
        c[i] = 0;
    }

    cudaMalloc((void**)&d_a, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_b, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_c, sizeof(DATATYPE) * arraySize);
    //复制数据到GPU
    cudaMemcpy(d_a, a, sizeof(DATATYPE) * arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(DATATYPE) * arraySize, cudaMemcpyHostToDevice);
    //计算
    vector_dot_product_gpu_5 << <blocknum, threadnum >> > (d_a, d_b, d_c, arraySize);
    //复制结果到CPU
    cudaMemcpy(c, d_c, sizeof(DATATYPE) * arraySize, cudaMemcpyDeviceToHost);
    //释放空间
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    printf("atomic_1 multiple block(GPU):{1,2,3,4,5} · {10,20,30,40,50} = {%f}\n", c[0]);
    c[0] = 0;
    for (int i = 0; i < arraySize; ++i) {
        c[i] = 0;
    }


    // Add vectors in parallel.
    /*cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%f,%f,%f,%f,%f}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    */
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, int *a, int *b, int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

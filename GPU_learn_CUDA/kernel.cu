
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
#define arraysize 5
#define single 1
//第八章 矩阵乘法
#define arraysizeM 10
#define arraysizeL 10
#define arraysizeN 10
#define threadnx 2

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

//计数法实现多block向量内积
__device__ void vector_dot(DATATYPE* out, volatile DATATYPE* tmp) {
    const int tidx = threadIdx.x;
    int i = blockDim.x / 2;
    while (i != 0) {
        if (tidx < i) {
            tmp[tidx] += tmp[tidx + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (tidx == 0) {
        out[0] = tmp[0];
    }
}
__device__ unsigned int lockcount = 0;
__global__ void vector_dot_product_gpu_6(DATATYPE* a, DATATYPE* b, DATATYPE* c_tmp, DATATYPE* c, int n) {
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
    vector_dot(&c_tmp[blockIdx.x], tmp);
    __shared__ bool lock;
    __threadfence();
    if (tidx == 0) {
        unsigned int lockiii = atomicAdd(&lockcount, 1);
        lock = (lockcount == gridDim.x);
    }
    __syncthreads();
    if (lock) {
        tmp[tidx] = c_tmp[tidx];
        __syncthreads();
        vector_dot(c, tmp);
        lockcount = 0;
    }

}

//串行向量加法
void vector_add_serial(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}
/*
int main()
{
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
    DATATYPE* d_a, * d_b, * d_c,* d_c_tmp;
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
    //DATATYPE(*d_aaa)[arrayNsize], (*d_bbb)[arrayNsize], (*d_ccc)[arrayNsize];
    //cudaMalloc((void**)&d_aaa, sizeof(DATATYPE) * arrayMsize * arrayNsize);
    //cudaMalloc((void**)&d_bbb, sizeof(DATATYPE) * arrayMsize * arrayNsize);
    //cudaMalloc((void**)&d_ccc, sizeof(DATATYPE) * arrayMsize * arrayNsize);
    //cudaMemcpy(d_aaa, aa, sizeof(DATATYPE) * arrayNsize * arrayMsize, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_bbb, bb, sizeof(DATATYPE)* arrayNsize* arrayMsize, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_ccc, cc, sizeof(DATATYPE)* arrayNsize* arrayMsize, cudaMemcpyHostToDevice);
    //vector_add_gpu_4 << <blocknum, threadnum >> > (d_aaa, d_bbb, d_ccc, arrayMsize, arrayNsize);
    //cudaMemcpy(cc, d_ccc, sizeof(DATATYPE)* arrayNsize*arrayMsize, cudaMemcpyDeviceToHost);
    //cudaFree(d_aaa);
    //cudaFree(d_bbb);
    //cudaFree(d_ccc);
    //std::cout << "m×n matrix add\n";
    //for (int i = 0; i < arrayMsize; ++i) {
    //    for (int j = 0; j < arrayNsize; ++j) {
    //        std::cout << cc[i][j] << " ";
    //        cc[i][j] = 0;
    //    }
    //    std::cout << "\n";
    //}
    //m×n加法结果验证
    vector_add_mn(aa, bb, cc, arrayMsize, arrayNsize);
    //std::cout << "m×n matrix valid\n";
    //for (int i = 0; i < arrayMsize; ++i) {
    //    for (int j = 0; j < arrayNsize; ++j) {
    //        std::cout << cc[i][j] << " ";
    //        cc[i][j] = 0;
    //    }
    //    std::cout << "\n";
    //}

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
    //printf("single block:{1,2,3,4,5} · {10,20,30,40,50} = {%f}\n",ccccd);
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
    //printf("single block:{1,2,3,4,5} · {10,20,30,40,50} = {%f}\n",c[0]);
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
    //printf("multiple block(CPU):{1,2,3,4,5} · {10,20,30,40,50} = {%f}\n", c[0]);
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
    //printf("multiple block(GPU):{1,2,3,4,5} · {10,20,30,40,50} = {%f}\n", c[0]);
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
    //printf("atomic_0 multiple block(GPU):{1,2,3,4,5} · {10,20,30,40,50} = {%f}\n", c[0]);
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
    //printf("atomic_1 multiple block(GPU):{1,2,3,4,5} · {10,20,30,40,50} = {%f}\n", c[0]);
    c[0] = 0;
    for (int i = 0; i < arraySize; ++i) {
        c[i] = 0;
    }

    //计数法实现多block向量内积
    cudaMalloc((void**)&d_a, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_b, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_c, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_c_tmp, sizeof(DATATYPE) * arraySize);
    //复制数据到GPU
    cudaMemcpy(d_a, a, sizeof(DATATYPE) * arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(DATATYPE) * arraySize, cudaMemcpyHostToDevice);
    //计算
    vector_dot_product_gpu_6 << <blocknum, threadnum >> > (d_a, d_b,d_c_tmp, d_c, arraySize);
    //复制结果到CPU
    cudaMemcpy(c, d_c, sizeof(DATATYPE) * arraySize, cudaMemcpyDeviceToHost);
    //释放空间
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_c_tmp);
    //printf("counting method multiple block(GPU):{1,2,3,4,5} · {10,20,30,40,50} = {%f}\n", c[0]);
    c[0] = 0;
    for (int i = 0; i < arraySize; ++i) {
        c[i] = 0;
    }

    //cublas库向量内积
    cublasCreate(&handle);
    cudaMalloc((void**)&d_aa, sizeof(DATATYPE) * arraySize);
    cudaMalloc((void**)&d_bb, sizeof(DATATYPE) * arraySize);
    cublasSetVector(arraySize, sizeof(DATATYPE), a, 1, d_aa, 1);
    cublasSetVector(arraySize, sizeof(DATATYPE), b, 1, d_bb, 1);
    cublasSdot_v2(handle, arraySize, d_aa, 1, d_bb, 1,&c1[0]);
    //cublasGetVector(arraySize, sizeof(DATATYPE), d_bb, 1, c1, 1);
    cudaFree(d_aa);
    cudaFree(d_bb);
    cublasDestroy(handle);
    //printf("cublas :{1,2,3,4,5} · {10,20,30,40,50} = {%f}\n",c1[0]);
    for (int i = 0; i < arraySize; ++i) {
        c1[i] = 0;
    }
    

    // Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}
//
    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%f,%f,%f,%f,%f}\n",
    //    c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}
    
    return 0;
}
*/

//第八章

//a：m行l列，b：l行n列，c：m行n列,串行计算
void matrix_multiplication_serial_1(DATATYPE* a, DATATYPE* b, DATATYPE* c, int m, int n, int l) {
    int i, j, k;
    double temp = 0.0;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            temp = 0.0;
            for (k = 0; k < l; k++) {
                temp += a[i * l + k] * b[k * n + j];
            }
            c[i * n + j] = temp;
        }
    }
}

//循环交换矩阵乘法
void matrix_multiplication_serial_2(DATATYPE* a, DATATYPE* b, DATATYPE* c, int m, int n, int l) {
    int i, j, k;
    double temp = 0.0;
    for (i = 0; i < m; i++) {
        for (k = 0; k < l; k++) {
            temp = a[i * l + k];
            for (j = 0; j < n; j++) {
                c[i * n + j] += temp * b[k * n + j];
            }
        }
    }
}

//转置矩阵乘法
void matrix_multiplication_serial_3(DATATYPE* a, DATATYPE* b, DATATYPE* c, int m, int n, int l) {
    int i, j, k;
    double temp = 0.0;
    DATATYPE* b1;
    b1 = (DATATYPE*)malloc(sizeof(DATATYPE) * l * n);
    for (int i = 0; i < l; i++) {
        for (int j = 0; j < n; j++) {
            b1[i * l + j] = b[j * n + i];
        }
    }
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            temp = 0.0;
            for (k = 0; k < l; k++) {
                temp += a[i * l + k] * b1[j * n + k];
            }
            c[i * n + j] = temp;
        }
    }
    free(b1);
}

//grid线程循环矩阵乘法 A:n行lda列;C:n行ldc列；其中ldb=ldc
__global__ void matrix_multiplication_gpu_1(const DATATYPE* a, size_t lda, const DATATYPE* b, size_t ldb, DATATYPE* c, size_t ldc, int n) {
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int idx = bidx * blockDim.x + tidx;
    const int row = idx / n;
    const int column = idx % n;
    if (row < n && column < ldc) {
        double tmp = 0.0;
        for (int i = 0; i < n; i++) {
            tmp += a[row * lda + i] * b[i * ldb + column];
        }
        c[row * ldc + column] = tmp;
    }

}

//block线程循环矩阵乘法，A：n行lda列，B：lda行ldb列；C：n行ldc列，其中ldb=ldc
__global__ void matrix_multiplication_gpu_1_0(const DATATYPE* a, size_t lda, const DATATYPE* b, size_t ldb, DATATYPE* c, size_t ldc, int n) {
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    double tmp = 0.0;
    int i;
    //for (; bidx < n; bidx += gridDim.x) //若矩阵行数超过block上限，则取消该句注释
    {
        for (tidx = threadIdx.x; tidx < ldc; tidx += blockDim.x) {
            tmp = 0.0;
            for (i = 0; i < lda; i++) {
                tmp += a[bidx * lda + i] * b[i * ldb + tidx];
            }
            c[bidx * ldc + tidx] = tmp;
        }
    }
}

//行共享存储矩阵乘法
__global__ void matrix_multiplication_gpu_2(const DATATYPE* a, size_t lda, const DATATYPE* b, size_t ldb, DATATYPE* c, size_t ldc, int n) {
    extern __shared__ DATATYPE data[];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    int i, j;
    for (i = tid; i < n; i += blockDim.x) {
        data[i] = a[row * lda + i];
    }
    __syncthreads();
    double tmp = 0.0;
    for (j = tid; j < n; j += blockDim.x) {
        tmp = 0.0;
        for (i = 0; i < n; i++) {
            tmp += data[i] * b[i * ldb + j];
            //printf("%lf\n",tmp);
        }
        c[row * ldc + j] = tmp;
    }
}

//棋盘阵列矩阵乘法
__global__ void matrix_multiplication_gpu_3(const DATATYPE* a, size_t lda, const DATATYPE* b, size_t ldb, DATATYPE* c, size_t ldc, int n) {
    __shared__ DATATYPE matA[threadnx][threadnx];
    __shared__ DATATYPE matB[threadnx][threadnx];
    const int tidc = threadIdx.x;
    const int tidr = threadIdx.y;
    const int bidc = blockIdx.x * threadnx;
    const int bidr = blockIdx.y * threadnx;
    int i, j;
    double results = 0.0;
    for (j = 0;j < n;j += threadnx) {
        if (tidr + bidr < n && tidc + j < n) {
            matA[tidr][tidc] = a[(tidr + bidr) * lda + tidc + j];
        }
        else {
            matA[tidr][tidc] = 0;
        }
        if (tidr + j < n && tidc + bidc < n) {
            matB[tidr][tidc] = b[(tidr + j) * ldb + tidc + bidc];
        }
        else {
            matB[tidr][tidc] = 0;
        }
        __syncthreads();
        for (i = 0;i < threadnx;i++) {
            results += matA[tidr][i] * matB[i][tidc];
        }
        __syncthreads();
    }
    if (tidr + bidr < n && tidc + bidc < n) {
        c[(tidr + bidr) * ldc + tidc + bidc] = results;
    }

}

//移除棋盘阵列中的判断
__global__ void matrix_multiplication_gpu_4(const DATATYPE* a, size_t lda, const DATATYPE* b, size_t ldb, DATATYPE* c, size_t ldc, int n) {
    __shared__ DATATYPE matA[threadnx][threadnx];
    __shared__ DATATYPE matB[threadnx][threadnx];
    const int tidc = threadIdx.x;
    const int tidr = threadIdx.y;
    const int bidc = blockIdx.x * threadnx;
    const int bidr = blockIdx.y * threadnx;
    int i, j;
    double results = 0.0;
    for (j = 0;j < n;j += threadnx) {
        matA[tidr][tidc] = a[(tidr + bidr) * lda + tidc + j];
        matB[tidr][tidc] = b[(tidr + j) * ldb + tidc + bidc];
        __syncthreads();
        for (i = 0;i < threadnx;i++) {
            results += matA[tidr][i] * matB[i][tidc];
        }
        __syncthreads();
    }
    if (tidr + bidr < n && tidc + bidc < n) {
        c[(tidr + bidr) * ldc + tidc + bidc] = results;
    }
}

int main() {
    DATATYPE* a, * b, * c, * d_a, * d_b, * d_c, * d_c3, * c3;
    a = (DATATYPE*)malloc(sizeof(DATATYPE*) * arraysizeM * arraysizeL);
    b = (DATATYPE*)malloc(sizeof(DATATYPE*) * arraysizeL * arraysizeN);
    c = (DATATYPE*)malloc(sizeof(DATATYPE*) * arraysizeM * arraysizeN);
    c3 = (DATATYPE*)malloc(sizeof(DATATYPE*) * arraysizeM * arraysizeN);
    for (int i = 0; i < arraysizeM * arraysizeL; i++) {
        a[i] = i;
    }
    for (int i = 0; i < arraysizeL * arraysizeN; i++) {
        b[i] = i;
    }
    for (int i = 0; i < arraysizeM * arraysizeL; i++) {
        c[i] = 0;
    }
    //串行
    matrix_multiplication_serial_1(a, b, c, arraysizeM, arraysizeN, arraysizeL);
    printf("serial a × b = { \n");
    for (int i = 0; i < arraysizeM; i++) {
        for (int j = 0; j < arraysizeN; j++) {
            printf("%7.0f ", c[i * arraysizeM + j]);
        }
        printf("\n");
    }
    for (int i = 0; i < arraysizeM * arraysizeL; i++) {
        c[i] = 0;
    }
    //循环交换
    matrix_multiplication_serial_2(a, b, c, arraysizeM, arraysizeN, arraysizeL);
    printf("serial_mod a × b = { \n");
    for (int i = 0; i < arraysizeM; i++) {
        for (int j = 0; j < arraysizeN; j++) {
            printf("%7.0f ", c[i * arraysizeM + j]);
        }
        printf("\n");
    }
    for (int i = 0; i < arraysizeM * arraysizeL; i++) {
        c[i] = 0;
    }
    //转置矩阵乘法
    matrix_multiplication_serial_3(a, b, c, arraysizeM, arraysizeN, arraysizeL);
    printf("serial_transpose a × b = { \n");
    for (int i = 0; i < arraysizeM; i++) {
        for (int j = 0; j < arraysizeN; j++) {
            printf("%7.0f ", c[i * arraysizeM + j]);
        }
        printf("\n");
    }
    for (int i = 0; i < arraysizeM * arraysizeL; i++) {
        c[i] = 0;
    }

    //grid线程循环矩阵乘法
    cudaMalloc((void**)&d_a, sizeof(DATATYPE) * arraysizeM * arraysizeL);
    cudaMalloc((void**)&d_b, sizeof(DATATYPE) * arraysizeL * arraysizeN);
    cudaMalloc((void**)&d_c, sizeof(DATATYPE) * arraysizeM * arraysizeN);
    cudaMemcpy(d_a, a, sizeof(DATATYPE) * arraysizeM * arraysizeL, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(DATATYPE) * arraysizeL * arraysizeN, cudaMemcpyHostToDevice);
    int blocks = (arraysizeN + threadnum - 1) / threadnum;
    matrix_multiplication_gpu_1 << <blocks * arraysizeN, threadnum >> > (d_a,arraysizeN,d_b,arraysizeN,d_c,arraysizeN,arraysizeN);
    cudaMemcpy(c, d_c, sizeof(DATATYPE) * arraysizeM * arraysizeN, cudaMemcpyDeviceToHost);
    printf("serial_transpose a × b = { \n");
    for (int i = 0; i < arraysizeM; i++) {
        for (int j = 0; j < arraysizeN; j++) {
            printf("%7.0f ", c[i * arraysizeM + j]);
        }
        printf("\n");
    }
    for (int i = 0; i < arraysizeM * arraysizeL; i++) {
        c[i] = 0;
    }

    //block线程循环矩阵乘法
    cudaMalloc((void**)&d_a, sizeof(DATATYPE) * arraysizeM * arraysizeL);
    cudaMalloc((void**)&d_b, sizeof(DATATYPE) * arraysizeL * arraysizeN);
    cudaMalloc((void**)&d_c, sizeof(DATATYPE) * arraysizeM * arraysizeN);
    cudaMemcpy(d_a, a, sizeof(DATATYPE) * arraysizeM * arraysizeL, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(DATATYPE) * arraysizeL * arraysizeN, cudaMemcpyHostToDevice);
    matrix_multiplication_gpu_1_0 << <arraysizeN, threadnum >> > (d_a, arraysizeN, d_b, arraysizeN, d_c, arraysizeN, arraysizeN);
    cudaMemcpy(c, d_c, sizeof(DATATYPE) * arraysizeM * arraysizeN, cudaMemcpyDeviceToHost);
    printf("block thread a × b = { \n");
    for (int i = 0; i < arraysizeM; i++) {
        for (int j = 0; j < arraysizeN; j++) {
            printf("%7.0f ", c[i * arraysizeM + j]);
        }
        printf("\n");
    }
    for (int i = 0; i < arraysizeM * arraysizeL; i++) {
        c[i] = 0;
    }

    //对齐存储
    size_t pitch_a, pitch_b, pitch_c;
    cudaMallocPitch((void**)&d_a, &pitch_a, sizeof(DATATYPE) * arraysizeN, arraysizeN);
    cudaMallocPitch((void**)&d_b, &pitch_b, sizeof(DATATYPE) * arraysizeN, arraysizeN);
    cudaMallocPitch((void**)&d_c3, &pitch_c, sizeof(DATATYPE) * arraysizeN, arraysizeN);
    //printf("%d,%d,%d", pitch_a, pitch_b, pitch_c);
    cudaMemcpy2D(d_a, pitch_a, a, sizeof(DATATYPE) * arraysizeN, sizeof(DATATYPE) * arraysizeN, arraysizeN, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_b, pitch_b, b, sizeof(DATATYPE) * arraysizeN, sizeof(DATATYPE) * arraysizeN, arraysizeN, cudaMemcpyHostToDevice);
    matrix_multiplication_gpu_2 << <arraysizeN, threadnum, sizeof(DATATYPE)* arraysizeN >> > (d_a, pitch_a / sizeof(DATATYPE), d_b, pitch_b / sizeof(DATATYPE), d_c3, pitch_c / sizeof(DATATYPE), arraysizeN);
    cudaMemcpy2D(c3, sizeof(DATATYPE) * arraysizeN, d_c3, pitch_c, sizeof(DATATYPE) * arraysizeN, arraysizeN, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c3);
    printf("aligned storage a × b = { \n");
    for (int i = 0; i < arraysizeM; i++) {
        for (int j = 0; j < arraysizeN; j++) {
            printf("%7.0f ", c3 [i * arraysizeM + j]);
        }
        printf("\n");
    }
    for (int i = 0; i < arraysizeM * arraysizeL; i++) {
        c3 [i] = 0;
    }

    //棋盘阵列矩阵乘法
    int bx = (arraysizeN + threadnx - 1) / threadnx;
    dim3 blockns(bx, bx);
    dim3 threadns(threadnx, threadnx);
    cudaMallocPitch((void**)&d_a, &pitch_a, sizeof(DATATYPE)* arraysizeN, arraysizeN);
    cudaMallocPitch((void**)&d_b, &pitch_b, sizeof(DATATYPE)* arraysizeN, arraysizeN);
    cudaMallocPitch((void**)&d_c3, &pitch_c, sizeof(DATATYPE)* arraysizeN, arraysizeN);
    cudaMemcpy2D(d_a, pitch_a, a, sizeof(DATATYPE)* arraysizeN, sizeof(DATATYPE)* arraysizeN, arraysizeN, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_b, pitch_b, b, sizeof(DATATYPE)* arraysizeN, sizeof(DATATYPE)* arraysizeN, arraysizeN, cudaMemcpyHostToDevice);
    matrix_multiplication_gpu_3 << <blockns, threadns >> > (d_a, pitch_a / sizeof(DATATYPE), d_b, pitch_b / sizeof(DATATYPE), d_c3, pitch_c / sizeof(DATATYPE), arraysizeN);
    cudaMemcpy2D(c3, sizeof(DATATYPE)* arraysizeN, d_c3, pitch_c, sizeof(DATATYPE)* arraysizeN, arraysizeN, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c3);
    printf("checkboard array a × b = { \n");
    for (int i = 0; i < arraysizeM; i++) {
        for (int j = 0; j < arraysizeN; j++) {
            printf("%7.0f ", c3[i * arraysizeM + j]);
        }
        printf("\n");
    }
    for (int i = 0; i < arraysizeM * arraysizeL; i++) {
        c3[i] = 0;
    }

    //优化的棋盘阵列矩阵乘法
    cudaMallocPitch((void**)&d_a, &pitch_a, sizeof(DATATYPE) * arraysizeN, arraysizeN);
    cudaMallocPitch((void**)&d_b, &pitch_b, sizeof(DATATYPE) * arraysizeN, arraysizeN);
    cudaMallocPitch((void**)&d_c3, &pitch_c, sizeof(DATATYPE)* arraysizeN, arraysizeN);
    cudaMemcpy2D(d_a, pitch_a, a, sizeof(DATATYPE)* arraysizeN, sizeof(DATATYPE)* arraysizeN, arraysizeN, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_b, pitch_b, b, sizeof(DATATYPE)* arraysizeN, sizeof(DATATYPE)* arraysizeN, arraysizeN, cudaMemcpyHostToDevice);
    matrix_multiplication_gpu_4 << <blockns, threadns >> > (d_a, pitch_a / sizeof(DATATYPE), d_b, pitch_b / sizeof(DATATYPE), d_c3, pitch_c / sizeof(DATATYPE), arraysizeN);
    cudaMemcpy2D(c3, sizeof(DATATYPE)* arraysizeN, d_c3, pitch_c, sizeof(DATATYPE)* arraysizeN, arraysizeN, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c3);
    printf("improved checkboard a × b = { \n");
    for (int i = 0; i < arraysizeM; i++) {
        for (int j = 0; j < arraysizeN; j++) {
            printf("%7.0f ", c3[i * arraysizeM + j]);
        }
        printf("\n");
    }
    for (int i = 0; i < arraysizeM * arraysizeL; i++) {
        c3[i] = 0;
    }
    //cublas矩阵乘法
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaMalloc((void**)&d_a, sizeof(DATATYPE)* arraysizeN* arraysizeN);
    cudaMalloc((void**)&d_b, sizeof(DATATYPE)* arraysizeN* arraysizeN);
    cudaMalloc((void**)&d_c3, sizeof(DATATYPE)* arraysizeN* arraysizeN);
    float alpha = 1.0;
    float beta = 0.0;
    cublasSetVector(arraysizeN* arraysizeN, sizeof(DATATYPE), a, 1, d_a, 1);
    cublasSetVector(arraysizeN* arraysizeN, sizeof(DATATYPE), b, 1, d_b, 1);
    cublasSetVector(arraysizeN * arraysizeN, sizeof(DATATYPE), c3, 1, d_c3, 1);
    cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, arraysizeN, arraysizeN, arraysizeN, &alpha, d_b, arraysizeN, d_a, arraysizeN, &beta, d_c3, arraysizeN);
    cublasGetVector(arraysizeN * arraysizeN, sizeof(DATATYPE), d_c3, 1, c3, 1);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c3);
    cublasDestroy(handle);
    printf("cublas a × b = { \n");
    for (int i = 0; i < arraysizeM; i++) {
        for (int j = 0; j < arraysizeN; j++) {
            printf("%7.0f ", c3[i * arraysizeM + j]);
        }
        printf("\n");
    }
    for (int i = 0; i < arraysizeM * arraysizeL; i++) {
        c3[i] = 0;
    }

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

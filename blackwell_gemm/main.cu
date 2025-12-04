// System includes
#include <assert.h>
#include <stdio.h>
#include <unordered_map>

const std::unordered_map<u_int8_t,float>FP4_VALUES{
    {0x00,+0.0f},{0x01,+0.5f},{0x02,+1.0f},{0x03,+1.5f},
    {0x04,+2.0f},{0x05,+3.0f},{0x06,+4.0f},{0x07,+6.0f},
    {0x08,-0.0f},{0x09,-0.5f},{0x0A,-1.0f},{0x0B,-1.5f},
    {0x0C,-2.0f},{0x0D,-3.0f},{0x0E,-4.0f},{0x0F,-6.0f},
};
// gridDim 1
// blockDim 32
__global__ void gemm_kernel(
    const u_int32_t* dA,
    const u_int8_t* dA_Scale,
    const u_int32_t* dB,
    const u_int8_t* dB_Scale,
    float* dC,int M,int N,int K
)
{
    const int idx = threadIdx.x;
    u_int32_t a_fp4[4];
    u_int32_t b_fp4[2];
    u_int32_t scaleA = 0xFFFFFFFF;
    u_int32_t scaleB = 0xFFFFFFFF;
    float C[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float D[4];
    //TODO: load A and B matrix tiles by ldmatrix instruction
    int A_row_id = idx >> 2;
    int A_col_id = (idx % 4);
    a_fp4[0] = dA[(A_row_id + 0)* (K/8) + A_col_id + 0];
    a_fp4[1] = dA[(A_row_id + 8)* (K/8) + A_col_id + 0];
    a_fp4[2] = dA[(A_row_id + 0)* (K/8) + A_col_id + 4];
    a_fp4[3] = dA[(A_row_id + 8)* (K/8) + A_col_id + 4];

    int B_col_id = idx >> 2;
    int B_row_id = (idx % 4);
    b_fp4[0] = dB[(B_col_id + 0)* (K/8) + B_row_id + 0];
    b_fp4[1] = dB[(B_col_id + 0)* (K/8) + B_row_id + 4];

    if(idx%4 == 0){
        scaleA &= (static_cast<u_int32_t>(dA_Scale[(A_row_id)*(K/32) + 0]));
        scaleA &= (static_cast<u_int32_t>(dA_Scale[(A_row_id)*(K/32) + 1]) << 8);

        scaleB &= (static_cast<u_int32_t>(dB_Scale[(B_col_id)*(K/32) + 0]));
        scaleB &= (static_cast<u_int32_t>(dB_Scale[(B_col_id)*(K/32) + 1]) << 8);
    }
    else if(idx%4 == 1){
        scaleA &= (static_cast<u_int32_t>(dA_Scale[(A_row_id + 8)*(K/32) + 0]));
        scaleA &= (static_cast<u_int32_t>(dA_Scale[(A_row_id + 8)*(K/32) + 1]) << 8);
    }

    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0"
        "{%0, %1, %2, %3},\n\t"  // D matrix FP32
        "{%4, %5, %6, %7},"      // A matrix FP4
        "{%8, %9},"              // B matrix FP4
        "{%10, %11, %12, %13},"  // C matrix FP32
        "{%14}," "{0,0},"         // A scale UE8
        "{%15}," "{0,0};"         // B scale UE8
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(a_fp4[0]), "r"(a_fp4[1]), "r"(a_fp4[2]), "r"(a_fp4[3]),
          "r"(b_fp4[0]), "r"(b_fp4[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]),
          "r"(scaleA),
          "r"(scaleB)
    );

    // Store results back to global memory
    dC[(A_row_id + 0) * N + B_row_id + 0] = D[0];
    dC[(A_row_id + 0) * N + B_row_id + 1] = D[1];
    dC[(A_row_id + 8) * N + B_row_id + 0] = D[2];
    dC[(A_row_id + 8) * N + B_row_id + 1] = D[3];

}

void initFP4Matrix(u_int32_t* matrix, const int matrixSize) {
    for (int i = 0; i < matrixSize; i++) {
        u_int32_t packedValue = 0;
        for (int j = 0; j < 8; j++) {
            u_int8_t fp4Value = rand() % 16; // Random FP4 value between 0 and 15
            packedValue |= (static_cast<u_int32_t>(fp4Value) << (j * 4));
        }
        matrix[i] = packedValue;
    }
}

void initScaleMatrix(u_int8_t* scaleMatrix, const int scaleSize) {
    for (int i = 0; i < scaleSize; i++) {
        scaleMatrix[i] = rand() % 6 - 6 + 128; // Random scale value between -2 and 3
    }
}


void matmul_mxfp4_cpu(const u_int32_t* A,
                      const u_int8_t* A_Scale,
                      const u_int32_t* B,
                      const u_int8_t* B_Scale,
                      float* C, const int M, const int N, const int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                // Extract FP4 values from packed u_int32_t
                u_int8_t a_fp4 = (A[(m * K + k) / 8] >> ((k % 8) * 4)) & 0x0F;
                u_int8_t b_fp4 = (B[(n * K + k) / 8] >> ((k % 8) * 4)) & 0x0F;

                // printf("A_fp4_raw=%x B_fp4_raw=%x\n", a_fp4, b_fp4);

                // Convert FP4 to float and apply scaling
                float a_val = FP4_VALUES.at(a_fp4) * powf(2.0f, A_Scale[m * K / 32 + k / 32]-128);
                float b_val = FP4_VALUES.at(b_fp4) * powf(2.0f, B_Scale[n * K / 32 + k / 32]-128);

                sum += a_val * b_val;
                // printf("A[%d,%d]=%f B[%d,%d]=%f Partial Sum=%f\n", m, k, a_val, k, n, b_val, sum);

                // printf("A_scale[%d,%d]=%d B_scale[%d,%d]=%d \n", m, k, A_Scale[m * K / 32 + k / 32]-128, n, k, B_Scale[n * K / 32 + k / 32]-128);
            }
            C[m * N + n] = sum;
        }
    }

}

int main(int argc, char **argv)
{
    const int M = 16;
    const int N = 8;
    const int K = 64;

    const int matrixASize = M * K / 8;
    const int matrixBSize = K * N / 8;
    const int matrixCSize = M * N;
    const int matrixAscaleSize = M * K / 32;
    const int matrixBscaleSize = K * N / 32;

    const u_int32_t* matrixA = new u_int32_t[matrixASize];
    const u_int32_t* matrixB = new u_int32_t[matrixBSize];
    const u_int8_t* matrixA_Scale = new u_int8_t[matrixAscaleSize];
    const u_int8_t* matrixB_Scale = new u_int8_t[matrixBscaleSize];
    float* matrixC = new float[matrixCSize];
    
    // Initialize matrices
    initFP4Matrix((u_int32_t*)matrixA, matrixASize);
    initFP4Matrix((u_int32_t*)matrixB, matrixBSize);
    initScaleMatrix((u_int8_t*)matrixA_Scale, matrixAscaleSize);
    initScaleMatrix((u_int8_t*)matrixB_Scale, matrixBscaleSize);

    // CPU computation for verification
    matmul_mxfp4_cpu(matrixA, matrixA_Scale, matrixB, matrixB_Scale, matrixC, M, N, K);

    // Allocate device memory
    u_int32_t *d_A, *d_B;
    u_int8_t *d_A_Scale, *d_B_Scale;
    float *d_C;
    cudaMalloc(&d_A, matrixASize * sizeof(u_int32_t));
    cudaMalloc(&d_B, matrixBSize * sizeof(u_int32_t));
    cudaMalloc(&d_C, matrixCSize * sizeof(float));
    cudaMalloc(&d_A_Scale, matrixAscaleSize * sizeof(u_int8_t));
    cudaMalloc(&d_B_Scale, matrixBscaleSize * sizeof(u_int8_t));

    // Copy matrices to device
    cudaMemcpy(d_A, matrixA, matrixASize * sizeof(u_int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, matrixB, matrixBSize * sizeof(u_int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_Scale, matrixA_Scale, matrixAscaleSize * sizeof(u_int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_Scale, matrixB_Scale, matrixBscaleSize * sizeof(u_int8_t), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 gridDim(1);
    dim3 blockDim(32);
    gemm_kernel<<<gridDim, blockDim>>>(d_A, d_A_Scale, d_B, d_B_Scale, d_C,M,N,K);

    // Copy result back to host
    float* matrixC_GPU = new float[matrixCSize];
    cudaMemcpy(matrixC_GPU, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results (this is just a placeholder, actual verification code should compare CPU and GPU results)
    for (int i = 0; i < M * N; i++)
    {
        if(abs(matrixC[i] - matrixC_GPU[i]) > 1e-5) {
            printf("Mismatch at index %d: CPU=%f, GPU=%f\n", i, matrixC[i], matrixC_GPU[i]);
            break;
        }
    }

    // Clean up
    delete[] matrixA;
    delete[] matrixB;
    delete[] matrixC;
    delete[] matrixA_Scale;
    delete[] matrixB_Scale;
    delete[] matrixC_GPU;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_Scale);
    cudaFree(d_B_Scale);

    return 0;
}
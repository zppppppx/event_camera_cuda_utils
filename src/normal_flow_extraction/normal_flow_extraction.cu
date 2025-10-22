#include "normal_flow_extraction/normal_flow_extraction.cuh"
#include "normal_flow_extraction/math_utils.cuh"
#include "cuda_utils.cuh"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <chrono>

__global__ void normalFlowExtraction(
    const double* __restrict__ d_sae,
    int H, int W,
    int R,
    int ransac_iters,
    float inlier_threshold,
    float min_inlier_ratio,
    unsigned global_seed,
    float* __restrict__ d_nx,
    float* __restrict__ d_ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= W || y >= H) return;

    const int idx = y * W + x;

    const double t0 = d_sae[idx];
    if(t0 <= 0.) { d_nx[idx] = 0.; d_ny[idx] = 0.; return; }

    constexpr int n_local = (2 * MAX_LOCAL_RADIUS + 1) * (2 * MAX_LOCAL_RADIUS + 1);
    double xs_local[n_local];
    double ys_local[n_local];
    double ts_local[n_local];
    double xs_mean = 0;
    double ys_mean = 0;
    double ts_mean = 0;
    
    int N = 0;
    const int x0 = x;
    const int y0 = y;

    R = (R < MAX_LOCAL_RADIUS) ? R : MAX_LOCAL_RADIUS;
    for(int dy = -R; dy <= R; ++dy)
    {
        int yy = y0 + dy;
        if(yy < 0 || yy >= H) continue;

        for(int dx = -R; dx <= R; ++dx)
        {
            int xx = x0 + dx;
            if(xx < 0 || xx >= W) continue;

            double t = d_sae[yy * W + xx];
            if(t <= 0.) continue;

            xs_local[N] = (double)xx;
            xs_mean += (double)xx;
            ys_local[N] = (double)yy;
            ys_mean += (double)yy;
            ts_local[N] = t;
            ts_mean += (double)t;
            ++N;
        }
    }

    xs_mean /= N;
    ys_mean /= N;
    ts_mean /= N;

    for(int i = 0; i < N; ++i)
    {
        xs_local[i] -= xs_mean;
        ys_local[i] -= ys_mean;
        ts_local[i] -= ts_mean;
    }

    if(N < 3) { d_nx[idx] = 0.; d_ny[idx] = 0.; return; }

    // RANSAC to fit a plane
    int best_cnt = -1;
    double best_a = 0., best_b = 0., best_c = 0.;
    unsigned seed = (global_seed ^ (unsigned)(idx * 747796405u + 2891336453u));
    for(int iter = 0; iter < ransac_iters; ++iter)
    {
        int i1 = rand_int(seed, 0, N-1);
        int i2 = rand_int(seed, 0, N-1);
        int i3 = rand_int(seed, 0, N-1);

        if(i1 == i2 || i1 == i3 || i2 == i3) continue;

        double a, b, c;
        if(!fitPlaneFromThreePoints(xs_local[i1], ys_local[i1], ts_local[i1], 
                                    xs_local[i2], ys_local[i2], ts_local[i2], 
                                    xs_local[i3], ys_local[i3], ts_local[i3], 
                                    a, b, c))
        {
            continue;
        }

        int cnt = 0;
        for(int i = 0; i < N; ++i)
        {
            double distance = distanceToPlane(xs_local[i], ys_local[i], ts_local[i], a, b, c);
            if(distance < inlier_threshold)
            {
                cnt++;
            }
        }
        if(cnt > best_cnt)
        {
            best_cnt = cnt;
            best_a = a;
            best_b = b;
            best_c = c;
        }
    }

    float inlier_ratio = (float)best_cnt / (float)N;
    if(inlier_ratio < min_inlier_ratio) 
    {
        d_nx[idx] = 0.; d_ny[idx] = 0.; 
        return;
    }

    int inliers[n_local];
    int M = 0;
    for(int k = 0; k < N; ++k)
    {
        if(distanceToPlane(xs_local[k], ys_local[k], ts_local[k], best_a, best_b, best_c) < inlier_threshold)
        {
            inliers[M++] = k;
        }
    }


    bool ok = refinePlaneWithInliers(xs_local, ys_local, ts_local,
                                    inliers, M, best_a, best_b, best_c);
    if(!ok) 
    { 
        d_nx[idx] = 0.; d_ny[idx] = 0.; 
        return; 
    }

    double g2 = best_a * best_a + best_b * best_b;
    if(!(g2 > 1e-12)) 
    { 
        d_nx[idx] = 0.; d_ny[idx] = 0.; 
        return; 
    }

    d_nx[idx] = (float)(-best_a / g2);
    d_ny[idx] = (float)(-best_b / g2);
}


void computeNormalFlow(
    const double* h_sae,
    int H, int W,
    const NormalFlowParams& params,
    std::vector<float>& out_nx,
    std::vector<float>& out_ny)
{
    assert(H > 0 && W > 0);

#ifdef CUDA_TEST_TIME
    // CUDA timing events
    cudaEvent_t evStart, evAfterHtoD, evAfterKernel, evAfterDtoH;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evAfterHtoD));
    CUDA_CHECK(cudaEventCreate(&evAfterKernel));
    CUDA_CHECK(cudaEventCreate(&evAfterDtoH));
#endif

    const size_t sae_bytes = static_cast<size_t>(H) * static_cast<size_t>(W) * sizeof(double);
    double* d_sae = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sae, sae_bytes));
#ifdef CUDA_TEST_TIME
    CUDA_CHECK(cudaEventRecord(evStart));
#endif
    CUDA_CHECK(cudaMemcpy(d_sae, h_sae, sae_bytes, cudaMemcpyHostToDevice));
#ifdef CUDA_TEST_TIME
    CUDA_CHECK(cudaEventRecord(evAfterHtoD));
    CUDA_CHECK(cudaEventSynchronize(evAfterHtoD));
#endif

    const size_t n_bytes = static_cast<size_t> (H) * static_cast<size_t> (W) * sizeof(float);
    float* d_nx = nullptr;
    float* d_ny = nullptr;
    CUDA_CHECK(cudaMalloc(&d_nx, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_ny, n_bytes));

    const dim3 block(16, 16);
    const dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    // dynamic shared memory size: (blockDim + 2*R) tile of doubles
    normalFlowExtraction<<<grid, block>>>(d_sae, H, W, 
                                            static_cast<int>(params.neighbor_radius), 
                                            static_cast<int>(params.ransac_iters), 
                                            static_cast<float>(params.inlier_threshold), 
                                            static_cast<float>(params.min_inlier_ratio), 
                                            static_cast<unsigned>(params.seed), 
                                            d_nx, d_ny);
    
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
#ifdef CUDA_TEST_TIME
    CUDA_CHECK(cudaEventRecord(evAfterKernel));
    CUDA_CHECK(cudaEventSynchronize(evAfterKernel));
#endif

    out_nx.resize(H * W);
    out_ny.resize(H * W);

    CUDA_CHECK(cudaMemcpy(out_nx.data(), d_nx, n_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_ny.data(), d_ny, n_bytes, cudaMemcpyDeviceToHost));
#ifdef CUDA_TEST_TIME
    CUDA_CHECK(cudaEventRecord(evAfterDtoH));
    CUDA_CHECK(cudaEventSynchronize(evAfterDtoH));
#endif

#ifdef CUDA_TEST_TIME
    float h2d_ms = 0.0f, kernel_ms = 0.0f, d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, evStart, evAfterHtoD));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, evAfterHtoD, evAfterKernel));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, evAfterKernel, evAfterDtoH));
    std::cout << "computeNormalFlow timing - H2D: " << h2d_ms
              << " ms, Kernel: " << kernel_ms
              << " ms, D2H: " << d2h_ms
              << " ms, Memcpy total: " << (h2d_ms + d2h_ms) << " ms" << std::endl;
#endif

    CUDA_CHECK(cudaFree(d_sae));
    CUDA_CHECK(cudaFree(d_nx));
    CUDA_CHECK(cudaFree(d_ny));

#ifdef CUDA_TEST_TIME
    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evAfterHtoD));
    CUDA_CHECK(cudaEventDestroy(evAfterKernel));
    CUDA_CHECK(cudaEventDestroy(evAfterDtoH));
#endif
}

// Zero-copy host output variant
void computeNormalFlow(
    const double* h_sae,
    int H, int W,
    const NormalFlowParams& params,
    float* out_nx,
    float* out_ny)
{
    assert(H > 0 && W > 0);

#ifdef CUDA_TEST_TIME
    cudaEvent_t evStart, evAfterHtoD, evAfterKernel, evAfterDtoH;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evAfterHtoD));
    CUDA_CHECK(cudaEventCreate(&evAfterKernel));
    CUDA_CHECK(cudaEventCreate(&evAfterDtoH));
#endif

    const size_t sae_bytes = static_cast<size_t>(H) * static_cast<size_t>(W) * sizeof(double);
    double* d_sae = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sae, sae_bytes));
#ifdef CUDA_TEST_TIME
    CUDA_CHECK(cudaEventRecord(evStart));
#endif
    CUDA_CHECK(cudaMemcpy(d_sae, h_sae, sae_bytes, cudaMemcpyHostToDevice));
#ifdef CUDA_TEST_TIME
    CUDA_CHECK(cudaEventRecord(evAfterHtoD));
    CUDA_CHECK(cudaEventSynchronize(evAfterHtoD));
#endif

    const size_t n_bytes = static_cast<size_t> (H) * static_cast<size_t> (W) * sizeof(float);
    float* d_nx = nullptr;
    float* d_ny = nullptr;
    CUDA_CHECK(cudaMalloc(&d_nx, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_ny, n_bytes));

    const dim3 block(16, 16);
    const dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    normalFlowExtraction<<<grid, block>>>(d_sae, H, W, 
                                            static_cast<int>(params.neighbor_radius), 
                                            static_cast<int>(params.ransac_iters), 
                                            static_cast<float>(params.inlier_threshold), 
                                            static_cast<float>(params.min_inlier_ratio), 
                                            static_cast<unsigned>(params.seed), 
                                            d_nx, d_ny);
    
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
#ifdef CUDA_TEST_TIME
    CUDA_CHECK(cudaEventRecord(evAfterKernel));
    CUDA_CHECK(cudaEventSynchronize(evAfterKernel));
#endif

    // Direct D2H into user buffers (RowMajor expected)
    CUDA_CHECK(cudaMemcpy(out_nx, d_nx, n_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_ny, d_ny, n_bytes, cudaMemcpyDeviceToHost));
#ifdef CUDA_TEST_TIME
    CUDA_CHECK(cudaEventRecord(evAfterDtoH));
    CUDA_CHECK(cudaEventSynchronize(evAfterDtoH));
#endif

#ifdef CUDA_TEST_TIME
    float h2d_ms = 0.0f, kernel_ms = 0.0f, d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, evStart, evAfterHtoD));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, evAfterHtoD, evAfterKernel));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, evAfterKernel, evAfterDtoH));
    std::cout << "computeNormalFlow timing - H2D: " << h2d_ms
              << " ms, Kernel: " << kernel_ms
              << " ms, D2H: " << d2h_ms
              << " ms, Memcpy total: " << (h2d_ms + d2h_ms) << " ms" << std::endl;
#endif

    CUDA_CHECK(cudaFree(d_sae));
    CUDA_CHECK(cudaFree(d_nx));
    CUDA_CHECK(cudaFree(d_ny));

#ifdef CUDA_TEST_TIME
    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evAfterHtoD));
    CUDA_CHECK(cudaEventDestroy(evAfterKernel));
    CUDA_CHECK(cudaEventDestroy(evAfterDtoH));
#endif
}

// Zero-copy Eigen overload: require RowMajor outputs and copy D2H directly into them
void computeNormalFlow(
    const Eigen::Ref<RowMajorMatrixXd>& sae,
    const NormalFlowParams& params,
    Eigen::Ref<RowMajorMatrixXf> nx,
    Eigen::Ref<RowMajorMatrixXf> ny
)
{
    const int H = sae.rows();
    const int W = sae.cols();
    assert(nx.rows() == H && nx.cols() == W);
    assert(ny.rows() == H && ny.cols() == W);

    // Call zero-copy host variant with direct pointers
    computeNormalFlow(sae.data(), H, W, params, nx.data(), ny.data());
}
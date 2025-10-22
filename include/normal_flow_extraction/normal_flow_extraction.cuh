#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

#define MAX_LOCAL_RADIUS 7

using RowMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

__global__ void normalFlowExtraction(
    const double* __restrict__ d_sae,
    int H, int W,
    int R,
    int ransac_iters,
    float inlier_threshold,
    float min_inlier_ratio,
    float max_normal_length,
    unsigned global_seed,
    float* __restrict__ d_nx,
    float* __restrict__ d_ny
);

struct NormalFlowParams
{
    unsigned neighbor_radius = 1;
    unsigned ransac_iters = 7;
    float inlier_threshold = 2.f;
    float min_inlier_ratio = 0.75f;
    float max_normal_length = 4e2f;
    unsigned seed = 12345u;
};

void computeNormalFlow(
    const double* h_sae,
    int H, int W,
    const NormalFlowParams& params,
    std::vector<float>& out_nx,
    std::vector<float>& out_ny
);

// Zero-copy host output variant: write directly into provided row-major buffers
void computeNormalFlow(
    const double* h_sae,
    int H, int W,
    const NormalFlowParams& params,
    float* out_nx,
    float* out_ny
);

// Zero-copy Eigen overload: require RowMajor outputs and copy D2H directly into them
void computeNormalFlow(
    const Eigen::Ref<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& sae,
    const NormalFlowParams& params,
    Eigen::Ref<RowMajorMatrixXf> nx,
    Eigen::Ref<RowMajorMatrixXf> ny
);
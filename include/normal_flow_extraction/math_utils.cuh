#pragma once



// Gaussian elimination to solve Ax = b
__device__ bool solve3x3(double A[3][3], double b[3], double x[3]);


// Fit a plane with 3 points
__device__ bool fitPlaneFromThreePoints(double x1, double y1, double t1,
                                               double x2, double y2, double t2,
                                               double x3, double y3, double t3,
                                               double& a, double& b, double& c);


__device__ double distanceToPlane(double x, double y, double t, double a, double b, double c);

// Refine the plane with inliers
__device__ bool refinePlaneWithInliers(
    const double* xs, const double* ys, const double* ts,
    const int* inliers, int M,
    double& a, double& b, double& c
);
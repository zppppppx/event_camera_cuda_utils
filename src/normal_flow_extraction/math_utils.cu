#include "normal_flow_extraction/math_utils.cuh"
#include <cuda_runtime.h>

__device__ bool solve3x3(double A[3][3], double b[3], double x[3])
{
    int piv[3] = {0, 1, 2};
    // select pivot column by column
    for (int col = 0; col < 3; ++col)
    {
        int best = col;
        double maxv = fabs(A[piv[col]][col]);
        for (int r = col + 1; r < 3; ++r)
        {
            double v = fabs(A[piv[r]][col]);
            if (v > maxv)
            {
                maxv = v;
                best = r;
            }
        }
        if (maxv < 1e-12f)
            return false;

        // swap pivot row
        if (best != col)
        {
            int tmp = piv[col];
            piv[col] = piv[best];
            piv[best] = tmp;
        }
        int pr = piv[col];
        double diag = A[pr][col];
        double invd = 1.0f / diag;

        // elimination
        for (int r = col + 1; r < 3; ++r)
        {
            int rr = piv[r];
            double factor = A[rr][col] * invd;
            if (fabs(factor) < 1e-20f)
                continue;
            for (int c = col; c < 3; ++c)
            {
                A[rr][c] -= factor * A[pr][c];
            }
            b[rr] -= factor * b[pr];
        }
    }

    for (int i = 2; i >= 0; --i)
    {
        int ri = piv[i];
        double sum = b[ri];
        for (int c = i + 1; c < 3; ++c)
        {
            sum -= A[ri][c] * x[c];
        }
        double diag = A[ri][i];
        if (fabs(diag) < 1e-12f)
            return false;
        x[i] = sum / diag;
    }
    return isfinite(x[0]) && isfinite(x[1]) && isfinite(x[2]);
}

__device__ bool fitPlaneFromThreePoints(double x1, double y1, double t1,
                                               double x2, double y2, double t2,
                                               double x3, double y3, double t3,
                                               double& a, double& b, double& c)
{
    double A[3][3] = {{x1, y1, 1}, {x2, y2, 1}, {x3, y3, 1}};
    double bb[3] = {t1, t2, t3};
    double p[3];

    bool ok = solve3x3(A, bb, p);

    if(!ok) return false;

    a = p[0];
    b = p[1];
    c = p[2];

    return true;
}

__device__ double distanceToPlane(double x, double y, double t, double a, double b, double c)
{
    double t_pred = (a * x + b * y + c);
    return fabs(t_pred - t);
}

__device__ bool refinePlaneWithInliers(
    const double* xs, const double* ys, const double* ts,
    const int* inliers, int M,
    double& a, double& b, double& c
)
{
    if(M < 3) return false;

    double Sx=0, Sy=0, St=0, Sxx=0, Sxy=0, Syy=0, Sxt=0, Syt=0;
    for (int i = 0; i < M; ++i) {
        int k = inliers[i];
        double x = xs[k], y = ys[k], t = ts[k];
        Sx  += x;   Sy  += y;   St  += t;
        Sxx += x*x; Sxy += x*y; Syy += y*y;
        Sxt += x*t; Syt += y*t;
    }
    double N = (double)M;

    double A[3][3] = {
        {(double)Sxx, (double)Sxy, (double)Sx},
        {(double)Sxy, (double)Syy, (double)Sy},
        {(double)Sx,  (double)Sy,  (double)N }
    };
    double bvec[3] = {(double)Sxt, (double)Syt, (double)St};
    double p[3];

    const double lambda = 1e-6f;
    A[0][0] += lambda; A[1][1] += lambda; A[2][2] += lambda;

    bool ok = solve3x3(A, bvec, p);
    if (!ok) return false;
    a = p[0]; b = p[1]; c = p[2];
    return true;
}
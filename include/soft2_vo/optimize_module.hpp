#ifndef SOFT2_VO_OPTIMIZE_MODULE_HPP_
#define SOFT2_VO_OPTIMIZE_MODULE_HPP_

#include <ceres/ceres.h>

// Helper: E = [t]_x * R  →  epipolar line = E * p_norm
template <typename T>
void ComputeEpipolarLine(const T* rel_t, const T* rel_q, const T* p_prev_norm, T* line) {
  // [t]_x
  T tx[9] = {
      T(0), -rel_t[2], rel_t[1],
      rel_t[2], T(0), -rel_t[0],
      -rel_t[1], rel_t[0], T(0)
  };

  // R from quaternion
  T R[9];
  ceres::QuaternionToRotation(rel_q, R);

  // E = [t]_x * R
  T E[9] = {T(0)};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        E[i * 3 + j] += tx[i * 3 + k] * R[k * 3 + j];

  // line = E * p_prev_norm
  line[0] = E[0] * p_prev_norm[0] + E[1] * p_prev_norm[1] + E[2] * p_prev_norm[2];
  line[1] = E[3] * p_prev_norm[0] + E[4] * p_prev_norm[1] + E[5] * p_prev_norm[2];
  line[2] = E[6] * p_prev_norm[0] + E[7] * p_prev_norm[1] + E[8] * p_prev_norm[2];
}

#endif  // SOFT2_VO_OPTIMIZE_MODULE_HPP_

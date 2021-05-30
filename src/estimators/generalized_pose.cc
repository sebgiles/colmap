// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "estimators/generalized_pose.h"

#include "base/camera_models.h"
#include "base/cost_functions.h"
#include "base/essential_matrix.h"
#include "base/pose.h"
#include "estimators/generalized_absolute_pose.h"
#include "estimators/essential_matrix.h"
#include "optim/bundle_adjustment.h"
#include "util/matrix.h"
#include "util/misc.h"
#include "util/threading.h"

namespace colmap {
namespace {

typedef RANSAC<GP3PEstimator> GeneralizedAbsolutePoseRANSAC;

}  // namespace

bool EstimateGeneralizedAbsolutePose(
                        const GeneralizedAbsolutePoseEstimationOptions& options,
                        const std::vector<Eigen::Vector2d>& points2D,
                        const std::vector<Eigen::Vector3d>& points3D,
                        const std::vector<size_t>& cam_idxs,                                
                        const std::vector<Eigen::Matrix3x4d>& rel_camera_poses,
                        const std::vector<Camera>& cameras,
                        Eigen::Vector4d* qvec, Eigen::Vector3d* tvec,
                        size_t* num_inliers,
                        std::vector<char>* inlier_mask) {
  options.Check();

  // TODO(sebgiles): check cameras and tforms are same length

  // Normalize image coordinates.
  std::vector<Eigen::Vector2d> points2D_normalized(points2D.size());
  for (size_t i = 0; i < points2D.size(); ++i) {
    points2D_normalized[i] = cameras[cam_idxs[i]].ImageToWorld(points2D[i]);
  }

  // Format data for the solver.
  std::vector<GP3PEstimator::X_t> points2D_with_tf;
  for (size_t i = 0; i < points2D_normalized.size(); ++i) {
    points2D_with_tf.emplace_back();
    points2D_with_tf.back().rel_tform = rel_camera_poses[cam_idxs[i]];
    points2D_with_tf.back().xy = points2D_normalized[i];
  }

  // Estimate pose for given focal length.
  auto custom_options = options;
  custom_options.ransac_options.max_error = 
      cameras[0].ImageToWorldThreshold(options.ransac_options.max_error);
  GeneralizedAbsolutePoseRANSAC ransac(custom_options.ransac_options);
  auto report = ransac.Estimate(points2D_with_tf, points3D);

  Eigen::Matrix3x4d proj_matrix;
  inlier_mask->clear();

  *num_inliers = report.support.num_inliers;
  proj_matrix = report.model;
  *inlier_mask = report.inlier_mask;

  if (*num_inliers == 0) {
    return false;
  }

  // Extract pose parameters.
  *qvec = RotationMatrixToQuaternion(proj_matrix.leftCols<3>());
  *tvec = proj_matrix.rightCols<1>();

  if (IsNaN(*qvec) || IsNaN(*tvec)) {
    return false;
  }

  return true;
}
}  // namespace colmap

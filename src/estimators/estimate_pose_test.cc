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

#define TEST_NAME "base/estimate_pose"
#include "util/testing.h"

#include <array>

#include <Eigen/Core>

#include "base/pose.h"
#include "base/projection.h"
#include "base/similarity_transform.h"
#include "estimators/generalized_pose.h"

#include "optim/ransac.h"
#include "util/random.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(Estimate) {
  SetPRNGSeed(0);

  std::vector<Eigen::Vector3d> points3D;
  points3D.emplace_back(1, 1, 1);
  points3D.emplace_back(0, 1, 1);
  points3D.emplace_back(3, 1.0, 4);
  points3D.emplace_back(3, 1.1, 4);
  points3D.emplace_back(3, 1.2, 4);
  points3D.emplace_back(3, 1.3, 4);
  points3D.emplace_back(3, 1.4, 4);
  points3D.emplace_back(2, 1, 7);

  size_t ptCount = points3D.size();

  auto points3D_faulty = points3D;
  for (size_t i = 0; i < ptCount; ++i) {
    points3D_faulty[i](0) = 20;
  }

  for (double qx = 0; qx < 1; qx += 0.2) {
    for (double tx = 0; tx < 1; tx += 0.1) {

      const size_t kCamCount = 3;
      // generate a pose for each camera (absolute)
      // first camera simulates ground truth position for the rig
      const std::array<SimilarityTransform3, kCamCount> camera_poses = {{
          SimilarityTransform3(1, Eigen::Vector4d(1, qx, 0, 0),
                               Eigen::Vector3d(tx, -0.1, 0)),
          SimilarityTransform3(1, Eigen::Vector4d(1, qx, 0, 0),
                               Eigen::Vector3d(tx, 0, 0)), 
          SimilarityTransform3(1, Eigen::Vector4d(1, qx, 0, 0),
                               Eigen::Vector3d(tx, 0.1, 0)),
      }};

      std::vector<double> intrinsics_0 = {
          8.68993378e+02, 8.66063001e+02, 5.25942323e+02, 4.20042529e+02, 
          -3.99431000e-01, 1.88924000e-01, 1.53000000e-04, 5.71000000e-04
      };

      std::vector<Camera> cameras(kCamCount);
      for (Camera& cam: cameras){
        cam.SetModelIdFromName("OPENCV");
        cam.SetWidth(1024);
        cam.SetHeight(768);
        cam.SetParams(intrinsics_0);
      }

      std::vector<int> cam_idxs(ptCount);
      for (size_t i = 0; i < ptCount; i++) cam_idxs[i] = i%kCamCount;

      // compute extrinsics: camera poses wrt the first camera
      std::vector<Eigen::Matrix3x4d> extrinsics(kCamCount);
      for (size_t i = 0; i < kCamCount; ++i) {
        Eigen::Vector4d rel_qvec;
        Eigen::Vector3d rel_tvec;
        ComputeRelativePose(camera_poses[0].Rotation(),
                            camera_poses[0].Translation(),
                            camera_poses[i].Rotation(),
                            camera_poses[i].Translation(), &rel_qvec, &rel_tvec);
        extrinsics[i] = ComposeProjectionMatrix(rel_qvec, rel_tvec);
      }



      // Project points to respective camera's pixel coordinates.
      std::vector<Eigen::Vector2d> points2D;
      for (size_t i = 0; i < ptCount; ++i) {
        auto pt3D = points3D[i];
        auto cam_idx = cam_idxs[i];
        camera_poses[cam_idx].TransformPoint(&pt3D);
        auto normed = pt3D.hnormalized();
        auto camera = cameras[cam_idx];
        points2D.emplace_back();
        points2D.back() = camera.WorldToImage(normed);
      }
      
      // Check that both vectors have the same size.
      assert(points2D.size() == points3D.size());

      // Absolute pose estimation parameters.
      GeneralizedAbsolutePoseEstimationOptions abs_pose_options;
      abs_pose_options.estimate_focal_length = false;
      abs_pose_options.ransac_options.max_error = 1e-5; //max_error_px;
      abs_pose_options.ransac_options.min_inlier_ratio = 0.01;
      abs_pose_options.ransac_options.min_num_trials = 1000;
      abs_pose_options.ransac_options.max_num_trials = 100000;
      abs_pose_options.ransac_options.confidence = 0.9999;
      
      // Absolute pose estimation.
      Eigen::Vector4d qvec;
      Eigen::Vector3d tvec;
      size_t num_inliers;
      std::vector<char> inlier_mask;

      bool success = EstimateGeneralizedAbsolutePose(abs_pose_options, 
              points2D, points3D, cam_idxs, extrinsics, cameras, 
              &qvec, &tvec, &num_inliers, &inlier_mask);

      //   RANSACOptions options;
      //   options.max_error = 1e-5;
      //   RANSAC<GP3PEstimator> ransac(options);
      //   const auto report = ransac.Estimate(points2D, points3D);

      BOOST_CHECK_EQUAL(success, true);

      auto estimated_pose = ComposeProjectionMatrix(qvec, tvec);

      // Test if correct transformation has been determined.
      const double matrix_diff =
          (camera_poses[0].Matrix().topLeftCorner<3, 4>() - estimated_pose)
              .norm();

      BOOST_CHECK(matrix_diff < 1e-2);

      // // Test residuals of exact points.
      // std::vector<double> residuals;
      // GP3PEstimator::Residuals(points2D_with_cali, points3D, estimated_pose, &residuals);
      // for (auto residual: residuals) BOOST_CHECK(residual < 1e-10);

      // // Test residuals of faulty points.
      // GP3PEstimator::Residuals(points2D_with_cali, points3D_faulty, estimated_pose, &residuals);
      // for (auto residual: residuals) BOOST_CHECK(residual > 1e-10);
    }
  }
}

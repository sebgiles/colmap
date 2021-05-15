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

#define TEST_NAME "base/estimate_pose_from_csv"
#include "util/testing.h"

#include <array>

#include <Eigen/Core>

#include "base/pose.h"
#include "base/projection.h"
#include "base/similarity_transform.h"
#include "estimators/generalized_pose.h"

#include "optim/ransac.h"
#include "util/random.h"


#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

#define MAXBUFSIZE  ((int) 1e6)

std::vector<Eigen::Vector2d> readPoints2d(const char *filename) {
  int cols = 0, rows = 0;
  double buff[MAXBUFSIZE];

  // Read numbers from file into buffer.
  ifstream infile;
  infile.open(filename);
  while (! infile.eof()) {
    string line;
    getline(infile, line);
    int temp_cols = 0;
    stringstream stream(line);
    while(! stream.eof()) stream >> buff[cols*rows+temp_cols++];
    if (temp_cols == 0) continue;
    if (cols == 0) cols = temp_cols;
    rows++;
  }
  infile.close();

  rows--;

  // Populate matrix with numbers.
  std::vector<Eigen::Vector2d> result(rows);
  for (int i = 0; i < rows; i++){
    result[i][0] = buff[2*i];
    result[i][1] = buff[2*i+1];
  }

  return result;
};

std::vector<Eigen::Vector3d> readPoints3d(const char *filename) {
  int cols = 0, rows = 0;
  double buff[MAXBUFSIZE];

  // Read numbers from file into buffer.
  ifstream infile;
  infile.open(filename);
  while (! infile.eof()) {
    string line;
    getline(infile, line);
    int temp_cols = 0;
    stringstream stream(line);
    while(! stream.eof()) stream >> buff[cols*rows+temp_cols++];
    if (temp_cols == 0) continue;
    if (cols == 0) cols = temp_cols;
    rows++;
  }
  infile.close();

  rows--;

  // Populate matrix with numbers.
  std::vector<Eigen::Vector3d> result(rows);
  for (int i = 0; i < rows; i++){
    result[i][0] = buff[3*i];
    result[i][1] = buff[3*i+1];
    result[i][2] = buff[3*i+2];
  }

  return result;
};

std::vector<size_t> readCamIdxs(const char *filename) {
  int cols = 0, rows = 0;
  double buff[MAXBUFSIZE];

  // Read numbers from file into buffer.
  ifstream infile;
  infile.open(filename);
  while (! infile.eof()) {
    string line;
    getline(infile, line);
    int temp_cols = 0;
    stringstream stream(line);
    while(! stream.eof()) stream >> buff[cols*rows+temp_cols++];
    if (temp_cols == 0) continue;
    if (cols == 0) cols = temp_cols;
    rows++;
  }
  infile.close();

  rows--;

  // Populate matrix with numbers.
  std::vector<size_t> result(rows);
  for (int i = 0; i < rows; i++){
    result[i] = (size_t)buff[i];
  }

  return result;
};

using namespace colmap;

BOOST_AUTO_TEST_CASE(Estimate) {
  SetPRNGSeed(0);

  std::vector<Eigen::Vector3d> points3D = readPoints3d("/home/seb/3dv/mp3d.csv");
  std::vector<Eigen::Vector2d> points2D = readPoints2d("/home/seb/3dv/mkpq.csv");
  std::vector<size_t> cam_idxs = readCamIdxs("/home/seb/3dv/cam_idxs.csv");

  // Check that both vectors have the same size.
  assert(points2D.size() == points3D.size());
  assert(points2D.size() == cam_idxs.size());

  size_t ptCount = points3D.size();

  // generate a pose for each camera (absolute)
  // first camera simulates ground truth position for the rig
  const std::vector<Eigen::Matrix3x4d> extrinsics = {
      SimilarityTransform3(
        1, Eigen::Vector4d(1, 0, 0, 0), Eigen::Vector3d(0, 0, 0)
      ).Matrix().topLeftCorner<3, 4>()
  };

  std::vector<std::vector<double>> intrinsics = {{
      8.68993378e+02, 8.66063001e+02, 5.25942323e+02, 4.20042529e+02, 
      -3.99431000e-01, 1.88924000e-01, 1.53000000e-04, 5.71000000e-04
  }};

  assert(intrinsics.size() == extrinsics.size());

  size_t camCount = extrinsics.size();

  std::vector<Camera> cameras(camCount);
  for (size_t i=0; i<camCount; i++){
    Camera& cam = cameras[i];
    cam.SetModelIdFromName("OPENCV");
    cam.SetWidth(1024);
    cam.SetHeight(768);
    cam.SetParams(intrinsics[i]);
  }

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

  BOOST_CHECK_EQUAL(success, true);

}

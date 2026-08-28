# auto_calib_v2.0

## Modifications

Modifications were made to lidar2camera/auto_calib_v2.0 from [PJLab-ADG/SensorsCalibration](https://github.com/PJLab-ADG/SensorsCalibration).

Since execution on Jetson was too slow with only a single CPU core being utilized, OMP was used to optimize various for-loops to fully load the CPU. Point cloud segmentation was not optimized because it is iterative.

Since the test point cloud is merged from multiple files, there may be duplicate points, so downsampling was applied. However, voxel filtering overflows when the point cloud is too large, so octree was used instead, computing the centroid for each leaf node.

Support for left-multiplying extrinsic parameters by deltaT was added in BruteForceSearch and RandomSearch, because it feels like the search should be performed on the point cloud transformed to the camera coordinate system before projection? Not sure... it probably has no practical impact.

Added BruteForceSearch for translation in Calibrate().

---

## Testing

Testing was conducted in CARLA, so the intensity of LiDAR points is inaccurate and only related to distance. I also tried removing the intensity weight from the score, but the results were still not satisfactory.

The final translation t in the extrinsics is not very accurate, differing from ground truth by ten to twenty centimeters; the angle error is small, and the projection is roughly aligned.

The final score differs from ground truth by a very small margin, only around 0.00x, but the search still deviated from ground truth. Projection really is less sensitive to translation than to rotation?
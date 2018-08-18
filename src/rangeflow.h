#include <Eigen/Dense>
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CameraInfo.h"

#include "ros/ros.h"
#include "ros/console.h"

#include "sensor_msgs/PointCloud2.h"
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <image_geometry/pinhole_camera_model.h>
#include "tf/transform_broadcaster.h"
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include "nav_msgs/Odometry.h"


using namespace Eigen;
using namespace sensor_msgs;

class rangeflow
{
   public:
      ros::NodeHandle& n;
      ros::NodeHandle& nh;

      // Subscribers
      ros::Subscriber depth_sub;
      ros::Publisher cloud_pub;
      ros::Publisher down_cloud_pub;
      ros::Publisher valid_cloud_pub;
      ros::Publisher poseCovPub;

      ros::Publisher odomPub;
      ros::Publisher rel_odomPub;
      ros::Publisher inv_odomPub;
      ros::Publisher voStatePub;

      ros::Publisher pub_dpt_fovis;
      ros::Publisher pub_img_fovis;


      tf::TransformListener m_tfListener;
      tf::TransformBroadcaster tfBroadcaster;

      typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
      bool FirstFrame;
      bool dvo_refine;
      int pyramid_level;
      
      int level0_iterCounts;
      int level1_iterCounts;
      int level2_iterCounts;
      int level3_iterCounts;

      int level0_minGradMagnitudes;
      int level1_minGradMagnitudes;
      int level2_minGradMagnitudes;
      int level3_minGradMagnitudes;

      tf::Transform Transform_last;
      tf::Transform Transform_cur;
      tf::Transform Relative_trans;
      tf::Transform last_relative_trans;
      
      nav_msgs::Odometry rel_odom;
      nav_msgs::Odometry inv_odom; // inversed relative odometry for UKF filtering
      double estimation_quality;
      double gray_threshold;
      double cIndex;

      message_filters::Subscriber<sensor_msgs::Image> imgsub;
      message_filters::Subscriber<sensor_msgs::Image> depthsub;
      message_filters::Subscriber<sensor_msgs::CameraInfo> camInfosub;
      typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,sensor_msgs::Image,sensor_msgs::CameraInfo> MySyncPolicy;
      message_filters::Synchronizer<MySyncPolicy>* sync1;

      std::string frame_id;
      ros::Time current_ts_, last_ts_;

      FILE* fileVar;

   protected:

      /** Matrices that store the original and filtered depth frames with the image resolution */
      MatrixXf depth_ft;
      MatrixXf depth_wf;

      /** Matrices that store the point coordinates after downsampling. */
      MatrixXf depth;
      MatrixXf depth_old;
      MatrixXf depth_inter;
      MatrixXf xx;
      MatrixXf xx_inter;
      MatrixXf xx_old;
      MatrixXf yy;
      MatrixXf yy_inter;
      MatrixXf yy_old;

      /** Matrices that store the depth derivatives */
      MatrixXf du;
      MatrixXf dv;
      MatrixXf dt;

      /** Weights used to ponder equations in the least square solution */
      MatrixXf weights;

      /** Matrix which indicates wheter the depth of a pixel is zero (null = 1) or not (null = 00). and border and noisy points */
      MatrixXi null;

      /** Matrix which indicates wheter a point is in a border or has an inaccurate depth (border =1, border = 0 otherwise) */
      MatrixXi border;

      /** Least squares covariance matrix */
      Matrix<float, 6, 6> est_cov;

      /** Camera properties: */
      double f_dist;		//!<Focal lenght (meters)
      float x_incr;//!<Separation between pixels (cols) in the sensor array (meters)
      float y_incr;//!<Separation between pixels (rows) in the sensor array (meters)
      double fovh;			//!<Horizontal field of view (rad)
      double fovv;			//!<Vertical field of view (rad)

      /** Number of rows and cols of the depth image that will be considered by the visual odometry method.
       * As a rule, the more rows and cols the slower and more accurate the method becomes.
       * They always have to be less or equal to the size of the original depth image. */
      int rows;
      int cols;

      /** Size (rows) of the gaussian kernel used to filter the depth image */
      int gaussian_mask_size;

      /** Speed filter parameters:
       * Previous_speed_const_weight directly ponders the previous speed in order to calculate the filtered speed. Recommended range - (0, 0.5)
       * Previous_speed_eig_weight ponders the product of the corresponding eigenvalue and the previous speed in order to calculate the filtered speed*/
      double previous_speed_const_weight;	//!<Default 0.2
      double previous_speed_eig_weight;	//!<Default 300

      /** Solution from the solver: kai before applying the filter in local coordinates */
      Matrix<float, 6, 1> kai_solver;

      /** Last filtered speed in absolute coordinates */
      Matrix<float, 6, 1> kai_abs;

      /** It filters the depth image with a gaussian kernel and downsample this filtered image according to the values of "rows" and "cols" */
      void filterAndDownsample();

      /** It calculates the "average" coordinates of the points observed by the camera between two consecutive frames */
      void calculateCoord();

      /** It calculates the depth derivatives respect to u,v (rows and cols) and t (time) */
      void calculateDepthDerivatives();

      /** This method finds pixels whose depth is zero to subsequently discard them */
      void findNullPoints();

      /** This method finds pixels which are not in planar or smooth surfaces, and also inaccurate (noisy) pixels */
      void findBorders();

      /** This method discards the pixels found by 'findNullPoints()' and 'findBorders()' */
      void findValidPoints();

      /** The Solver. It buils the overdetermined system and gets the least-square solution.
       * It also calculates the least-square covariance matrix */
      void solveDepthSystem();

      /**  Virtual method to filter the speed and update the camera pose. */
      void filterSpeedAndPoseUpdate ( const ros::Time& stamp, ros::WallTime start );
      
      /** Publish the visual odometry state */
      void publishVOState(const ros::Time& stamp);

   public:

      // Determine wheter the equation is ill-conditioned.
      double condition_index;

      /** Camera properties */
      double lens_disp;//Lateral displacement of the lens with respect to the center of the camera (meters)
      double focal_length;

      /** Frames per second (Hz) */
      double fps;

      /** Resolution of the images taken by the range camera */
      int cam_mode;	// (1 - 640 x 480, 2 - 320 x 240, 4 - 160 x 120)

      /** Downsample the image taken by the range camera. Useful to reduce the computational burden,
       * as this downsampling is applied before the gaussian filter */
      int downsample; // (1 - original size, 2 - res/2, 4 - res/4)

      /** Num of valid points after removing null pixels and borders */
      unsigned int num_valid_points;

      /** Thresholds used to remove borders and noisy points */
      float duv_threshold;		//!< Threshold to du*du + dv*dv
      float dt_threshold;		//!< Threshold to dt
      float dif_threshold;//!< Threshold to [abs(final_dx-ini_dx) + abs(final_dy-ini_dy)]
      float difuv_surroundings;//!< Threshold to the difference of (du,dv) at a pixel and the values of (du,dv) at its surroundings
      float dift_surroundings;//!< Threshold to the difference of dt at a pixel and the value of dt at its surroundings

      /** Execution time (ms) */
      float execution_time;

      // camera pose: x, y, z, yaw, pitch, roll
      Matrix<double, 6, 1> cam_pose; //!< Last camera pose
      Matrix<double, 6, 1> cam_oldpose; //!< Previous camera pose

      image_geometry::PinholeCameraModel model_;

      /** This method performs the necessary steps to estimate the camera speed in local coordinates once the depth image has been loaded */
      void OdometryCalculation();


      /** This method gets the coordinates of the points regarded by the visual odometry method */
      inline void getPointsCoord ( MatrixXf &x, MatrixXf &y, MatrixXf &z );

      /** This method gets the depth derivatives respect to u,v and t respectively */
      inline void getDepthDerivatives ( MatrixXf &cur_du, MatrixXf &cur_dv,
                                        MatrixXf &cur_dt );

      /** It gets the matrix of weights */
      inline void getWeights ( MatrixXf &we );

      /** It resets the border thresholds to their default values */
      void bordersThresholdToDefault();

      void callback ( const sensor_msgs::Image::ConstPtr& msg_image, const sensor_msgs::Image::ConstPtr& msg_depth, const sensor_msgs::CameraInfo::ConstPtr& msg_camInfo );

      //Constructor. Initialize variables and matrix sizes
      rangeflow ( ros::NodeHandle& n, ros::NodeHandle& nh );

};

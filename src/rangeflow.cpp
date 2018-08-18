#include "rangeflow.h"
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <ros/ros.h>
#include <math.h>
#include <limits>


#include <pcl_ros/transforms.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <sensor_msgs/image_encodings.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include "dvo.h"
#include "rangeflow_odom/vo_state.h"

using namespace std;
using namespace Eigen;
using namespace cv;

int ImageHeight;
int ImageWidth ;

float vals[9] ;

const Mat cameraMatrix = Mat ( 3,3,CV_32FC1,vals );
const Mat distCoeff ( 1,5,CV_32FC1,Scalar ( 0 ) );

cv::Mat image0;
cv::Mat depth0;

bool first_frame = true;


rangeflow::rangeflow ( ros::NodeHandle& n, ros::NodeHandle& nh ) :
   n ( n ), nh ( nh )
{
   nh.param ( "cam_mode", cam_mode, 1 );
   nh.param ( "downsample", downsample, 2 );
   nh.param ( "fps", fps, 30.0 );
   nh.param ( "rows", rows, 60 );
   nh.param ( "cols", cols, 80 );
   nh.param ( "gaussian_mask_size", gaussian_mask_size, 7 );
   nh.param ( "fovh", fovh, M_PI * 57.5 / 180.0 );
   nh.param ( "fovv", fovv, M_PI * 45.0 / 180.0 );
   nh.param ( "lens_disp", lens_disp, 0.022 );
   nh.param ( "focal_length", focal_length, 525.0 );
   nh.param ( "previous_speed_const_weight", previous_speed_const_weight, 0.2 );
   nh.param ( "previous_speed_eig_weight", previous_speed_eig_weight, 300.0 );
   nh.param ( "dvo_refine", dvo_refine, false );
   nh.param ( "gray_threshold", gray_threshold, 10.0 );
   nh.param ( "cIndex", cIndex, 1500.0 );

   nh.param ( "pyramid_level", pyramid_level, 4 );
   nh.param ( "level0_iterCounts", level0_iterCounts, 7 );
   nh.param ( "level1_iterCounts", level1_iterCounts, 7 );
   nh.param ( "level2_iterCounts", level2_iterCounts, 7 );
   nh.param ( "level3_iterCounts", level3_iterCounts, 10 );

   nh.param ( "level0_minGradMagnitudes", level0_minGradMagnitudes, 12 );
   nh.param ( "level1_minGradMagnitudes", level1_minGradMagnitudes, 5 );
   nh.param ( "level2_minGradMagnitudes", level2_minGradMagnitudes, 3 );
   nh.param ( "level3_minGradMagnitudes", level3_minGradMagnitudes, 1 );

   odomPub = n.advertise<nav_msgs::Odometry> ( "rangeflow_odometry", 50, true );
   rel_odomPub = n.advertise<nav_msgs::Odometry> ( "rangeflow_measurement", 50, true );
   inv_odomPub = n.advertise<nav_msgs::Odometry> ( "rangeflow_inv_measurement", 50, true );
   voStatePub =  n.advertise<rangeflow_odom::vo_state> ( "rangeflow_state",50, true );

   pub_dpt_fovis = n.advertise<sensor_msgs::Image> ( "/fovis/depth", 5 );
   pub_img_fovis = n.advertise<sensor_msgs::Image> ( "/fovis/image", 5 );

   cloud_pub = n.advertise<PointCloud> ( "/camera/downsampling/points", 100 );
   down_cloud_pub = n.advertise<PointCloud> ( "/camera/filtering/points", 100 );
   valid_cloud_pub = n.advertise<PointCloud> ( "/camera/valid_points", 100 );
   poseCovPub = n.advertise<geometry_msgs::PoseWithCovarianceStamped> ( "pose_cov", 10 );


   imgsub.subscribe ( n, "rgbimage", 10 );
   depthsub.subscribe ( n, "depthimage", 10 );
   camInfosub.subscribe ( n, "depthcamerainfo", 10 );
   sync1 = new message_filters::Synchronizer<MySyncPolicy> ( 100 );
   sync1->connectInput ( imgsub, depthsub, camInfosub );
   sync1->registerCallback ( boost::bind ( &rangeflow::callback, this, _1, _2, _3 ) );

   const unsigned int resh = 640 / ( cam_mode * downsample );
   const unsigned int resv = 480 / ( cam_mode * downsample );

   depth.resize ( rows, cols );
   depth_old.resize ( rows, cols );
   depth_inter.resize ( rows, cols );
   depth_ft.resize ( resv, resh );
   depth_wf.resize ( resv, resh );
   depth_wf.setZero();
   depth.setZero();
   depth_old.setZero();
   depth_inter.setZero();
   depth_ft.setZero();

   du.resize ( rows, cols );
   dv.resize ( rows, cols );
   dt.resize ( rows, cols );
   xx.resize ( rows, cols );
   xx_inter.resize ( rows, cols );
   xx_old.resize ( rows, cols );
   yy.resize ( rows, cols );
   yy_inter.resize ( rows, cols );
   yy_old.resize ( rows, cols );

   border.resize ( rows, cols );
   border.setZero();
   null.resize ( rows, cols );
   null.setZero();
   weights.resize ( rows, cols );
   weights.setZero();

   est_cov.setZero ();
   kai_solver.setZero ();
   kai_abs.setZero ();
   cam_pose.setZero();

   f_dist = 1.0 / focal_length;								  //In meters
   x_incr = 2.0 * f_dist * ( floor ( float ( resh ) / float ( cols ) ) * cols / float ( resh ) )
            * tan ( 0.5 * fovh ) / ( cols - 1 );						  //In meters
   y_incr = 2.0 * f_dist * ( floor ( float ( resv ) / float ( rows ) ) * rows / float ( resv ) )
            * tan ( 0.5 * fovv ) / ( rows - 1 );				  		  //In meters							      				  //In Hz

   //Depth thresholds
   const int dy = floor ( float ( resv ) / float ( rows ) );
   const int dx = floor ( float ( resh ) / float ( cols ) );

   duv_threshold = 0.002 * ( dx + dy ) * ( cam_mode * downsample );
   dt_threshold = 0.4 * fps;
   dif_threshold = 0.002 * ( dx + dy ) * ( cam_mode * downsample );

   difuv_surroundings = 0.005 * ( dx + dy ) * ( cam_mode * downsample );
   dift_surroundings = 0.01 * fps * ( dx + dy ) * ( cam_mode * downsample );

   num_valid_points = 0;
   FirstFrame = true;
   estimation_quality = 10000;

   Transform_last = tf::Transform ( tf::createIdentityQuaternion(), tf::Vector3 ( 0, 0, 0 ) );
   Transform_cur = tf::Transform ( tf::createIdentityQuaternion(), tf::Vector3 ( 0, 0, 0 ) );
   Relative_trans = tf::Transform ( tf::createIdentityQuaternion(), tf::Vector3 ( 0, 0, 0 ) );
   last_relative_trans = tf::Transform ( tf::createIdentityQuaternion(), tf::Vector3 ( 0, 0, 0 ) );
   fileVar= fopen ( "dfLog.txt", "w" );
}

void rangeflow::calculateCoord()
{
   for ( int x = 0; x < cols; x++ )
      for ( int y = 0; y < rows; y++ )
      {
         if ( ( depth ( y, x ) ) == 0 || ( depth_old ( y, x ) == 0 ) )
         {
            depth_inter ( y, x ) = 0;
            xx_inter ( y, x ) = 0;
            yy_inter ( y, x ) = 0;
         }
         else
         {
            depth_inter ( y, x ) = 0.5 * ( depth ( y, x ) + depth_old ( y, x ) );
            xx_inter ( y, x ) = 0.5 * ( xx ( y, x ) + xx_old ( y, x ) );
            yy_inter ( y, x ) = 0.5 * ( yy ( y, x ) + yy_old ( y, x ) );
         }
      }
}

void rangeflow::calculateDepthDerivatives()
{
   for ( int x = 1; x < cols - 1; x++ )
      for ( int y = 1; y < rows - 1; y++ )
      {
         du ( y, x ) = 0.5 * ( depth_inter ( y, x + 1 ) - depth_inter ( y, x - 1 ) );
         dv ( y, x ) = 0.5 * ( depth_inter ( y + 1, x ) - depth_inter ( y - 1, x ) );
         dt ( y, x ) = fps * ( depth ( y, x ) - depth_old ( y, x ) );
      }
}

void rangeflow::filterAndDownsample()
{
   ros::WallTime startTime = ros::WallTime::now();

   //Push the frames back
   depth_old = depth;
   xx_old = xx;
   yy_old = yy;

   //					Create the kernel
   //==========================================================
   Eigen::MatrixXf kernel ( gaussian_mask_size, 1 );

   const float sigma = 0.2f * gaussian_mask_size;
   float r, s = 2.0f * sigma * sigma;
   float ksum = 0.0f;

   // Generate kernel
   if ( ( gaussian_mask_size % 2 == 0 ) || ( gaussian_mask_size < 3 ) )
   {
      cout << endl << "Mask size must be odd and bigger than 2";
      depth_ft = depth_wf;
      return;
   }

   const int lim_mask = ( gaussian_mask_size - 1 ) / 2;
   for ( int x = -lim_mask; x <= lim_mask; x++ )
   {
      r = std::sqrt ( float ( x * x ) );
      kernel ( x + lim_mask, 0 ) = ( exp ( - ( r * r ) / s ) ) / ( M_PI * s );
      ksum += kernel ( x + lim_mask, 0 );
   }

   // normalize the Kernel
   for ( int x = -lim_mask; x <= lim_mask; x++ )
   {
      kernel ( x + lim_mask, 0 ) /= ksum;
      //cout << kernel(x + lim_mask, 1) << "  ";
   }

   const int width = depth_wf.cols();
   const int height = depth_wf.rows();
   MatrixXf depth_if;
   depth_if.resize ( height, width );


   //Apply gaussian filter (separately)
   //rows
   for ( int i = 0; i < height; i++ )
      for ( int j = 0; j < width; j++ )
      {
         if ( ( j >= lim_mask ) && ( j < width - lim_mask ) )
         {
            float sum = 0.0f;
            float ponder = 1.0f;
            for ( int k = -lim_mask; k <= lim_mask; k++ )
            {

               if ( depth_wf ( i, j + k ) == 0 )
                  ponder -= kernel ( k + lim_mask, 0 );
               else
                  sum += kernel ( k + lim_mask, 0 ) * depth_wf ( i, j + k );
            }
            if ( ponder == 1.0f )
               depth_if ( i, j ) = sum;
            else if ( ponder > 0.0001f )
               depth_if ( i, j ) = sum / ponder;
            else
               depth_if ( i, j ) = 0;
         }
         else
            depth_if ( i, j ) = depth_wf ( i, j );

      }

   //cols
   for ( int i = 0; i < height; i++ )
      for ( int j = 0; j < width; j++ )
      {
         if ( ( i >= lim_mask ) && ( i < height - lim_mask ) )
         {
            float sum = 0.0f;
            float ponder = 1.0f;

            for ( int k = -lim_mask; k <= lim_mask; k++ )
            {

               if ( depth_if ( i + k, j ) == 0 )
                  ponder -= kernel ( k + lim_mask, 0 );
               else
                  sum += kernel ( k + lim_mask, 0 ) * depth_if ( i + k, j );
            }

            if ( ponder == 1.0f )
               depth_wf ( i, j ) = sum;
            else if ( ponder > 0.0001f )
               depth_wf ( i, j ) = sum / ponder;
            else
               depth_wf ( i, j ) = 0;
         }
         else
            depth_wf ( i, j ) = depth_if ( i, j );

      }

   //Downsample the pointcloud
   const float inv_f = float ( 640 / width ) / 525.0f;
   const float disp_x = 0.5 * ( width - 1 );
   const float disp_y = 0.5 * ( height - 1 );

   const int dy = floor ( float ( height ) / float ( rows ) );
   const int dx = floor ( float ( width ) / float ( cols ) );
   const unsigned int iniy = ( height - dy * rows ) / 2;
   const unsigned int inix = ( width - dx * cols ) / 2;


   PointCloud::Ptr cloud ( new PointCloud );
   cloud->header.frame_id = frame_id;
   cloud->height = 1;
   cloud->width = 1;
   cloud->is_dense = true;
   int point_counter = 0;
   pcl::PointXYZ pt;

   for ( int y = 0; y < rows; y++ )
      for ( int x = 0; x < cols; x++ )
      {
         pt.z = depth ( y, x ) = depth_wf ( iniy + y * dy, inix + x * dx );
         pt.x = xx ( y, x )    = ( inix + x * dx - disp_x ) * depth_wf ( iniy + y * dy, inix + x * dx ) * inv_f + lens_disp;
         pt.y = yy ( y, x )    = ( iniy + y * dy - disp_y ) * depth_wf ( iniy + y * dy, inix + x * dx ) * inv_f;

         point_counter++;
         cloud->points.push_back ( pt );
      }

   cloud->width = point_counter;
   down_cloud_pub.publish ( cloud );


   double dwalltime = ( ros::WallTime::now() - startTime ).toSec();
   ROS_DEBUG_STREAM ( "Execution time - filter + downsample (ms): " << dwalltime << "s " );
}

void rangeflow::findBorders()
{
   border.setZero();

   //Detect borders
   for ( int x = 1; x < cols - 1; x++ )
      for ( int y = 1; y < rows - 1; y++ )
      {
         if ( null ( y, x ) == 0 )
         {
            const float aver_duv = du ( y, x ) * du ( y, x ) + dv ( y, x ) * dv ( y, x );
            const float ini_dx   = 0.5 * ( depth_old ( y, x + 1 ) - depth_old ( y, x - 1 ) );
            const float ini_dy   = 0.5 * ( depth_old ( y + 1, x ) - depth_old ( y - 1, x ) );
            const float final_dx = 0.5 * ( depth ( y, x + 1 ) - depth ( y, x - 1 ) );
            const float final_dy = 0.5 * ( depth ( y + 1, x ) - depth ( y - 1, x ) );

            //Derivative too high (the average derivative)
            if ( aver_duv > duv_threshold )
               border ( y, x ) = 1;

            else if ( abs ( dt ( y, x ) ) > dt_threshold )
               border ( y, x ) = 1;

            //Big difference between initial and final derivatives
            else if ( abs ( final_dx - ini_dx ) + abs ( final_dy - ini_dy ) > dif_threshold )
               border ( y, x ) = 1;

            else //Difference between derivatives in the surroundings
            {
               float sum_duv = 0;
               float sum_dift = 0;
               float sum_difdepth = 0;
               for ( int k = -1; k < 2; k++ )
                  for ( int l = -1; l < 2; l++ )
                  {
                     sum_duv      += abs ( du ( y, x ) - du ( y + k, x + l ) ) + abs ( dv ( y, x ) - dv ( y + k, x + l ) );
                     sum_dift     += abs ( dt ( y, x ) - dt ( y + k, x + l ) );
                     sum_difdepth += abs ( depth_inter ( y, x ) - depth_inter ( y + k, x + l ) );
                  }

               if ( sum_dift > depth_inter ( y, x ) * dift_surroundings )
                  border ( y, x ) = 1;

               else if ( sum_duv > ( 4.0 * sum_difdepth + depth_inter ( y, x ) ) * difuv_surroundings )
                  border ( y, x ) = 1;

            }
         }
      }

   //Delete sparse points
   for ( int x = 1; x < cols - 1; x++ )
      for ( int y = 1; y < rows - 1; y++ )
      {
         if ( ( null ( y, x ) == 0 ) && ( border ( y, x ) == 0 ) )
         {
            float sum_alone = 0;
            for ( int k = -1; k < 2; k++ )
               for ( int l = -1; l < 2; l++ )
               {
                  sum_alone += ( border ( y + k, x + l ) || null ( y + k, x + l ) );
               }

            if ( sum_alone > 6 )
               border ( y, x ) = 1;

         }
      }
}

void rangeflow::findNullPoints()
{
   null.setZero();
   for ( int x = 0; x < cols; x++ )
      for ( int y = 0; y < rows; y++ )
         if ( depth_inter ( y, x ) == 0 )
            null ( y, x ) = 1;

}

void rangeflow::findValidPoints()
{
   num_valid_points = 0;

   //valid pointcloud
   const int width = depth_wf.cols();
   const int height = depth_wf.rows();

   const float inv_f = float ( 640 / width ) / 525.0f;
   const float disp_x = 0.5 * ( width - 1 );
   const float disp_y = 0.5 * ( height - 1 );

   const int dy = floor ( float ( height ) / float ( rows ) );
   const int dx = floor ( float ( width ) / float ( cols ) );
   const unsigned int iniy = ( height - dy * rows ) / 2;
   const unsigned int inix = ( width - dx * cols ) / 2;

   PointCloud::Ptr cloud ( new PointCloud );
   cloud->header.frame_id = frame_id;
   cloud->height = 1;
   cloud->width = 1;
   cloud->is_dense = true;
   int point_counter = 0;
   pcl::PointXYZ pt;

   for ( int y = 1; y < rows - 1; y++ )
      for ( int x = 1; x < cols - 1; x++ )
         if ( ( border ( y, x ) == 0 ) && ( null ( y, x ) == 0 ) )
         {
            num_valid_points++;

            // Fill in XYZ
            pt.x = ( inix + x * dx - disp_x ) * depth_wf ( iniy + y * dy, inix + x * dx ) * inv_f + lens_disp;
            pt.y = ( iniy + y * dy - disp_y ) * depth_wf ( iniy + y * dy, inix + x * dx ) * inv_f ;
            pt.z = depth_wf ( iniy + y * dy, inix + x * dx );

            point_counter++;
            cloud->points.push_back ( pt );
         }

   cloud->width = point_counter;
   valid_cloud_pub.publish ( cloud );
}

void rangeflow::solveDepthSystem()
{
   ros::WallTime startTime = ros::WallTime::now();

   unsigned int cont = 0;
   MatrixXf A;
   MatrixXf Var;
   MatrixXf B;
   A.resize ( num_valid_points, 6 );
   B.resize ( num_valid_points, 1 );
   Var.resize ( 6, 1 );

   ROS_DEBUG_STREAM ( "num_valid_points is: " << num_valid_points );

   //Fill the matrix A and the vector B
   //The order of the variables will be (vz, vx, vy, wz, wx, wy)
   //The points order will be (1,1), (1,2)...(1,cols-1), (2,1), (2,2)...(row-1,cols-1). Points at the borders are not included

   const float f_inv_x = f_dist / x_incr;
   const float f_inv_y = f_dist / y_incr;
   const float kz2 = ( 1.425e-5 / gaussian_mask_size ) * ( 1.425e-5 / gaussian_mask_size );

   //We need to express the last camera velocity respect to the current camera reference frame
   Matrix<double, 3, 3> inv_trans;
   inv_trans.setZero();
   Matrix<float, 6, 1> kai_abs;
   kai_abs.setZero();
   Matrix<float, 3, 1> v_old, w_old;
   v_old.setZero();
   w_old.setZero();

   Quaterniond q =
      AngleAxis<double> ( cam_pose ( 3,0 ),  Matrix<double,3,1>::UnitZ() ) *
      AngleAxis<double> ( cam_pose ( 4,0 ),  Matrix<double,3,1>::UnitY() ) *
      AngleAxis<double> ( cam_pose ( 5,0 ),  Matrix<double,3,1>::UnitX() );
   inv_trans = q.toRotationMatrix();

   v_old = inv_trans.inverse().cast<float>() * kai_abs.topRows ( 3 );
   w_old = inv_trans.inverse().cast<float>() * kai_abs.bottomRows ( 3 );

   //Create the weighted least squares system
   for ( int y = 1; y < rows - 1; y++ )
      for ( int x = 1; x < cols - 1; x++ )
         if ( ( border ( y, x ) == 0 ) && ( null ( y, x ) == 0 ) )
         {
            //Precomputed expressions
            const float inv_d = 1.0f / depth_inter ( y, x );
            const float dxcomp = du ( y, x ) * f_inv_x * inv_d;
            const float dycomp = dv ( y, x ) * f_inv_y * inv_d;
            const float z2 = depth_inter ( y, x ) *  depth_inter ( y, x );
            const float z4 = z2 * z2;


            //Weights calculation
            const float var11 = kz2 * z4;
            const float var12 = kz2 * xx_inter ( y, x ) * z2
                                * depth_inter ( y, x );
            const float var13 = kz2 * yy_inter ( y, x ) * z2
                                * depth_inter ( y, x );
            const float var22 = kz2 * xx_inter ( y, x ) * xx_inter ( y, x ) * z2;
            const float var23 = kz2 * xx_inter ( y, x ) * yy_inter ( y, x ) * z2;
            const float var33 = kz2 * yy_inter ( y, x ) * yy_inter ( y, x ) * z2;
            const float var44 = kz2 * z4 * fps * fps;
            const float var55 = kz2 * z4 * 0.25;
            const float var66 = kz2 * z4 * 0.25;

            const float j1 = -2.0 * inv_d * inv_d
                             * ( xx_inter ( y, x ) * dxcomp + yy_inter ( y, x ) * dycomp )
                             * ( v_old[0] + yy_inter ( y, x ) * w_old[1]
                                 - xx_inter ( y, x ) * w_old[2] )
                             + inv_d * dxcomp
                             * ( v_old[1] - yy_inter ( y, x ) * w_old[0] )
                             + inv_d * dycomp
                             * ( v_old[2] + xx_inter ( y, x ) * w_old[0] );

            const float j2 = inv_d * dxcomp
                             * ( v_old[0] + yy_inter ( y, x ) * w_old[1]
                                 - 2.0 * xx_inter ( y, x ) * w_old[2] )
                             - dycomp * w_old[0];

            const float j3 = inv_d * dycomp
                             * ( v_old[0] + 2 * yy_inter ( y, x ) * w_old[1]
                                 - xx_inter ( y, x ) * w_old[2] )
                             + dxcomp * w_old[0];

            const float j4 = 1;

            const float j5 = xx_inter ( y, x ) * inv_d * inv_d * f_inv_y
                             * ( v_old[0] + yy_inter ( y, x ) * w_old[1]
                                 - xx_inter ( y, x ) * w_old[2] )
                             + inv_d * f_inv_x
                             * ( -v_old[1] - depth_inter ( y, x ) * w_old[2]
                                 + yy_inter ( y, x ) * w_old[0] );

            const float j6 = yy_inter ( y, x ) * inv_d * inv_d * f_inv_y
                             * ( v_old[0] + yy_inter ( y, x ) * w_old[1]
                                 - xx_inter ( y, x ) * w_old[2] )
                             + inv_d * f_inv_y
                             * ( -v_old[2] + depth_inter ( y, x ) * w_old[1]
                                 - xx_inter ( y, x ) * w_old[0] );

            weights ( y, x ) = sqrt ( 1.0 / ( j1 * ( j1 * var11 + j2 * var12 + j3 * var13 )
                                              + j2 * ( j1 * var12 + j2 * var22 + j3 * var23 )
                                              + j3 * ( j1 * var13 + j2 * var23 + j3 * var33 )
                                              + j4 * j4 * var44 + j5 * j5 * var55
                                              + j6 * j6 * var66 ) );

            A ( cont, 0 ) = weights ( y, x ) * ( 1.0f + dxcomp * xx_inter ( y, x ) * inv_d + dycomp * yy_inter ( y, x ) * inv_d );
            A ( cont, 1 ) = weights ( y, x ) * ( -dxcomp );
            A ( cont, 2 ) = weights ( y, x ) * ( -dycomp );
            A ( cont, 3 ) = weights ( y, x ) * ( dxcomp * yy_inter ( y, x ) - dycomp * xx_inter ( y, x ) );
            A ( cont, 4 ) = weights ( y, x ) * ( yy_inter ( y, x )
                                                 + dxcomp * inv_d * yy_inter ( y, x ) * xx_inter ( y, x )
                                                 + dycomp * ( yy_inter ( y, x ) * yy_inter ( y, x ) * inv_d + depth_inter ( y, x ) ) );
            A ( cont, 5 ) = weights ( y, x ) * ( -xx_inter ( y, x )
                                                 - dxcomp * ( xx_inter ( y, x ) * xx_inter ( y, x ) * inv_d + depth_inter ( y, x ) )
                                                 - dycomp * inv_d * yy_inter ( y, x ) * xx_inter ( y, x ) );

            B ( cont, 0 ) = weights ( y, x ) * ( -dt ( y, x ) );

            cont++;
         }

   //Solve the linear system of equations using a minimum least squares method
   MatrixXf atrans = A.transpose();
   MatrixXf a_ls = atrans * A;
   Var = a_ls.ldlt().solve ( atrans * B );
   kai_solver = Var;

   //Covariance matrix calculation
   MatrixXf residuals ( num_valid_points, 1 );
   residuals = A * Var - B;
   est_cov = ( 1.0f / float ( num_valid_points - 6 ) ) * a_ls.inverse()
             * residuals.squaredNorm();


   //Calculate the condition index to check whether the linearized optimization is in degeneration condition.
   EigenSolver<MatrixXf> es ( a_ls );
   //   cout << "The eigenvalues are:" << endl << es.eigenvalues() << endl;
   //   cout << "The eigenvectors, V, is:" << endl << es.eigenvectors() << endl << endl;

   double eigen_max = 0.0;
   double eigen_min = numeric_limits <double> ::max();

   for ( int i = 0; i < 6; i ++ )
   {
      complex<float> lambda = es.eigenvalues() [i];
      if ( lambda.imag() == 0 )
      {
         if ( lambda.real() > eigen_max )
            eigen_max = lambda.real();
         if ( lambda.real() < eigen_min )
            eigen_min = lambda.real();
      }
   }
   condition_index = eigen_max / eigen_min;

   ROS_DEBUG_STREAM ( "condition index is: " << condition_index );
   ROS_DEBUG_STREAM ( "estimation covariance is:" << est_cov );
}

void rangeflow::OdometryCalculation()
{
   ros::WallTime startTime = ros::WallTime::now();
   filterAndDownsample();
   calculateCoord();
   calculateDepthDerivatives();
   findNullPoints();
   findBorders();
   findValidPoints();

   if ( num_valid_points > 6 )
   {
      solveDepthSystem();
   }
   else
   {
      kai_solver.setZero();
   }

   double dwalltime = ( ros::WallTime::now() - startTime ).toSec();
   ROS_DEBUG_STREAM ( "Odometry Calculation Time is: " << dwalltime << "s " );

}

void rangeflow::filterSpeedAndPoseUpdate ( const ros::Time& stamp, ros::WallTime start )
{
   ros::WallTime startTime = ros::WallTime::now();

   //Calculate Eigenvalues and Eigenvectors
   SelfAdjointEigenSolver<MatrixXf> eigensolver ( est_cov );
   if ( eigensolver.info() != Success )
   {
      printf ( "Eigensolver couldn't find a solution. Pose is not updated" );
      estimation_quality = 10000;
      return;
   }

   //First, we have to describe both the new linear and angular speeds in the "eigenvector" basis
   Eigen::MatrixXf Bii, kai_b;
   Bii.resize ( 6, 6 );
   kai_b.resize ( 6, 1 );
   Bii = eigensolver.eigenvectors();

   kai_b = Bii.colPivHouseholderQr().solve ( kai_solver );

   //Second, we have to describe both the old linear and angular speeds in the "eigenvector" basis too
   //-------------------------------------------------------------------------------------------------
   Matrix<double, 3, 3> inv_trans;
   inv_trans.setZero();

   Matrix<float, 3, 1> v_loc_old, w_loc_old;
   v_loc_old.setZero();
   w_loc_old.setZero();

   //Express them in the local reference frame first
   Quaterniond q =
      AngleAxis<double> ( cam_pose ( 3,0 ),  Matrix<double,3,1>::UnitZ() ) *
      AngleAxis<double> ( cam_pose ( 4,0 ),  Matrix<double,3,1>::UnitY() ) *
      AngleAxis<double> ( cam_pose ( 5,0 ),  Matrix<double,3,1>::UnitX() );
   inv_trans = q.toRotationMatrix();

   v_loc_old = inv_trans.inverse().cast<float>() * kai_abs.topRows ( 3 );
   w_loc_old = inv_trans.inverse().cast<float>() * kai_abs.bottomRows ( 3 );

   //Then transform that local representation to the "eigenvector" basis
   MatrixXf kai_b_old;
   kai_b_old.resize ( 6, 1 );
   Matrix<float, 6, 1> kai_loc_old;
   kai_loc_old.setZero();

   kai_loc_old.topRows<3>() = v_loc_old;
   kai_loc_old.bottomRows<3>() = w_loc_old;

   kai_b_old = Bii.colPivHouseholderQr().solve ( kai_loc_old );


   //Filter speed
   const float c = 400.0;
   MatrixXf kai_b_fil;
   kai_b_fil.resize ( 6, 1 );
   for ( unsigned int i = 0; i < 6; i++ )
   {
      kai_b_fil ( i, 0 ) = ( kai_b ( i, 0 )
                             + ( c * eigensolver.eigenvalues() ( i, 0 ) + 0.2 ) * kai_b_old ( i, 0 ) )
                           / ( 1.0 + c * eigensolver.eigenvalues() ( i, 0 ) + 0.2 );
      //kai_b_fil_d(i,0) = (kai_b_d(i,0) + 0.2*kai_b_old_d(i,0))/(1.0 + 0.2);
   }

   //Transform filtered speed to local and then absolute reference systems
   MatrixXf kai_loc_fil;
   Matrix<float, 3, 1> v_abs_fil, w_abs_fil;
   v_abs_fil.setZero();
   w_abs_fil.setZero();

   kai_loc_fil.resize ( 6, 1 );
   kai_loc_fil = Bii.inverse().colPivHouseholderQr().solve ( kai_b_fil );

   q = AngleAxis<double> ( cam_pose ( 3,0 ),  Matrix<double,3,1>::UnitZ() ) *
       AngleAxis<double> ( cam_pose ( 4,0 ),  Matrix<double,3,1>::UnitY() ) *
       AngleAxis<double> ( cam_pose ( 5,0 ),  Matrix<double,3,1>::UnitX() );
   inv_trans = q.toRotationMatrix();

   v_abs_fil = inv_trans.cast<float>() * kai_loc_fil.topRows ( 3 );
   w_abs_fil = inv_trans.cast<float>() * kai_loc_fil.bottomRows ( 3 );

   kai_abs.topRows<3>() = v_abs_fil;
   kai_abs.bottomRows<3>() = w_abs_fil;

   double pitch, roll;
   Matrix<double, 3, 1> w_euler_d;
   w_euler_d.setZero();
   pitch = cam_pose ( 4,0 );
   roll = cam_pose ( 5,0 );

   w_euler_d ( 0, 0 ) = kai_loc_fil ( 4, 0 ) * sin ( roll ) / cos ( pitch )
                        + kai_loc_fil ( 5, 0 ) * cos ( roll ) / cos ( pitch );
   w_euler_d ( 1, 0 ) = kai_loc_fil ( 4, 0 ) * cos ( roll )
                        - kai_loc_fil ( 5, 0 ) * sin ( roll );
   w_euler_d ( 2, 0 ) = kai_loc_fil ( 3, 0 )
                        + kai_loc_fil ( 4, 0 ) * sin ( roll ) * tan ( pitch )
                        + kai_loc_fil ( 5, 0 ) * cos ( roll ) * tan ( pitch );

   //update camera pose
   cam_pose ( 0,0 ) += v_abs_fil ( 0, 0 ) / fps; //x
   cam_pose ( 1,0 ) += v_abs_fil ( 1, 0 ) / fps; //y
   cam_pose ( 2,0 ) += v_abs_fil ( 2, 0 ) / fps; //z
   cam_pose ( 3,0 ) += w_euler_d ( 0, 0 ) / fps; //yaw
   cam_pose ( 4,0 ) += w_euler_d ( 1, 0 ) / fps; //pitch
   cam_pose ( 5,0 ) += w_euler_d ( 2, 0 ) / fps; //roll

   //Originally, the odometry coordinate is: X-forward, Z-downward
   //Translation and Rotation in Msg form
   geometry_msgs::Point t_msg;
   geometry_msgs::Quaternion r_msg;

   t_msg.x = cam_pose ( 0,0 );
   t_msg.y = cam_pose ( 1,0 );
   t_msg.z = cam_pose ( 2,0 );

   //Translation and Rotation in tf form
   tf::Vector3 t_tf;
   tf::Quaternion q_tf;

   q_tf.setRPY ( cam_pose ( 5,0 ) , cam_pose ( 4,0 ), cam_pose ( 3,0 ) );

   tf::pointMsgToTF ( t_msg,t_tf );
   tf::quaternionTFToMsg ( q_tf, r_msg );

   //Publish tf
   //Here, convert the odometry coordinate as: Z-forward, Y-downward, the same as the camera_rgb_optical_frame
   tf::Transform initial_base2sensor = tf::Transform ( tf::createQuaternionFromRPY ( 0, -1.57 , -1.57 ), tf::Vector3 ( 0,0,0 ) );
   Transform_cur = initial_base2sensor * tf::Transform ( q_tf, t_tf ) * initial_base2sensor.inverse();
   tfBroadcaster.sendTransform ( tf::StampedTransform ( Transform_cur, stamp, "camera_init", "camera" ) );


   //Publish accumulated odometry
   nav_msgs::Odometry odom;
   odom.header.stamp = stamp;
   odom.header.frame_id = "/camera_init";
   odom.child_frame_id = "/camera";


   tf::pointTFToMsg ( Transform_cur.getOrigin(), t_msg );
   tf::quaternionTFToMsg ( Transform_cur.getRotation(),r_msg );

   odom.pose.pose.position = t_msg;
   odom.pose.pose.orientation = r_msg;
   if ( odomPub.getNumSubscribers() )
      odomPub.publish ( odom );

//    fprintf ( fileVar,"%f %f %f %f %f %f %f %f %f\n",
//              odom.header.stamp.toSec(),
//              odom.pose.pose.position.x,
//              odom.pose.pose.position.y,
//              odom.pose.pose.position.z,
//              odom.pose.pose.orientation.x,
//              odom.pose.pose.orientation.y,
//              odom.pose.pose.orientation.z,
//              odom.pose.pose.orientation.w,
//              ( ros::WallTime::now()-start ).toSec() );

   //Publish pose with covariance
   geometry_msgs::PoseWithCovarianceStamped pcov;
   pcov.header.stamp = stamp;
   pcov.header.frame_id = "camera_init";
   pcov.pose.pose.position = t_msg;
   pcov.pose.pose.orientation  = r_msg;

   for ( int i = 0; i < 3; i ++ )
      for ( int j = 0; j < 3; j ++ )
         pcov.pose.covariance[i * 3 + j] = est_cov ( i,j );

   if ( poseCovPub.getNumSubscribers() )
      poseCovPub.publish ( pcov );


   // Publish Rangeflow relative odometry
   Relative_trans = Transform_last.inverse() * Transform_cur;
   last_relative_trans = Relative_trans;

   rel_odom.header.stamp = stamp;
   rel_odom.header.frame_id = "/camera_pre";
   rel_odom.child_frame_id = "/camera";

   inv_odom.header.stamp     =  stamp;
   inv_odom.header.frame_id  =  "/camera_pre";
   inv_odom.child_frame_id   =  "/camera";

   tf::poseTFToMsg ( Relative_trans,rel_odom.pose.pose );
   tf::poseTFToMsg ( Relative_trans.inverse(), inv_odom.pose.pose );

   for ( int i = 0; i < 3; i ++ )
      for ( int j = 0; j < 3; j ++ )
         rel_odom.pose.covariance[i * 3 + j] = est_cov ( i,j );

   for ( int i = 0; i < 3; i ++ )
      for ( int j = 0; j < 3; j ++ )
         inv_odom.pose.covariance[i * 3 + j] = est_cov ( i,j );


   double dt = ( current_ts_ - last_ts_ ).toSec();
   if ( dt > 0 )
   {
      rel_odom.twist.twist.linear.x = rel_odom.pose.pose.position.x / dt;
      rel_odom.twist.twist.linear.y = rel_odom.pose.pose.position.y / dt;
      rel_odom.twist.twist.linear.z = rel_odom.pose.pose.position.z / dt;

      inv_odom.twist.twist.linear.x = inv_odom.pose.pose.position.x / dt;
      inv_odom.twist.twist.linear.y = inv_odom.pose.pose.position.y / dt;
      inv_odom.twist.twist.linear.z = inv_odom.pose.pose.position.z / dt;

      tf::Quaternion tfq = Relative_trans.getRotation();
      double angle = tfq.getAngle();
      tf::Vector3 axis = tfq.getAxis();
      tf::Vector3 twist = axis * ( angle/dt );
      rel_odom.twist.twist.angular.x = twist [0];
      rel_odom.twist.twist.angular.y = twist [1];
      rel_odom.twist.twist.angular.z = twist [2];

      tf::Quaternion inv_tfq = Relative_trans.inverse().getRotation();
      double inv_angle = inv_tfq.getAngle();
      tf::Vector3 inv_axis = inv_tfq.getAxis();
      tf::Vector3 inv_twist = inv_axis * ( inv_angle/dt );
      inv_odom.twist.twist.angular.x = inv_twist [0];
      inv_odom.twist.twist.angular.y = inv_twist [1];
      inv_odom.twist.twist.angular.z = inv_twist [2];

   }

//    if ( rel_odomPub.getNumSubscribers() )
//       rel_odomPub.publish ( rel_odom );

   if ( inv_odomPub.getNumSubscribers() )
      inv_odomPub.publish ( inv_odom );


   // set the condition number to the index nubmer from leas squares optimization
   estimation_quality = condition_index;

   Transform_last = Transform_cur;

   double dwalltime = ( ros::WallTime::now() - startTime ).toSec();
   ROS_DEBUG_STREAM ( "Filter Speed And Pose Update Took:  " << dwalltime << "s " );
}

void rangeflow::publishVOState(const ros::Time& stamp)
{

   //Complete fovis_state msg, with estimation status, pose with covariance etc.
   rangeflow_odom::vo_state state_msg;
   state_msg.header.stamp = stamp;
   state_msg.header.frame_id = rel_odom.header.frame_id;
   state_msg.referenceStamp = last_ts_;
   state_msg.conditionNumber = estimation_quality;
   if ( std::isnan ( rel_odom.pose.pose.position.x )
         || std::isnan ( rel_odom.pose.pose.position.y )
         || std::isnan ( rel_odom.pose.pose.position.z )
         || std::isnan ( rel_odom.pose.pose.orientation.x )
         || std::isnan ( rel_odom.pose.pose.orientation.y )
         || std::isnan ( rel_odom.pose.pose.orientation.z )
         || std::isnan ( rel_odom.pose.pose.orientation.w )
         || std::isnan ( rel_odom.twist.twist.angular.x )
         || std::isnan ( rel_odom.twist.twist.angular.y )
         || std::isnan ( rel_odom.twist.twist.angular.z )
         || std::isnan ( rel_odom.twist.twist.linear.x )
         || std::isnan ( rel_odom.twist.twist.linear.y )
         || std::isnan ( rel_odom.twist.twist.linear.z ) )
   {
      state_msg.pose.pose.position.x = 0;
      state_msg.pose.pose.position.y = 0;
      state_msg.pose.pose.position.z = 0;

      state_msg.pose.pose.orientation.x = 0;
      state_msg.pose.pose.orientation.y = 0;
      state_msg.pose.pose.orientation.z = 0;
      state_msg.pose.pose.orientation.w = 1;

      state_msg.twist.twist.angular.x = 0;
      state_msg.twist.twist.angular.y = 0;
      state_msg.twist.twist.angular.z = 0;

      state_msg.twist.twist.linear.x = 0;
      state_msg.twist.twist.linear.y = 0;
      state_msg.twist.twist.linear.z = 0;
   }
   else if(fabs(rel_odom.twist.twist.linear.x) < 0.75 
	&& fabs(rel_odom.twist.twist.linear.y) < 0.75  
	&& fabs(rel_odom.twist.twist.linear.z) < 0.75)
   {
      state_msg.pose = rel_odom.pose;
      state_msg.twist = rel_odom.twist;
   }
   else 
   {
     state_msg.conditionNumber = 1000000;
   }   
   voStatePub.publish ( state_msg );
}


void rangeflow::callback ( const sensor_msgs::Image::ConstPtr& msg_image,
                           const sensor_msgs::Image::ConstPtr& msg_depth,
                           const sensor_msgs::CameraInfo::ConstPtr& msg_camInfo )
{
   ros::WallTime startTime = ros::WallTime::now();

   frame_id = msg_depth->header.frame_id;
   current_ts_ = msg_depth->header.stamp;

   PointCloud::Ptr cloud ( new PointCloud );
   cloud->header.frame_id = frame_id;
   cloud->height = 1;
   cloud->width = 1;
   cloud->is_dense = true;

   int point_counter = 0;
   pcl::PointXYZ pt;

   // Update camera model
   model_.fromCameraInfo ( msg_camInfo );

   vals[0] = model_.fx();
   vals[1] = 0.0;
   vals[2] = model_.cx();
   vals[3] = 0.0;
   vals[4] = model_.fy();
   vals[5] = model_.cy();
   vals[6] = 0.0;
   vals[7] = 0.0;
   vals[8] = 1.0;

   // Use correct principal point from calibration
   float center_x = model_.cx();
   float center_y = model_.cy();

   float constant_x = 1.0 / model_.fx();
   float constant_y = 1.0 / model_.fy();

   ROS_DEBUG_STREAM ( "center_x: " << center_x );
   ROS_DEBUG_STREAM ( "center_y: " << center_y );
   ROS_DEBUG_STREAM ( "constant_x: " << 1/constant_x );
   ROS_DEBUG_STREAM ( "constant_y: " << 1/constant_y );
   ROS_DEBUG_STREAM ( "encoding is: " << msg_depth->encoding );


   if ( msg_depth->encoding == sensor_msgs::image_encodings::TYPE_16UC1 )
   {
      int row_step = msg_depth->step / sizeof ( uint16_t );
      const uint16_t* depth_row = reinterpret_cast<const uint16_t*> ( &msg_depth->data[0] );

      for ( int v = ( int ) msg_depth->height - 1; v >= 0; --v )
      {
         for ( int u = ( int ) msg_depth->width - 1; u >= 0; --u )
         {
            uint16_t raw = depth_row[v * row_step + u];
            if ( ( v % downsample == 0 ) && ( u % downsample == 0 ) && ( raw!=0 ) )
            {
               float depth_value = raw * 0.001f;

               depth_wf ( v / downsample, u / downsample ) = depth_value;

               // Fill in XYZ
               pt.x = ( u - center_x ) * depth_value * constant_x;
               pt.y = ( v - center_y ) * depth_value * constant_y;
               pt.z = depth_value;

               point_counter++;
               cloud->points.push_back ( pt );
            }
         }
      }
   }
   else if ( msg_depth->encoding == sensor_msgs::image_encodings::TYPE_32FC1 )
   {
      int row_step = msg_depth->step / sizeof ( float );
      const float* depth_row = reinterpret_cast<const float*> ( &msg_depth->data[0] );

      for ( int v = ( int ) msg_depth->height - 1; v >= 0; --v )
      {
         for ( int u = ( int ) msg_depth->width - 1; u >= 0; --u )
         {
            float depth_value = depth_row[v * row_step + u];
            if ( ( v % downsample == 0 ) && ( u % downsample == 0 ) && ( !std::isnan ( depth_value ) ) )
            {
               depth_wf ( v / downsample, u / downsample ) = depth_value;

               // Fill in XYZ
               pt.x = ( u - center_x ) * depth_value * constant_x;
               pt.y = ( v - center_y ) * depth_value * constant_y;
               pt.z = depth_value;

               point_counter++;
               cloud->points.push_back ( pt );
            }
         }
      }
   }

   OdometryCalculation();


   cv_bridge::CvImagePtr cv_image;
   cv_bridge::CvImagePtr cv_depth;

   cv::Mat image1;
   cv::Mat depth1;

   try
   {
      if ( msg_image->encoding == image_encodings::RGB8 )
      {
         cv_image = cv_bridge::toCvCopy ( msg_image, sensor_msgs::image_encodings::RGB8 );
         cv::Mat opencv_image = cv_image->image;
         cvtColor ( cv_image->image, image1,CV_RGB2GRAY );
      }
      else
      {
         cv_image   = cv_bridge::toCvCopy ( msg_image , image_encodings::MONO8 );
         image1 = cv_image->image;
      }
      cv_depth   = cv_bridge::toCvCopy ( msg_depth );
      depth1 = cv_depth->image;
   }
   catch ( cv_bridge::Exception& e )
   {
      ROS_ERROR ( "cv_bridge exception: %s", e.what() );
      estimation_quality = 10000;
      publishVOState(msg_image->header.stamp);
      return;
   }

   if ( first_frame )
   {
      image0 = image1;
      depth0 = depth1;

      first_frame = false;
      publishVOState(msg_image->header.stamp);
      return;
   }

   //Calculate the Average intensity of the gray image.
   cv::Scalar avgPixelIntensity = cv::mean ( image1 );
   ROS_DEBUG_STREAM ( "Average Intensity is: " << avgPixelIntensity.val[0] );

   if ( condition_index > cIndex && avgPixelIntensity.val[0] > gray_threshold && dvo_refine )
   {
      ros::WallTime start = ros::WallTime::now();

      if ( image0.empty() || depth0.empty() || image1.empty() || depth1.empty() )
      {
         cout << "Data (rgb or depth images) is empty." << endl;
         estimation_quality = 10000;
	 publishVOState(msg_image->header.stamp);
         return;
      }


      Mat Rt;

      vector<int> iterCounts ( pyramid_level );
      if ( pyramid_level == 4 )
      {
         iterCounts[0] = level0_iterCounts;
         iterCounts[1] = level1_iterCounts;
         iterCounts[2] = level2_iterCounts;
         iterCounts[3] = level3_iterCounts;
      }
      if ( pyramid_level == 3 )
      {
         iterCounts[0] = level0_iterCounts;
         iterCounts[1] = level1_iterCounts;
         iterCounts[2] = level2_iterCounts;
      }
      if ( pyramid_level == 2 )
      {
         iterCounts[0] = level0_iterCounts;
         iterCounts[1] = level1_iterCounts;
      }
      if ( pyramid_level == 1 )
      {
         iterCounts[0] = level0_iterCounts;
      }

      vector<float> minGradMagnitudes ( pyramid_level );
      if ( pyramid_level == 4 )
      {
         minGradMagnitudes[0] = level0_minGradMagnitudes;
         minGradMagnitudes[1] = level1_minGradMagnitudes;
         minGradMagnitudes[2] = level2_minGradMagnitudes;
         minGradMagnitudes[3] = level3_minGradMagnitudes;
      }
      if ( pyramid_level == 3 )
      {
         minGradMagnitudes[0] = level0_minGradMagnitudes;
         minGradMagnitudes[1] = level1_minGradMagnitudes;
         minGradMagnitudes[2] = level2_minGradMagnitudes;
      }
      if ( pyramid_level == 2 )
      {
         minGradMagnitudes[0] = level0_minGradMagnitudes;
         minGradMagnitudes[1] = level1_minGradMagnitudes;
      }
      if ( pyramid_level == 1 )
      {
         minGradMagnitudes[0] = level0_minGradMagnitudes;
      }



      const float minDepth = 0.f; //in meters
      const float maxDepth = 7.f; //in meters
      const float maxDepthDiff = 0.07f; //in meters

      Mat initRt = Mat::eye ( 4,4,CV_64FC1 );
      Eigen::Matrix4f lastRt;
      pcl_ros::transformAsMatrix ( last_relative_trans.inverse(), lastRt );

      Eigen::Matrix4d lstRt ( lastRt.cast<double>() );
      eigen2cv ( lstRt, initRt );
      ROS_DEBUG_STREAM ( "Last relative transform is: " << initRt );

      //Estimate the rigid body motion from frame0 to frame1.
      bool isFound = DVO::RGBDOdometry ( Rt, Mat(),
                                         image0, depth0, Mat(),
                                         image1, depth1, Mat(),
                                         cameraMatrix, minDepth, maxDepth, maxDepthDiff,
                                         iterCounts, minGradMagnitudes, 0 );
//      cout << "Rt = " << Rt << endl;
      if ( !isFound )
      {
         cout << "Rigid body motion cann't be estimated for given RGBD data."  << endl;
         estimation_quality = 10000;
	 
	 publishVOState(msg_image->header.stamp);
         return;
      }

      Eigen::Matrix4f rel_trans = Eigen::Matrix4f::Identity();
      cv2eigen ( Rt,rel_trans );

      ROS_DEBUG_STREAM ( "Relative transform is: " << rel_trans );
      ROS_DEBUG_STREAM ( "Dense visual estimation took: " << ( ros::WallTime::now() - start ).toSec() << "s " );


      //Translation and Rotation in tf form
      tf::Vector3 t_tf ( rel_trans ( 0,3 ), rel_trans ( 1,3 ), rel_trans ( 2,3 ) );
      tf::Matrix3x3 r_tf ( rel_trans ( 0,0 ), rel_trans ( 0,1 ), rel_trans ( 0,2 ),
                           rel_trans ( 1,0 ), rel_trans ( 1,1 ), rel_trans ( 1,2 ),
                           rel_trans ( 2,0 ), rel_trans ( 2,1 ), rel_trans ( 2,2 ) );

      if ( t_tf.length() < 0.2 )
      {
         //   Total_tf = tf::Transform ( r_tf,t_tf ) * Total_tf;
         Transform_cur = Transform_cur * tf::Transform ( r_tf,t_tf ).inverse();
      }
      else
      {
         // assume constant motion constraint
         // Here is a bug problem, it sometimes gives "nan" value.
         // Need to be considered in the future.
//          Transform_cur = Transform_cur * last_relative_trans;
         Transform_cur = Transform_cur ;
      }


      //Depthflow odometry coordinate is: Z-downward, X-forwad, We should change back to this cooridate
      tf::Transform initial_base2sensor = tf::Transform ( tf::createQuaternionFromRPY ( 0.0, -1.57 , -1.57 ),
                                          tf::Vector3 ( 0,0,0 ) );
      tf::Transform cam_trans = initial_base2sensor.inverse() * Transform_cur * initial_base2sensor;

      double roll, pitch, yaw;
      cam_trans.getBasis().getRPY ( roll, pitch, yaw );

//       cam_pose.setFromValues ( cam_trans.getOrigin().getX(),
//                                cam_trans.getOrigin().getY(),
//                                cam_trans.getOrigin().getZ(),
//                                yaw, pitch, roll );
      cam_pose ( 0,0 ) = cam_trans.getOrigin().getX();
      cam_pose ( 1,0 ) = cam_trans.getOrigin().getY();
      cam_pose ( 2,0 ) = cam_trans.getOrigin().getZ();
      cam_pose ( 3,0 ) = yaw;
      cam_pose ( 4,0 ) = pitch;
      cam_pose ( 5,0 ) = roll;

      tfBroadcaster.sendTransform ( tf::StampedTransform ( Transform_cur, msg_image->header.stamp, "camera_init", "camera" ) );


      //Publish accumulated odometry
      nav_msgs::Odometry odom;
      odom.header.stamp = msg_image->header.stamp;
      odom.header.frame_id = "/camera_init";
      odom.child_frame_id = "/camera";

      geometry_msgs::Point t_msg;
      geometry_msgs::Quaternion r_msg;

      tf::pointTFToMsg ( Transform_cur.getOrigin(), t_msg );
      tf::quaternionTFToMsg ( Transform_cur.getRotation(),r_msg );

      odom.pose.pose.position = t_msg;
      odom.pose.pose.orientation = r_msg;
      if ( odomPub.getNumSubscribers() )
         odomPub.publish ( odom );


      // Publish Dense relative odometry
      Relative_trans = Transform_last.inverse() * Transform_cur;
      last_relative_trans = Relative_trans;

      rel_odom.header.stamp = msg_image->header.stamp;
      rel_odom.header.frame_id = "/camera_pre";
      rel_odom.child_frame_id = "/camera";

      inv_odom.header.stamp     =  msg_image->header.stamp;
      inv_odom.header.frame_id  =  "/camera_pre";
      inv_odom.child_frame_id   =  "/camera";

      tf::poseTFToMsg ( Relative_trans,rel_odom.pose.pose );
      tf::poseTFToMsg ( Relative_trans.inverse(), inv_odom.pose.pose );

      double dt = ( current_ts_ - last_ts_ ).toSec();
      if ( dt > 0 )
      {
         rel_odom.twist.twist.linear.x = rel_odom.pose.pose.position.x / dt;
         rel_odom.twist.twist.linear.y = rel_odom.pose.pose.position.y / dt;
         rel_odom.twist.twist.linear.z = rel_odom.pose.pose.position.z / dt;

         inv_odom.twist.twist.linear.x = inv_odom.pose.pose.position.x / dt;
         inv_odom.twist.twist.linear.y = inv_odom.pose.pose.position.y / dt;
         inv_odom.twist.twist.linear.z = inv_odom.pose.pose.position.z / dt;

         tf::Quaternion tfq = Relative_trans.getRotation();
         double angle = tfq.getAngle();
         tf::Vector3 axis = tfq.getAxis();
         tf::Vector3 twist = axis * ( angle/dt );
         rel_odom.twist.twist.angular.x = twist [0];
         rel_odom.twist.twist.angular.y = twist [1];
         rel_odom.twist.twist.angular.z = twist [2];

         tf::Quaternion inv_tfq = Relative_trans.inverse().getRotation();
         double inv_angle = inv_tfq.getAngle();
         tf::Vector3 inv_axis = inv_tfq.getAxis();
         tf::Vector3 inv_twist = inv_axis * ( inv_angle/dt );
         inv_odom.twist.twist.angular.x = inv_twist [0];
         inv_odom.twist.twist.angular.y = inv_twist [1];
         inv_odom.twist.twist.angular.z = inv_twist [2];

      }

//       if ( rel_odomPub.getNumSubscribers() )
//          rel_odomPub.publish ( rel_odom );

      if ( inv_odomPub.getNumSubscribers() )
         inv_odomPub.publish ( inv_odom );

      // set a constant condition number
      estimation_quality = 600;

      Transform_last = Transform_cur;

   }
   else
   {
      filterSpeedAndPoseUpdate ( msg_depth->header.stamp , startTime );
   }

   image0 = image1;
   depth0 = depth1;

   cloud->width = point_counter;
   cloud->header.seq = msg_depth->header.seq;
   cloud->header.stamp = pcl_conversions::toPCL ( msg_depth->header ).stamp;
   cloud_pub.publish ( cloud );

   publishVOState(msg_image->header.stamp);
   
//    //Complete fovis_state msg, with estimation status, pose with covariance etc.
//    rangeflow_odom::vo_state state_msg;
//    state_msg.header.stamp = msg_image->header.stamp;
//    state_msg.header.frame_id = rel_odom.header.frame_id;
//    state_msg.referenceStamp = last_ts_;
//    state_msg.conditionNumber = estimation_quality;
//    if ( std::isnan ( rel_odom.pose.pose.position.x )
//      || std::isnan ( rel_odom.pose.pose.position.y )
//      || std::isnan ( rel_odom.pose.pose.position.z )
//      || std::isnan ( rel_odom.pose.pose.orientation.x )
//      || std::isnan ( rel_odom.pose.pose.orientation.y )
//      || std::isnan ( rel_odom.pose.pose.orientation.z )
//      || std::isnan ( rel_odom.pose.pose.orientation.w )
//      || std::isnan ( rel_odom.twist.twist.angular.x )
//      || std::isnan ( rel_odom.twist.twist.angular.y )
//      || std::isnan ( rel_odom.twist.twist.angular.z )
//      || std::isnan ( rel_odom.twist.twist.linear.x )
//      || std::isnan ( rel_odom.twist.twist.linear.y )
//      || std::isnan ( rel_odom.twist.twist.linear.z ) )
//    {
//       state_msg.pose.pose.position.x = 0;
//       state_msg.pose.pose.position.y = 0;
//       state_msg.pose.pose.position.z = 0;
//
//       state_msg.pose.pose.orientation.x = 0;
//       state_msg.pose.pose.orientation.y = 0;
//       state_msg.pose.pose.orientation.z = 0;
//       state_msg.pose.pose.orientation.w = 1;
//
//       state_msg.twist.twist.angular.x = 0;
//       state_msg.twist.twist.angular.y = 0;
//       state_msg.twist.twist.angular.z = 0;
//
//       state_msg.twist.twist.linear.x = 0;
//       state_msg.twist.twist.linear.y = 0;
//       state_msg.twist.twist.linear.z = 0;
//    }
//    else
//    {
//       state_msg.pose = rel_odom.pose;
//       state_msg.twist = rel_odom.twist;
//    }
//    voStatePub.publish ( state_msg );

   last_ts_ = current_ts_;

   double dwalltime = ( ros::WallTime::now() - startTime ).toSec();
   ROS_DEBUG_STREAM ( "Condition number is: " << estimation_quality );
   ROS_DEBUG_STREAM ( "Rangeflow Odometry Estimation took " << dwalltime << "s " );

//   fprintf (fileVar, "%f\n", dwalltime);

//    fprintf ( fileVar,"%f %f %f %f %f %f %f %f %f\n",
//              odom.header.stamp.toSec(),
//              odom.pose.pose.position.x,
//              odom.pose.pose.position.y,
//              odom.pose.pose.position.z,
//              odom.pose.pose.orientation.x,
//              odom.pose.pose.orientation.y,
//              odom.pose.pose.orientation.z,
//              odom.pose.pose.orientation.w,
//              ( ros::WallTime::now()-start ).toSec() );
}

int main ( int argc, char** argv )
{
   ros::init ( argc, argv, "range_flow" );

   ros::NodeHandle n;
   ros::NodeHandle nh ( "~" );

   rangeflow odom ( n, nh );

   ros::spin();
   return 0;
}

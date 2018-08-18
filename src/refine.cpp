#include <cassert>
#include <iostream>
#include <fstream>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <boost/version.hpp>
#include <boost/thread.hpp>
#include <boost/thread/future.hpp>
#include <boost/format.hpp>
#include "boost/filesystem/path.hpp"
#include "boost/filesystem/operations.hpp"

#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

#include "nav_msgs/Odometry.h"
#include "tf/transform_broadcaster.h"
#include "tf/transform_listener.h"

#include <pcl/filters/normal_space.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/sampling_surface_normal.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/registration/transformation_estimation_lm.h>

#include <pcl/registration/correspondence_rejection_distance.h>
#include <pcl/registration/correspondence_estimation_backprojection.h>
#include <pcl/registration/correspondence_estimation_normal_shooting.h>

#include <pcl/features/normal_3d.h>

using namespace std;
using namespace pcl;

class Odometer
{
    typedef pcl::PointCloud<pcl::PointNormal> DP;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorX;
    typedef Eigen::Matrix<double, 4, 1> Vector4;
    typedef Eigen::Matrix<double, 4, 4> Matrix4;

    FILE* fileVar;

    ros::NodeHandle& n;
    ros::NodeHandle& nh;

    // Subscribers
    ros::Subscriber cloudSub;

    // Publisher
    ros::Publisher voxGrid_cloud;
    ros::Publisher passThrough_cloud;
    ros::Publisher normalSpaceSampling_cloud;
    ros::Publisher localmapPub;
    ros::Publisher globalmapPub;
    ros::Publisher odomPub;
    ros::Publisher do_pub;

    ros::Time mapCreationTime;

    // Main algorithm definition
    DP *localMapPointCloud;

    int inputQueueSize;

    tf::TransformListener tfListener;
    tf::TransformBroadcaster tfBroadcaster;

    boost::thread publishThread;
    boost::mutex publishLock;
    ros::Time publishStamp;

    // multi-threading mapper
    typedef boost::packaged_task<DP*> MapBuildingTask;
    typedef boost::unique_future<DP*> MapBuildingFuture;
    boost::thread mapBuildingThread;
    MapBuildingTask mapBuildingTask;
    MapBuildingFuture mapBuildingFuture;

    bool processingNewCloud;
    bool mapBuildingInProgress;

    int LocalMapNum;
    unsigned int minReadingPointCount;

    string odomFrame;
    string mapFrame;

    //Define some transformation matrix
    Eigen::Matrix4f Trans_Odom2Map;
    Eigen::Matrix4f Trans_Odom2Camera;
    Eigen::Matrix4f Trans_Camera2Map;
    Eigen::Matrix4f Trans_AsusInit2AsusFrame;
    Eigen::Matrix4f Trans_AsusInit2AsusFrame_Last;

    Eigen::Matrix4f LastTrans_Camera2Map;
    Eigen::Matrix4f CurrTrans_Camera2Map;
    Eigen::Matrix4f Relative_Trans;

    int random_sampling_num;
    double voxel_leafsize;
    bool useMap;


    double max_corresdist;
    int max_iterations;
    double max_trans_epsilon;
    int min_number_correspondences;
    double euclidean_fitness_epsilon;
    double rej_max_corresdist;
    double mapVoxelLeafSize;

    int localMapNum;

    double prvTime;
    double currTime;

    DP *tempCloud;

    nav_msgs::Odometry measurementMsg;

    int skip;
    int skipNum;
    bool refine;
    bool useRandom_sampling;
    bool useNDT;

    double max_epsilon;
    double step_size;
    double resolution;
    int max_iterations_ndt;

    tf::Transform rel_transform_last;
    Eigen::Matrix4f Trans_InterFrame_last;

  public:
    Odometer ( ros::NodeHandle& n, ros::NodeHandle& nh );
    ~Odometer();

  protected:
    void gotCloud ( const pcl::PCLPointCloud2ConstPtr& cloudMsgIn );
    void processCloud ( std::unique_ptr<DP> cloud, const std::string& scannerFrame, const ros::Time& stamp, uint32_t seq );
    DP* updateMap ( DP* newPointCloud, Eigen::Matrix4f Ticp, bool updateExisting );
    void setMap ( DP* newPointCloud );
    void publishLoop ( double publishPeriod );
    void publishTransform();
    nav_msgs::Odometry EigenMatrix2OdomMsg ( const Eigen::Matrix4f& inTr, double deltaT, const std::string& frame_id,
        const std::string& child_frame_id, const ros::Time& stamp );

};

Odometer::Odometer ( ros::NodeHandle& n, ros::NodeHandle& nh ) :
  n ( n ), nh ( nh ), localMapPointCloud ( new DP ), mapBuildingInProgress ( false ), minReadingPointCount (
    500 ), odomFrame ( "odom" ), mapFrame ( "world" ), Trans_Camera2Map ( Eigen::Matrix4f::Identity ( 4, 4 ) ), prvTime (
      0.0 ), currTime ( 0.0 ), tempCloud ( new DP )
{

  nh.param<string> ( "odomFrame", odomFrame, "camera_rgb_optical_frame" );
  nh.param<string> ( "mapFrame", mapFrame, "world" );

  nh.param<int> ( "localMapNum", localMapNum, 6000 );

  nh.param<bool> ( "useRandom_sampling", useRandom_sampling, false );
  nh.param<int> ( "random_sampling_num", random_sampling_num, 3500 );
  nh.param<double> ( "voxel_leafsize", voxel_leafsize, 0.05 );

  nh.param<double> ( "setMaxCorresDist", max_corresdist, 0.05 );
  nh.param<int> ( "setMaxIterationNum", max_iterations, 50 );
  nh.param<double> ( "setTransEpsilon", max_trans_epsilon, 1e-8 );
  nh.param<int> ( "min_number_correspondences", min_number_correspondences, 300 );
  nh.param<double> ( "euclidean_fitness_epsilon", euclidean_fitness_epsilon, 0.001 );

  nh.param<double> ( "rej_max_corresdist", rej_max_corresdist, 0.3 );
  nh.param<double> ( "mapVoxelLeafSize", mapVoxelLeafSize, 0.05 );

  nh.param<int> ( "skipNum", skipNum, 3 );
  nh.param<bool> ( "refine", refine, true );


  cloudSub = n.subscribe ( "input_cloud", inputQueueSize, &Odometer::gotCloud, this );

  voxGrid_cloud = n.advertise<sensor_msgs::PointCloud2> ( "/voxelgrid/cloud", 2, true );
  passThrough_cloud = n.advertise<sensor_msgs::PointCloud2> ( "/passThrough/cloud", 2, true );

  localmapPub = n.advertise<sensor_msgs::PointCloud2> ( "local_point_map", 2, true );
  odomPub = n.advertise<nav_msgs::Odometry> ( "icp_odometry", 50, true );
  do_pub = n.advertise<nav_msgs::Odometry> ( "icp_measurement", 50, true );

  //refreshing tf transform thread
// publishThread = boost::thread(boost::bind(&Odometer::publishLoop, this, 0.01));
  Trans_AsusInit2AsusFrame = Eigen::Matrix4f::Identity();
  Trans_AsusInit2AsusFrame_Last = Eigen::Matrix4f::Identity();
  skip = 0;

}

Odometer::~Odometer()
{
  delete tempCloud;
}

void Odometer::gotCloud ( const pcl::PCLPointCloud2ConstPtr& cloudMsgIn )
{
  if ( skip < skipNum )
  {
    skip ++;
    return;
  }
  skip = 0;

  ros::WallTime startTime = ros::WallTime::now();

  ROS_INFO_STREAM ( "Input point cloud size: "<< cloudMsgIn->width * cloudMsgIn->height );
  pcl::PCLPointCloud2 cloud_filtered;

  if ( useRandom_sampling )
  {
    pcl::RandomSample<pcl::PCLPointCloud2> sor;
    sor.setInputCloud ( cloudMsgIn );
    sor.setSample ( random_sampling_num );
    sor.setSeed ( rand() );
    sor.filter ( cloud_filtered );
    double deltaT = ( ros::WallTime::now() - startTime ).toSec();
    ROS_INFO_STREAM ( "Random downsampling took: " << deltaT << "s " );
  }
  else
  {
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    sor.setInputCloud ( cloudMsgIn );
    sor.setLeafSize ( voxel_leafsize, voxel_leafsize, voxel_leafsize );
    sor.filter ( cloud_filtered );
    double deltaT = ( ros::WallTime::now() - startTime ).toSec();
    ROS_INFO_STREAM ( "VoxelGrid downsampling took: " << deltaT << "s " );
    ROS_INFO_STREAM ( "Cloud size after downsampling: " << cloud_filtered.width * cloud_filtered.height );
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr src ( new pcl::PointCloud<pcl::PointXYZ> );
  pcl::fromPCLPointCloud2 ( cloud_filtered, *src );

  pcl::PointCloud<pcl::PointNormal>::Ptr ptCloud ( new pcl::PointCloud<pcl::PointNormal> );
  pcl::copyPointCloud ( *src, *ptCloud );

  std::unique_ptr<DP> cloud ( new DP ( *ptCloud ) );
  processCloud ( std::move ( cloud ), cloudMsgIn->header.frame_id, pcl_conversions::fromPCL ( cloud_filtered.header ).stamp,
                 cloudMsgIn->header.seq );

  double dt = ( ros::WallTime::now() - startTime ).toSec();
  ROS_INFO_STREAM ( "Total ICP Odometry Estimation took: " << dt << "s " << endl );
}

void Odometer::processCloud ( unique_ptr<DP> newPointCloud, const std::string& scannerFrame, const ros::Time& stamp,
                              uint32_t seq )
{
  currTime = stamp.toSec();

  processingNewCloud = true;
  mapCreationTime = stamp;

  // Convert point cloud
  if ( newPointCloud->size() == 0 )
  {
    ROS_ERROR ( "I found no good points in the cloud" );
    return;
  }

  string reason;
  if ( localMapPointCloud->size() == 0 )
  {
    publishLock.lock();
    Trans_Odom2Map = Eigen::Matrix4f::Identity();
    publishLock.unlock();
  }

  // Fetch transformation from scanner to odom
  if ( !tfListener.canTransform ( scannerFrame, odomFrame, stamp, &reason ) )
  {
    ROS_ERROR_STREAM ( "Cannot lookup TOdomToScanner(" << odomFrame<< " to " << scannerFrame << "):\n" << reason );
    return;
  }
  ROS_INFO_STREAM ( "scannerFrame is: " << scannerFrame );


  tf::StampedTransform trans_camera_to_Odom;
  try
  {
    tfListener.lookupTransform ( scannerFrame, odomFrame, stamp, trans_camera_to_Odom );
  }
  catch ( tf::TransformException ex )
  {
    ROS_ERROR ( "%s", ex.what() );
  }
  pcl_ros::transformAsMatrix ( trans_camera_to_Odom, Trans_Odom2Camera );
  ROS_DEBUG_STREAM ( "Trans_Odom2Camera is: " << Trans_Odom2Camera );

  pcl::PointCloud<pcl::PointNormal>::Ptr Original_Reading ( new pcl::PointCloud<pcl::PointNormal> );
  pcl::copyPointCloud ( *newPointCloud, *Original_Reading );

  //Step1: Filtering the points along a specified dimension and and the accepted interval values are set to (0, 7)
  //Because the measurement bigger than 7 meters are too noisy

  pcl::PassThrough<pcl::PointNormal> pass;
  pass.setInputCloud ( Original_Reading );
  pass.setFilterFieldName ( "z" );
  pass.setFilterLimits ( 0.0, 7.0 );
  pass.filter ( *Original_Reading );
  ROS_INFO_STREAM ( "Point cloud size after PassThrough filtering: " << Original_Reading->points.size() );

  sensor_msgs::PointCloud2 cloud_filtered;
  pcl::toROSMsg ( *Original_Reading, cloud_filtered );
  passThrough_cloud.publish ( cloud_filtered );


  if ( Original_Reading->size() < minReadingPointCount )
  {
    ROS_ERROR_STREAM ( "Not enough points in the newPointCloud: only " << Original_Reading->size() << "pts." );
    return;
  }

  // Initialize the map if empty
  if ( localMapPointCloud->size() == 0 )
  {
    DP *tempCloud ( new DP );
    pcl::copyPointCloud ( *Original_Reading, *tempCloud );
    setMap ( updateMap ( tempCloud, Trans_Camera2Map, false ) ); //The map's Coordinate is Global
    ROS_DEBUG_STREAM ( "create initial map successfuly!" );
    LastTrans_Camera2Map = Eigen::Matrix4f::Identity ( 4, 4 );
    return;
  }

  // if the future has completed, use the new map
  if ( mapBuildingInProgress && useMap && ( mapBuildingFuture.has_value() ) )
  {
    setMap ( mapBuildingFuture.get() );
    mapBuildingInProgress = false;
  }

  //Step4: Transform the selected new reading cloud into Global Map coordinate
  pcl::PointCloud<pcl::PointNormal>::Ptr src ( new pcl::PointCloud<pcl::PointNormal> );
  pcl::PointCloud<pcl::PointNormal>::Ptr tgt ( new pcl::PointCloud<pcl::PointNormal> );


  if ( refine )
  {
    // Fetch the relative transform between previous frame and current frame computed by depth flow method
    tfListener.waitForTransform ( "/camera_base", "/cam_zforward",stamp, ros::Duration ( 0.01 ) );
    if ( !tfListener.canTransform ( "camera_base", "cam_zforward",  stamp, &reason ) )
    {
      ROS_ERROR_STREAM ( "Cannot lookup camera_init to camera):\n" << reason );
      return;
    }
    tf::StampedTransform trans_asusInit2asusFrame;
    try
    {
      tfListener.lookupTransform ( "camera_base", "cam_zforward",  stamp, trans_asusInit2asusFrame );
    }
    catch ( tf::TransformException ex )
    {
      ROS_ERROR ( "%s", ex.what() );
    }
    pcl_ros::transformAsMatrix ( trans_asusInit2asusFrame, Trans_AsusInit2AsusFrame );
    ROS_DEBUG_STREAM ( "Trans_AsusInit2AsusFrame is: " << endl << Trans_AsusInit2AsusFrame );

    // The relative transform from current to previous frame, compute its inverse because the direction is opposite
    Eigen::Matrix4f Trans_InterFrame;
    Trans_InterFrame = ( Trans_AsusInit2AsusFrame_Last.inverse () * Trans_AsusInit2AsusFrame ).inverse();

    ROS_DEBUG_STREAM ( "Trans_AsusInit2AsusFrame_Last: " << endl << Trans_AsusInit2AsusFrame_Last );
    ROS_DEBUG_STREAM ( "Trans_InterFrame is: " << endl << Trans_InterFrame );


//     // Add the relative transform to previous global transform as the initial guess of ICP
//     tf::Transform rel_transform;
//     tf::Matrix3x3 tf_r;
//     for ( int i = 0; i < 3; i ++ )
//       for ( int j = 0; j < 3; j ++ )
//         tf_r[i][j] = Trans_InterFrame ( i,j );
//
//     tf::Vector3 tf_t;
//     tf_t[0] = Trans_InterFrame ( 0,3 );
//     tf_t[1] = Trans_InterFrame ( 1,3 );
//     tf_t[2] = Trans_InterFrame ( 2,3 );
//
//     rel_transform.setOrigin ( tf_t );
//     rel_transform.setBasis ( tf_r );
//
//     double yaw, pitch, roll;
//     rel_transform.getBasis().getRPY ( roll, pitch, yaw );
//
//     if ( rel_transform.getOrigin().length() <= 0.05
//          && std::abs ( yaw ) <= 0.05 )
//     {
//       Trans_Camera2Map = Trans_InterFrame * Trans_Camera2Map;
//     }
//     else
//     {
//       Trans_Camera2Map = Trans_InterFrame_last * Trans_Camera2Map;
//     }
//
//     Trans_InterFrame_last = Trans_InterFrame;

    Trans_Camera2Map = Trans_InterFrame * Trans_Camera2Map;
  }

  Eigen::Matrix4f Incr_Trans_Camera2Map;
  Incr_Trans_Camera2Map = LastTrans_Camera2Map.inverse() * Trans_Camera2Map;

  // Add the relative transform to previous global transform as the initial guess of ICP
  tf::Transform rel_transform;
  tf::Matrix3x3 tf_r;
  for ( int i = 0; i < 3; i ++ )
    for ( int j = 0; j < 3; j ++ )
      tf_r[i][j] = Incr_Trans_Camera2Map ( i,j );

  tf::Vector3 tf_t;
  tf_t[0] = Incr_Trans_Camera2Map ( 0,3 );
  tf_t[1] = Incr_Trans_Camera2Map ( 1,3 );
  tf_t[2] = Incr_Trans_Camera2Map ( 2,3 );

  rel_transform.setOrigin ( tf_t );
  rel_transform.setBasis ( tf_r );

  double yaw, pitch, roll;
  rel_transform.getBasis().getRPY ( roll, pitch, yaw );

  if ( rel_transform.getOrigin().length() >= 0.05 || std::abs ( yaw ) >= 0.05 || refine == false)
  {

    pcl::transformPointCloud ( *Original_Reading, *src,  Trans_Camera2Map );
    pcl::copyPointCloud ( *localMapPointCloud, *tgt );


    ROS_INFO_STREAM ( "reading cloud size is: " << src->size() );
    ROS_INFO_STREAM ( "reference cloud size is: " << tgt->size() );


    // Reading cloud prepared for iteration
    pcl::PointCloud<pcl::PointNormal>::Ptr input_transformed ( src );

    //Define temporary transformation and final transformation
//     Eigen::Matrix4f transformation_ = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f final_transformation_ = Eigen::Matrix4f::Identity ( 4, 4 );
//     Eigen::Matrix4f pre_transformation_ = Eigen::Matrix4f::Identity ( 4, 4 );

      /*****************************************PCL ICP Solution************************************************
      int nr_iterations_ = 0;
      bool converged_ = false;

      //Step5: Find closest point correspondences in source and target point cloud
      boost::shared_ptr<pcl::Correspondences> correspondences ( new pcl::Correspondences );
      pcl::registration::CorrespondenceEstimation<pcl::PointNormal, pcl::PointNormal> corr_est;
      corr_est.setInputSource ( input_transformed );
      corr_est.setInputTarget ( tgt );


      pcl::registration::DefaultConvergenceCriteria<float>::Ptr convergence_criteria_ (
        new pcl::registration::DefaultConvergenceCriteria<float> ( nr_iterations_, transformation_,
            *correspondences ) );
      convergence_criteria_->setMaximumIterations ( max_iterations );
      convergence_criteria_->setRelativeMSE ( euclidean_fitness_epsilon );
      convergence_criteria_->setTranslationThreshold ( max_trans_epsilon );
      convergence_criteria_->setRotationThreshold ( 1.0 - max_trans_epsilon );

      //Step6: Reject correspondence outliers using different methods
      typedef pcl::registration::CorrespondenceRejector::Ptr CorrespondenceRejectorPtr;
      std::vector<CorrespondenceRejectorPtr> correspondence_rejectors_;

      boost::shared_ptr<pcl::registration::CorrespondenceRejectorDistance> rejector_distance (
        new pcl::registration::CorrespondenceRejectorDistance );
      rejector_distance->setInputSource<pcl::PointNormal> ( input_transformed );
      rejector_distance->setInputTarget<pcl::PointNormal> ( tgt );
      rejector_distance->setMaximumDistance ( rej_max_corresdist );
      correspondence_rejectors_.push_back ( rejector_distance );


      ros::WallTime startTime = ros::WallTime::now();
      // Step7: Repeat until convergence
      do
      {
        ros::WallTime t0 = ros::WallTime::now();
        corr_est.determineCorrespondences ( *correspondences, max_corresdist );

        ros::WallTime t1 = ros::WallTime::now();
        ROS_DEBUG_STREAM ( "Finding correspondence took: " << ( t1 - t0 ).toSec() << "s " );

        //corr_est.determineReciprocalCorrespondences(*correspondences, max_corresdist);
        //ROS_DEBUG("No. of correspondences: %i ", (int) correspondences->size());

        //Reject some potential wrong correspondence according to different kind of methods
        pcl::CorrespondencesPtr temp_correspondences ( new pcl::Correspondences ( *correspondences ) );
        boost::shared_ptr<pcl::Correspondences> selected_correspondences ( new pcl::Correspondences );
        for ( size_t i = 0; i < correspondence_rejectors_.size(); ++i )
        {
          correspondence_rejectors_[i]->setInputCorrespondences ( temp_correspondences );
          correspondence_rejectors_[i]->getCorrespondences ( *correspondences );
          // Modify input for the next iteration
          if ( i < correspondence_rejectors_.size() - 1 )
            *temp_correspondences = *correspondences;
        }
        size_t cnt = correspondences->size();
        //ROS_INFO("Selected point correspondence number is: %i ", (int ) cnt);

        ros::WallTime t2 = ros::WallTime::now();
        ROS_DEBUG_STREAM ( "Rejecting correspondence took: " << ( t2 - t1 ).toSec() << "s " );


        // Check whether we have enough correspondences
        if ( ( int ) cnt < min_number_correspondences )
        {
          ROS_ERROR_STREAM (
            "[pcl::%s::computeTransformation] Not enough correspondences found. Relax your threshold parameters.\n" );
          convergence_criteria_->setConvergenceState (
            pcl::registration::DefaultConvergenceCriteria<float>::CONVERGENCE_CRITERIA_NO_CORRESPONDENCES );
          converged_ = false;

          final_transformation_ = pre_transformation_;
          break;
        }

        pcl::registration::TransformationEstimationPointToPlaneLLS<pcl::PointNormal, pcl::PointNormal, float> trans_est_lm;
        //pcl::registration::TransformationEstimationLM<pcl::PointNormal, pcl::PointNormal, float> trans_est_lm;

        trans_est_lm.estimateRigidTransformation ( *input_transformed, *tgt, *correspondences, transformation_ );

        ros::WallTime t3 = ros::WallTime::now();
        ROS_DEBUG_STREAM ( "Solving the equation took " << ( t3 - t2 ).toSec() << "s " );
        ROS_DEBUG_STREAM ( "                                                         " );

        // Tranform the iterative input cloud for next iteration
        pcl::transformPointCloud ( *input_transformed, *input_transformed, transformation_ );

        // Obtain the final transformation
        final_transformation_ = transformation_ * final_transformation_;

        ++nr_iterations_;

        //Check whether the iteration should be stopped
        converged_ = convergence_criteria_->hasConverged();
      }
      while ( !converged_ );

      ROS_DEBUG_STREAM ( "ICP iteration number is: " << nr_iterations_ );
      double dt = ( ros::WallTime::now() - startTime ).toSec();
      ROS_INFO_STREAM ( "ICP Iteration took: " << dt << "s " );
       **************************************************************************************************************************************/

      
    pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal> icp;
    typedef pcl::registration::TransformationEstimationPointToPlaneLLS<pcl::PointNormal, pcl::PointNormal> PointToPlane;
    boost::shared_ptr<PointToPlane> point_to_plane ( new PointToPlane );
    icp.setTransformationEstimation ( point_to_plane );

    boost::shared_ptr<pcl::registration::CorrespondenceRejectorDistance> rejector_distance ( new pcl::registration::CorrespondenceRejectorDistance );
    rejector_distance->setInputSource<pcl::PointNormal> ( src );
    rejector_distance->setInputTarget<pcl::PointNormal> ( tgt );
    rejector_distance->setMaximumDistance (rej_max_corresdist );

    icp.addCorrespondenceRejector ( rejector_distance );


    icp.setInputSource ( src );
    icp.setInputTarget ( tgt );

    // Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
    icp.setMaxCorrespondenceDistance (max_corresdist);
    // Set the maximum number of iterations (criterion 1)
    icp.setMaximumIterations ( max_iterations );
    // Set the transformation epsilon (criterion 2)
    icp.setTransformationEpsilon ( max_trans_epsilon );
    // Set the euclidean distance difference epsilon (criterion 3)
    icp.setEuclideanFitnessEpsilon ( euclidean_fitness_epsilon );


    pcl::PointCloud<pcl::PointNormal> Final;
    icp.align ( Final );
    final_transformation_ = icp.getFinalTransformation();
    
    
//     //GICP
//     pcl::GeneralizedIterativeClosestPoint<pcl::PointNormal, pcl::PointNormal> gicp;
//     //gicp.setMaxCorrespondenceDistance(0.05);
//     gicp.setInputSource ( input_transformed );
//     gicp.setInputTarget ( tgt );
//     pcl::PointCloud<pcl::PointNormal> Final1;
//     gicp.align ( Final1 );
//     std::cout << "GICP has converged:" << gicp.hasConverged() << " score: " << gicp.getFitnessScore() << std::endl;
//     final_transformation_ = gicp.getFinalTransformation();




    //get the transformation matrix between reading cloud and map cloud in Global Map Coordinate
    Trans_Camera2Map = final_transformation_ * Trans_Camera2Map;
    //std::cout << "Trans_Camera2Map" << std::endl << Trans_Camera2Map << std::endl;


    Eigen::Matrix4f Final_Trans = Trans_Camera2Map;

    nav_msgs::Odometry accumOdom;
    accumOdom = EigenMatrix2OdomMsg ( Final_Trans, 1, mapFrame, "", stamp );
    // Publish accumulate odometry
    if ( odomPub.getNumSubscribers() )
      odomPub.publish ( accumOdom );


    //Compute the relative odometry between current frame and previous frame
    double deltaT = currTime - prvTime;
    if ( deltaT > 0 )
    {
      ROS_DEBUG ( "deltaT:  %f", deltaT );

      CurrTrans_Camera2Map = Trans_Camera2Map;
      Eigen::Matrix4f temp_Trans = LastTrans_Camera2Map.inverse() * CurrTrans_Camera2Map;
      Relative_Trans = temp_Trans.inverse();

      if ( do_pub.getNumSubscribers() )
        do_pub.publish ( EigenMatrix2OdomMsg ( Relative_Trans, deltaT, "prev_asus_frame", "asus_frame", stamp ) );

      LastTrans_Camera2Map = CurrTrans_Camera2Map;
    }

    // Compute tf and Publish tf
    publishLock.lock();
    tf::Matrix3x3 rot_mat ( Final_Trans ( 0, 0 ), Final_Trans ( 0, 1 ), Final_Trans ( 0, 2 ),
                            Final_Trans ( 1, 0 ), Final_Trans ( 1, 1 ), Final_Trans ( 1, 2 ),
                            Final_Trans ( 2, 0 ), Final_Trans ( 2, 1 ), Final_Trans ( 2, 2 ) );
    tf::Vector3 t ( Final_Trans ( 0, 3 ), Final_Trans ( 1, 3 ), Final_Trans ( 2, 3 ) );
    tf::Transform transform ( rot_mat, t );
    tfBroadcaster.sendTransform ( tf::StampedTransform ( transform, stamp, mapFrame, odomFrame ) );
    publishLock.unlock();
    processingNewCloud = false;


    //Local Map Based ICP. Construct global map using another thread
    if ( !mapBuildingInProgress )
    {

      tempCloud->clear();
      pcl::copyPointCloud ( *Original_Reading, *tempCloud );
      // make sure we process the last available map
      ROS_DEBUG_STREAM ( "Adding new points to the map in background" );
      ROS_DEBUG_STREAM ( "The added cloud size is: " << tempCloud->size() );

      mapBuildingTask = MapBuildingTask (
                          boost::bind ( &Odometer::updateMap, this, tempCloud, Trans_Camera2Map, true ) );
      mapBuildingFuture = mapBuildingTask.get_future();
      mapBuildingThread = boost::thread ( boost::move ( boost::ref ( mapBuildingTask ) ) );
      mapBuildingInProgress = true;
    }

  }

  prvTime = currTime;
  Trans_AsusInit2AsusFrame_Last = Trans_AsusInit2AsusFrame;
  LastTrans_Camera2Map = Trans_Camera2Map;
}

Odometer::DP* Odometer::updateMap ( DP* newPointCloud, Eigen::Matrix4f Ticp, bool updateExisting )
{
  pcl::PointCloud<pcl::PointNormal>::Ptr mapCloud ( new pcl::PointCloud<pcl::PointNormal> );
  pcl::PointCloud<pcl::PointNormal>::Ptr downSizeCloud ( new pcl::PointCloud<pcl::PointNormal> );
  pcl::PointCloud<pcl::PointNormal>::Ptr truncated_Cloud ( new pcl::PointCloud<pcl::PointNormal> );
  pcl::PointCloud<pcl::PointNormal>::Ptr finalCloud ( new pcl::PointCloud<pcl::PointNormal> );
  //DP *integralMap(new DP);

  std::unique_ptr<DP> integralMap ( new DP );

  ROS_DEBUG ( "Previous Map point cloud size is: %i", localMapPointCloud->height * localMapPointCloud->width );
  //Transform the new point cloud into Global Map Coordinate and concatenate with previous global map
  pcl::transformPointCloud ( *newPointCloud, *mapCloud, Ticp );
  ROS_DEBUG_STREAM ( "New added point cloud size is: " << mapCloud->points.size() );

  if ( useMap )
  {
    // Merge point clouds to map
    if ( updateExisting )
    {
      *mapCloud += *localMapPointCloud;
    }

    // Downsampling the global map point cloud to a fixed density
    pcl::VoxelGrid<pcl::PointNormal> sor;
    sor.setInputCloud ( mapCloud );
    sor.setLeafSize ( mapVoxelLeafSize, mapVoxelLeafSize, mapVoxelLeafSize );
    sor.filter ( *downSizeCloud );
    ROS_DEBUG_STREAM ( "Map size after downsampling is: " << downSizeCloud->points.size() );

    pcl::copyPointCloud ( *mapCloud, *downSizeCloud );

    if ( ( int ) downSizeCloud->points.size() > localMapNum )
    {
      truncated_Cloud->header.frame_id = downSizeCloud->header.frame_id;
      truncated_Cloud->height = 1;
      truncated_Cloud->is_dense = true;
      for ( int i = 0; i < localMapNum; i++ )
      {
        truncated_Cloud->push_back ( downSizeCloud->points[i] );
      }
      truncated_Cloud->header.seq = downSizeCloud->header.seq;
      truncated_Cloud->header.stamp = downSizeCloud->header.stamp;
      pcl::copyPointCloud ( *truncated_Cloud, *finalCloud );
    }
    else
    {
      pcl::copyPointCloud ( *downSizeCloud, *finalCloud );
    }

    // Calculate the Normals of the point cloud using KD-tree
    pcl::NormalEstimation<pcl::PointNormal, pcl::PointNormal> norm_est;
    norm_est.setSearchMethod ( pcl::search::KdTree<pcl::PointNormal>::Ptr ( new pcl::search::KdTree<pcl::PointNormal> ) );
    norm_est.setKSearch ( 5 );
    norm_est.setInputCloud ( finalCloud );
    norm_est.compute ( *finalCloud );

    pcl::copyPointCloud ( *finalCloud, *integralMap );
  }
  else
  {
    pcl::copyPointCloud ( *mapCloud, *integralMap );
  }

  return integralMap.release();
}

void Odometer::setMap ( DP* newPointCloud )
{
  // delete old map
  if ( localMapPointCloud )
    delete localMapPointCloud;

  // set new map
  localMapPointCloud = newPointCloud;

  sensor_msgs::PointCloud2 localMapCloudMsg;
  pcl::toROSMsg ( *localMapPointCloud, localMapCloudMsg );
  localMapCloudMsg.header.frame_id = mapFrame;

  // Publish map point cloud
  if ( localmapPub.getNumSubscribers() )
    localmapPub.publish ( localMapCloudMsg );
}

void Odometer::publishLoop ( double publishPeriod )
{
  if ( publishPeriod == 0 )
    return;
  ros::Rate r ( 1.0 / publishPeriod );
  while ( ros::ok() )
  {
    publishTransform();
    r.sleep();
  }
}

void Odometer::publishTransform()
{

  if ( processingNewCloud == false )
  {
    this->publishLock.lock();

    tf::Matrix3x3 rot_mat ( Trans_Camera2Map ( 0, 0 ), Trans_Camera2Map ( 0, 1 ), Trans_Camera2Map ( 0, 2 ),
                            Trans_Camera2Map ( 1, 0 ), Trans_Camera2Map ( 1, 1 ), Trans_Camera2Map ( 1, 2 ),
                            Trans_Camera2Map ( 2, 0 ), Trans_Camera2Map ( 2, 1 ), Trans_Camera2Map ( 2, 2 ) );
    tf::Vector3 t ( Trans_Camera2Map ( 0, 3 ), Trans_Camera2Map ( 1, 3 ), Trans_Camera2Map ( 2, 3 ) );
    tf::Transform transform ( rot_mat, t );
    // Note: we use now as timestamp to refresh the tf and avoid other buffer to be empty
    tfBroadcaster.sendTransform ( tf::StampedTransform ( transform, ros::Time::now(), mapFrame, odomFrame ) );

    this->publishLock.unlock();
  }

}

nav_msgs::Odometry Odometer::EigenMatrix2OdomMsg ( const Eigen::Matrix4f& inTr, double deltaT,
    const std::string& frame_id, const std::string& child_frame_id, const ros::Time& stamp )
{
  nav_msgs::Odometry odom;
  odom.header.stamp = stamp;
  odom.header.frame_id = frame_id;
  odom.child_frame_id = child_frame_id;

  odom.pose.pose.position.x = inTr ( 0, 3 );
  odom.pose.pose.position.y = inTr ( 1, 3 );
  odom.pose.pose.position.z = inTr ( 2, 3 );
  odom.twist.twist.linear.x = inTr ( 0, 3 ) / deltaT;
  odom.twist.twist.linear.y = inTr ( 1, 3 ) / deltaT;
  odom.twist.twist.linear.z = inTr ( 2, 3 ) / deltaT;


  Eigen::Matrix3d R;
  for ( int row = 0; row < 3; row++ )
  {
    for ( int col = 0; col < 3; col++ )
    {
      R ( row, col ) = inTr ( row, col );
    }
  }

  // rotate coordinate frame so that look vector is +X, and up is +Z
  Eigen::Matrix3d M;
  M <<  0,  0, 1,
    1,  0, 0,
    0,  1, 0;

  R = R * M.transpose ();

  std::vector<float> euler ( 3 );
  euler[0] = atan ( R ( 1, 2 ) / R ( 2, 2 ) );
  euler[1] = asin ( -R ( 0, 2 ) );
  euler[2] = atan ( R ( 0, 1 ) / R ( 0, 0 ) );

  Eigen::Quaterniond quat ( R );

  odom.pose.pose.orientation.x = ( double ) quat.x();
  odom.pose.pose.orientation.y = ( double ) quat.y();
  odom.pose.pose.orientation.z = ( double ) quat.z();
  odom.pose.pose.orientation.w = ( double ) quat.w();

  odom.twist.twist.angular.x = euler[0] / deltaT;
  odom.twist.twist.angular.y = euler[1] / deltaT;
  odom.twist.twist.angular.z = euler[2] / deltaT;

  odom.pose.covariance[0 + 0 * 6] = 0.1;
  odom.pose.covariance[1 + 1 * 6] = 0.1;
  odom.pose.covariance[2 + 2 * 6] = 0.1;
  odom.pose.covariance[3 + 3 * 6] = 0.1;
  odom.pose.covariance[4 + 4 * 6] = 0.1;
  odom.pose.covariance[5 + 5 * 6] = 0.1;
  odom.twist.covariance[0 + 0 * 6] = 0.01;
  odom.twist.covariance[1 + 1 * 6] = 0.01;
  odom.twist.covariance[2 + 2 * 6] = 0.01;
  odom.twist.covariance[3 + 3 * 6] = 0.01;
  odom.twist.covariance[4 + 4 * 6] = 0.01;
  odom.twist.covariance[5 + 5 * 6] = 0.01;

  return odom;
}

// Main function supporting the Mapper class
int main ( int argc, char **argv )
{
  ros::init ( argc, argv, "odometer" );
  ros::NodeHandle n;
  ros::NodeHandle nh ( "~" );

  //vis.setBackgroundColor(0,0,0);

  Odometer Odometer ( n, nh );
  ros::spin();

  return 0;
}



#include <boost/algorithm/clamp.hpp>
#include <ros/time.h>
#include <cmath>
#include <functional>
#include "myboat_gazebo/myboat_thrust_plugin.hh"

using namespace gazebo;

double KP = 0.0, KI = 0.0, KD = 0.0;

Thruster::Thruster(UsvThrust *_parent)
{

  // std::cout << "************Thruster MYBOAT PID****************" << KP << " " << KI << " " << KD << std::endl;

  this->plugin = _parent;
  this->engineJointPID.Init(KP, KI, KD);
  this->currCmd = 0.0;
  this->desiredAngle = 0.0;
  this->lastCmdTime = this->plugin->world->SimTime();

}

Thruster::Thruster(UsvThrust *_parent, double kp, double ki, double kd)
{
  this->plugin = _parent;
  this->engineJointPID.Init(kp, ki, kd);
  this->currCmd = 0.0;
  this->desiredAngle = 0.0;
  this->lastCmdTime = this->plugin->world->SimTime();
  this->lastAngleUpdateTime = this->plugin->world->SimTime();
}

void Thruster::OnThrustCmd(const std_msgs::Float32::ConstPtr &_msg)
{
  ROS_DEBUG_STREAM("New thrust command! " << _msg->data);
  std::lock_guard<std::mutex> lock(this->plugin->mutex);
  this->lastCmdTime = this->plugin->world->SimTime();
  this->currCmd = _msg->data;
}

void Thruster::OnThrustAngle(const std_msgs::Float32::ConstPtr &_msg)
{
  ROS_DEBUG_STREAM("New thrust angle! " << _msg->data);
  std::lock_guard<std::mutex> lock(this->plugin->mutex);
  this->desiredAngle = boost::algorithm::clamp(_msg->data, -this->maxAngle,
                                               this->maxAngle);
}

double UsvThrust::SdfParamDouble(sdf::ElementPtr _sdfPtr,
  const std::string &_paramName, const double _defaultVal) const
{
  if (!_sdfPtr->HasElement(_paramName))
  {
    ROS_INFO_STREAM("Parameter <" << _paramName << "> not found: "
                    "Using default value of <" << _defaultVal << ">.");
    return _defaultVal;
  }

  double val = _sdfPtr->Get<double>(_paramName);
  ROS_DEBUG_STREAM("Parameter found - setting <" << _paramName <<
                   "> to <" << val << ">.");
  return val;
}

void UsvThrust::Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf)
{
  ROS_DEBUG("Loading usv_gazebo_thrust_plugin");
  this->model = _parent;
  this->world = this->model->GetWorld();

  std::string nodeNamespace = "";
  if (_sdf->HasElement("robotNamespace"))
  {
    nodeNamespace = _sdf->Get<std::string>("robotNamespace") + "/";
    ROS_INFO_STREAM("Thruster namespace <" << nodeNamespace << ">");
  }

  ros::NodeHandle nh(nodeNamespace);
  nh.param<double>("/myboat/kp", KP, 300.0);
  nh.param<double>("/myboat/ki", KI, 0.0);
  nh.param<double>("/myboat/kd", KD, 20.0);
  ROS_INFO_STREAM("Loaded PID: " << KP << ", " << KI << ", " << KD);

  this->cmdTimeout = this->SdfParamDouble(_sdf, "cmdTimeout", 1.0);
  this->publisherRate = this->SdfParamDouble(_sdf, "publisherRate", 100.0);
  ROS_DEBUG_STREAM("Loading thrusters from SDF");

  int thrusterCounter = 0;
  if (_sdf->HasElement("thruster"))
  {
    
    sdf::ElementPtr thrusterSDF = _sdf->GetElement("thruster");  

    while (thrusterSDF)
    {
        
      Thruster thruster(this);
      // Thruster thruster(this, KP, KI, KD);

      if (thrusterSDF->HasElement("linkName"))
      {
        std::string linkName = thrusterSDF->Get<std::string>("linkName");
        thruster.link = this->model->GetLink(linkName);
        if (!thruster.link)
        {
          ROS_ERROR_STREAM("Could not find a link by the name <" << linkName
            << "> in the model!");
        }
        else
        {
          ROS_DEBUG_STREAM("Thruster added to link <" << linkName << ">");
        }
      }
      else
      {
        ROS_ERROR_STREAM("Please specify a link name for each thruster!");
      }

      if (thrusterSDF->HasElement("propJointName"))
      {
        std::string propName =
          thrusterSDF->GetElement("propJointName")->Get<std::string>();
        thruster.propJoint = this->model->GetJoint(propName);
        if (!thruster.propJoint)
        {
          ROS_ERROR_STREAM("Could not find a propellor joint by the name of <"
            << propName << "> in the model!");
        }
        else
        {
          ROS_DEBUG_STREAM("Propellor joint <" << propName <<
            "> added to thruster");
        }
      }
      else
      {
        ROS_ERROR_STREAM("No propJointName SDF parameter for thruster #"
          << thrusterCounter);
      }

      if (thrusterSDF->HasElement("engineJointName"))
      {
        std::string engineName =
          thrusterSDF->GetElement("engineJointName")->Get<std::string>();
        thruster.engineJoint = this->model->GetJoint(engineName);
        if (!thruster.engineJoint)
        {
          ROS_ERROR_STREAM("Could not find a engine joint by the name of <" <<
            engineName << "> in the model!");
        }
        else
        {
          ROS_DEBUG_STREAM("Engine joint <" << engineName <<
            "> added to thruster");
        }
      }
      else
      {
        ROS_ERROR_STREAM("No engineJointName SDF parameter for thruster #"
          << thrusterCounter);
      }

      if (thrusterSDF->HasElement("cmdTopic"))
      {
        thruster.cmdTopic = thrusterSDF->Get<std::string>("cmdTopic");
      }
      else
      {
        ROS_ERROR_STREAM("Please specify a cmdTopic (for ROS subscription) "
          "for each thruster!");
      }

      if (thrusterSDF->HasElement("angleTopic"))
      {
        thruster.angleTopic = thrusterSDF->Get<std::string>("angleTopic");
      }
      else
      {
        ROS_ERROR_STREAM("Please specify a angleTopic (for ROS subscription) "
          "for each thruster!");
      }

      if (thrusterSDF->HasElement("enableAngle"))
      {
        thruster.enableAngle = thrusterSDF->Get<bool>("enableAngle");
        // ROS_WARN_STREAM_ONCE("Debug : enableAngle : " << thruster.enableAngle );

      }
      else
      {
        ROS_ERROR_STREAM("Please specify for each thruster if it should enable "
          "angle adjustment (for ROS subscription)!");
      }

      if (thrusterSDF->HasElement("mappingType"))
      {
        thruster.mappingType = thrusterSDF->Get<int>("mappingType");
        ROS_DEBUG_STREAM("Parameter found - setting <mappingType> to <" <<
          thruster.mappingType << ">.");
      }
      else
      {
        thruster.mappingType = 0;
        ROS_INFO_STREAM("Parameter <mappingType> not found: "
          "Using default value of <" << thruster.mappingType << ">.");
      }

      thruster.maxCmd = this->SdfParamDouble(thrusterSDF, "maxCmd", 6.0);
      thruster.maxForceFwd =
        this->SdfParamDouble(thrusterSDF, "maxForceFwd", 250.0);
      thruster.maxForceRev =
        this->SdfParamDouble(thrusterSDF, "maxForceRev", -100.0);
      thruster.maxAngle = this->SdfParamDouble(thrusterSDF, "maxAngle",
                                               M_PI / 2);

      this->thrusters.push_back(thruster);
      thrusterSDF = thrusterSDF->GetNextElement("thruster");
      thrusterCounter++;


    }
  }
  else
  {
    ROS_WARN_STREAM("No 'thruster' tags in description - how will you move?");
  }
  ROS_DEBUG_STREAM("Found " << thrusterCounter << " thrusters");

  this->rosnode.reset(new ros::NodeHandle(nodeNamespace));
  this->prevUpdateTime = this->world->SimTime();

  this->jointStatePub =
    this->rosnode->advertise<sensor_msgs::JointState>("joint_states", 1);
  this->jointStateMsg.name.resize(2 * thrusters.size());
  this->jointStateMsg.position.resize(2 * thrusters.size());
  this->jointStateMsg.velocity.resize(2 * thrusters.size());
  this->jointStateMsg.effort.resize(2 * thrusters.size());

  for (size_t i = 0; i < this->thrusters.size(); ++i)
  {
    this->jointStateMsg.name[2 * i] = this->thrusters[i].engineJoint->GetName();
    this->jointStateMsg.name[2 * i + 1] =
      this->thrusters[i].propJoint->GetName();

    this->thrusters[i].cmdSub = this->rosnode->subscribe(
      this->thrusters[i].cmdTopic, 1, &Thruster::OnThrustCmd,
      &this->thrusters[i]);

    if (this->thrusters[i].enableAngle)
    {
      this->thrusters[i].angleSub = this->rosnode->subscribe(
        this->thrusters[i].angleTopic, 1, &Thruster::OnThrustAngle,
        &this->thrusters[i]);
    }

    std::string forceTopic = this->thrusters[i].link->GetName() + "/curForce";
    this->thrusters[i].forcePub = this->rosnode->advertise<std_msgs::Float32>(forceTopic, 10);
    ROS_INFO_STREAM("Publishing thrust force on topic: " << forceTopic);
  }

  this->updateConnection = event::Events::ConnectWorldUpdateBegin(
    std::bind(&UsvThrust::Update, this));
}

double UsvThrust::ScaleThrustCmd(
    const double _desired_velocity,
    const double _maxCmd,
    const double _maxPos,
    const double _maxNeg
) const
{
    static const double v_data[] = {
        0.272, 0.508, 0.754, 1.001, 1.252, 1.500, 1.750, 2.000,
        2.251, 2.502, 3.002, 3.247, 3.499, 3.750, 4.001, 4.251,
        4.500, 4.751, 5.000, 5.249, 5.500, 5.750, 5.999
    };
    static const double f_data[] = {
        1.219, 2.761, 4.837, 7.407, 10.522, 14.109, 18.202, 22.799,
        27.927, 33.544, 46.245, 53.218, 60.844, 68.989, 77.620, 86.699,
        96.250, 106.355, 116.875, 127.929, 139.505, 151.547, 163.996
    };
    const int N = sizeof(v_data) / sizeof(v_data[0]);

    double v = _desired_velocity;

    bool is_negative = false;
    if (v < 0) {
        is_negative = true;
        v = -v;
    }

    if (v <= 0.0) {
        return 0.0;
    }

    if (v >= v_data[N-1]) {
        double thrust = is_negative ? -_maxNeg : _maxPos;
        return thrust;
    }

    if (v <= v_data[0]) {
        double ratio = v / v_data[0];
        double thrust = f_data[0] * ratio;
        thrust = is_negative ? -thrust : thrust;
        if (is_negative) {
            thrust = std::max(thrust, -std::abs(_maxNeg));
        } else {
            thrust = std::min(thrust, _maxPos);
        }
        return thrust;
    }

    int i = 0;
    while (i < N-1 && v > v_data[i+1]) {
        i++;
    }

    double t = (v - v_data[i]) / (v_data[i+1] - v_data[i]);
    double thrust = f_data[i] + t * (f_data[i+1] - f_data[i]);
    thrust = is_negative ? -thrust : thrust;

    if (is_negative) {
        thrust = std::max(thrust, -std::abs(_maxNeg));
    } else {
        thrust = std::min(thrust, _maxPos);
    }

    return thrust;
}

double UsvThrust::Glf(const double _x, const float _A, const float _K,
    const float _B, const float _v, const float _C, const float _M) const
{
  return _A + (_K - _A) / (pow(_C + exp(-_B * (_x - _M)), 1.0 / _v));
}

double UsvThrust::GlfThrustCmd(const double _cmd,
                               const double _maxPos,
                               const double _maxNeg) const
{
  double val = 0.0;
  if (_cmd > 0.01)
  {
    val = this->Glf(_cmd, 0.01f, 59.82f, 5.0f, 0.38f, 0.56f, 0.28f);
    val = std::min(val, _maxPos);
  }
  else if (_cmd < 0.01)
  {
    val = this->Glf(_cmd, -199.13f, -0.09f, 8.84f, 5.34f, 0.99f, -0.57f);
    val = std::max(val, _maxNeg);
  }
  return val;
}

void UsvThrust::Update()
{
  common::Time now = this->world->SimTime();

  for (size_t i = 0; i < this->thrusters.size(); ++i)
  {
    {
      std::lock_guard<std::mutex> lock(this->mutex);
      double dtc = (now - this->thrusters[i].lastCmdTime).Double();
      if (dtc > this->cmdTimeout && this->cmdTimeout > 0.0)
      {
        this->thrusters[i].currCmd = 0.0;
        ROS_DEBUG_STREAM_THROTTLE(1.0, "[" << i << "] Cmd Timeout");
      }

      this->RotateEngine(i, now - this->thrusters[i].lastAngleUpdateTime);

      ignition::math::Vector3d tforcev(0, 0, 0);
      switch (this->thrusters[i].mappingType)
      {
        case 0:
          tforcev.X() = this->ScaleThrustCmd(this->thrusters[i].currCmd,
                                            this->thrusters[i].maxCmd,
                                            this->thrusters[i].maxForceFwd,
                                            this->thrusters[i].maxForceRev);
          break;
        case 1:
          tforcev.X() = this->GlfThrustCmd(this->thrusters[i].currCmd/
                                          this->thrusters[i].maxCmd,
                                          this->thrusters[i].maxForceFwd,
                                          this->thrusters[i].maxForceRev);
          break;
        default:
            ROS_FATAL_STREAM("Cannot use mappingType=" <<
              this->thrusters[i].mappingType);
            break;
      }

      // add link force
      this->thrusters[i].link->AddLinkForce(tforcev);

      // just rotate propeller
      this->SpinPropeller(i);

      // pub force
      std_msgs::Float32 forceMsg;
      forceMsg.data = tforcev.X();
      this->thrusters[i].forcePub.publish(forceMsg);
    }
  }

  if (now - this->prevUpdateTime >= (1 / this->publisherRate))
  {
    this->jointStateMsg.header.stamp = ros::Time::now();
    this->jointStatePub.publish(this->jointStateMsg);
    this->prevUpdateTime = now;
  }
}

void UsvThrust::RotateEngine(size_t _i, common::Time _stepTime)
{
  // ROS_WARN_STREAM_ONCE("Debug : Thruster[" << _i << "] First dt: " << _stepTime.Double());

  double desiredAngle = this->thrusters[_i].desiredAngle;
  double currAngle = this->thrusters[_i].engineJoint->Position(0);
  double angleError = currAngle - desiredAngle;

  double effort = this->thrusters[_i].engineJointPID.Update(angleError,
                                                           _stepTime);
  this->thrusters[_i].engineJoint->SetForce(0, effort);

  ignition::math::Angle position = this->thrusters[_i].engineJoint->Position(0);
  position.Normalize();
  this->jointStateMsg.position[2 * _i] = position.Radian();
  this->jointStateMsg.velocity[2 * _i] =
    this->thrusters[_i].engineJoint->GetVelocity(0);
  this->jointStateMsg.effort[2 * _i] = effort;

  this->thrusters[_i].lastAngleUpdateTime += _stepTime;
}

void UsvThrust::SpinPropeller(size_t _i)
{
  const double kMinInput = 0.1;
  const double kMaxEffort = 2.0;
  double effort = 0.0;

  physics::JointPtr propeller = this->thrusters[_i].propJoint;

  if (std::abs(this->thrusters[_i].currCmd/
              this->thrusters[_i].maxCmd) > kMinInput)
    effort = (this->thrusters[_i].currCmd /
              this->thrusters[_i].maxCmd) * kMaxEffort;

  propeller->SetForce(0, effort);

  ignition::math::Angle position = propeller->Position(0);
  position.Normalize();
  this->jointStateMsg.position[2 * _i + 1] = position.Radian();
  this->jointStateMsg.velocity[2 * _i + 1] = propeller->GetVelocity(0);
  this->jointStateMsg.effort[2 * _i + 1] = effort;
}

GZ_REGISTER_MODEL_PLUGIN(UsvThrust);
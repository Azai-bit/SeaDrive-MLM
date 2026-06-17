#ifndef USV_GAZEBO_PLUGINS_DYNAMICS_PLUGIN_HH_
#define USV_GAZEBO_PLUGINS_DYNAMICS_PLUGIN_HH_

#include <Eigen/Core>
#include <string>
#include <vector>

#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>
#include <gazebo/physics/physics.hh>
#include <sdf/sdf.hh>

#include "wave_gazebo_plugins/Wavefield.hh"
#include "wave_gazebo_plugins/WavefieldEntity.hh"
#include "wave_gazebo_plugins/WavefieldModelPlugin.hh"

namespace gazebo
{
  class UsvDynamicsPlugin : public ModelPlugin
  {
    public: UsvDynamicsPlugin();
    public: virtual ~UsvDynamicsPlugin() = default;
    public: virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf);
    protected: virtual void Update();
    private: double SdfParamDouble(sdf::ElementPtr _sdfPtr,
                                   const std::string &_paramName,
                                   const double _defaultVal) const;
    private: double CircleSegment(double R, double h);
    private: physics::WorldPtr world;
    private: physics::LinkPtr link;
    private: common::Time prevUpdateTime;
    private: ignition::math::Vector3d prevLinVel;
    private: ignition::math::Vector3d prevAngVel;
    private: double paramXdotU;
    private: double paramYdotV;
    private: double paramZdotW;
    private: double paramKdotP;
    private: double paramMdotQ;
    private: double paramNdotR;
    private: double paramXu;
    private: double paramXuu;
    private: double paramYv;
    private: double paramYvv;
    private: double paramZw;
    private: double paramZww;
    private: double paramKp;
    private: double paramKpp;
    private: double paramMq;
    private: double paramMqq;
    private: double paramNr;
    private: double paramNrr;
    private: double waterLevel;
    private: double waterDensity;
    private: double paramBoatLength;
    private: double paramBoatWidth;
    private: double paramHullRadius;
    private: int paramLengthN;
    private: Eigen::MatrixXd Ma;
    protected: std::string waveModelName;
    private: event::ConnectionPtr updateConnection;
    private: std::shared_ptr<const asv::WaveParameters> waveParams;
  };
}

#endif
#pragma once
#include "EKF.h"

#define N_STATES 3
#define N_OBS_V 1

class BattEKF : public EKF<N_STATES>
{
public:
    BattEKF() {
        _state << 1.5f,1.0f,0.0f;
        _covariance << 0.01f,  0.0f,  0.0f,
                        0.0f, 0.01f,  0.0f,
                        0.0f,  0.0f, 0.01f;
    }
    class VoltageObs : public Observation<N_OBS_V> {
    public:
        VoltageObs(EKF<N_STATES>& ekf, const Eigen::Matrix<float,N_OBS_V,1>& z, const Eigen::Matrix<float,N_OBS_V,N_OBS_V>& R):
        Observation(ekf, z, R)
        {
            const Eigen::Matrix<float,N_STATES,1>& state = _ekf.getState();

            _h[0] = state[0] - state[1]*state[2];

            _H[0] = 1;
            _H[1] = -state[2];
            _H[2] = -state[1];

            init();
        }
    };
};

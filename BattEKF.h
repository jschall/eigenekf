#pragma once
#include "ExtendedKalmanFilter.h"

#define N_STATES 3
#define N_OBS_V 1

class BattEKF : public ExtendedKalmanFilter<N_STATES>
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
        VoltageObs(ExtendedKalmanFilter<N_STATES>& ekf, const Matrix_Ox1& z, const Matrix_OxO& R) : Observation(ekf, z, R) {}

        void computePredictedObs(Matrix_Ox1& h)
        {
            const Eigen::Matrix<float,N_STATES,1>& state = _ekf.getState();
            h[0] = state[0] - state[1]*state[2];
        }

        void computeObsSensitivity(Matrix_OxS& H)
        {
            const Eigen::Matrix<float,N_STATES,1>& state = _ekf.getState();
            H[0] = 1;
            H[1] = -state[2];
            H[2] = -state[1];
        }
    };
};

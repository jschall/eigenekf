#pragma once
#include <Eigen/Dense>

template<int N_STATES> class EKF {
public:
    template<int N_OBS> class Observation {
    public:
        Observation(EKF<N_STATES>& ekf, const Eigen::Matrix<float,N_OBS,1>& z, const Eigen::Matrix<float,N_OBS,N_OBS>& R) :
        _ekf(ekf),
        _z(z),
        _R(R) {}

        void init()
        {
            _y.noalias() = _z-_h;

            // compute the inverse of the S matrix
            _S_inv.noalias() = _H*_ekf._covariance*_H.transpose();
            _S_inv.noalias() += _R;
            _S_inv = _S_inv.inverse();
        }

        const Eigen::Matrix<float,N_OBS,1>& getInnov() const
        {
            return _y;
        }

        float getNIS() const
        {
            return (_y.transpose()*_S_inv*_y)[0];
        }

        void commit()
        {
            Eigen::Matrix<float,N_STATES, 1>& x = _ekf._state;
            Eigen::Matrix<float,N_STATES,N_STATES>& P = _ekf._covariance;

            Eigen::Matrix<float,N_STATES,N_OBS> K;
            K.noalias() = P*_H.transpose()*_S_inv;

            // Update the state and covariance
            x.noalias() += K*_y;
            P = (Eigen::Matrix<float,N_STATES,N_STATES>::Identity()-K*_H)*P;
        }

    protected:
        EKF<N_STATES> &_ekf;

        // The subclass constructor is expected to set these and then call init
        Eigen::Matrix<float,N_OBS,1>        _z;
        Eigen::Matrix<float,N_OBS,N_OBS>    _R;
        Eigen::Matrix<float,N_OBS,1>        _h;
        Eigen::Matrix<float,N_OBS,N_STATES> _H;

    private:
        Eigen::Matrix<float,N_OBS,1>        _y;
        Eigen::Matrix<float,N_OBS,N_OBS>    _S_inv;

    };

    const Eigen::Matrix<float,N_STATES,1>& getState() { return _state; }

protected:
    Eigen::Matrix<float,N_STATES,1> _state;
    Eigen::Matrix<float,N_STATES,N_STATES> _covariance;
};

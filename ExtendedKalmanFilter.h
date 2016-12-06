#pragma once
#include <Eigen/Dense>

template<int N_STATES> class ExtendedKalmanFilter {
public:
    typedef Eigen::Matrix<float,N_STATES,1> Matrix_Sx1;
    typedef Eigen::Matrix<float,N_STATES,N_STATES> Matrix_SxS;

    template<int N_OBS> class Observation {
    public:
        typedef Eigen::Matrix<float,N_OBS,1> Matrix_Ox1;
        typedef Eigen::Matrix<float,N_OBS,N_OBS> Matrix_OxO;
        typedef Eigen::Matrix<float,N_STATES,N_OBS> Matrix_SxO;
        typedef Eigen::Matrix<float,N_OBS,N_STATES> Matrix_OxS;
        typedef Matrix_Ox1 z_type;
        typedef Matrix_OxO R_type;

        Observation(ExtendedKalmanFilter<N_STATES>& ekf, const Matrix_Ox1& z, const Matrix_OxO& R) :
        _ekf(ekf),
        _z(z),
        _R(R),
        _y_computed(false),
        _S_inv_computed(false)
        {}

        const Matrix_Ox1& getInnov()
        {
            compute_y();
            return _y;
        }

        float getNIS()
        {
            compute_y();
            compute_S_inv();
            return (_y.transpose()*_S_inv*_y)[0];
        }

        void commit()
        {
            compute_y();
            compute_S_inv();
            Matrix_Sx1& x = _ekf._state;
            Matrix_SxS& P = _ekf._covariance;

            Matrix_SxO K;
            K.noalias() = P*_H.transpose()*_S_inv;

            // Update the state and covariance
            x.noalias() += K*_y;
            P = (Matrix_SxS::Identity()-K*_H)*P;
        }

    protected:
        virtual void computePredictedObs(Matrix_Ox1& h) = 0;
        virtual void computeObsSensitivity(Matrix_OxS& H) = 0;

        ExtendedKalmanFilter<N_STATES>&    _ekf;

    private:
        void compute_y() {
            if (!_y_computed) {
                Matrix_Ox1 h;
                computePredictedObs(h);
                _y.noalias() = _z-h;
                _y_computed = true;
            }
        }

        void compute_S_inv() {
            if (!_S_inv_computed) {
                computeObsSensitivity(_H);
                _S_inv.noalias() = _H*_ekf._covariance*_H.transpose();
                _S_inv.noalias() += _R;
                _S_inv = _S_inv.inverse();
                _S_inv_computed = true;
            }
        }

        bool              _y_computed;
        bool              _S_inv_computed;
        Matrix_Ox1        _z;     // Observation vector
        Matrix_OxO        _R;     // Observation covariance
        Matrix_Ox1        _y;     // Innovation vector
        Matrix_OxO        _S_inv; // Innovation covariance inverse
        Matrix_OxS        _H;     // Observation sensitivity matrix, dh/dx
    };

    const Matrix_Sx1& getState() { return _state; }
    const Matrix_SxS& getCovariance() { return _covariance; }

protected:
    Matrix_Sx1 _state;
    Matrix_SxS _covariance;
};

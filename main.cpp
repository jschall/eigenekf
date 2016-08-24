#include <iostream>
#include "BattEKF.h"

using namespace std;
int main()
{
    BattEKF ekf;

    Eigen::Matrix<float,1,1> z;
    z << 1.0f;

    Eigen::Matrix<float,1,1> R;
    R << 0.1f;

    BattEKF::VoltageObs v_obs(ekf, z, R);
    cout << v_obs.getNIS() << endl;
    v_obs.commit();
}

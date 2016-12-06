#include <iostream>
#include "BattEKF.h"

using namespace std;
int main()
{
    BattEKF ekf;

    BattEKF::VoltageObs::z_type z;
    z << 1.0f;

    BattEKF::VoltageObs::R_type R;
    R << 0.1f;

    BattEKF::VoltageObs v_obs(ekf, z, R);
    cout << v_obs.getNIS() << endl;
    v_obs.commit();
}

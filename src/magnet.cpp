#include "magnet.hpp"
#include <math.h>
#include <iostream>

Magnet::Magnet(Vector3 p, Vector3 m_dir, scalar_t m_mag):Magnet(p, p, 1, m_dir, m_mag){
}

Magnet::Magnet(Vector3 p0, Vector3 p1, scalar_t trip_time, Vector3 m_dir, scalar_t m_mag) : p0(p0), p1(p1), trip_time(trip_time){
    m = arma::normalise(m_dir,2);
    m = m*m_mag;
    v = (p1 - p0)/trip_time;
    std::cout << "Magnet velocity " << v << std::endl;
}

Vector3 Magnet::get_field(Vector3 p, scalar_t t){
    Vector3 r = p - get_pos(t);

    if(flat_field) {
        return {0, 0, arma::norm(m)};
    }

    return (1/(4.0*PI))*( (3.0*r*(arma::dot(m,r)))/std::pow(arma::norm(r),5) - m / std::pow(arma::norm(r),3) );
}

scalar_t Magnet::get_psi(Vector3 p, scalar_t t){
    Vector3 r = p - get_pos(t);

    if(flat_field) {
        return arma::norm(m)*(100-r(2));
    }

    return 1.0*arma::dot(m,r) / (4.0*PI*std::pow(arma::norm(r), 3));
}

Vector3 Magnet::get_pos(scalar_t t){
    if (t <= trip_time){
        return p0 + v*t;
    } else {
        return p1;
    }
}

void Magnet::get_field_arrays(CubeX &Bx, CubeX &By, CubeX &Bz, scalar_t t, scalar_t const dx){
    for (unsigned int k = 0; k < Bx.n_slices; k++){
        for (unsigned int j = 0; j < Bx.n_cols; j++){
            for (unsigned int i = 0; i < Bx.n_rows; i++){
                Vector3 B = get_field({i*dx, j*dx, k*dx}, t);
                Bx(i, j, k) = B(0);
                By(i, j, k) = B(1);
                Bz(i, j, k) = B(2);
            }
        }
    }
}

scalar_t Magnet::get_element(Vector3 p, scalar_t t, int i, int j){
    Vector3 B = get_field(p, t);
    if ( i == 0 && j == 0 ){
        return B(0)*B(0) - 0.5*std::pow(arma::norm(B,2), 2);
    } else if ( i == 1 && j == 1 ){
        return B(1)*B(1) - 0.5*std::pow(arma::norm(B,2), 2);
    } else {
        return B(1)*B(0);
    }
}

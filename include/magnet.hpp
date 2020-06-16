#ifndef MAGNET_HPP
#define MAGNET_HPP

#include "sim_params_ferro.hpp"
#include "sim_external_types.hpp"

class Magnet {
public:
    Magnet(Vector3 p, Vector3 m_dir, scalar_t m_mag);
    Magnet(Vector3 p0, Vector3 p1, scalar_t trip_time, Vector3 m_dir, scalar_t m_mag);

    Vector3 get_field(Vector3 p, scalar_t t);
    scalar_t get_element(Vector3 p, scalar_t t, int i, int j);
    scalar_t get_psi(Vector3 p, scalar_t t);
    Vector3 get_pos(scalar_t t);
    void get_field_arrays(CubeX &Bx, CubeX &By, CubeX &Bz, scalar_t t, scalar_t const dx);

    scalar_t trip_time;
    Vector3 p0, p1, v;
    Vector3 m;
    Vector3 vel;
    bool flat_field = false;
};

#endif

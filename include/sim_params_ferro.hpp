#ifndef SIM_PARAMS_FERRO_HPP
#define SIM_PARAMS_FERRO_HPP

#include "sim_params_water.hpp"

scalar_t const MU0 = 0.0000004*PI;

typedef struct SimFerroParams{
    SimFerroParams(){}

    SimFerroParams( scalar_t M_s, scalar_t m, scalar_t kb, scalar_t T, scalar_t interface_reg,
            int grid_w_em, int grid_h_em, int grid_d_em, scalar_t appStrength) :
            M_s(M_s), m(m), kb(kb), T(T), interface_reg(interface_reg),
            grid_w_em(grid_w_em), grid_h_em(grid_h_em), grid_d_em(grid_d_em),
            appStrength(appStrength)
    {}

    scalar_t M_s = 10.0*22.8*1000.0; //Determined visually from Afkhami 2008.
//    scalar_t M_s = 1923026.4768879882;
    scalar_t m = 1.0*2.0E-19; //total magnetic moment (gross approximation)
    scalar_t kb = 1.38064852E-23; //Boltzmann's Constant
    scalar_t T = 273.15; //Temperature (Kevlin)
    scalar_t interface_reg = 1.0;

    int const grid_w_em = 62;
    int const grid_h_em = 62;
    int const grid_d_em = 62;

    scalar_t appStrength = 3000;

    // Calculated
    int const n_cells_em = grid_w_em*grid_h_em*grid_d_em;
} SimFerroParams;

#endif

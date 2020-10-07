#ifndef SIM_FERRO_HPP
#define SIM_FERRO_HPP

#include <fstream>
#include "sim_params.hpp"
#include "sim_params_ferro.hpp"
#include "sim_external_types.hpp"
#include "sim_water.hpp"

class CudaCG;
class Magnet;

class SimFerro : public SimWater{
public:
    SimFerro(SimParams C, SimWaterParams CW, SimFerroParams CF, Magnet *magnet);
    static void create_ferro_params_from_args(int argc, char** argv, int& n_steps, SimParams *&retC, SimWaterParams *&retCW, SimFerroParams *&retCF);
    void step();
    scalar_t MUF = 1.0*MU0;
    void save_data() override;
    void load_data() override;

protected:
    Magnet *magnet;
    VectorXs psi;
    CubeX Psi;
    CubeX mu;
    void update_mu_from_labels();
    void output_rhs_psi();
    void check_psi(CubeX &Psi);
    MatrixA A_psi;
    MatrixA A_psid;
    VectorXs b_psi;
    CubeX Tmag_xx;
    CubeX Tmag_yy;
    CubeX Tmag_zz;
    CubeX Tmag_xy;
    CubeX Tmag_xz;
    CubeX Tmag_yz;
    CubeX Bmag_x;
    CubeX Bmag_y;
    CubeX Bmag_z;
    recorder* save_psi;
    recorder* save_B;
    DisplayWindow* plot2;
    double total_real_time = 0;
    CudaCG* cudacg_mag;

    // for the mag potential solve
    void build_A_psi_and_b(); // no input params since all var are members of this class
    void update_mag_tensor_entries(CubeX &Psi);
    void apply_force_from_tensor();
    void add_magnet_potential(CubeX &Q, Magnet &M, scalar_t time);
    scalar_t get_mu_with_Langevin(scalar_t Hx, scalar_t Hy, scalar_t Hz);
    scalar_t get_mu(int i, int j, int k);
    scalar_t get_mu_half(Vector3i Ip, Vector3i dir);
    scalar_t get_mu_frac(Vector3i Ip, Vector3i dir, scalar_t dx);
    void solve_ferro();
    scalar_t get_psi(int i, int j, int k); // Coordinates are fluid based.
    scalar_t get_mu_coeff(Vector3 pos);
    void check_divergence_mag(const CubeX &Q);
    void print_mag_tensor_entries();

private:
    std::ofstream frameData;
    unsigned int frameNumber = 0;
    // Ferrofluid parameters
    scalar_t M_s = 2.0*22.8*1000.0; //Determined visually from Afkhami 2008.
    //scalar_t M_s = 0;
    scalar_t m = 1.0*2.0E-19; //total magnetic moment (gross approximation)
    scalar_t kb = 1.38064852E-23; //Boltzmann's Constant
    scalar_t T = 273.15; //Temperature (Kevlin)
    scalar_t interface_reg = 1.0;

    int const grid_w_em = 64;
    int const grid_h_em = 64;
    int const grid_d_em = 34;
    int const offset_w_em = (grid_w_em-grid_w)/2;
    int const offset_h_em = (grid_h_em-grid_h)/2;
    int const offset_d_em = (grid_d_em-grid_d)/2;
    int const n_cells_em;
    const Vector3i offset_em = {offset_w_em, offset_h_em, offset_d_em};
    int n_cells_em_use;
};

#endif

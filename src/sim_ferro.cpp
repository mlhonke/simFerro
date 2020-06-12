#include "sim_ferro.hpp"
#include "sim_label.hpp"
#include "interpolate.hpp"
#include <limits>
#include <iostream>
#include <iomanip>
#include "Eigen/OrderingMethods"
#include "Eigen/Eigenvalues"
#include "fd_math.hpp"
#include "sim_pls.hpp"
#include "sim_pls_cuda.cuh"
#include "sim_levelset.hpp"
//#include "sim_levelset/sim_mesh.hpp"
#include "CudaCG.hpp"
#include "sim.hpp"
//#include <time.h>

//static Eigen::SparseQR<MatrixA, Eigen::COLAMDOrdering<int>> CG_psi;
//static Eigen::ConjugateGradient<MatrixA, Eigen::Lower | Eigen::Upper, Eigen::SimplicialCholesky<MatrixA>> CG_psi;
static Eigen::ConjugateGradient<MatrixA, Eigen::Lower|Eigen::Upper> CG_psi;
//static Eigen::ConjugateGradient<MatrixA, Eigen::Lower|Eigen::Upper, Eigen::IncompleteCholesky<scalar_t>> CG_psi;
//static Eigen::SimplicialLLT<MatrixA, Eigen::Lower|Eigen::Upper> CG_psi;

SimFerro::SimFerro(SimParams C, SimWaterParams CW, SimFerroParams CF)
    :
    SimWater(C, CW),
    M_s(CF.M_s),
    m(CF.m),
    kb(CF.kb),
    T(CF.T),
    interface_reg(CF.interface_reg),
    grid_w_em(CF.grid_w_em),
    grid_h_em(CF.grid_h_em),
    grid_d_em(CF.grid_d_em),
    n_cells_em(CF.n_cells_em)
    {

    std::cout << "Left EM bound " << -offset_w_em*scale_w << std::endl;
    std::cout << "Down EM bound " << -offset_h_em*scale_h << std::endl;
    std::cout << "Back EM bound " << -offset_d_em*scale_d << std::endl;

    // Setup the CG CUDA solver! Since mag solve has same number of cells every iteration only allocate memory once.
    n_cells_em_use = 7*n_cells_em - (2*grid_w_em*grid_h_em + 2*grid_h_em*grid_d_em + 2*grid_w_em*grid_d_em);
    std::cout << "Non-zero cells predicted: " << n_cells_em_use << std::endl;
    cudacg_mag = new CudaCG(n_cells_em, n_cells_em_use);
    cudacg_mag->project = true; // enable projection for pure neumann boundary problems

    // Create a magnet to interact with our ferrofluid. // Was 3000-7000 // 500 for flat is nice
    magnet = new Magnet({sim_w / 2.0, sim_h/2.0, -0.2}, {sim_w / 2.0, sim_h/2.0, -0.2}, 2.0, {0, 0, 1}, CF.appStrength);
    magnet->flat_field = false; // Enable for a constant field through all space (eg. homogenous from e-magnet).

    // Init memory for constant sized arrays / matricies.
    mu = CubeX(grid_w_em, grid_h_em, grid_d_em); //Initialize to permeability of free space.
    mu.fill(MU0);
    psi = VectorXs::Constant(n_cells_em, 1, 0);
    Psi = CubeX(psi.data(), grid_w_em, grid_h_em, grid_d_em, false);
    A_psi = MatrixA(n_cells_em, n_cells_em);
    A_psid = MatrixA(n_cells_em, n_cells_em);
    Tmag_xx = CubeX(grid_w, grid_h, grid_d, arma::fill::zeros);
    Tmag_yy = CubeX(grid_w, grid_h, grid_d, arma::fill::zeros);
    Tmag_zz = CubeX(grid_w, grid_h, grid_d, arma::fill::zeros);
    Tmag_xy = CubeX(grid_w+1, grid_h+1, grid_d, arma::fill::zeros);
    Tmag_xz = CubeX(grid_w+1, grid_h, grid_d+1, arma::fill::zeros);
    Tmag_yz = CubeX(grid_w, grid_h+1, grid_d+1, arma::fill::zeros);
    Bmag_x = CubeX(grid_w, grid_h, grid_d);
    Bmag_y = CubeX(grid_w, grid_h, grid_d);
    Bmag_z = CubeX(grid_w, grid_h, grid_d);

    // Initialize components of total magnetic field (initially only due to magnet since I assume ferrofluid starts unmagnetized).
    add_magnet_potential(Psi, *magnet, 0); // Magnet is in initial starting position, since t = 0;
    calc_grad(Psi.subcube(offset_w_em, offset_h_em, offset_d_em, offset_w_em+grid_w-1, offset_h_em+grid_h-1, offset_d_em+grid_d-1), Bmag_x, Bmag_y, Bmag_z, scale_w); // Calculate magnetic fields
    psi = VectorXs::Constant(n_cells_em, 1, 0); // Put psi back to zero although it will be overwritten in solve_ferro() just to be safe.
    Bmag_x = -Bmag_x;
    Bmag_y = -Bmag_y;
    Bmag_z = -Bmag_z;

    for (int i = 0; i < grid_d; i++){
        Vector3 B;
        B(0) = grid_trilerp({sim_w/2.0, sim_h/2.0, i*scale_d}, Bmag_x, scale_w);
        B(1) = grid_trilerp({sim_w/2.0, sim_h/2.0, i*scale_d}, Bmag_y, scale_w);
        B(2) = grid_trilerp({sim_w/2.0, sim_h/2.0, i*scale_d}, Bmag_z, scale_w);
        std::cout << "Magnetic field is " << B(0) << " " << B(1) <<  " " << B(2) << std::endl;
        std::cout << "Relative Mu at " << i << " is " << get_mu_with_Langevin(B(0), B(1), B(2)) / MU0 << std::endl;
    }

    Vector3 B;
    B(0) = grid_trilerp({sim_w/2.0, sim_h/2.0, sim_d/2.0}, Bmag_x, scale_w);
    B(1) = grid_trilerp({sim_w/2.0, sim_h/2.0, sim_d/2.0}, Bmag_y, scale_w);
    B(2) = grid_trilerp({sim_w/2.0, sim_h/2.0, sim_d/2.0}, Bmag_z, scale_w);
    scalar_t mu = get_mu_with_Langevin(B(0), B(1), B(2));
    scalar_t H_crit = std::sqrt((2/MU0)*((1+MU0/mu)/std::pow((MU0/mu -1),2))*std::sqrt(density*std::abs(g)*lambda));
    std::cout << "The critical field value for this simulation is " << H_crit << std::endl;

    CubeX Bx(grid_w, grid_h, grid_d);
    CubeX By(grid_w, grid_h, grid_d);
    CubeX Bz(grid_w, grid_h, grid_d);
    magnet->get_field_arrays(Bx, By, Bz, 0, scale_w);
//    std::cout << Bz.slice(grid_d/2) << std::endl;

//    for (int i = 0; i < n_cells; i++){
//        std::cout << convert_index_to_coords(i, grid_w, grid_h) << std::endl;
//    }

//    load_data(); // load a previous simulation data
    frameData.open("/home/graphics/Dev/Ferro3D/screens/frameData.txt");
}

void SimFerro::create_params_from_args(int argc, char **argv, int &n_steps, SimParams *&retC, SimWaterParams *&retCW, SimFerroParams *&retCF) {
    int i;
    SimWater::create_params_from_args(argc, argv, n_steps, retC, retCW, i);

    if (argc > 1) {
        retCF = new SimFerroParams{
                std::stof(argv[i++]),
                std::stof(argv[i++]),
                std::stof(argv[i++]),
                std::stof(argv[i++]),
                std::stof(argv[i++]),
                std::stoi(argv[i++]),
                std::stoi(argv[i++]),
                std::stoi(argv[i++]),
                std::stof(argv[i++])
        };
    }
    // Otherwise use default arguments.
    else {
        n_steps = 1000000;
        retCF = new SimFerroParams();
    }
}

void SimFerro::save_data(){
    SimWater::save_data();
    // Because I don't want to deal with C files, and this makes it easy to store a vector of parameters if needed.
    VectorX save_t(1);
    save_t[0] = t;
    save_t.save("ferro_state.bin");
    VectorXi save_cur_step(1);
    save_cur_step[0] = cur_step;
    save_cur_step.save("ferro_state_2.bin");
}

void SimFerro::load_data(){
    SimWater::load_data();
    VectorX save_t(1);
    save_t.load("ferro_state.bin");
    t = save_t[0];
    VectorXi save_cur_step(1);
    save_cur_step.load("ferro_state_2.bin");
    cur_step = save_cur_step[0];
}

void SimFerro::step() {
    ExecTimer *execTimer = new ExecTimer("Total sim solve");
    if (cur_step % 100 == 0 && cur_step != 0){
        save_data();
    }

    elapsed_time += dt;
//    dt = 0.1*get_new_timestep(V);
    dt = 0.0005;
//    dt = 0.5;
//    if (elapsed_time + dt >= render_time - 1E-14) {
//        dt = render_time - elapsed_time;
//        render_time += render_dt;
////        frameData << dt << std::endl;
//        do_render = true;
//    }
    std::cout << "Time Step = " << dt << " Time I" << cur_step << " Total elapsed: " << elapsed_time << " Render Step: " << do_render << std::endl;

    scalar_t max_vel = get_max_vel(V);
    if (max_vel > scale_w/dt){
        std::cout << "SIMULATION HAS EXCEEDED MAXIMUM ALLOWED VELOCITY!" << std::endl;
    }
//    scalar_t dt_sft = std::sqrt(density/(8.0*PI*lambda))*std::pow(scale_w, 1.5);
//    std::cout << "Theoretical surface tension time step constract: " << dt_sft << std::endl;
    // Make sure gravity isn't going to overshoot the dt.
//    if (dt*g > scale_w){
//        std::cout << "Restricting time step for gravity" << std::endl;
//        dt = std::abs(scale_w/g);
//    }

//    if (cur_step == 0){
//        dt = -0.1*scale_w/g;
//    } else {
//        dt = 0.1 * get_new_timestep(V);
//    }

    std::cout << "Magnet position: " << magnet->get_pos(t) << std::endl;

#ifdef DEBUGSIM
    std::cout << "Velocities " << std::endl;
    std::cout << std::setprecision(5) << u << std::endl << std::endl;
    std::cout << v << std::endl << std::endl;
#endif

//    auto* execTimer = new ExecTimer("Extrapolate velocities");
    extrapolate_velocities_from_LS();
//    delete execTimer;
//    set_boundary_velocities();

#ifdef DEBUGSIM
    std::cout << "Velocities after extrapolation" << std::endl;
    std::cout << std::setprecision(5) << u << std::endl << std::endl;
    std::cout << v << std::endl << std::endl;
#endif

//    struct timespec begin_ls, end_ls;
//    clock_gettime(CLOCK_MONOTONIC, &begin_ls);
//    update_triangle_mesh();
//    std::cout << "Volume before levelset advancement " << -calc_mesh_volume(mesh) << std::endl;
//    execTimer = new ExecTimer("Update velocities on device");
    update_velocities_on_device();
//    delete execTimer;
//    execTimer = new ExecTimer("Advance LS");
    simLS->advance(cur_step, dt);
//    delete execTimer;
//    clock_gettime(CLOCK_MONOTONIC, &end_ls);
//    double time_spent_ls = (double)(end_ls.tv_sec-begin_ls.tv_sec);
//    time_spent_ls += (end_ls.tv_nsec - begin_ls.tv_nsec) / 1000000000.0;
//    std::cout << "Total time spent on surface: " << time_spent_ls << std::endl;
//    execTimer = new ExecTimer("Update labels");
    update_labels_from_level_set();
//    delete execTimer;
//    update_triangle_mesh();
//    std::cout << "Volume after levelset advancement " << -calc_mesh_volume(mesh) << std::endl;
//    extrapolate_velocities_from_LS(); // Questionable since now quite detached from divergence free pressure field...
//    set_boundary_velocities();

#ifdef DEBUGSIM
    std::cout << "Velocities after second (debatable) extrapolation" << std::endl;
    std::cout << std::setprecision(5) << u << std::endl << std::endl;
    std::cout << v << std::endl << std::endl;
#endif

    // labels are in their final position now
//    execTimer = new ExecTimer("Update mu from labels");
    update_mu_from_labels();
//    delete execTimer;

    std::cout << "Solving for magnetic potential." << std::endl;
//    execTimer = new ExecTimer("Solve magnetic potential");
    solve_ferro();
//    delete execTimer;

//    execTimer = new ExecTimer("Add magnetic potential.");
//    Psi.zeros();
    add_magnet_potential(Psi, *magnet, t);
//    delete execTimer;
//    check_divergence(Psi);
//    std::cout << "Total potential of fluid and magnet" << std::endl;
//    std::cout << Psi << std::endl;

//    execTimer = new ExecTimer("Calculate gradients");
    calc_grad(Psi.subcube(offset_w_em, offset_h_em, offset_d_em, offset_w_em+grid_w-1, offset_h_em+grid_h-1, offset_d_em+grid_d-1), Bmag_x, Bmag_y, Bmag_z, scale_w);
    Bmag_x = -Bmag_x;
    Bmag_y = -Bmag_y;
    Bmag_z = -Bmag_z;
//    delete execTimer;
//    std::cout << "Total mag field" << std::endl;
//    std::cout << Bmag_z << std::endl;

//    execTimer = new ExecTimer("Update mag tensor entries");
    update_mag_tensor_entries(Psi);
//    std::cout << "Txx" << std::endl;
//    std::cout << Tmag_xx << std::endl;
//    std::cout << "Txy" << std::endl;
//    std::cout << Tmag_xy << std::endl;
//    std::cout << "Txz" << std::endl;
//    std::cout << Tmag_xz << std::endl;
//    std::cout << "Tyy" << std::endl;
//    std::cout << Tmag_yy << std::endl;
//    std::cout << "Tyz" << std::endl;
//    std::cout << Tmag_yz << std::endl;
//    std::cout << "Tzz" << std::endl;
//    std::cout << Tmag_zz << std::endl;

//    delete execTimer;

//    std::cout << magnet->get_psi({})

//    std::cout << "Psi Matrix Full" << std::endl;
//    std::cout << std::setprecision(5) << *Psi << std::endl;

#ifdef DEBUGSIM
    std::cout << "T tensor entries" << std::endl;
    std::cout << Tmag_xx << std::endl;
    std::cout << Tmag_yy << std::endl;
    std::cout << Tmag_xy << std::endl;
#endif

    //output_rhs_psi();
//    check_psi(*Psi);
//    execTimer = new ExecTimer("Update velocities on device");
    update_velocities_on_device();
//    delete execTimer;
//    execTimer = new ExecTimer("Advect velocities");
    advect_velocity();
//    delete execTimer;
//    set_boundary_velocities();

#ifdef DEBUGSIM
    std::cout << "Velocities After Advection and before Mag Forces" << std::endl;
    std::cout << std::setprecision(5) << u << std::endl << std::endl;
    std::cout << v << std::endl << std::endl;
    std::cout << "Label" << std::endl;
    std::cout << label << std::endl << std::endl;
#endif

    //apply_force_from_magnets();
//    execTimer = new ExecTimer("Apply force from tensor");
//    std::cout << "Velocity before tensor." << std::endl;
//    std::cout << V[0] << std::endl;
//    std::cout << V[1] << std::endl;
//    std::cout << V[2] << std::endl;
    apply_force_from_tensor();
//    std::cout << "Velocity after tensor." << std::endl;
//    std::cout << V[0] << std::endl;
//    std::cout << V[1] << std::endl;
//    std::cout << V[2] << std::endl;
//    delete execTimer;
//    set_boundary_velocities();

#ifdef DEBUGSIM
    std::cout << "Velocities After Mag Forces Applied" << std::endl;
    std::cout << std::setprecision(5) << u << std::endl << std::endl;
    std::cout << v << std::endl << std::endl;
#endif

//    execTimer = new ExecTimer("Add gravity");
    add_gravity_to_velocity(V[2], dt);
//    delete execTimer;
//    set_boundary_velocities();

//    execTimer = new ExecTimer("Solve viscosity");
    solve_viscosity(); // Should respect boundary velocities (returning them as is).
//    set_boundary_velocities();
//    delete execTimer;
//    set_boundary_velocities();

//    execTimer = new ExecTimer("Solve pressure");
    solve_pressure(fluid_label, true, false, false);
//    delete execTimer;
#ifdef DEBUGSIM
    std::cout << "Pressure after solve" << std::endl;
    std::cout << *P << std::endl << std::endl;
#endif

//    CubeX divs = CubeX(grid_w, grid_h, grid_d, arma::fill::zeros);
//    check_divergence(divs);
//    std::cout << "Divergence of velocity field after pressure solve." << std::endl;
//    std::cout << divs << std::endl;

#ifdef DEBUGSIM
    std::cout << "Velocities After Pressure Forces Applied" << std::endl;
    std::cout << std::setprecision(5) << u << std::endl << std::endl;
    std::cout << v << std::endl << std::endl;
#endif

    t += dt;
    cur_step++;

    if (do_render || cur_step % 5 == 0) {
        frameData << frameNumber << " " << elapsed_time << std::endl;
//        execTimer = new ExecTimer("Render");
        update_triangle_mesh();
        update_viewer_triangle_mesh();
        do_render = false;
//        delete execTimer;
        frameNumber++;
    }
    delete execTimer;
}

//TODO: Simpliciation as of Nov.30 notes, maybe get some better runtime...
scalar_t SimFerro::get_mu_with_Langevin(scalar_t Hx, scalar_t Hy, scalar_t Hz){
    scalar_t H_mag = std::sqrt(Hx*Hx + Hy*Hy + Hz*Hz);
    if (H_mag < 1E-10)
        H_mag = 1E-10;
    scalar_t Mx = M_s * get_Langevin((MU0*m*H_mag)/(kb*T))*(Hx/H_mag);
    scalar_t My = M_s * get_Langevin((MU0*m*H_mag)/(kb*T))*(Hy/H_mag);
    scalar_t Mz = M_s * get_Langevin((MU0*m*H_mag)/(kb*T))*(Hz/H_mag);
    scalar_t M_mag = std::sqrt(Mx*Mx + My*My + Mz*Mz);
    scalar_t susceptibility = M_mag/H_mag;
    scalar_t new_mu = MU0*(1+susceptibility);
//    new_mu = 1.25*MU0;
    return new_mu;
}

void SimFerro::add_magnet_potential(CubeX &Q, Magnet &M, scalar_t time){
    for (int k = 0; k < grid_d_em; k++) {
        for (int j = 0; j < grid_h_em; j++) {
            for (int i = 0; i < grid_w_em; i++) {
                Q(i, j, k) += M.get_psi(
                        {i * scale_w - offset_w_em * scale_w, j * scale_h - offset_h_em * scale_h,
                         k * scale_d - offset_d_em * scale_d}, time);
            }
        }
    }
}

void SimFerro::solve_ferro() {
#ifdef DEBUGSIM
    // Check quantities before solving
    std::cout << "Level Set before psi solve" << std::endl;
    std::cout << simLS->LS << std::endl;
    std::cout << "Mu values" << std::endl;
    std::cout << mu << std::endl;
    std::cout << "B field values" << std::endl;
    std::cout << Bmag_x << std::endl;
    std::cout << Bmag_y << std::endl;
    std::cout << "Check the magnet" << std::endl;
    MatrixXs mag_temp(grid_w_em, grid_h_em);
    for (int i = 0; i < grid_w_em; i++){
        for (int j = 0; j < grid_h_em; j++){
            mag_temp(i,j) = magnet->get_psi(Vector2(i*scale_w-offset_w_em*scale_w, j*scale_h - offset_h_em*scale_h), t);
        }
    }
    std::cout << mag_temp << std::endl;
#endif

    A_psi.setZero();
    A_psid.setZero();
    A_psi.reserve(Eigen::VectorXi::Constant(n_cells_em, 7));
    A_psid.reserve(Eigen::VectorXi::Constant(n_cells_em, 1));
    b_psi = VectorXs::Zero(n_cells_em, 1);
//    ExecTimer *execTimer = new ExecTimer("Solve potential build A and B");
    build_A_psi_and_b();
//    delete execTimer;

//    CubeX b(b_psi.data(), grid_w_em, grid_h_em, grid_d_em);
//    std::cout << "RHS" << std::endl;
//    std::cout << b << std::endl;

//    Eigen::MatrixXd A_dense;
//    A_dense = Eigen::MatrixXd(A_psi);
//    Eigen::EigenSolver<Eigen::MatrixXd> es;
//    es.compute(A_dense, false);
//    std::cout << "Eigenvalues are " << es.eigenvalues().transpose() << std::endl;

#ifdef DEBUGFERRO
    std::cout << "Right hand side" << std::endl;
    std::cout << Eigen::Map<MatrixXs>(b_psi.data(), grid_w_em, grid_h_em) << std::endl;
#endif
    scalar_t sum = 0;
    for (int j = 0; j < n_cells_em; j++) {
        sum += b_psi(j);
    }
    sum = sum / ((scalar_t) n_cells_em);
    for (int j = 0; j < n_cells_em; j++) {
        b_psi(j) -= sum;
    }

    A_psi.makeCompressed();
    A_psid.makeCompressed();
//    std::cout << A_psi.valuePtr() << std::endl; //values in matrix
//    for (int i = 0; i < A_psi.outerSize(); i++) {
//        std::cout << A_psi.outerIndexPtr()[i] << std::endl; //n values in each row (i+1, starts with zero).
//    }
//    std::cout << A_psi.innerIndexPtr() << std::endl; //indices of non-zero elements in each row.
//    CG_psi.setMaxIterations(100000);
//    CG_psi.compute(A_psi);
//    if (CG_psi.info() != Eigen::Success) {
//        std::cout << "Failed decomposition!" << std::endl;
//    }
//    CG_psi.project = true;
//    psi.setZero();
//    psi = CG_psi.solve(b_psi);
    std::cout << "N cells being used in sparse matrix " << A_psi.nonZeros() << std::endl;

    // CUDA CG Steps
//    execTimer = new ExecTimer("Do CUDA potential solve");
    psi.setZero();
    cudacg_mag->load_matrix(A_psi.outerIndexPtr(), A_psi.innerIndexPtr(), A_psi.valuePtr(), (psi.data()), (b_psi.data()), A_psi.rows(), A_psi.nonZeros());
    cudacg_mag->load_diagonal(A_psid.outerIndexPtr(), A_psid.innerIndexPtr(), A_psid.valuePtr());
//    auto *execTimer2 = new ExecTimer("Actual just solve part of CUDA potential solve");
    cudacg_mag->solve();
//    delete execTimer2;
    cudacg_mag->get_error();
//    delete execTimer;

//    std::ofstream badsolvefile;
//    badsolvefile.open("badsolve.txt");
//    badsolvefile << A_psi << "\n";
//    badsolvefile << b_psi << "\n";

//    std::cout << Psi << std::endl;

//    if (CG_psi.info() != Eigen::Success) {
//        std::cout << "============================== Solve failed! =================================" << std::endl;
//        std::ofstream badsolvefile;
//        badsolvefile.open("badsolve.txt");
//        badsolvefile << A_psi << "\n";
//        badsolvefile << b_psi << "\n";
////        output_rhs_psi();
//        badsolvefile.close();
////        char temp;
////        std::cin >> temp;
//        // Set to zero, so gradient is zero and we don't apply false velocities to anything.
//        psi = MatrixXs::Zero(n_cells_em, 1);
//    }
//    psi = -1.0*psi;

//    std::cout << "Stats for last potential solve." << std::endl;
//    std::cout << CG_psi.error() << std::endl;
//    std::cout << CG_psi.iterations() << std::endl;
}

void SimFerro::apply_force_from_tensor() {
    for (unsigned int k = 0; k < grid_d; k++) {
        for (unsigned int j = 0; j < grid_h; j++) {
            for (unsigned int i = 0; i < grid_w + 1; i++) {
                if (fluid_label->get_label_center(i, j, k) == 1 || fluid_label->get_label_center(i - 1, j, k) == 1) {
                    scalar_t Fx = (Tmag_xx(i, j, k) - Tmag_xx(i - 1, j, k)) / scale_w +
                                  (Tmag_xy(i, j+1, k) - Tmag_xy(i, j, k)) / scale_h +
                                  (Tmag_xz(i, j, k+1) - Tmag_xz(i, j, k)) / scale_d;
                    V[0](i, j, k) += dt * Fx/density;
                }
            }
        }
    }

    for (unsigned int k = 0; k < grid_d; k++) {
        for (unsigned int j = 0; j < grid_h+1; j++) {
            for (unsigned int i = 0; i < grid_w; i++) {
                if (fluid_label->get_label_center(i, j, k) == 1 || fluid_label->get_label_center(i, j-1, k) == 1) {
                    scalar_t Fy = (Tmag_yy(i, j, k) - Tmag_yy(i, j-1, k)) / scale_h +
                                  (Tmag_xy(i+1, j, k) - Tmag_xy(i, j, k)) / scale_w +
                                  (Tmag_yz(i, j, k+1) - Tmag_yz(i, j, k)) / scale_d;
                    V[1](i, j, k) += dt * Fy/density;
                }
            }
        }
    }

    for (unsigned int k = 0; k < grid_d+1; k++) {
        for (unsigned int j = 0; j < grid_h; j++) {
            for (unsigned int i = 0; i < grid_w; i++) {
                if (fluid_label->get_label_center(i, j, k) == 1 || fluid_label->get_label_center(i, j, k-1) == 1) {
                    scalar_t Fz = (Tmag_zz(i, j, k) - Tmag_zz(i, j, k-1)) / scale_d +
                                  (Tmag_xz(i+1, j, k) - Tmag_xz(i, j, k)) / scale_w +
                                  (Tmag_yz(i, j+1, k) - Tmag_yz(i, j, k)) / scale_h;
                    V[2](i, j, k) += dt * Fz/density;
                }
            }
        }
    }
}

scalar_t SimFerro::get_psi(int i, int j, int k){
    return Psi(i+offset_w_em, j + offset_h_em, k + offset_d_em);
}

scalar_t SimFerro::get_mu_coeff(Vector3 pos){
    scalar_t Bquadx = grid_trilerp(pos, Bmag_x, scale_w);
    scalar_t Bquady = grid_trilerp(pos, Bmag_y, scale_w);
    scalar_t Bquadz = grid_trilerp(pos, Bmag_z, scale_w);
    scalar_t mu_fluid = get_mu_with_Langevin(Bquadx, Bquady, Bquadz);
    scalar_t mu_coeff = mu_fluid + (MU0 - mu_fluid)*((std::tanh(((interface_reg*PI)/scale_w)*grid_tricerp(pos, simLS->LS, scale_w, false))+1.0)/2.0);

    return mu_coeff;
}

void SimFerro::update_mag_tensor_entries(CubeX &Psi) {
    for (unsigned int k = 0; k < grid_d; k++) {
        for (unsigned int j = 0; j < grid_h; j++) {
            for (unsigned int i = 0; i < grid_w; i++) {
                int il = i + offset_w_em;
                int jl = j + offset_h_em;
                int kl = k + offset_d_em;

                scalar_t Bx = Bmag_x(i, j, k);
                scalar_t By = Bmag_y(i, j, k);
                scalar_t Bz = Bmag_z(i, j, k);
                scalar_t Bnorm2 = (Bx * Bx + By * By + Bz * Bz);
                scalar_t mu_fluid = get_mu_with_Langevin(Bx, By, Bz);
//                scalar_t mu_val = mu_fluid + (MU0 - mu_fluid)*((std::tanh(((interface_reg*PI)/scale_w)*simLS->LS(i, j, k))+1.0)/2.0);
                scalar_t mu_val = get_mu(i,j,k);
                Tmag_xx(i, j, k) = mu_val * (Bx * Bx - 0.5 * Bnorm2);
                Tmag_yy(i, j, k) = mu_val * (By * By - 0.5 * Bnorm2);
                Tmag_zz(i, j, k) = mu_val * (Bz * Bz - 0.5 * Bnorm2);
            }
        }
    }

    /* Couldn't think of a way to do these as one function call three times, different optimal ways of interpolating
     * values, different derivatives to take, wouldn't have been clean. Update this is if better method is possible. */
    for (int k = 0; k < grid_d; k++) {
        for (int j = 0; j < grid_h+1; j++) {
            for (int i = 0; i < grid_w+1; i++) {
                scalar_t mu_coeff;
//                Vector2 pos = {(i-0.5)*scale_w, (j-0.5)*scale_h};
                Vector3 pos = {(i-0.5)*scale_w, (j-0.5)*scale_h, (k)*scale_d};

//                if (grid_bilerp(pos, simLS->LS.slice(k)) < eps){
//                    scalar_t Bquadx = grid_bilerp(pos, Bmag_x.slice(k));
//                    scalar_t Bquady = grid_bilerp(pos, Bmag_y.slice(k));
//                    scalar_t Bquadz = grid_bilerp(pos, Bmag_z.slice(k));
//                    mu_coeff = get_mu_with_Langevin(Bquadx, Bquady, Bquadz);
//                } else {
//                    mu_coeff = MU0;
//                }
//              mu_coeff = get_mu_coeff(pos);
                mu_coeff = 0.25*(get_mu(i-1, j-1, k) + get_mu(i, j-1, k) + get_mu(i-1, j, k) + get_mu(i,j,k));
                Tmag_xy(i, j ,k) = mu_coeff * ((get_psi(i, j, k) - get_psi(i-1, j, k) + get_psi(i, j-1, k) - get_psi(i-1, j-1, k))/(2*scale_w))
                        * ((get_psi(i,j,k) - get_psi(i, j-1, k) + get_psi(i-1, j, k) - get_psi(i-1, j-1, k))/(2*scale_h));
            }
        }
    }

    for (int k = 0; k < grid_d+1; k++) {
        for (int j = 0; j < grid_h; j++) {
            for (int i = 0; i < grid_w+1; i++) {
                scalar_t mu_coeff;
                Vector3 pos = {(i-0.5)*scale_w, j*scale_h, (k-0.5)*scale_d};
//                if (grid_trilerp(pos, simLS->LS) < eps){
//                    scalar_t Bquadx = grid_trilerp(pos, Bmag_x);
//                    scalar_t Bquady = grid_trilerp(pos, Bmag_y);
//                    scalar_t Bquadz = grid_trilerp(pos, Bmag_z);
//                    mu_coeff = get_mu_with_Langevin(Bquadx, Bquady, Bquadz);
//                } else {
//                    mu_coeff = MU0;
//                }
//                mu_coeff = get_mu_coeff(pos);
                mu_coeff = 0.25*(get_mu(i-1, j, k-1) + get_mu(i, j, k-1) + get_mu(i-1, j, k) + get_mu(i,j,k));
                Tmag_xz(i, j ,k) = mu_coeff * ((get_psi(i, j, k) - get_psi(i, j, k-1) + get_psi(i-1, j, k) - get_psi(i-1, j, k-1))/(2*scale_w))
                                   * ((get_psi(i,j,k) - get_psi(i-1, j, k) + get_psi(i, j, k-1) - get_psi(i-1, j, k-1))/(2*scale_h));
            }
        }
    }

    for (int k = 0; k < grid_d+1; k++) {
        for (int j = 0; j < grid_h+1; j++) {
            for (int i = 0; i < grid_w; i++) {
                scalar_t mu_coeff;
                Vector3 pos = {i*scale_w, (j-0.5)*scale_h, (k-0.5)*scale_d};
//                if (grid_trilerp(pos, simLS->LS) < eps){
//                    scalar_t Bquadx = grid_trilerp(pos, Bmag_x);
//                    scalar_t Bquady = grid_trilerp(pos, Bmag_y);
//                    scalar_t Bquadz = grid_trilerp(pos, Bmag_z);
//                    mu_coeff = get_mu_with_Langevin(Bquadx, Bquady, Bquadz);
//                } else {
//                    mu_coeff = MU0;
//                }
//                mu_coeff = get_mu_coeff(pos);
                mu_coeff = 0.25*(get_mu(i, j-1, k) + get_mu(i, j-1, k-1) + get_mu(i, j, k-1) + get_mu(i,j,k));
                Tmag_yz(i, j ,k) = mu_coeff * ((get_psi(i, j, k) - get_psi(i, j-1, k) + get_psi(i, j, k-1) - get_psi(i, j-1, k-1))/(2*scale_w))
                                   * ((get_psi(i,j,k) - get_psi(i, j, k-1) + get_psi(i, j-1, k) - get_psi(i, j-1, k-1))/(2*scale_h));
            }
        }
    }
}

void SimFerro::update_mu_from_labels() {
    for (int k = 0; k < grid_d; k++) {
        for (int j = 0; j < grid_h; j++) {
            for (int i = 0; i < grid_w; i++) {
                int iem = i + offset_w_em;
                int jem = j + offset_h_em;
                int kem = k + offset_d_em;
                // Air and solids have MU0 permeability
//                if (get_label_center(i, j, k) == 0 || get_label_center(i, j, k) == 2) {
//                    mu(iem, jem, kem) = MU0;
//                } else if (get_label_center(i, j, k) == 1) {
//                    // mu(iem, jem) = MUF;
//                    mu(iem, jem, kem) = get_mu_with_Langevin(Bmag_x(i, j, k), Bmag_y(i, j, k), Bmag_z(i,j,k));
//                }

                scalar_t mu_fluid = get_mu_with_Langevin(Bmag_x(i, j, k), Bmag_y(i, j, k), Bmag_z(i,j,k));
                Vector3 pos = {i*scale_w, j*scale_h, k*scale_d};
                scalar_t new_mu = mu_fluid + (MU0 - mu_fluid)*((std::tanh(((interface_reg*PI)/scale_w)*grid_tricerp(pos, simLS->LS, scale_w, false))+1.0)/2.0);
                mu(iem, jem, kem) = new_mu;
            }
        }
    }
}

scalar_t SimFerro::get_mu(int i, int j, int k){
    return mu(i + offset_w_em, j + offset_h_em, k + offset_d_em);
}

scalar_t SimFerro::get_mu_half(Vector3i Ip, Vector3i dir){
    Vector3i I2p = Ip + dir;
    scalar_t mu_val;
    if (is_coord_valid(Ip) && is_coord_valid(I2p)){
        Vector3ui I = arma::conv_to<arma::Col<unsigned int>>::from(Ip);
        Vector3ui I2 = arma::conv_to<arma::Col<unsigned int>>::from(I2p);

        mu_val = 0.5*(get_mu(I(0), I(1), I(2)) + get_mu(I2(0), I2(1), I2(2)));

//        scalar_t Bhalfx = 0.5*(Bmag_x(I(0), I(1), I(2)) + Bmag_x(I2(0), I2(1), I2(2)));
//        scalar_t Bhalfy = 0.5*(Bmag_y(I(0), I(1), I(2)) + Bmag_y(I2(0), I2(1), I2(2)));
//        scalar_t Bhalfz = 0.5*(Bmag_z(I(0), I(1), I(2)) + Bmag_z(I2(0), I2(1), I2(2)));
//        scalar_t mu_fluid = get_mu_with_Langevin(Bhalfx, Bhalfy, Bhalfz);
//        scalar_t ls_val = grid_tricerp((get_position(I)+get_position(I2))/2.0, simLS->LS, scale_w, false);//(simLS->LS(I(0), I(1), I(2)) + simLS->LS(I2(0), I2(1), I2(2)))/2.0;
//        mu_val = mu_fluid + (MU0 - mu_fluid)*((std::tanh(((2.0*PI)/scale_w)*ls_val)+1.0)/2.0);

//        if (get_label_center(I) == 1 && get_label_center(I2) == 1){
//            scalar_t Bhalfx = 0.5*(Bmag_x(I(0), I(1), I(2)) + Bmag_x(I2(0), I2(1), I2(2)));
//            scalar_t Bhalfy = 0.5*(Bmag_y(I(0), I(1), I(2)) + Bmag_y(I2(0), I2(1), I2(2)));
//            scalar_t Bhalfz = 0.5*(Bmag_z(I(0), I(1), I(2)) + Bmag_z(I2(0), I2(1), I2(2)));
//            mu_val = get_mu_with_Langevin(Bhalfx, Bhalfy, Bhalfz);
//        } else if (get_label_center(I) == 1 && get_label_center(I2) != 1){
//            scalar_t ls_val = (simLS->LS(I(0), I(1), I(2)) + simLS->LS(I2(0), I2(1), I2(2)))/2.0;
//            mu_val = get_mu(I(0),I(1),I(2)) + (get_mu(I2(0), I2(1), I2(2)) - get_mu(I(0), I(1), I(2)))*((std::tanh(((2.0*PI)/scale_w)*ls_val)+1.0)/2.0);
//        } else if (get_label_center(I) != 1 && get_label_center(I2) == 1){
//            scalar_t ls_val = (simLS->LS(I(0), I(1), I(2)) + simLS->LS(I2(0), I2(1), I2(2)))/2.0;
//            mu_val = get_mu(I2(0),I2(1),I2(2)) + (get_mu(I(0), I(1), I(2)) - get_mu(I2(0), I2(1), I2(2)))*((std::tanh(((2.0*PI)/scale_w)*ls_val)+1.0)/2.0);
//        } else {
//            mu_val = MU0;
//        }
    } else {
        mu_val = MU0;
    }
    return mu_val;
}

scalar_t SimFerro::get_mu_frac(Vector3i Ip, Vector3i dir, scalar_t dx){
    Vector3 pos = {Ip(0)*scale_w + dir(0)*dx, Ip(1)*scale_h + dir(1)*dx, Ip(2)*scale_d + dir(2)*dx};
    scalar_t mu_val;
    if (is_coord_valid(Ip)){
        scalar_t Bhalfx = grid_trilerp(pos, Bmag_x, scale_w);
        scalar_t Bhalfy = grid_trilerp(pos, Bmag_y, scale_w);
        scalar_t Bhalfz = grid_trilerp(pos, Bmag_z, scale_w);
        scalar_t mu_fluid = get_mu_with_Langevin(Bhalfx, Bhalfy, Bhalfz);
        scalar_t ls_val = grid_trilerp(pos, simLS->LS, scale_w);
        mu_val = mu_fluid + (MU0 - mu_fluid)*((std::tanh(((interface_reg*PI)/scale_w)*ls_val)+1.0)/2.0);
    } else {
        mu_val = MU0;
    }
    return mu_val;
}


void SimFerro::build_A_psi_and_b() {
    /* Build a block matrix for solving the magnetic potential.*/
    for (unsigned int d = 0; d < grid_w_em*grid_h_em*grid_d_em; d++){
        Vector3ui IS = convert_index_to_coords(d, grid_w_em, grid_h_em); //EM spatial coordinates
        Vector3i ISF = IS - offset_em; //Fluid spatial coordinates (previously called isl, jsl etc.). May not be in grid.
        Vector3 X = {ISF(0)*scale_w, ISF(1)*scale_h, ISF(2)*scale_d};

        // assume equal size grid in both x, y directions eg. 1 / deltaX^2
        scalar_t scale = -1.0 / (scale_w * scale_w);

        // interpolated mu values (half step in each direction)
        scalar_t mu_up, mu_down, mu_left, mu_right, mu_front, mu_back;
        mu_up = get_mu_half(ISF, {-1, 0, 0});
        mu_down = get_mu_half(ISF, {1, 0, 0});
        mu_left = get_mu_half(ISF, {0, -1, 0});
        mu_right = get_mu_half(ISF, {0, 1, 0});
        mu_front = get_mu_half(ISF, {0, 0, -1});
        mu_back = get_mu_half(ISF, {0, 0, 1});

        // coefficients
        scalar_t coeff_mu_diag = -scale * (mu_up + mu_down + mu_left + mu_right + mu_front + mu_back);
        scalar_t coeff_mu_left = scale * mu_left;
        scalar_t coeff_mu_right = scale * mu_right;
        scalar_t coeff_mu_up = scale * mu_up;
        scalar_t coeff_mu_down = scale * mu_down;
        scalar_t coeff_mu_front = scale * mu_front;
        scalar_t coeff_mu_back = scale * mu_back;

        scalar_t h = 1.0;
        scalar_t mu_up_rhs, mu_down_rhs, mu_left_rhs, mu_right_rhs, mu_front_rhs, mu_back_rhs;
        mu_up_rhs = mu_up; mu_down_rhs = mu_down; mu_left_rhs = mu_left, mu_right_rhs = mu_right, mu_back_rhs = mu_back, mu_front_rhs = mu_front;
//        mu_up_rhs = get_mu_frac(ISF, {-1, 0, 0}, 0.5*h*scale_w);
//        mu_down_rhs = get_mu_frac(ISF, {1, 0, 0}, 0.5*h*scale_w);
//        mu_left_rhs = get_mu_frac(ISF, {0, -1, 0}, 0.5*h*scale_h);
//        mu_right_rhs = get_mu_frac(ISF, {0, 1, 0}, 0.5*h*scale_h);
//        mu_front_rhs = get_mu_frac(ISF, {0, 0, -1}, 0.5*h*scale_d);
//        mu_back_rhs = get_mu_frac(ISF, {0, 0, 1}, 0.5*h*scale_d);

        // calculate the right hand side of the system
        scalar_t h_scale = (-1.0 / (h * scale_w * h * scale_w));
        scalar_t psi_down = mu_down_rhs * magnet->get_psi({X(0) + h * scale_w, X(1), X(2)}, t);
        scalar_t psi_up = mu_up_rhs * magnet->get_psi({X(0) - h * scale_w, X(1), X(2)}, t);
        scalar_t psi_left = mu_left_rhs * magnet->get_psi({X(0), X(1) - h * scale_h, X(2)}, t);
        scalar_t psi_right = mu_right_rhs * magnet->get_psi({X(0), X(1) + h * scale_h, X(2)}, t);
        scalar_t psi_front = mu_front_rhs * magnet->get_psi({X(0), X(1), X(2) - h*scale_d}, t);
        scalar_t psi_back = mu_back_rhs * magnet->get_psi({X(0), X(1), X(2) + h*scale_d}, t);
        scalar_t psi_center =
                -(mu_up + mu_down + mu_left + mu_right + mu_back + mu_front) * magnet->get_psi(X, t);
        scalar_t perm_mag_rhs = psi_down + psi_up + psi_left + psi_right + psi_back + psi_front + psi_center;

        if (is_coord_valid(ISF)){
            Vector3ui I = arma::conv_to<arma::Col<unsigned int>>::from(ISF); // Now know these are correct spatial coordinates
            std::set<int> neighbours = get_neighbours<CubeXi, int>(fluid_label->label, I(0)+1, I(1)+1, I(2)+1);
            if ((fluid_label->get_label_center(I) == 0 || fluid_label->get_label_center(I) == 2) && (neighbours.find(1) != neighbours.end())){
                b_psi(d) = -h_scale * perm_mag_rhs;
            } else if (fluid_label->get_label_center(I) == 1 && (neighbours.find(0) != neighbours.end() || neighbours.find(2) != neighbours.end())){
                b_psi(d) = -h_scale * perm_mag_rhs;
            }
        }
//        b_psi(d) = -h_scale * perm_mag_rhs;

        A_psi.coeffRef(d, d) += coeff_mu_diag;

        if (IS(0) == 0) {
            A_psi.coeffRef(d, d) += coeff_mu_up;
        } else {
            A_psi.coeffRef(d, d - 1) = coeff_mu_up;
        }

        if (IS(0) == grid_w_em - 1) {
            A_psi.coeffRef(d, d) += coeff_mu_down;
        } else {
            A_psi.coeffRef(d, d + 1) = coeff_mu_down;
        }

        // Check boundary conditions along j (y)
        if (IS(1) == 0) {
            A_psi.coeffRef(d, d) += coeff_mu_left;
        } else {
            A_psi.coeffRef(d, d - grid_w_em) = coeff_mu_left;
        }

        if (IS(1) == grid_h_em - 1) {
            A_psi.coeffRef(d, d) += coeff_mu_right;
        } else {
            A_psi.coeffRef(d, d + grid_w_em) = coeff_mu_right;
        }

        // Check boundary conditions along k (z)
        if (IS(2) == 0) {
            A_psi.coeffRef(d, d) += coeff_mu_front;
        } else {
            A_psi.coeffRef(d, d - grid_w_em*grid_h_em) = coeff_mu_front;
        }

        if (IS(2) == grid_d_em - 1) {
            A_psi.coeffRef(d, d) += coeff_mu_back;
        } else {
            A_psi.coeffRef(d, d + grid_w_em*grid_h_em) = coeff_mu_back;
        }

        scalar_t diag = A_psi.coeffRef(d,d);
        if (diag != 0) {
            A_psid.coeffRef(d, d) = 1.0 / diag;
        }
    }
}

void SimFerro::check_divergence_mag(const CubeX &Q){
    CubeX R(grid_w_em, grid_h_em, grid_d_em);

    for (int k = 1; k < grid_d_em-1; k++){
        for (int j = 1; j < grid_h_em-1; j++){
            for (int i = 1; i < grid_w_em-1; i++){
                Vector3i IS = {i, j, k};
                Vector3i ISF = IS - offset_em; //Fluid spatial coordinates (previously called isl, jsl etc.). May not be in grid.
                Vector3 X = {ISF(0)*scale_w, ISF(1)*scale_h, ISF(2)*scale_d};

                scalar_t mu_up, mu_down, mu_left, mu_right, mu_front, mu_back;
                mu_up = get_mu_half(ISF, {-1, 0, 0});
                mu_down = get_mu_half(ISF, {1, 0, 0});
                mu_left = get_mu_half(ISF, {0, -1, 0});
                mu_right = get_mu_half(ISF, {0, 1, 0});
                mu_front = get_mu_half(ISF, {0, 0, -1});
                mu_back = get_mu_half(ISF, {0, 0, 1});

                scalar_t psi_down = mu_down * Q(i+1,j,k);
                scalar_t psi_up = mu_up * Q(i-1,j,k);
                scalar_t psi_left = mu_left * Q(i,j-1,k);
                scalar_t psi_right = mu_right * Q(i,j+1,k);
                scalar_t psi_front = mu_front * Q(i,j,k-1);
                scalar_t psi_back = mu_back * Q(i,j,k+1);
                scalar_t psi_center =
                        -(mu_up + mu_down + mu_left + mu_right + mu_back + mu_front) * Q(i,j,k);
                scalar_t perm_mag = psi_down + psi_up + psi_left + psi_right + psi_back + psi_front + psi_center;

                R(i,j,k) = perm_mag;
            }
        }
    }

    std::cout << "Result of divergence test." << std::endl;
    std::cout << R << std::endl;
}
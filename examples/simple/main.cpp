#include "sim_ferro.hpp"
#include "sim_levelset.hpp"
#include "sim_pls_cuda.hpp"
#include "magnet.hpp"

int main(int argc, char** argv){
    // Build the parameter structures to setup the simulation.
    SimParams* params; // Grid and global physics (gravity) parameters
    SimWaterParams* water_params; // Fluid parameters (viscosity, surface tension...)
    SimFerroParams* ferro_params; // Ferrofluid parameters (magnetic susceptibility...)
    int n_steps;
    SimFerro::create_ferro_params_from_args(argc, argv, n_steps, params, water_params, ferro_params);

    // Create a magnet to interact with the ferrofluid.
    auto magnet = new Magnet({params->sim_w / 2.0, params->sim_h/2.0, -0.2},
            {params->sim_w / 2.0, params->sim_h/2.0, -0.2}, 2.0, {0, 0, 1}, ferro_params->appStrength);
    magnet->flat_field = false; // Enable for a homogenous magnet field.

    // Create the simulation.
    auto sim = SimFerro(*params, *water_params, *ferro_params, magnet);

    // Level set must be added after, since it needs to know simulation velocities for self advection.
    auto level_set = new SimLevelSet(*params, sim.DEV_C, sim.DEV_V);
    level_set->initialize_level_set_rectangle({1,1,1}, {(double) params->grid_w-2, (double) params->grid_h-2, (double) params->grid_d / 4.0});
    sim.initialize_fluid(level_set);

    // Run the simulation.
    for (int i = 0; i < n_steps; i++){
        sim.step();
    }

    return 0;
}

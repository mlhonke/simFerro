#include "sim_ferro.hpp"
#include "sim_levelset.hpp"

int main(int argc, char** argv){
    // Build the parameter structures to setup the simulation.
    SimParams* params; // Grid and global physics (gravity) parameters
    SimWaterParams* water_params; // Fluid parameters (viscosity, surface tension...)
    SimFerroParams* ferro_params; // Ferrofluid parameters (magnetic susceptibility...)
    int n_steps;
    SimFerro::create_params_from_args(argc, argv, n_steps, params, water_params, ferro_params);

    // Create the simulation.
    SimFerro sim = SimFerro(*params, *water_params, *ferro_params);

    auto level_set = new SimLevelSet(*params, sim.DEV_C, sim.DEV_V);
    level_set->initialize_level_set_rectangle({1,1,1}, {(double) params->grid_w-2, (double) params->grid_h-2, (double) params->grid_d / 4.0});
    sim.initialize_fluid(level_set);

    // Run the simulation.
    for (int i = 0; i < n_steps; i++){
        sim.step();
    }

    return 0;
}

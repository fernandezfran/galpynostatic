// This file is part of galpynostatic
//    https://github.com/fernandezfran/galpynostatic/
// Copyright (c) 2024, Francisco Fernandez, Maximiliano Gavilán, Andrés
// Ruderman License: MIT
//    https://github.com/fernandezfran/galpynostatic/blob/master/LICENSE

#include <omp.h>

#include <cmath>

extern "C" void
run_map(bool model, const double g_pot, const int nthreads,
        const int grid_size, const int time_steps, const int isotherm_len,
        const int num_logell, const int num_logxi, const double temperature,
        const double mass, const double density, const double vcut,
        const double specific_capacity, const double geometry_param,
        const double *logell_grid, const double *logxi_grid,
        const double *spl_ai, const double *spl_bi, const double *spl_ci,
        const double *spl_di, const double *soc_eq, double *res_logell,
        double *res_logxi, double *res_socmax)
{
    const double faraday = 96484.5561;
    const double gas_constant = 8.314472;
    const double t_hour = 3600.0;
    const double rfaraday = gas_constant * temperature / faraday;

    (nthreads == -1 ? omp_set_num_threads(omp_get_num_procs())
                    : omp_set_num_threads(nthreads));

#pragma omp parallel
    {
        int index;
#pragma omp for collapse(2) firstprivate(logxi_grid, logell_grid)
        for (int logell = 0; logell < num_logell; logell++) {
            for (int logxi = 0; logxi < num_logxi; logxi++) {
                index = logell * num_logxi + logxi;

                double c_rate = t_hour * (geometry_param - 1) /
                                pow(pow(10, logxi_grid[logxi]), 2);

                double particle_size =
                    2.0 * sqrt((pow(10, logell_grid[logell]) * geometry_param *
                                t_hour) /
                               c_rate);

                double surface_area =
                    2.0 * geometry_param * mass / (density * particle_size);

                double ccd = -c_rate * specific_capacity * mass /
                             (1000.0 * surface_area);

                double maximum_capacity =
                    specific_capacity * density * 3.6 / faraday;

                double time_step = -specific_capacity * mass * 3.6 /
                                   (ccd * surface_area) /
                                   static_cast<double>(time_steps - 1);
                double space_step =
                    0.5 * particle_size / static_cast<double>(grid_size - 1);

                double position[grid_size];
                for (int i = 0; i < grid_size; i++) {
                    position[i] = i * space_step;
                }

                double intercepts[grid_size], coefs[grid_size],
                    gamma[grid_size], previous_soc[grid_size],
                    actual_soc[grid_size];
                for (int i = 0; i < grid_size; i++) {
                    intercepts[i] = coefs[i] = gamma[i] = previous_soc[i] =
                        actual_soc[i] = 0.0;
                }

                double alpha = time_step / (2.0 * space_step * space_step);
                double beta =
                    (geometry_param - 1) * time_step / (4.0 * space_step);
                double alpha_0 = 1.0 + (2.0 * alpha);
                double gamma0 = 1.0 - (2.0 * alpha);
                coefs[1] = 2.0 * alpha / alpha_0;
                for (int i = 2; i < grid_size; i++) {
                    coefs[i] = (alpha + (beta / position[i - 1])) /
                               (alpha_0 - (alpha - (beta / position[i - 1])) *
                                              coefs[i - 1]);
                }
                double add[grid_size];
                double sub[grid_size];
                for (int i = 0; i < grid_size; i++) {
                    add[i] = alpha + (beta / position[i]);
                    sub[i] = alpha - (beta / position[i]);
                }

                if (model) {
                    for (int i = 0; i < grid_size; i++) {
                        actual_soc[i] = 1.0e-4;
                    }
                }
                else {
                    for (int i = 0; i < grid_size; i++) {
                        actual_soc[i] = (soc_eq[0] == 0.0) ? 1e-4 : soc_eq[0];
                    }
                }

                double pot_i = vcut + 1.0;

                while (pot_i > vcut) {
                    double pot_eq = 0.0;

                    if (model) {
                        pot_eq = rfaraday *
                                 (g_pot * (0.5 - actual_soc[grid_size - 1]) +
                                  log((1.0 - actual_soc[grid_size - 1]) /
                                      actual_soc[grid_size - 1]));
                    }
                    else {
                        double ai = 0.0;
                        double bi = 0.0;
                        double ci = 0.0;
                        double di = 0.0;
                        double socd = 0.0;
                        for (int i = 0; i < isotherm_len; i++) {
                            if ((actual_soc[grid_size - 1] >= soc_eq[i]) &&
                                (actual_soc[grid_size - 1] < soc_eq[i + 1])) {
                                ai = spl_ai[i];
                                bi = spl_bi[i];
                                ci = spl_ci[i];
                                di = spl_di[i];
                                socd = soc_eq[i];
                                break;
                            }
                            else {
                                ai = spl_ai[isotherm_len - 1];
                                bi = spl_bi[isotherm_len - 1];
                                ci = spl_ci[isotherm_len - 1];
                                di = spl_di[isotherm_len - 1];
                                socd = soc_eq[isotherm_len - 1];
                            }
                        }
                        double dsocs = actual_soc[grid_size - 1] - socd;

                        pot_eq = di + ci * dsocs + bi * dsocs * dsocs +
                                 ai * dsocs * dsocs * dsocs;
                    }

                    double i0 = faraday * maximum_capacity *
                                sqrt(actual_soc[grid_size - 1] *
                                     (1.0 - actual_soc[grid_size - 1]));

                    pot_i = pot_eq + 2.0 * rfaraday * asinh(ccd / (2.0 * i0));

                    for (int i = 0; i < grid_size; i++) {
                        previous_soc[i] = actual_soc[i];
                    }

                    // Vector of solutions and Thomas coefficients
                    gamma[0] = gamma0 * previous_soc[0] +
                               2.0 * alpha * previous_soc[1];
                    gamma[grid_size - 1] =
                        gamma0 * previous_soc[grid_size - 1] +
                        2 * alpha * previous_soc[grid_size - 2] -
                        add[grid_size - 1] * 4.0 * space_step *
                            (ccd / (faraday * maximum_capacity));
                    for (int i = 1; i < grid_size - 1; i++) {
                        gamma[i] = gamma0 * previous_soc[i] +
                                   add[i] * previous_soc[i + 1] +
                                   sub[i] * previous_soc[i - 1];
                    }

                    intercepts[1] = gamma[0] / alpha_0;
                    for (int i = 2; i < grid_size; i++) {
                        intercepts[i] =
                            (gamma[i - 1] + sub[i - 1] * intercepts[i - 1]) /
                            (alpha_0 - sub[i - 1] * coefs[i - 1]);
                    }

                    // Concentration calculation
                    actual_soc[grid_size - 1] =
                        (gamma[grid_size - 1] +
                         2.0 * alpha * intercepts[grid_size - 1]) /
                        (alpha_0 - 2.0 * alpha * coefs[grid_size - 1]);
                    for (int i = 2; i < grid_size + 1; i++) {
                        actual_soc[grid_size - i] =
                            (coefs[grid_size - (i - 1)] *
                             actual_soc[grid_size - (i - 1)]) +
                            intercepts[grid_size - (i - 1)];
                    }
                }

                double socmax = 0.0;
                for (int i = 0; i < grid_size; i++) {
                    socmax += previous_soc[i];
                }
                socmax /= static_cast<double>(grid_size);

                res_logell[index] = logell_grid[logell];
                res_logxi[index] = logxi_grid[logxi];
                res_socmax[index] = socmax;
            }
        }
    }
}

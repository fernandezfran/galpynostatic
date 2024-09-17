// This file is part of galpynostatic
//    https://github.com/fernandezfran/galpynostatic/
// Copyright (c) 2024, Francisco Fernandez, Maximiliano Gavilán and Andrés
//    Ruderman
// License: MIT
//    https://github.com/fernandezfran/galpynostatic/blob/master/LICENSE

#include <omp.h>

#include <cmath>

extern "C" void
run_profile(const bool model, const double g_pot, const int grid_size,
            const int time_steps, const int each, const int isotherm_len,
            const double temperature, const double mass, const double density,
            const double vcut, const double specific_capacity,
            const double geometry_param, const double logxi,
            const double logell, const double profile_soc,
            const double *spl_ai, const double *spl_bi, const double *spl_ci,
            const double *spl_di, const double *soceq, double *res_soc,
            double *res_pot, double *res_r_norm, double *res_cons)
{
    const double faraday = 96484.5561;
    const double gas_constant = 8.314472;
    const double t_hour = 3600.0;
    const double rfaraday = gas_constant * temperature / faraday;

    double c_rate = t_hour * (geometry_param - 1) / pow(pow(10, logxi), 2);

    double particle_size =
        2.0 * sqrt((pow(10, logell) * 2.0 * t_hour) / c_rate);

    double surface_area =
        2.0 * geometry_param * mass / (density * particle_size);

    double ccd = -c_rate * specific_capacity * mass / (1000.0 * surface_area);

    double maximum_capacity = specific_capacity * density * 3.6 / faraday;

    double time_step = -specific_capacity * mass * 3.6 / (ccd * surface_area) /
                       static_cast<double>(time_steps - 1);
    double space_step =
        0.5 * particle_size / static_cast<double>(grid_size - 1);

    double intercepts[grid_size], coefs[grid_size], gamma[grid_size],
        previous_soc[grid_size], actual_soc[grid_size], position[grid_size];

    for (int i = 0; i < grid_size; i++) {
        intercepts[i] = coefs[i] = gamma[i] = previous_soc[i] = actual_soc[i] =
            0.0;
        position[i] = i * space_step;
    }

    double alpha = time_step / (2.0 * space_step * space_step);
    double beta = (geometry_param - 1) * time_step / (4.0 * space_step);

    double alpha_0 = 1.0 + (2.0 * alpha);
    double gamma0 = 1.0 - (2.0 * alpha);

    coefs[1] = 2.0 * alpha / alpha_0;
    for (int i = 2; i < grid_size; i++) {
        coefs[i] =
            (alpha + (beta / position[i - 1])) /
            (alpha_0 - (alpha - (beta / position[i - 1])) * coefs[i - 1]);
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
            actual_soc[i] = (soceq[0] == 0.0) ? 1e-4 : soceq[0];
        }
    }

    double soc;
    double pot_i = vcut + 1.0;

    int steps = 0;
    int res_index = 0;
    int profile_index = 0;

    while (pot_i > vcut) {
        double pot_eq = 0.0;
        if (model) {
            pot_eq = rfaraday * (g_pot * (0.5 - actual_soc[grid_size - 1]) +
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
                if ((actual_soc[grid_size - 1] >= soceq[i]) &&
                    (actual_soc[grid_size - 1] < soceq[i + 1])) {
                    ai = spl_ai[i];
                    bi = spl_bi[i];
                    ci = spl_ci[i];
                    di = spl_di[i];
                    socd = soceq[i];
                    break;
                }
                else {
                    ai = spl_ai[isotherm_len - 1];
                    bi = spl_bi[isotherm_len - 1];
                    ci = spl_ci[isotherm_len - 1];
                    di = spl_di[isotherm_len - 1];
                    socd = soceq[isotherm_len - 1];
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

        soc = 0.0;
        for (int i = 0; i < grid_size; i++) {
            soc += actual_soc[i];
        }
        soc /= static_cast<double>(grid_size);

        if (steps % (time_steps / each) == 0) {
            if (res_index == 0) {
                res_index++;
            }
            else {
                res_soc[res_index] = soc;
                res_pot[res_index] = pot_i;
                res_index++;
            }
        }

        if ((soc > profile_soc - 1.0e-4) && (soc < profile_soc + 1.0e-4)) {
            if (profile_index == 0) {
                for (int i = 0; i < grid_size; i++) {
                    res_r_norm[i] = position[i] / (0.5 * particle_size);
                    res_cons[i] = actual_soc[i];
                }
                profile_index++;
            }
        }

        for (int i = 0; i < grid_size; i++) {
            previous_soc[i] = actual_soc[i];
            intercepts[i] = gamma[i] = actual_soc[i] = 0.0;
        }

        gamma[0] = gamma0 * previous_soc[0] + 2.0 * alpha * previous_soc[1];
        gamma[grid_size - 1] = gamma0 * previous_soc[grid_size - 1] +
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
            intercepts[i] = (gamma[i - 1] + sub[i - 1] * intercepts[i - 1]) /
                            (alpha_0 - sub[i - 1] * coefs[i - 1]);
        }

        actual_soc[grid_size - 1] =
            (gamma[grid_size - 1] + 2.0 * alpha * intercepts[grid_size - 1]) /
            (alpha_0 - 2.0 * alpha * coefs[grid_size - 1]);
        for (int i = 2; i < grid_size + 1; i++) {
            actual_soc[grid_size - i] = (coefs[grid_size - (i - 1)] *
                                         actual_soc[grid_size - (i - 1)]) +
                                        intercepts[grid_size - (i - 1)];
        }
        steps++;
    }

    res_soc[res_index + 1] = soc;
    res_pot[res_index + 1] = pot_i;
}

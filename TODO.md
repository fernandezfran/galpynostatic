# TODO

0. [ ] Add Jupyter Notebook tutorial for `simulation.py` module to documentation.
1. [ ] Review documentation for `simulation.py`.
2. [ ] Check warning in `test_plot.py` tests.
3. [ ] In `utils.py` replace `l` with `ell` and simplify the last function to be a one-line return.
4. [ ] In `test_plot.py`, mock the slowest test, or give as args initial parameters closer to the minimum.
5. [ ] In `simulation.py` classes, move the args that relate to the specifics of the simulation and not to the system from the instantiation to the `run()` method.
6. [ ] In `plot.py` create `MapPlotter` and `ProfilePlotter` superclasses and use them to inherit in a `RegressorPlotter`, `SimulationMapPlotter` and `SimulationProfilePlotter` for use in respective modules.
7. [ ] In `base.py`, in addition to `MapSpline`, add a similar class to the `Profile` that can be used in other modules.
8. [ ] In `preprocessing.py` add the `GetChargingCapacities` class.

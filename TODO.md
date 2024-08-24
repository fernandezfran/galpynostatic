# TODO

- [ ] Add Jupyter Notebook tutorial for `simulation.py` module to documentation.
- [ ] Review documentation for `simulation.py`.
- [ ] Fix warnings in `test_plot.py` and `test_make_prediction.py` tests.
- [ ] In `utils.py` replace `l` with `ell` and simplify the last function to be a one-line return.
- [ ] In `test_plot.py`, mock the slowest test, or give as args initial parameters closer to the minimum.
- [ ] In `simulation.py` classes, move the args that relate to the specifics of the simulation and not to the system from the instantiation to the `run()` method.
- [ ] In `plot.py` create `MapPlotter` and `ProfilePlotter` superclasses and use them to inherit in a `RegressorPlotter`, `SimulationMapPlotter` and `SimulationProfilePlotter` for use in respective modules.
- [ ] In `base.py`, in addition to `MapSpline`, add a similar class to the `Profile` that can be used in other modules.
- [ ] In `preprocessing.py` add the `GetChargingCapacities` class.

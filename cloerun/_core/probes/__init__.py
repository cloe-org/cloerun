"""The probe extension point: one module per supported likelihood class.

Available: `gcspectro_pls`, `gcspectro_bao`, `gcspectro_pls_bao` and
`photo_cls` (WL, GCph, GGL, 2x2pt and 3x2pt).

A new probe module must expose `obs_labels(cfg)`, `build_likelihood(cfg)`,
`vector_parameter_names(likelihood)`, `loglike(likelihood, parameters)`,
`tracked_packages(cfg)`, and be registered in `cloerun._core.loader.PROBES`.

It may also expose `parameter_names(likelihood)`, the parameter names the
likelihood reads, which enables the `{i}` templates of `cloerun._core.templates`,
and `use_noise_free_mock(likelihood, fiducial)`, called when the config sets
`data.noise_free_mock: true`: it returns the masked theory vector, which
`set_data_vector_masked(likelihood, vector)` sets on the likelihood rebuilt in
each sampler worker. `theory_spectra(likelihood, cfg, parameters)` and
`write_spectra(path, spectra)` enable the `sdv` mode, and
`derived_outputs(likelihood, parameters, names)` the `derived_outputs` section.
"""

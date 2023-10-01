Test:
nfactortest_2dim.m
nfactortest_3dim.m
nfactortest_5dim.m
are the tests in the simulation section of the paper.

nfactortest.m
nfactortest_random.m
nfactortest_5dim_old.m
are superseded, old version of the test to select number of factors.

findsigma.m
findsigma2.m
findsigma3.m
are functions to estimate the scale component by minimizing the sum squared errors.

Empirical Chen Zimmerman data:
tensor_CZ.m
tensor_CZ_v2.m
tensor_CZ_barplots.m
tensor_CZ_bs.m
weights_heatmap.m
are rolling window estimation, superseded.

tensor_CZ_full.m
tensor_CZ_full_v2.m
tensor_CZ_full_v3.m
tensor_CZ_full_v4.m
tensor_CZ_full_v5.m
are full sample estimation, only v5 is used in the paper.

tensor_CZ_test.m
is testing the number of factors after taking out market, to be 3, used in the paper.

tensor_CZ_mkt.m
regresses the latent factor onto market.

acronym.m
is creating the table of all acronyms, in appendix.

tensor_FT.m
tensor_FT_v2.m
are using FT data, superseded.

MC simulations:
tensor_pca_sim.m
tensor_pca_sim_v2.m
are verifying convergence rates, v2 used in the paper.

tensor_pca_sim_v3.m
is comparing model fit and model complexity in the paper.

tensor_pca_sim_v4.m
is comparing PCA to ALS in the paper.

Portfolio Sort simulation, superseded:
portsort_sim.m
portsort_sim_no.m
portsort_sim_v1.m
portsort_sim_v2.m

PCAgraph.m
plots the PCA graphical illustration used in slides.

from povalt.training.training import TrainPotential

tr_pot = TrainPotential(atoms_filename='complete.xyz', order=2, compact_clusters='T', nb_cutoff=5.0, n_sparse=20,
                        nb_covariance_type='ard_se', nb_delta=0.5, theta_uniform=1.0, nb_sparse_method='uniform',
                        l_max=8, n_max=8, atom_sigma=0.5, zeta=4, soap_cutoff=5.0, central_weight=1.0,
                        config_type_n_sparse='{fcc:50:bcc:50:hcp:50:sc:50}', soap_delta=0.1, f0=0.0,
                        soap_covariance_type='dot_product', soap_sparse_method='cur_points',
                        default_sigma='{0.002 0.2 0.2 0.2}',
                        config_type_sigma='{dimer:0.002:0.2:0.2:0.2:fcc:0.002:0.2:0.2:0.2:bcc:0.002:0.2:0.2:0.2:'
                                          'hcp:0.002:0.2:0.2:0.2:sc:0.002:0.2:0.2:0.2}',
                        energy_parameter_name='free_energy',
                        force_parameter_name='forces',
                        force_mask_parameter_name='force_mask',
                        sparse_jitter=1E-8, do_copy_at_file='F', sparse_separate_file='T', gp_file='Pt_test.xml')

tr_pot.train()

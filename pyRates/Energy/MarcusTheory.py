import numpy as np

def marcus_symmetric(delta_g, lambda_acceptor_singlet, lambda_donor_triplet):
    """
    Calculate the Delta G^# using the symmetric Marcus approach.
    """
    numerator = (delta_g + lambda_acceptor_singlet + lambda_donor_triplet)**2
    denominator = 4 * (lambda_acceptor_singlet + lambda_donor_triplet)

    return numerator / denominator

def marcus_asymmetric(delta_g, lambda_substrate_singlet, lambda_substrate_triplet, lambda_pc_singlet, lambda_pc_triplet):
    """
    Calculate the Delta G^# using the asymmetric Marcus approach.
    """

    l_r = lambda_substrate_singlet + lambda_pc_triplet
    l_p = lambda_substrate_triplet + lambda_pc_singlet

    numerator = -l_p + np.sqrt(l_r*l_p + (l_r-l_p)*delta_g)
    denominator = l_r - l_p

    return l_r * (numerator / denominator)**2
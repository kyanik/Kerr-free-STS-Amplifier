import numpy as np


def jpa_params(F_JPA, EC_J, EJ):
    phi_zpf_JPA = (2 * EC_J / EJ) ** 0.25
    ratio = phi_zpf_JPA**2 / 6
    Kerr = -EC_J * np.cos(F_JPA) / 2
    return ratio, Kerr, phi_zpf_JPA


def STS_params(EC, EL):
    phi_zps = (2 * EC / EL) ** 0.25
    ratio = phi_zps**2 / 6
    return ratio, phi_zps
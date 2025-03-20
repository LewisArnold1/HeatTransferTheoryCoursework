import numpy as np

## --- CONSTANTS AND PARAMETERS ---

# Constants
sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m²·K⁴)
g = 9.81  # Acceleration due to gravity (m/s²)

# System parameters
m_dot = 0.05  # Mass flow rate of water (kg/s)
Cp = 4186  # Specific heat capacity of water (J/kg.K)
T_in = 24  # Inlet temperature of water (°C)
T_a = 21  # Ambient temperature (°C)
A_s = 1.85  # Absorbing surface area of metal plate (m²)
tau_c = 0.92  # Transmissivity of glass
alpha_s = 0.96  # Absorptivity of metal plate
G_rec = np.mean(np.array([602, 810, 970, 1059, 1080, 1001, 864]))  # Solar radiation received (W/m²) (example value from table)
error_threshold = 0.000001  # Convergence error threshold
incline = np.radians(10) # Incline of surface
L_pg = 0.035  # Distance between metal plate and glass (m)
V_wind = 2  # Mean wind velocity (m/s)
L_tube = 1.8

# Emissivities
epsilon_s = 0.95  # Emissivity of metal plate
epsilon_c = 0.89  # Emissivity of glass cover

# Air properties from 0°C to 100°C (from table)
air_properties = {
    -20:   {"k": 0.02211, "nu": 1.169e-5, "Pr": 0.7408},
    -10:   {"k": 0.02288, "nu": 1.252e-5, "Pr": 0.7387},
    0:   {"k": 0.02364, "nu": 1.338e-5, "Pr": 0.7362},
    5:   {"k": 0.02401, "nu": 1.382e-5, "Pr": 0.7350},
    10:  {"k": 0.02439, "nu": 1.426e-5, "Pr": 0.7336},
    15:  {"k": 0.02476, "nu": 1.470e-5, "Pr": 0.7323},
    20:  {"k": 0.02514, "nu": 1.516e-5, "Pr": 0.7309},
    25:  {"k": 0.02551, "nu": 1.562e-5, "Pr": 0.7296},
    30:  {"k": 0.02588, "nu": 1.608e-5, "Pr": 0.7282},
    35:  {"k": 0.02625, "nu": 1.655e-5, "Pr": 0.7268},
    40:  {"k": 0.02662, "nu": 1.702e-5, "Pr": 0.7255},
    45:  {"k": 0.02699, "nu": 1.750e-5, "Pr": 0.7241},
    50:  {"k": 0.02735, "nu": 1.798e-5, "Pr": 0.7228},
    60:  {"k": 0.02808, "nu": 1.896e-5, "Pr": 0.7202},
    70:  {"k": 0.02881, "nu": 1.995e-5, "Pr": 0.7177},
    80:  {"k": 0.02953, "nu": 2.097e-5, "Pr": 0.7154},
    90:  {"k": 0.03024, "nu": 2.201e-5, "Pr": 0.7132},
    100: {"k": 0.03095, "nu": 2.306e-5, "Pr": 0.7111}, 
    120: {"k": 0.03235, "nu": 2.522e-5, "Pr": 0.7073},
    140: {"k": 0.03374, "nu": 2.745e-5, "Pr": 0.7041},
    160: {"k": 0.03511, "nu": 2.975e-5, "Pr": 0.7014},
    180: {"k": 0.03646, "nu": 3.212e-5, "Pr": 0.6992},
    200: {"k": 0.03779, "nu": 3.455e-5, "Pr": 0.6974},
}

## --- UTILITY FUNCTIONS ---

# --- TEMPERATURE ---

def mean_film_temperature(T1, T2):
    """ Computes mean film temperature given two temperatures. """
    return (T1 + T2) / 2

def to_kelvin(T_celsius):
    """ Converts Celsius to Kelvin. """
    return T_celsius + 273.15

def to_degrees(T_kelvin):
    """ Converts Celsius to Kelvin. """
    return T_kelvin - 273.15


# --- LINEAR INTERPOLATION ---

def find_temperature_bounds(T_mf):
    """ Finds the nearest lower (T1) and upper (T2) temperatures for interpolation. """
    temperatures = sorted(air_properties.keys())  # Get sorted temperature list
    if T_mf < temperatures[0] or T_mf > temperatures[-1]:  
        raise ValueError(f"T_mf = {T_mf} is out of the table range ({temperatures[0]} to {temperatures[-1]}).")

    for i in range(len(temperatures) - 1):
        if temperatures[i] <= T_mf <= temperatures[i + 1]:
            return temperatures[i], temperatures[i + 1]
    
    # This line should never be reached if the input is within range
    raise RuntimeError("Unexpected error in find_temperature_bounds. Check input values.")

def get_properties(T1, T2):
    """ Retrieves thermal conductivity (k), kinematic viscosity (nu), and Prandtl number (Pr) for T1 and T2. """
    prop1 = air_properties[T1]
    prop2 = air_properties[T2]
    return prop1["k"], prop2["k"], prop1["nu"], prop2["nu"], prop1["Pr"], prop2["Pr"]

def linear_interpolation(T_mf):
    """ Finds k, nu, Pr by interpolating between the nearest temperatures. """
    T1, T2 = find_temperature_bounds(T_mf)
    k1, k2, nu1, nu2, Pr1, Pr2 = get_properties(T1, T2)

    k = k1 + (k2 - k1) * ((T_mf - T1) / (T2 - T1))
    nu = nu1 + (nu2 - nu1) * ((T_mf - T1) / (T2 - T1))
    Pr = Pr1 + (Pr2 - Pr1) * ((T_mf - T1) / (T2 - T1))

    return k, nu, Pr


## --- HEAT TRANSFER COEFFICIENT FUNCTIONS ---

# --- CONVECTIVE HEAT TRANSFER COEFFICIENT (METAL TO COVER) ---

def convective_h_sc(T_s, T_c):
    """ Computes the convective heat transfer coefficient (metal to cover). """
    T_mf = mean_film_temperature(T_s, T_c)
    print(f"T_mf: {T_mf}") 
    k, nu, Pr = linear_interpolation(T_mf)
    print(f"k: {k}, nu: {nu}, Pr: {Pr}")
    beta = 1 / to_kelvin(T_mf)
    print(f"beta: {beta}")
    Gr = grashof_number(beta, T_s, T_c, nu)
    print(f"Gr: {Gr}")
    Ra = rayleigh_number(Gr, Pr)
    print(f"Ra: {Ra}")
    Nusselt = hollands_correlation(Ra)
    print(f"Nu: {Nusselt}")
    return Nusselt * k / L_pg

def grashof_number(beta, T_s, T_c, nu):
    """ Computes Grashof number using temperature difference, length, and kinematic viscosity. """
    return (g * beta * (T_s - T_c) * L_pg**3) / (nu**2)

def rayleigh_number(Gr, Pr):
    """ Computes Rayleigh number as the product of Grashof and Prandtl numbers. """
    return Gr * Pr

def hollands_correlation(Ra):
    """ Computes Nusselt number using Holland's correlation. """
    return 1 + 1.44 * (1 - (1708 * (np.sin(1.8 * incline)) ** 1.6) / (Ra * np.cos(incline))) * (1 - 1708 / (Ra * np.cos(incline))) + ((((Ra * np.cos(incline)) / 5830) ** (1/3)) - 1)


# --- RADIATIVE HEAT TRANSFER COEFFICIENT (METAL TO COVER) ---

def radiative_h_sc(T_s, T_c):
    """ Computes the radiative heat transfer coefficient (metal to cover). """
    T_s_K = to_kelvin(T_s)
    T_c_K = to_kelvin(T_c)
    return (sigma * (T_s_K**2 + T_c_K**2) * (T_s_K + T_c_K)) / ((1/epsilon_s) + (1/epsilon_c) - 1)


# --- CONVECTIVE HEAT TRANSFER COEFFICIENT (COVER TO AMBIENT) ---

def convective_h_ca(T_c, T_a):
    """ Computes the convective heat transfer coefficient (cover to ambient). """
    T_mf = mean_film_temperature(T_c, T_a) 
    print(f"T_mf: {T_mf}")
    k, nu, Pr = linear_interpolation(T_mf)
    print(f"k: {k}, nu: {nu}, Pr: {Pr}")
    Re = reynolds_number(nu)
    print(f"Re: {Re}")
    Nusselt = nusselt_number_flat_plate(Re, Pr)
    print(f"Nu: {Nusselt}")
    return (k * Nusselt) / L_tube

def reynolds_number(nu):
    """ Computes Reynolds number given wind velocity, characteristic length, and kinematic viscosity. """
    return (V_wind * L_tube) / nu

def nusselt_number_flat_plate(Re, Pr):
    """ Computes Nusselt number using empirical correlation for laminar flow over a flat plate. """
    return 0.664 * (Re**0.5) * (Pr**(1/3))


# --- RADIATIVE HEAT TRANSFER COEFFICIENT (COVER TO AMBIENT) ---

def radiative_h_ca(T_c, T_a):
    """ Computes the radiative heat transfer coefficient (cover to ambient). """
    T_a_K = to_kelvin(T_a)
    T_c_K = to_kelvin(T_c)
    return epsilon_c * sigma * ((T_a_K ** 2) + (T_c_K ** 2)) * (T_a_K + T_c_K)


# --- OVERALL HEAT TRANSFER COEFFICIENT ---

def compute_U(T_s, T_c, T_a):
    """ Computes the overall heat transfer coefficient U using updated heat transfer coefficients. """
    h_c_sc = convective_h_sc(T_s, T_c)
    print(f"h_c_sc: {h_c_sc}")
    h_r_sc = radiative_h_sc(T_s, T_c)
    print(f"h_r_sc: {h_r_sc}")
    h_c_ca = convective_h_ca(T_c, T_a)
    print(f"h_c_ca: {h_c_ca}")
    h_r_ca = radiative_h_ca(T_c, T_a)
    print(f"h_r_ca: {h_r_ca}")
    U_overall = 1 / (1 / (h_c_sc + h_r_sc) + 1 / (h_c_ca + h_r_ca))
    print(f"U_overall: {U_overall}")

    return h_c_sc, h_r_sc, h_c_ca, h_r_ca, U_overall 


## --- ITERATIVE SOLUTION ---

def outlet_temperature(U, T_s):
    """ Computes outlet temperatue. """
    T_in_K = to_kelvin(T_in)
    return T_in_K + (G_rec * A_s * tau_c * alpha_s - U * A_s * (T_s - T_a)) / (m_dot * Cp)

def new_surface_temperature(T_c, T_a, h_c_sc, h_r_sc, U):
    T_c_K = to_kelvin(T_c)  # Convert Tc to Kelvin
    T_a_K = to_kelvin(T_a)
    T_s_K = (T_c_K * (h_r_sc + h_c_sc) - T_a_K * U) / (h_r_sc + h_c_sc - U)
    T_s = to_degrees(T_s_K)
    return T_s

def solution(T_s, T_c):
    """ Method to find the final temperatures of the metal surface (Ts), glass cover (Tc), and water outlet (T_out). """
    
    h_c_sc, h_r_sc, h_c_ca, h_r_ca, U_overall  = compute_U(T_s, T_c, T_a) 

    return T_s

if __name__ == "__main__":
    # Initial guesses for Ts (metal surface temp) and Tc (glass cover temp)
    T_s = 35  # Initial guess (°C)
    T_c = (T_a+T_s)/2  # Initial guess (°C)

    print(f"Initial Metal Surface Temperature (T_s): {T_s:.6f} °C")
    print(f"Initial Glass Cover Temperature (T_c): {T_c:.6f} °C")

    solution(T_s, T_c)
   

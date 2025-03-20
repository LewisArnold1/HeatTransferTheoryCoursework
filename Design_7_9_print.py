import numpy as np

## --- CONSTANTS AND PARAMETERS ---

# Constants
sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m²·K⁴)
g = 9.81  # Acceleration due to gravity (m/s²)

# System parameters
m_dot = 0.05  # Mass flow rate of water (kg/s)
Cp = 4183  # Specific heat capacity of water (J/kg.K)
T_in = 24  # Inlet temperature of water (°C)
T_a = 21  # Ambient temperature (°C)
A_s = 1.85  # Absorbing surface area of metal plate (m²)
tau_c = 0.92  # Transmissivity of glass
alpha_s = 0.96  # Absorptivity of metal plate
G_rec = np.mean(np.array([602, 810, 970, 1059, 1080, 1001, 864]))  # Solar radiation received (W/m²) (example value from table)
error_threshold = 0.0001  # Convergence error threshold
incline = np.radians(10) # Incline of surface
L_pg = 0.035  # Distance between metal plate and glass (m)
V_wind = 2  # Mean wind velocity (m/s)
L_tube = 1.8
n = 7
d_inner = 16.4e-3
A_t = n * np.pi * d_inner * L_tube

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

water_properties = {
    5:  {"k": 0.5576, "mu": 0.001519, "Pr": 11.44, "rho": 1000},
    10: {"k": 0.5674, "mu": 0.001307, "Pr": 9.642, "rho": 999.7},
    15: {"k": 0.5769, "mu": 0.001138, "Pr": 8.253, "rho": 999.1},
    20: {"k": 0.5861, "mu": 0.001002, "Pr": 7.152, "rho": 998.2},
    25: {"k": 0.5948, "mu": 0.0008905, "Pr": 6.263, "rho": 997.1},
    30: {"k": 0.603, "mu": 0.0007977, "Pr": 5.534, "rho": 995.7},
    35: {"k": 0.6107, "mu": 0.0007196, "Pr": 4.929, "rho": 994},
    40: {"k": 0.6178, "mu": 0.0006533, "Pr": 4.422, "rho": 992.2},
    45: {"k": 0.6244, "mu": 0.0005963, "Pr": 3.994, "rho": 990.2},
    50: {"k": 0.6305, "mu": 0.0005471, "Pr": 3.628, "rho": 988},
    55: {"k": 0.636, "mu": 0.0005042, "Pr": 3.315, "rho": 985.7},
    60: {"k": 0.641, "mu": 0.0004666, "Pr": 3.045, "rho": 983.2},
    65: {"k": 0.6455, "mu": 0.0004334, "Pr": 2.81, "rho": 980.6},
    70: {"k": 0.6495, "mu": 0.000404, "Pr": 2.605, "rho": 977.8},
    75: {"k": 0.653, "mu": 0.0003779, "Pr": 2.425, "rho": 974.9},
    80: {"k": 0.6562, "mu": 0.0003545, "Pr": 2.266, "rho": 971.8},
    85: {"k": 0.6589, "mu": 0.0003335, "Pr": 2.125, "rho": 968.6},
    90: {"k": 0.6613, "mu": 0.0003145, "Pr": 2.0, "rho": 965.3},
    95: {"k": 0.6634, "mu": 0.0002974, "Pr": 1.888, "rho": 961.9},
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
    k, nu, Pr = linear_interpolation(T_mf)
    beta = 1 / to_kelvin(T_mf)
    Gr = grashof_number(beta, T_s, T_c, nu)
    Ra = rayleigh_number(Gr, Pr)
    Nusselt = hollands_correlation(Ra)
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
    k, nu, Pr = linear_interpolation(T_mf)
    Re = reynolds_number(nu)
    Nusselt = nusselt_number_flat_plate(Re, Pr)
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
    h_r_sc = radiative_h_sc(T_s, T_c)
    h_c_ca = convective_h_ca(T_c, T_a)
    h_r_ca = radiative_h_ca(T_c, T_a)
    U_overall = 1 / (1 / (h_c_sc + h_r_sc) + 1 / (h_c_ca + h_r_ca))
    
    return h_c_sc, h_r_sc, h_c_ca, h_r_ca, U_overall 


## --- ITERATIVE SOLUTION ---

def outlet_temperature(U, T_s):
    """ Computes outlet temperatue. """
    return T_in + (G_rec * A_s * tau_c * alpha_s - U * A_s * (T_s - T_a)) / (m_dot * Cp)

def new_surface_temperature(T_out, T_s):
    """Computes the updated surface temperature (T_s). """

    # Compute h_T (internal convection coefficient)
    h_T = convective_h_t(T_out, T_s)

    exponent = - (h_T * A_t) / (m_dot * Cp)  
    T_s = (T_out - T_in * np.exp(exponent)) / (1 - np.exp(exponent))
    return T_s

## --- NEW WATER PROPERTY FUNCTIONS ---

def find_temperature_bounds_water(T_mf):
    """ Finds the nearest lower (T1) and upper (T2) temperatures for water interpolation. """
    temperatures = sorted(water_properties.keys())  # Get sorted list
    if T_mf < temperatures[0] or T_mf > temperatures[-1]:
        raise ValueError(f"T_mf = {T_mf} is out of range ({temperatures[0]} to {temperatures[-1]}).")

    for i in range(len(temperatures) - 1):
        if temperatures[i] <= T_mf <= temperatures[i + 1]:
            return temperatures[i], temperatures[i + 1]
    
    raise RuntimeError("Unexpected error in find_temperature_bounds_water.")

def get_properties_water(T1, T2):
    """ Retrieves thermal conductivity (k), kinematic viscosity (nu), and Prandtl number (Pr) for T1 and T2. """
    prop1 = water_properties[T1]
    prop2 = water_properties[T2]
    return prop1["k"], prop2["k"], prop1["mu"], prop2["mu"], prop1["Pr"], prop2["Pr"], prop1["rho"], prop2["rho"]

def linear_interpolation_water(T_mf):
    """ Finds k, nu, Pr for water by interpolating between the nearest temperatures. """
    T1, T2 = find_temperature_bounds_water(T_mf)
    k1, k2, mu1, mu2, Pr1, Pr2, rho1, rho2 = get_properties_water(T1, T2)

    k = k1 + (k2 - k1) * ((T_mf - T1) / (T2 - T1))
    mu = mu1 + (mu2 - mu1) * ((T_mf - T1) / (T2 - T1))
    Pr = Pr1 + (Pr2 - Pr1) * ((T_mf - T1) / (T2 - T1))
    rho = rho1 + (rho2 - rho1) * ((T_mf - T1) / (T2 - T1))

    return k, mu, Pr, rho

def reynolds_number_internal(v_water, mu, rho):
    """ Computes the Reynolds number for internal pipe flow. """
    return (rho * v_water * d_inner) / mu

def mass_flow_to_velocity(rho):
    """ Converts mass flow rate to velocity using water density. """
    
    # Compute cross-sectional area of the tube
    A_tube = (np.pi * d_inner**2) / 4
    
    # Compute velocity using continuity equation
    v_water = m_dot / (rho * A_tube)
    
    return v_water


def convective_h_t(T_out, T_s):
    """ Computes the convective heat transfer coefficient h_T inside the tube. """
    
    # Compute mean film temperature
    T_mf = mean_film_temperature(T_in, T_out)
    
    # Get interpolated properties for water at T_mf
    k, mu, Pr, rho = linear_interpolation_water(T_mf)
    _, mu_s, _, _ = linear_interpolation_water(T_s)

    # Convert mass flow rate to velocity
    v_water = mass_flow_to_velocity(rho)
    
    # Compute Reynolds number
    Re = reynolds_number_internal(v_water, mu, rho)
    
    # Check if flow is **laminar or turbulent**
    if Re >= 10000:
        # Turbulent Flow: Use Dittus-Boelter
        if (0.6 < Pr < 160):
            Nu = 0.023 * (Re**0.8) * (Pr**0.4)
        else:
            raise ValueError(f"Prandtl number {Pr} is out of range for Dittus-Boelter.")
    else:
        # Laminar Flow: Use Entry Length Correlation
        term = ((Re * Pr) / (L_tube / d_inner)) ** (1/3) * (mu / mu_s) ** 0.14
        
        if term > 2:
            # Use entry length correlation
            Nu = 1.86 * term
        else:
            # Fully developed assumption
            Nu = 3.66
    
    # Compute convective heat transfer coefficient
    h_T = (Nu * k) / d_inner
    
    return h_T


def iterative_solution(T_c, T_s):
    """ Iterative method to find the final temperatures of the metal surface (Ts), glass cover (Tc), and water outlet (T_out). """
    converged = False
    ts_iterations = 0  # Counter for T_s updates

    print("\nIterating for T_s and T_out...")
    print(f"{'Iter':<5}{'T_s (°C)':<12}{'T_c (°C)':<12}{'T_out (°C)':<12}{'T_c Error':<12}"
          f"{'h_c_sc':<12}{'h_r_sc':<12}{'h_c_ca':<12}{'h_r_ca':<12}{'U':<12}")

    while not converged:
        ts_iterations += 1

        h_c_sc, h_r_sc, h_c_ca, h_r_ca, U_overall  = compute_U(T_s, T_c, T_a)  # Update U

        # Calculate new outlet temperature
        T_out = outlet_temperature(U_overall, T_s)

        # Calculate new surface temperature
        T_s_new = new_surface_temperature(T_out, T_s)

        # Compute error
        ts_error = abs(T_s_new - T_s)

        # Print iteration results with heat transfer coefficients
        print(f"{ts_iterations:<5}{T_s:<12.6f}{T_c:<12.6f}{T_out:<12.6f}{ts_error:<12.6f}"
              f"{h_c_sc:<12.6f}{h_r_sc:<12.6f}{h_c_ca:<12.6f}{h_r_ca:<12.6f}{U_overall:<12.6f}")


        # Check convergence
        if ts_error < error_threshold:
            T_s = T_s_new
            converged = True
        else:
            T_s = T_s_new  # Update Ts for next iteration

    return T_s

def compute_Tc(T_s, T_c):
    """ Iterates until Tc converges using updated heat transfer coefficients. """
    converged = False
    tc_iterations = 0  # Counter for T_c updates

    print("\nIterating for T_c...")
    print(f"{'Iter':<5}{'T_s (°C)':<12}{'T_c (°C)':<12}{'T_out (°C)':<12}{'T_c Error':<12}"
          f"{'h_c_sc':<12}{'h_r_sc':<12}{'h_c_ca':<12}{'h_r_ca':<12}{'U':<12}")

    while not converged:
        tc_iterations += 1
        # Solve for Ts and T_out first using iterative solution
        T_s = iterative_solution(T_c, T_s)

        h_c_sc, h_r_sc, h_c_ca, h_r_ca, U_overall = compute_U(T_s, T_c, T_a)

        # Compute new Tc from equation 3
        T_c_new = T_s - (U_overall * (T_s - T_a)) / (h_c_sc + h_r_sc)

        # Compute error
        tc_error = abs(T_c_new - T_c)

        # Compute T_out for reference
        T_out = outlet_temperature(U_overall, T_s)

        # Print iteration results with heat transfer coefficients
        print(f"{tc_iterations:<5}{T_s:<12.6f}{T_c_new:<12.6f}{T_out:<12.6f}{tc_error:<12.6f}"
              f"{h_c_sc:<12.6f}{h_r_sc:<12.6f}{h_c_ca:<12.6f}{h_r_ca:<12.6f}{U_overall:<12.6f}")

        # Check convergence
        if tc_error < error_threshold:
            T_c = T_c_new
            converged = True
        else:
            T_c = T_c_new  # Update Tc for next iteration

    return T_s, T_c

if __name__ == "__main__":
    # Initial guesses for Ts (metal surface temp) and Tc (glass cover temp)
    T_s = 35  # Initial guess (°C)
    T_c = (T_s + T_a)/2  # Initial guess (°C)

    print("\nStarting Iterative Calculation...\n")
    print(f"Initial Metal Surface Temperature (T_s): {T_s:.6f} °C")
    print(f"Initial Glass Cover Temperature (T_c): {T_c:.6f} °C")

    T_s, T_c = compute_Tc(T_s, T_c)
   
    # Print results
    print("\nFinal Results:")
    print(f"Final Metal Surface Temperature (T_s): {T_s:.6f} °C")
    print(f"Final Glass Cover Temperature (T_c): {T_c:.6f} °C")
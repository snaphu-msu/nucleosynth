import os
import numpy as np
import h5py
from .config import elements
from SkyNet import SkyNetRoot, NetworkOptions, HelmholtzEOS, NuclideLibrary, \
    REACLIBReactionLibrary, ReactionType, LeptonMode, SkyNetScreening, ReactionNetwork, \
    NetworkConvergenceCriterion, PiecewiseLinearFunction
import pandas as pd
from tqdm import tqdm
import pynucastro as pyna
from pynucastro.rates import ReacLibRate, TabularRate

import numpy as np
import matplotlib.pyplot as plt
import yt
import pynucastro as pyna
import itertools
import seaborn as sns

NSEtemp = 6e9  # K
ncomps = 686

# Q: Why is this only 20 isotopes?
default_nse_network = [
    "n", "p",
    "he3", "he4",
    "n14",
    "c12", "o16",
    "ne20", "mg24", "si28", "s32", "ar36", "ca40", "ti44",
    "cr48", "fe52", "fe54", "fe56", "ni56",
    "cr56",
]

def get_initial_composition(progenitor, at_radius):
    """Get the composition of a progenitor at a given radius.

    parameters
    ----------
    progenitor : progs.ProgModel
        The progenitor model containing the composition and profile data. Can be created using the progs module.
    at_radius : float
        The radius at which to get the composition, in cm.
    verbose : bool, optional
        If True, print additional information about the composition and missing elements. Default is False.
    """

    # Interpolate the composition to an exact radius
    radii = progenitor.profile["radius"].values
    comp = progenitor.composition.values
    index = (np.where(radii - at_radius > 0))[0][0]
    fraction = (radii[index] - at_radius) / (radii[index] - radii[index-1])
    zone_data = fraction * comp[index-1] + (1.0 - fraction) * comp[index]

    # For every isotope in the SkyNet network, find the progenitor abundances
    initY = np.zeros(len(skynetA))
    network = progenitor.composition.columns
    for i in range(len(skynetA)):
        iso = 'neutrons' if skynetZ[i] == 0 else elements.elements[skynetZ[i]] + str(skynetA[i])
        initY[i] = max(zone_data[network.get_loc(iso)], 0.0) if iso in network else 0.0
            
    # Return the initial composition normalized by the total mass fractions in the zone
    return initY / (np.sum(initY) * np.array(skynetA))


def do_nse_nucleosynthesis(model_path, network=None):
    """Perform NSE nucleosynthesis on a set of tracers using PyNucAstro. Automatically compiles tracers.

    parameters
    ----------
    model_path : str
        The path to the STIR simulation's folder.
    network : list, optional
        The list of nuclei to include in the NSE network. If None, the default 21-isotope network will be used.
    """

    # If no network is provided, use the default 21-isotope network
    if network is None:
        network = default_nse_network

    # Load in the relevant STIR data
    last_checkpoint = model_path + "/output/" + sorted([f for f in os.listdir(model_path + "/output") if "chk" in f])[-1]
    stir_data = yt.load(last_checkpoint).all_data()
    dens = np.array(stir_data['dens'])
    temp = np.array(stir_data['temp'])
    ye = np.array(stir_data['ye  '])
    mass = np.cumsum(stir_data['cell_volume'] * dens) / 2.e33

    # Run pynucastro's NSE network to get composition of each cell
    nse = pyna.NSENetwork(inert_nuclei = network)
    comps = pd.DataFrame()
    for tracer in range(np.size(dens)):
        comp, sol = nse.get_comp_nse(dens[tracer], temp[tracer], ye[tracer], use_coulomb_corr=True, return_sol=True)
        comps = pd.concat([comps, pd.DataFrame([comp.data])], ignore_index=True)

    # Add an enclosed mass column
    comps["mass"] = mass

    return comps


def do_nucleosynthesis(model_path, stir_model, progenitor, domain_radius, tracers, output_path, verbose = 0):
    """Perform nucleosynthesis on a set of tracers using SkyNet or PyNucAstro.

    parameters
    ----------
    model_path : str
        The path to the STIR simulation's folder.
    stir_model : str
        The name of the STIR model, which is the filename (without extension) of the .dat file.
    progenitor : progs.ProgModel
        The progenitor model containing the composition and profile data. Can be created using the progs module.
    tracers : pandas.DataFrame
        Tracers created using STIR data, containing the time, temperature, density, and electron fraction for each mass element tracer. These should be created using the flashbang module's get_tracers function.
    output_path : str
        The path where the SkyNet output files will be saved.
    verbose : int, optional
        If 0, very little information is printed. If 1, some info is printed. If 2, skynet stdoutput is also printed.
    """

    num_tracers = len(tracers["mass"].values)

    # Find the end point of nucleosynthesis
    _, rshock = np.loadtxt(model_path + "/" + stir_model + ".dat", unpack=True, usecols=(0, 11))
    if verbose > 0: print("Shock Radius:", rshock)

    # If the shock radius is not far enough, the star hasn't exploded
    # TODO: Get r_shock_target as ~90% of the max radius from a stir checkpoint file
    if rshock[-1] < domain_radius * 0.9:
        print("Star failed to explode, no nucleosynthesis will be done.")
        return None
    
    # Otherwise, find the index of the shock radius closest to the target
    else:
        i = np.where(rshock <= domain_radius * 0.9)[0][0]

    # Pull the final radii of each tracer
    radii = tracers["r"].values[-1]
    if verbose > 0: print("Radii:", radii[0], "-->", radii[-1])

    # Calculate the volume of each shell
    volume = np.zeros(num_tracers)
    volume[0] = (4.0 * np.pi / 3.0) * (((radii[1] + radii[0]) / 2.0) ** 3 - radii[0] ** 3)
    volume[-1] = (4.0 * np.pi / 3.0) * (radii[-1] ** 3 - ((radii[-1] + radii[-2]) / 2.0) ** 3)
    for i in range(1, num_tracers - 1):
        volume[i] = (4.0 * np.pi / 3.0) * (
            ((radii[i + 1] + radii[i]) / 2.0) ** 3
            - ((radii[i] + radii[i - 1]) / 2.0) ** 3
        )

    if verbose > 0: print("Total Volume:", sum(volume))

    # Reports on which elements are missing from the progenitor composition
    if verbose > 0:
        missing_elements = []
        for i in range(len(skynetA)):
            iso = 'neutrons' if skynetZ[i] == 0 else elements.elements[skynetZ[i]] + str(skynetA[i])
            if iso not in progenitor.composition.columns:
                missing_elements.append(iso)
        if len(missing_elements) > 0:
            print("Elements missing from progenitor composition:", missing_elements)

    final_composition = np.zeros((ncomps + 1) * num_tracers).reshape(num_tracers, (ncomps + 1))

    for mass_element in tqdm(range(num_tracers)):

        pID = int(tracers["mass"].values[mass_element])

        # Creates a 2D array where each row is data from a time step, ordered from last to first step
        trajectory_data = np.asarray([
            tracers["time"].values,
            tracers["temp"].values[:, mass_element] / 1e9,
            tracers["dens"].values[:, mass_element],
            tracers["ye  "].values[:, mass_element]
        ]).T

        # Adds additional time steps in which the material cools down and diffuses 
        cooldown_steps = 20
        cooldown_timestep = 1
        cooldown_start_time = trajectory_data[-1][0]
        cooldown_start_temp = trajectory_data[-1][1]
        cooldown_start_dens = trajectory_data[-1][2]
        times = np.concatenate((cooldown_start_time + cooldown_timestep * np.arange(0, cooldown_steps), [1000.0, 2000.0]))

        for i in range(1, len(times)):
            if times[i] >= 1000:
                temp, dens = 1e-5, 1e-10
            else:
                t = (1 + times[i] - cooldown_start_time)
                temp = cooldown_start_temp * t ** -1
                dens = cooldown_start_dens * t ** -3
            new_element = np.array([times[i], temp, dens, trajectory_data[-1][3]])
            trajectory_data = np.concatenate((trajectory_data, [new_element]))

        os.makedirs(output_path, exist_ok = True)
        filebase = output_path + "/nucleo_" + stir_model + str(pID)

        # Runs SkyNet with NSE if temperatures are above 1e9, densities are not too high, and Ye is not too low
        if trajectory_data[0, 1] > NSEtemp / 1e9 and trajectory_data[0, 2] < 1e13 and trajectory_data[0, 3] >= 0.001:
            run_skynet(True, True, cooldown_start_time, trajectory_data, outfile = filebase, verbose = verbose > 1)

        # Otherwise, get the intial composition from the progenitor and run without NSE
        else:
            starting_radius = tracers["r"].values[-1, mass_element]
            initY = get_initial_composition(progenitor, starting_radius)
            run_skynet(True, True, cooldown_start_time, trajectory_data, outfile = filebase, 
                       do_NSE = False, init_composition = initY, verbose = verbose > 1)

        h5file = h5py.File(filebase + ".h5", "r")
        As = h5file["A"]
        final_massfracs = h5file["Y"][-1] * As[:]

        final_composition[mass_element, 0] = float(pID)
        final_composition[mass_element, 1:] = final_massfracs[:]

        h5file.close()

    # Create list of isotope names for DataFrame columns
    iso_names = []
    for mass_element in range(ncomps):
        if skynetZ[mass_element] == 0:
            iso_names.append("neut")
        else:
            iso_names.append(elements.elements[skynetZ[mass_element]] + str(skynetA[mass_element]))

    columns = ["mass"] + iso_names
    df = pd.DataFrame(columns=columns)
    df["mass"] = tracers["mass"].values
    df.iloc[:, 1:] = final_composition[:, 1:ncomps+1]

    return df

# TODO: Work in progress
def run_pynucastro(tracers):
    """Perform nucleosynthesis on a set of tracers using PyNucAstro."""

    rl = pyna.ReacLibLibrary()

    all_nuclei = ["p", "he4",
                "c12", "n13",
                "o16", "ne20", "na23",
                "mg24", "al27", "si28",
                "p31", "s32", "cl35",
                "ar36", "k39", "ca40",
                "sc43", "ti44", "v47",
                "cr48", "mn51",
                "fe52","co55","ni56"]

    lib = rl.linking_nuclei(all_nuclei)
    nse = pyna.NSENetwork(libraries=lib, use_unreliable_spins=False)


def run_skynet(do_inv, do_screen, end_time, trajectory_data, outfile, do_NSE=True, init_composition=None, verbose=False):

    with open(SkyNetRoot + "/examples/code_tests/X-ray_burst/sunet") as f:
        nuclides = [l.strip() for l in f.readlines()]

    nuclib = NuclideLibrary.CreateFromWinv(SkyNetRoot + "/examples/code_tests/winvne_v2.0.dat", nuclides)

    opts = NetworkOptions()
    opts.ConvergenceCriterion = NetworkConvergenceCriterion.Mass
    opts.MassDeviationThreshold = 1.0E-10
    opts.IsSelfHeating = False
    opts.EnableScreening = do_screen
    opts.DisableStdoutOutput = not(verbose)
    opts.MaxDt = 0.2

    reactionLibraries = [
        REACLIBReactionLibrary(SkyNetRoot + "/examples/code_tests/reaclib", ReactionType.Strong, do_inv, LeptonMode.TreatAllAsDecayExceptLabelEC, 
                               "Strong reactions", nuclib, opts, True, True), 
        REACLIBReactionLibrary(SkyNetRoot + "/examples/code_tests/reaclib", ReactionType.Weak, False, LeptonMode.TreatAllAsDecayExceptLabelEC,
                               "Weak reactions", nuclib, opts, True, True), 
        REACLIBReactionLibrary(SkyNetRoot + "/examples/code_tests/nfis", ReactionType.Strong, False, LeptonMode.TreatAllAsDecayExceptLabelEC,
                               "Symmetric neutron induced fission with 0 neutrons emitted", nuclib, opts, True, True), 
        REACLIBReactionLibrary(SkyNetRoot + "/examples/code_tests/sfis", ReactionType.Strong, False, LeptonMode.TreatAllAsDecayExceptLabelEC,
                               "Spontaneous fission", nuclib, opts, True, True)
    ]

    screen = SkyNetScreening(nuclib)
    helm = HelmholtzEOS(SkyNetRoot + "/data/helm_table.dat")
    net = ReactionNetwork(nuclib, reactionLibraries, helm, screen, opts)

    density_vs_time = PiecewiseLinearFunction(trajectory_data[:,0], trajectory_data[:,2], True)
    temperature_vs_time = PiecewiseLinearFunction(trajectory_data[:,0], trajectory_data[:,1], True)
    start_time = trajectory_data[0,0] + 1.0e-20

    if do_NSE:
        Ye0 = trajectory_data[0, 3]
        output = net.EvolveFromNSE(start_time, end_time, temperature_vs_time, density_vs_time, Ye0, outfile)
    else:
        output = net.Evolve(init_composition, start_time, end_time, temperature_vs_time, density_vs_time, outfile)

    return output


skynetA = (1, 1, 2, 3, 3, 4, 6, 6, 7, 8, 9, 7, 9, 8, 10, 11, 9, 10, 11, 12, 13,
   14, 12, 13, 14, 15, 13, 14, 15, 16, 17, 18, 17, 18, 19, 17, 18, 19,
   20, 21, 22, 23, 20, 21, 22, 23, 20, 21, 22, 23, 24, 25, 26, 22, 23,
   24, 25, 26, 27, 22, 23, 24, 25, 26, 27, 28, 29, 30, 26, 27, 28, 29,
   30, 31, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 31, 32, 33, 34, 35,
   36, 37, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 35, 36, 37, 38, 39,
   40, 41, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 40, 41, 42, 43, 44,
   45, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 43, 44, 45, 46,
   47, 48, 49, 50, 51, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
   46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 46, 47, 48, 49, 50, 51, 52,
   53, 54, 55, 56, 57, 58, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 50,
   51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 56, 57, 58,
   59, 60, 61, 62, 63, 64, 65, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
   65, 66, 67, 68, 69, 70, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
   71, 72, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
   74, 75, 76, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
   65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
   82, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 67, 68,
   69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
   86, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
   72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
   89, 90, 91, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
   91, 92, 93, 94, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
   89, 90, 91, 92, 93, 94, 95, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
   92, 93, 94, 95, 96, 97, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
   91, 92, 93, 94, 95, 96, 97, 98, 86, 87, 88, 89, 90, 91, 92, 93, 94,
   95, 96, 97, 98, 99, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
   98, 99, 100, 101, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
   101, 102, 103, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
   101, 102, 103, 104, 105, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
   103, 104, 105, 106, 107, 108, 109, 92, 93, 94, 95, 96, 97, 98, 99,
   100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 98, 99, 100,
   101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
   114, 115, 116, 117, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
   106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
   119, 120, 121, 122, 124, 104, 105, 106, 107, 108, 109, 110, 111,
   112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
   125, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
   116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
   129, 130, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
   119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
   108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
   121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
   134, 135, 136)

skynetZ =  (0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7,
   7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11,
   11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14,
   14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16,
   16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18,
   18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20,
   20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22,
   22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23,
   24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25,
   25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
   26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28,
   28, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29,
   29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
   30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 32,
   32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33,
   33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34,
   34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35,
   35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 36,
   36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37,
   37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38,
   38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 39, 39,
   39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 40,
   40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
   40, 40, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
   41, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
   42, 42, 42, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
   44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 45,
   45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 46,
   46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 47, 47,
   47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48,
   48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48,
   49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49,
   49, 49, 49, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
   50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51, 51, 51,
   51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51,
   51, 51, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52,
   52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53,
   53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53,
   53, 53, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
   54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54)
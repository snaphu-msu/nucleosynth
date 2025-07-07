import os
import numpy as np
import h5py
from .config import elements
from SkyNet import SkyNetRoot, NetworkOptions, HelmholtzEOS, NuclideLibrary, \
    REACLIBReactionLibrary, ReactionType, LeptonMode, SkyNetScreening, ReactionNetwork, \
    NetworkConvergenceCriterion, PiecewiseLinearFunction

NSEtemp = 6e9  # K
ncomps = 686

# TODO: Check with Sean - Should these values stay hardcoded or should we get the values from the progenitor/stir data?
r_start = 15e6
r_shock_target = 1.0e9
r_end = 1.1e9

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
    # TODO: Verify with Sean that it's okay to set comps less than 0 to 0.
    initY = np.zeros(len(skynetA))
    network = progenitor.composition.columns
    for i in range(len(skynetA)):
        iso = 'neutrons' if skynetZ[i] == 0 else elements.elements[skynetZ[i]] + str(skynetA[i])
        initY[i] = max(zone_data[network.get_loc(iso)], 0.0) if iso in network else 0.0
            
    # Return the initial composition normalized by the total mass fractions in the zone
    return initY / (np.sum(initY) * np.array(skynetA))

def do_nucleosynthesis(model_path, stir_model, progenitor, tracers, output_path, verbose = False):
    """Perform nucleosynthesis on a set of tracers using SkyNet.

    parameters
    ----------
    model_path : str
        The path to the STIR simulation's output folder.
    stir_model : str
        The name of the STIR model, which is the filename (without extension) of the .dat file.
    progenitor : progs.ProgModel
        The progenitor model containing the composition and profile data. Can be created using the progs module.
    tracers : pandas.DataFrame
        Tracers created using STIR data, containing the time, temperature, density, and electron fraction for each mass element tracer. These should be created using the flashbang module's get_tracers function.
    output_path : str
        The path where the output files will be saved.
    verbose : bool, optional
        If True, print additional information about the nucleosynthesis process. Default is False.
    """

    num_tracers = len(tracers["mass"].values)

    # Find the end point of nucleosynthesis
    time, rshock = np.loadtxt(model_path + "/" + stir_model + ".dat", unpack=True, usecols=(0, 11))
    if verbose: print("Shock Radius:", rshock)

    # If the shock radius is not far enough, the star hasn't exploded
    if rshock[-1] < r_shock_target:
        print("Star failed to explode, no nucleosynthesis will be done.")
        return
    
    # Otherwise, find the index of the shock radius closes to the target
    else:
        i = np.where(rshock <= r_shock_target)[0][0]

    # Reports on which elements are missing from the progenitor composition
    if verbose:
        missing_elements = []
        for i in range(len(skynetA)):
            iso = 'neutrons' if skynetZ[i] == 0 else elements.elements[skynetZ[i]] + str(skynetA[i])
            if iso not in progenitor.composition.columns:
                missing_elements.append(iso)
        if len(missing_elements) > 0:
            print("Elements missing from progenitor composition:", missing_elements)

    # TODO: Why is this in log space and then log10 of the radii?
    radii = np.logspace(np.log10(r_start), np.log10(r_end), num_tracers)
    if verbose: 
        print("Lowest Radius:", radii[0])
        print("Highest Radius:", radii[-1])

    # Calculate the volume of each shell
    # TODO: Could be simplified
    volume = np.zeros(num_tracers)
    volume[0] = (4.0 * np.pi / 3.0) * (((radii[1] + radii[0]) / 2.0) ** 3 - radii[0] ** 3)
    volume[-1] = (4.0 * np.pi / 3.0) * (radii[-1] ** 3 - ((radii[-1] + radii[-2]) / 2.0) ** 3)
    for i in range(1, num_tracers - 1):
        volume[i] = (4.0 * np.pi / 3.0) * (
            ((radii[i + 1] + radii[i]) / 2.0) ** 3
            - ((radii[i] + radii[i - 1]) / 2.0) ** 3
        )

    if verbose: 
        print("Total Volume:", sum(volume))

    final_composition = np.zeros((ncomps + 1) * num_tracers).reshape(num_tracers, (ncomps + 1))
    for j in range(num_tracers):

        print(f"Performing Nucleosynthesis: {j}/{num_tracers}")

        pID = int(tracers["chk"].values[j])

        # Creates a 2D array where each row is data from a time step, ordered from last to first step
        trajectory_data = np.flip(np.asarray([
            tracers["time"].values,
            tracers["temp"].values[:, j] / 1e9,
            tracers["dens"].values[:, j],
            tracers["ye  "].values[:, j]
        ]).T, 0)

        # TODO: Evan was setting the final time here before setting boundary conditions
        # But won't skynet stop at this time then, instead of allowing things to settle like below?
        final_time = trajectory_data[-1][0]

        # Sets boundary conditions in time as petering off to a stable nuclear state
        tstep = trajectory_data[-1][0] - trajectory_data[-2][0]
        times = np.concatenate((trajectory_data[-1][0] + tstep * np.arange(0, 10), [1000.0, 2000.0]))
        temps = np.concatenate((np.linspace(trajectory_data[-1][1], 1e-5, 10), [1e-5, 1e-5]))
        dens = np.concatenate((np.linspace(trajectory_data[-1][2], 1e-10, 10), [1e-10, 1e-10]))
        for i in range(1, len(times)):
            new_element = np.array([times[i], temps[i], dens[i], trajectory_data[-1][3]])
            trajectory_data = np.concatenate((trajectory_data, [new_element]))

        os.makedirs(output_path, exist_ok = True)
        filebase = output_path + "/nucleo_" + stir_model + str(pID)

        # Runs SkyNet with NSE if temperatures are above 1e9 and densities are not too high
        # TODO: Check with Sean if the added rho check is normal
        if trajectory_data[0, 1] > NSEtemp / 1e9 and trajectory_data[0, 2] < 1e13:
            run_skynet(True, True, final_time, trajectory_data, outfile = filebase)

        # Otherwise, get the intial composition from the progenitor and run without NSE
        else:
            starting_radius = tracers["r"].values[-1, j] # TODO: Should this actually be the first radius and not the last of that mass element?
            initY = get_initial_composition(progenitor, starting_radius)
            run_skynet(True, True, final_time, trajectory_data, outfile = filebase, do_NSE = False, init_composition = initY)

        h5file = h5py.File(filebase + ".h5", "r")
        As = h5file["A"]
        final_massfracs = h5file["Y"][-1] * As[:]

        final_composition[j, 0] = float(pID)
        final_composition[j, 1:] = final_massfracs[:]

        h5file.close()

    np.save(output_path + "/final_" + stir_model + "_comps.npy", final_composition)

    totalmasses = {}

    for j in range(ncomps):
        if skynetZ[j] == 0:
            iso = "neut"
        else:
            iso = elements.elements[skynetZ[j] - 1] + str(skynetA[j])

        totalmasses[iso] = sum(volume[:] * tracers["dens"].values[0, :] * final_composition[:, 1 + j])

    np.save(output_path + "/totalmasses_" + stir_model + ".npy", totalmasses)
    return totalmasses

def run_skynet(do_inv, do_screen, end_time, trajectory_data, outfile, do_NSE=True, init_composition=None):

    with open(SkyNetRoot + "/examples/code_tests/X-ray_burst/sunet") as f:
        nuclides = [l.strip() for l in f.readlines()]

    nuclib = NuclideLibrary.CreateFromWinv(SkyNetRoot + "/examples/code_tests/winvne_v2.0.dat", nuclides)

    opts = NetworkOptions()
    opts.ConvergenceCriterion = NetworkConvergenceCriterion.Mass
    opts.MassDeviationThreshold = 1.0E-10
    opts.IsSelfHeating = False
    opts.EnableScreening = do_screen
    opts.DisableStdoutOutput = True
    opts.MaxDt = 0.001

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
    Ye0 = trajectory_data[0,3]
    start_time = trajectory_data[0,0] + 1.0e-20

    if do_NSE:
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
import os
clear = lambda: os.system('cls')
clear() # for vscode only

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scstats
from scipy.optimize import minimize

# Global variable
N              = 100000 # quantity of synthetic data to be generated
synthetic_data = []    # global variable that stores the synthetic data

# Auxiliary function to treat the given data - trying to transform to float case by case
# The case where the given data is not convertible then it's assigned as ncd (non-convertible data)
def isConvertibleToFloat(value):
    try:
        float(str(value).replace(',', '.').strip())
        return True
    except ValueError:
        return False

# Auxiliary function to calculate the MSE of the Poisson distribution
def MSEPoisson(lambda_param, data):
    poisson_dist = scstats.poisson.pmf(np.arange(len(data)), lambda_param)
    poisson_dist = poisson_dist / np.sum(poisson_dist)
    return np.mean((data - poisson_dist)**2) # MSE

def MSERayleigh(sigma_param, data):
    max_value = int(max(data))
    ray_dist = scstats.rayleigh.pdf(np.arange(0, max_value + 1), scale=sigma_param)
    ray_dist = ray_dist / np.sum(ray_dist) * np.sum(data)
    
    if len(ray_dist) > len(data): 
        ray_dist = ray_dist[:len(data)]
    elif len(ray_dist) < len(data):
        ray_dist = np.pad(ray_dist, (0, len(data) - len(ray_dist)), 'constant')
    
    return np.mean((data - ray_dist)**2) # MSE


# case 1 (probability density functions approach)
# Import all the three .csv files:
def main():
    train_data_path = [
        './data1.csv',
        './data2.csv',
        './data3.csv'
    ]

    np.set_printoptions(precision=3, suppress=True) # suppressing orders of precision

    # Interpreting data
    # example of data structure: 2021-10-24 00:33:00,19,20850235,F,PLAQUETAS SANGUE,"316,00"
    # overall data structure: [ timestamp, age, register, sex, data_type, quantity ]
    
    # creating a dictionary containing the data pattern
    csv_columns = [
        'timestamp', # neglected
        'age',       # 0
        'register',  # 1
        'sex',       # 2
        'data_type', # 3
        'quantity'   # 4
    ]
    data_type_filter = 'PLAQUETAS SANGUE' # Platelets in blood

    data = []              # the overall existing & treated data
    data_type_labels = []  # the overall raw data (non-treated)
    data_original_rows = 0 # quantity of rows before every filter step
    data_filtered_rows = 0 # quantity of rows after filtering steps
    rm_rows_nan        = 0 # removed rows with NaN

    # Loading the csv files
    for tdp in train_data_path:
        # Very important to consider header=None to not consider the first row as being the column information
        # very common for .csv files
        df = pd.read_csv(tdp, header=None)
        
        # Treating the current file:
        print(f"File: {tdp}")

        # Step 1: Let's consider only the 'data_type' equals to some pattern, e.g., 'PLAQUETAS SANGUE'
        data_type_labels = df.iloc[:, 1:csv_columns.__len__()]
        data_original_rows = data_type_labels.__len__()

        # Step 1.1: Assigning all the column information by the csv_columns pattern variable
        data_type_labels.columns = csv_columns[1:csv_columns.__len__()]

        # Step 2: Replace any empty spaces in 'data_type' and 'quantity' w/ NaN (4th and 5th columns of the original data frame, respectively)
        data_type_labels.iloc[:, [0,3,4]] = data_type_labels.iloc[:, [0,3,4]].replace(['', ' '], pd.NA)
        
        # Step 3: Drop all NaN quantities found in the data frame (age, data_type, quantity)
        data_type_labels = data_type_labels.dropna(subset=[
            data_type_labels.columns[0],
            data_type_labels.columns[3],
            data_type_labels.columns[4]
        ])
        rm_rows_nan = data_original_rows - data_type_labels.__len__()

        print(f"Total (original): {data_original_rows}")
        print(f"Total (nan-filter): {data_type_labels.__len__().__str__()}")
        print(f"Removed NaN rows: {rm_rows_nan.__str__()}")

        # Step 4: Create a lambda function that assess if there's any non convertible data
        ncd = data_type_labels.iloc[:,4].apply(
            lambda x: False if pd.isna(x) else not isConvertibleToFloat(x)
        ) # non convertible data
        print(f"Non-convertible data in 'quantity': {data_type_labels[ncd].__len__()}") # showing non convertible data quantity

        data_type_labels = data_type_labels[~ncd] # removing non-convertible data
        print(f"Total (ncd filtered): {data_type_labels.__len__()}")
        
        # Step 5: Transform quantities to the right format (field quantity)
        # Any non-float data must be treated before converting the string to float that can lead to errors
        # The secondary solution is commented but this also lead to a NaN conversion of data, 
        # demands another NaN row exclusion, repeating the step 3
        data_type_labels.iloc[:, 4] = data_type_labels.iloc[:, 4].apply(
            lambda x: float(str(x).replace(',', '.'))# if str(x).strip() else np.nan
        )

        # Step 6: Removing any negative number that doesn't make any sense for these types of data
        # Only to ensure that there will not exist any negative number
        nn = data_type_labels.iloc[:,4].apply(
            lambda x: True if x < 0 else False
        ) # find out which are the negative number (nn) rows
        print(f"Negative data in 'quantity': {data_type_labels[nn].__len__()}")

        data_type_labels = data_type_labels[~nn]
        print(f"Total (nn+ncd+nan filtered): {data_type_labels.__len__()}\n")

        # Step 7: Concatening all the valid data
        data.append(data_type_labels)
        data_type_labels = [] # clear the variable
    
    # Step 8: Concatenating all the data
    data = pd.concat(data, ignore_index=True)
    data_original_rows = data.__len__()

    # Step 9: Filtering the data_type field
    data = data[data.iloc[:,3] == data_type_filter]
    data_filtered_rows = data.__len__()

    print(f"Total (concatenated data): {data_original_rows.__str__()}")
    print(f"Total (keyword+nn+ncd+nan filter): {data_filtered_rows.__str__()}")
    print("Dataset usage: {0:.2f}%".format(data_filtered_rows/data_original_rows*100))

    # Step 10: Grouping the data 'PLAQUETAS SANGUE' by age classes
    class_div = 5 # [years] age class division
    age_bins = list(range(0, data['age'].max() + class_div, class_div))
    age_labels = [f"{i}-{i+class_div}" for i in age_bins[:-1]] # age_bins must be subtracted from 1 - to avoid creating an additional class 105-110
    num_classes = len(age_labels)

    data['age_class'] = pd.cut(data['age'], bins=age_bins, labels=age_labels, right=False)
    gdata = data.groupby('age_class', observed=True) # grouped data

    # Step 10.1: Cleaning outliers (important to create a more reliable distribution)
    k = 1.4 # defining outlier threshold (mu + 2*std_dev) - all that is out of this range is gonna be neglected
    outliers = lambda x: x[(x >= x.mean() - k * x.std()) & (x <= x.mean() + k * x.std())]
    outliers_mps = lambda x: x[(x <= x.mean() + k * x.std())] # mean plus dev
    gdata = gdata.apply(
        lambda group: outliers_mps(group['quantity'])
    )
    gdata = gdata.reset_index().groupby('age_class', observed=True)

    print(f"Total age classes (grouped data): {num_classes} classes\n")

    print(f"age_class    #    (LLN - Law of Large Numbers - threshold: >= 10^3)")
    for age_class, count in gdata.size().items():
        LLNC = "LLN-compliant" if count >= 10**3 else "LLN-non-compliant (weak approx.)" # LLN compliance variable
        print(f"{age_class}    {count} - {LLNC}")

    # Step 11: Creating the Poisson distribution (optimized)
    print("\nCalculating the optimized Poisson & Rayleigh distributions\n")
    rsigma = []           # first sigma guess (std dev)
    optsigma = []         # stores all optimized sigmas (std devs)
    rminresult = []       # variable that will receive all minimized info from MMSE
    rayleigh_dist = []
    optrayleigh_dist = []

    plambda = []          # first guess (mean of the data by age classes)
    optlambda = []        # stores all optimized lambdas (approx the mean but optimized)
    pminresult = []       # variable that will receive all the minimize info
    poisson_dist = []
    opt_poisson_dist = [] # variable to store the optimized poisson distributions

    for i, age_class in enumerate(age_labels):
        if age_class in gdata.groups:
            # Poisson
            plambda.append(np.mean(gdata.get_group(age_class)['quantity']))
            pminresult.append(minimize(
                lambda lambda_param: MSEPoisson(lambda_param, gdata.get_group(age_class)['quantity']),
                plambda[i],
                method='Nelder-Mead' # optimizing method
            )) # MMSE method
            optlambda.append(pminresult[i].x[0])
            opt_poisson_dist.append(scstats.poisson.pmf(
                #np.arange(len(gdata.get_group(age_class)['quantity'])),
                np.arange(0, int(max(gdata.get_group(age_class)['quantity']))),
                optlambda[i]
            ))
            opt_poisson_dist[i] = opt_poisson_dist[i] / np.sum(opt_poisson_dist[i])
            poisson_dist.append(scstats.poisson.pmf(
                np.arange(0, int(max(gdata.get_group(age_class)['quantity']))),
                plambda[i]
            ))
            poisson_dist[i] = poisson_dist[i] / np.sum(poisson_dist[i])
            print(f"(Poisson) lambda {i}: {plambda[i]} -> {optlambda[i]} optimized (eval. under {len(gdata.get_group(age_class)['quantity'])} occurrences)")
            
    for i, age_class in enumerate(age_labels):
        if age_class in gdata.groups:
            # Rayleigh
            rsigma.append(1.0)
            rminresult.append(minimize(
                lambda sigma_param: MSERayleigh(sigma_param, gdata.get_group(age_class)['quantity']),
                rsigma[i],
                method='Nelder-Mead' # optimizing method (simplex)
            )) # MMSE method
            optsigma.append(rminresult[i].x[0])
            optrayleigh_dist.append(scstats.rayleigh.pdf(
                np.arange(0, int(max(gdata.get_group(age_class)['quantity'])) + 1),
                scale=optsigma[i]
            ))
            #optrayleigh_dist[i] = optrayleigh_dist[i] / np.sum(optrayleigh_dist[i]) * np.sum(gdata.get_group(age_class)['quantity'])
            print(f"(Rayleigh) sigma {i}: {rsigma[i]} -> {optsigma[i]} optimized (eval. under {len(gdata.get_group(age_class)['quantity'])} occurrences)")
    
    # Step 12: Generating the synthetic data frame
    #print(max(data['age']))
    synthetic_ages = np.random.randint(0, 105, N)
    synthetic_age_classes = pd.cut(synthetic_ages, bins=age_bins, labels=age_labels, right=False)
    
    synthetic_platelets = []

    # crossing the information between age_class and Rayleigh distribution which one must be used
    sigma_mapping = dict(zip(age_labels, optsigma))
    for age_class in synthetic_age_classes:
        if age_class in sigma_mapping:
            sigma = sigma_mapping[age_class]
            synth_platelet = scstats.rayleigh.rvs(scale=sigma)
            synthetic_platelets.append(synth_platelet)
        else:
            raise ValueError(f'The given age_class {age_class} does not correspond to any of the gdata groups')
    
    # Create the synthetic data frame
    synthetic_data = pd.DataFrame({
        'age': synthetic_ages,
        'age_class': synthetic_age_classes,
        'quantity': synthetic_platelets
    })

    synthetic_gdata = synthetic_data.groupby('age_class', observed=True)

    print(f"\n{synthetic_data}")

    # Step 12.1: Calculate the Wasserstein distance
    print("\nCalculating Wasserstein distance: \n")
    for i, age_class in enumerate(age_labels):
        if age_class in gdata.groups:
            distance_wass = scstats.wasserstein_distance(
                gdata.get_group(age_class)['quantity'],
                synthetic_gdata.get_group(age_class)['quantity']
            )
            print(f"[{age_class}]: {distance_wass}")

    # Step 13 & 14: Plotting the histograms & comparing with the Rayleigh distributions
    fig, axs = plt.subplots(4, 6, figsize=(20, 18))  # 5x4 grid of subplots
    axs = axs.flatten()

    for i, age_class in enumerate(age_labels):
        # Check if the age_class is in the gdata.groups (to avoid errors)
        if age_class in gdata.groups:
            class_data = gdata.get_group(age_class)['quantity']
            synth_data = synthetic_gdata.get_group(age_class)['quantity']
            axs[i].hist(class_data, bins=80, alpha=0.4, density=True)
            axs[i].hist(synth_data, bins=80, alpha=0.4, density=True)
            axs[i].plot(optrayleigh_dist[i], 'b-', label=f'Rayleigh dist opt. (\sigma={optsigma[i]:.2f})')
            #axs[i].plot(opt_poisson_dist[i], 'g-', label='Poisson dist. opt.')
            #axs[i].plot(poisson_dist[i], 'k--', label='Poisson dist.')
            axs[i].set_title(f'Age Class: {age_class}', fontsize=8)
            axs[i].set_xlabel('Platelets (in Blood) Quantity (mil/mm³)', fontsize=8)
            axs[i].set_ylabel('Occurrence', fontsize=8) # Occurrence
            axs[i].tick_params(axis='both', which='major', labelsize=7)
            axs[i].grid(False)
        else:
            axs[i].axis('off')  # Turn off the subplot if there's no data

    # Disable any remaining subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    #plt.suptitle("Histograms of Platelets in Blood by Age Classes", fontsize=12)
    plt.suptitle('')
    plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to fit the suptitle
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.rcParams.update({'font.size': 8})

    plt.figure(figsize=(10, 6))
    for i, age_class in enumerate(age_labels):
        if age_class in gdata.groups:
            plt.hist(gdata.get_group(age_class)['quantity'], bins=80, alpha=0.4, density=False, label=f'Age Class: {age_class}')

    plt.title('Platelets in Blood divided by Age Classes')
    plt.xlabel('Platelets (in Blood) Quantity (mil/mm³)')
    plt.ylabel('Occurrences')
    plt.legend(title='Age Classes')
    plt.grid(False)

    plt.figure(figsize=(10, 6))
    for i, age_class in enumerate(age_labels):
        if age_class in gdata.groups:
            plt.plot(optrayleigh_dist[i], label=f'Age Class: {age_class} (sigma={optsigma[i]:.2f})')
            
    plt.title('Rayleigh Distributions divided by Age Classes')
    plt.xlabel('Platelets (in Blood) Quantity (mil/mm³)')
    plt.ylabel('Density')
    plt.legend(title='Age Classes')
    plt.grid(False)

    plt.figure(figsize=(10, 6))
    for i, age_class in enumerate(age_labels):
        if (age_class in gdata.groups) & (len(gdata.get_group(age_class)) > 10**3):
            plt.plot(optrayleigh_dist[i], label=f'Age Class: {age_class} (sigma={optsigma[i]:.2f})')
            
    plt.title('Rayleigh Distributions divided by Age Classes - LLN-Compliant only')
    plt.xlabel('Platelets (in Blood) Quantity (mil/mm³)')
    plt.ylabel('Density')
    plt.legend(title='Age Classes')
    plt.grid(False)
    plt.show()

if __name__ == '__main__':
    main()
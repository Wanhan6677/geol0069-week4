# Geol0069-Week4：Unsupervised learning

Unsupervised learning helps us find patterns in data without using predefined labels. In this notebook, we focus on a practical workflow for classifying sea ice and leads in Earth observation data.

The tasks in this notebook will be mainly two:
1. Discrimination of Sea ice and lead based on image classification based on Sentinel-2 optical data.
2. Discrimination of Sea ice and lead based on altimetry data classification based on Sentinel-3 altimetry data.

### Prerequisites

The following software needs to be installed before running the code.



```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```python
pip install rasterio
```

    Requirement already satisfied: rasterio in /usr/local/lib/python3.12/dist-packages (1.5.0)
    Requirement already satisfied: affine in /usr/local/lib/python3.12/dist-packages (from rasterio) (2.4.0)
    Requirement already satisfied: attrs in /usr/local/lib/python3.12/dist-packages (from rasterio) (25.4.0)
    Requirement already satisfied: certifi in /usr/local/lib/python3.12/dist-packages (from rasterio) (2026.1.4)
    Requirement already satisfied: click!=8.2.*,>=4.0 in /usr/local/lib/python3.12/dist-packages (from rasterio) (8.3.1)
    Requirement already satisfied: cligj>=0.5 in /usr/local/lib/python3.12/dist-packages (from rasterio) (0.7.2)
    Requirement already satisfied: numpy>=2 in /usr/local/lib/python3.12/dist-packages (from rasterio) (2.0.2)
    Requirement already satisfied: pyparsing in /usr/local/lib/python3.12/dist-packages (from rasterio) (3.3.2)



```python
pip install netCDF4
```

    Collecting netCDF4
      Downloading netcdf4-1.7.4-cp311-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (2.1 kB)
    Collecting cftime (from netCDF4)
      Downloading cftime-1.6.5-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (8.7 kB)
    Requirement already satisfied: certifi in /usr/local/lib/python3.12/dist-packages (from netCDF4) (2026.1.4)
    Requirement already satisfied: numpy>=1.21.2 in /usr/local/lib/python3.12/dist-packages (from netCDF4) (2.0.2)
    Downloading netcdf4-1.7.4-cp311-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (10.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m10.1/10.1 MB[0m [31m78.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading cftime-1.6.5-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (1.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.6/1.6 MB[0m [31m86.7 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: cftime, netCDF4
    Successfully installed cftime-1.6.5 netCDF4-1.7.4


## Introduction to Unsupervised Learning Methods {cite}`bishop2006pattern`

### Introduction to K-means Clustering

K-means partitions a dataset into k clusters by assigning each point to the nearest centroid based on feature similarity {cite}macqueen1967some. The algorithm iteratively updates assignments and centroids to minimise within-cluster variation.

### Key steps?
- Choose k (number of clusters)
- Initialize centroids
- Assign points to nearest centroid (Euclidean distance)
- Update centroids as the mean of assigned points
- Repeat until centroids change very little (converges)

### Why K-means for Clustering?

K-means clustering is particularly well-suited for applications where:

- Works well when the data structure is unknown (exploratory)
- Simple and efficient for large datasets




### Basic Code Implementation

Below, you'll find a basic implementation of the K-means clustering algorithm. This serves as a foundational understanding and a starting point for applying the algorithm to your specific data analysis tasks.



```python
# Python code for K-means clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# K-means model
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('K-means')
plt.show()

```


    
![png](README_files/README_6_0.png)
    



```python
import os
import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)

plt.savefig("figures/gmm_s2.png", dpi=200, bbox_inches="tight")
```


    <Figure size 640x480 with 0 Axes>


## Gaussian Mixture Models (GMM) {cite}`bishop2006pattern`

Unsupervised Learning: Gaussian Mixture Models (GMM)

In this project, we use Gaussian Mixture Models (GMM) for unsupervised classification of sea ice vs leads. A GMM is a probabilistic model that represents the data as a mixture of several Gaussian (normal) distributions. Each Gaussian has its own mean and variance, so the overall dataset can be modelled as a combination of simpler distributions. This makes GMM useful for clustering and density estimation (GEOL0069 – Week 4 Jupyter Book).

A GMM has three main elements:

- Number of components: the number of clusters (e.g., 2 clusters for sea ice and lead).

- Expectation–Maximization (EM): an iterative algorithm that (i) estimates the probability of each data point belonging to each cluster (E-step), and (ii) updates the model parameters to maximise the likelihood (M-step).

- Covariance type: controls the shape and flexibility of clusters (e.g., spherical / diagonal / tied / full) (GEOL0069 – Week 4 Jupyter Book).

Compared with K-means, GMM provides soft clustering (probabilities) and can model non-spherical clusters.
Below is a basic code implementation of a Gaussian Mixture Model.-


```python
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# GMM model
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
y_gmm = gmm.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap='viridis')
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('Gaussian Mixture Model')
plt.show()

```


    
![png](README_files/README_9_0.png)
    


On randomly selected data, K-means and GMM give broadly similar results. However, in both cases the number of groups is a user-defined setting: clusters for K-means and components for GMM.

## Image Classification
Now, let's explore the application of these unsupervised methods to image classification tasks, focusing specifically on distinguishing between sea ice and leads in Sentinel-2 imagery.

### K-Means Implementation


```python
import rasterio
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

base_path = "/content/drive/MyDrive/GEOL0069/Week 4/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for K-means, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 1))

# K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels = kmeans.labels_

# Create an empty array for the result, filled with a no-data value (e.g., -1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Place cluster labels in the locations corresponding to valid data
labels_image[valid_data_mask] = labels

# Plotting the result
plt.imshow(labels_image, cmap='viridis')
plt.title('K-means clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')
plt.show()

del kmeans, labels, band_data, band_stack, valid_data_mask, X, labels_image

```


    
![png](README_files/README_12_0.png)
    


Using the clustering output, we create a pixel-wise label map where values are 0/1 for the two clusters and -1 for missing/invalid pixels.

### GMM Implementation


```python
import rasterio
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Paths to the band images
base_path = "/content/drive/MyDrive/GEOL0069/Week 4/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for GMM, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 1))

# GMM clustering
gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
labels = gmm.predict(X)

# Create an empty array for the result, filled with a no-data value (e.g., -1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Place GMM labels in the locations corresponding to valid data
labels_image[valid_data_mask] = labels

# Plotting the result
plt.imshow(labels_image, cmap='viridis')
plt.title('GMM clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')
plt.show()

```


    
![png](README_files/README_15_0.png)
    


In this Sentinel-2 example, GMM seems to highlight lead-like features more effectively, as the crack network within the sea ice appears more continuous and prominent.

## Altimetry Classification
Now, let's explore the application of these unsupervised methods to altimetry classification tasks, focusing specifically on distinguishing between sea ice and leads in Sentinel-3 altimetry dataset.

### Read in Functions Needed
Before delving into the modeling process, it's crucial to preprocess the data to ensure compatibility with our analytical models. This involves transforming the raw data into meaningful variables, such as peakniness and stack standard deviation (SSD), etc.


```python
#
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy.ma as ma
import glob
from matplotlib.patches import Polygon
import scipy.spatial as spatial
from scipy.spatial import KDTree
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster

#=========================================================================================================
#===================================  SUBFUNCTIONS  ======================================================
#=========================================================================================================

#*args and **kwargs allow you to pass an unspecified number of arguments to a function,
#so when writing the function definition, you do not need to know how many arguments will be passed to your function
#**kwargs allows you to pass keyworded variable length of arguments to a function.
#You should use **kwargs if you want to handle named arguments in a function.
#double star allows us to pass through keyword arguments (and any number of them).
def peakiness(waves, **kwargs):

    "finds peakiness of waveforms."

    #print("Beginning peakiness")
    # Kwargs are:
    #          wf_plots. specify a number n: wf_plots=n, to show the first n waveform plots. \

    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import time

    print("Running peakiness function...")

    size=np.shape(waves)[0] #.shape property is a tuple of length .ndim containing the length of each dimensions
                            #Tuple of array dimensions.

    waves1=np.copy(waves)

    if waves1.ndim == 1: #number of array dimensions
        print('only one waveform in file')
        waves2=waves1.reshape(1,np.size(waves1)) #numpy.reshape(a, newshape, order='C'), a=array to be reshaped
        waves1=waves2

    # *args is used to send a non-keyworded variable length argument list to the function
    def by_row(waves, *args):
        "calculate peakiness for each waveform"
        maximum=np.nanmax(waves)
        if maximum > 0:

            maximum_bin=np.where(waves==maximum)
            #print(maximum_bin)
            maximum_bin=maximum_bin[0][0]
            waves_128=waves[maximum_bin-50:maximum_bin+78]

            waves=waves_128

            noise_floor=np.nanmean(waves[10:20])
            where_above_nf=np.where(waves > noise_floor)

            if np.shape(where_above_nf)[1] > 0:
                maximum=np.nanmax(waves[where_above_nf])
                total=np.sum(waves[where_above_nf])
                mean=np.nanmean(waves[where_above_nf])
                peaky=maximum/mean

            else:
                peaky = np.nan
                maximum = np.nan
                total = np.nan

        else:
            peaky = np.nan
            maximum = np.nan
            total = np.nan

        if 'maxs' in args:
            return maximum
        if 'totals' in args:
            return total
        if 'peaky' in args:
            return peaky

    peaky=np.apply_along_axis(by_row, 1, waves1, 'peaky') #numpy.apply_along_axis(func1d, axis, arr, *args, **kwargs)

    if 'wf_plots' in kwargs:
        maximums=np.apply_along_axis(by_row, 1, waves1, 'maxs')
        totals=np.apply_along_axis(by_row, 1, waves1, 'totals')

        for i in range(0,kwargs['wf_plots']):
            if i == 0:
                print("Plotting first "+str(kwargs['wf_plots'])+" waveforms")

            plt.plot(waves1[i,:])#, a, col[i],label=label[i])
            plt.axhline(maximums[i], color='green')
            plt.axvline(10, color='r')
            plt.axvline(19, color='r')
            plt.xlabel('Bin (of 256)')
            plt.ylabel('Power')
            plt.text(5,maximums[i],"maximum="+str(maximums[i]))
            plt.text(5,maximums[i]-2500,"total="+str(totals[i]))
            plt.text(5,maximums[i]-5000,"peakiness="+str(peaky[i]))
            plt.title('waveform '+str(i)+' of '+str(size)+'\n. Noise floor average taken between red lines.')
            plt.show()


    return peaky

#=========================================================================================================
#=========================================================================================================
#=========================================================================================================


def unpack_gpod(variable):

    from scipy.interpolate import interp1d

    time_1hz=SAR_data.variables['time_01'][:]
    time_20hz=SAR_data.variables['time_20_ku'][:]
    time_20hzC = SAR_data.variables['time_20_c'][:]

    out=(SAR_data.variables[variable][:]).astype(float)  # convert from integer array to float.

    #if ma.is_masked(dataset.variables[variable][:]) == True:
    #print(variable,'is masked. Removing mask and replacing masked values with nan')
    out=np.ma.filled(out, np.nan)

    if len(out)==len(time_1hz):

        print(variable,'is 1hz. Expanding to 20hz...')
        out = interp1d(time_1hz,out,fill_value="extrapolate")(time_20hz)

    if len(out)==len(time_20hzC):
        print(variable, 'is c band, expanding to 20hz ku band dimension')
        out = interp1d(time_20hzC,out,fill_value="extrapolate")(time_20hz)
    return out


#=========================================================================================================
#=========================================================================================================
#=========================================================================================================

def calculate_SSD(RIP):

    from scipy.optimize import curve_fit
    # from scipy import asarray as ar,exp
    from numpy import asarray as ar, exp

    do_plot='Off'

    def gaussian(x,a,x0,sigma):
            return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    SSD=np.zeros(np.shape(RIP)[0])*np.nan
    x=np.arange(np.shape(RIP)[1])

    for i in range(np.shape(RIP)[0]):

        y=np.copy(RIP[i])
        y[(np.isnan(y)==True)]=0

        if 'popt' in locals():
            del(popt,pcov)

        SSD_calc=0.5*(np.sum(y**2)*np.sum(y**2)/np.sum(y**4))
        #print('SSD calculated from equation',SSD)

        #n = len(x)
        mean_est = sum(x * y) / sum(y)
        sigma_est = np.sqrt(sum(y * (x - mean_est)**2) / sum(y))
        #print('est. mean',mean,'est. sigma',sigma_est)

        try:
            popt,pcov = curve_fit(gaussian, x, y, p0=[max(y), mean_est, sigma_est],maxfev=10000)
        except RuntimeError as e:
            print("Gaussian SSD curve-fit error: "+str(e))
            #plt.plot(y)
            #plt.show()

        except TypeError as t:
            print("Gaussian SSD curve-fit error: "+str(t))

        if do_plot=='ON':

            plt.plot(x,y)
            plt.plot(x,gaussian(x,*popt),'ro:',label='fit')
            plt.axvline(popt[1])
            plt.axvspan(popt[1]-popt[2], popt[1]+popt[2], alpha=0.15, color='Navy')
            plt.show()

            print('popt',popt)
            print('curve fit SSD',popt[2])

        if 'popt' in locals():
            SSD[i]=abs(popt[2])


    return SSD

```


```python
path = '/content/drive/MyDrive/GEOL0069/Week 4/'
SAR_file = 'S3A_SR_2_LAN_SI_20190307T005808_20190307T012503_20230527T225016_1614_042_131______LN3_R_NT_005.SEN3'
SAR_data = Dataset(path + SAR_file + '/enhanced_measurement.nc')

SAR_lat = unpack_gpod('lat_20_ku')
SAR_lon = unpack_gpod('lon_20_ku')
waves   = unpack_gpod('waveform_20_ku')
sig_0   = unpack_gpod('sig0_water_20_ku')
RIP     = unpack_gpod('rip_20_ku')
flag = unpack_gpod('surf_type_class_20_ku')

# Filter out bad data points using criteria (here, lat >= -99999)
find = np.where(SAR_lat >= -99999)
SAR_lat = SAR_lat[find]
SAR_lon = SAR_lon[find]
waves   = waves[find]
sig_0   = sig_0[find]
RIP     = RIP[find]

# Calculate additional features
PP = peakiness(waves)
SSD = calculate_SSD(RIP)

# Convert to numpy arrays (if not already)
sig_0_np = np.array(sig_0)
PP_np    = np.array(PP)
SSD_np   = np.array(SSD)

# Create data matrix
data = np.column_stack((sig_0_np, PP_np, SSD_np))

# Standardize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)
```

    Running peakiness function...


    /tmp/ipython-input-448542667.py:63: RuntimeWarning: Mean of empty slice
      noise_floor=np.nanmean(waves[10:20])


    Gaussian SSD curve-fit error: Optimal parameters not found: Number of calls to function has reached maxfev = 10000.
    Gaussian SSD curve-fit error: Optimal parameters not found: Number of calls to function has reached maxfev = 10000.
    Gaussian SSD curve-fit error: Optimal parameters not found: Number of calls to function has reached maxfev = 10000.


There are some NaN values in the dataset so one way to deal with this is to delete them.


```python
# Remove any rows that contain NaN values
nan_count = np.isnan(data_normalized).sum()
print(f"Number of NaN values in the array: {nan_count}")

data_cleaned = data_normalized[~np.isnan(data_normalized).any(axis=1)]

mask = ~np.isnan(data_normalized).any(axis=1)
waves_cleaned = np.array(waves)[mask]
flag_cleaned = np.array(flag)[mask]

data_cleaned = data_cleaned[(flag_cleaned==1)|(flag_cleaned==2)]
waves_cleaned = waves_cleaned[(flag_cleaned==1)|(flag_cleaned==2)]
flag_cleaned = flag_cleaned[(flag_cleaned==1)|(flag_cleaned==2)]

```

    Number of NaN values in the array: 1283


Now, let's proceed with running the GMM model as usual. Remember, you have the flexibility to substitute this with K-Means or any other preferred model.


```python
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(data_cleaned)
clusters_gmm = gmm.predict(data_cleaned)
```

We can also inspect how many data points are there in each class of your clustering prediction.


```python
unique, counts = np.unique(clusters_gmm, return_counts=True)
class_counts = dict(zip(unique, counts))
print("Cluster counts:", class_counts)
```

    Cluster counts: {np.int64(0): np.int64(8880), np.int64(1): np.int64(3315)}


We can plot the mean waveform of each class.


```python
# mean and standard deviation for all echoes
mean_ice = np.mean(waves_cleaned[clusters_gmm==0],axis=0)
std_ice = np.std(waves_cleaned[clusters_gmm==0], axis=0)

plt.plot(mean_ice, label='ice')
plt.fill_between(range(len(mean_ice)), mean_ice - std_ice, mean_ice + std_ice, alpha=0.3)


mean_lead = np.mean(waves_cleaned[clusters_gmm==1],axis=0)
std_lead = np.std(waves_cleaned[clusters_gmm==1], axis=0)

plt.plot(mean_lead, label='lead')
plt.fill_between(range(len(mean_lead)), mean_lead - std_lead, mean_lead + std_lead, alpha=0.3)

plt.title('Plot of mean and standard deviation for each class')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7d1a2f6e0a10>




    
![png](README_files/README_28_1.png)
    



```python
x = np.stack([np.arange(1,waves_cleaned.shape[1]+1)]*waves_cleaned.shape[0])
plt.plot(x,waves_cleaned)  # plot of all the echos
plt.show()
```


    
![png](README_files/README_29_0.png)
    



```python
# plot echos for the lead cluster
x = np.stack([np.arange(1,waves_cleaned[clusters_gmm==1].shape[1]+1)]*waves_cleaned[clusters_gmm==1].shape[0])
plt.plot(x,waves_cleaned[clusters_gmm==1])  # plot of all the echos
plt.title('lead cluster')
plt.show()
```


    
![png](README_files/README_30_0.png)
    



```python
# plot echos for the sea ice cluster
x = np.stack([np.arange(1,waves_cleaned[clusters_gmm==0].shape[1]+1)]*waves_cleaned[clusters_gmm==0].shape[0])
plt.plot(x,waves_cleaned[clusters_gmm==0])  # plot of all the echos
plt.title('sea ice cluster')
plt.show()
```


    
![png](README_files/README_31_0.png)
    


## Scatter Plots of Clustered Data

This code visualizes the clustering results using scatter plots, where different colors represent different clusters (`clusters_gmm`).


```python
plt.scatter(data_cleaned[:,0],data_cleaned[:,1],c=clusters_gmm)
plt.xlabel("sig_0")
plt.ylabel("PP")
plt.title('sig_0 vs PP')
plt.show()
plt.scatter(data_cleaned[:,0],data_cleaned[:,2],c=clusters_gmm)
plt.xlabel("sig_0")
plt.ylabel("SSD")
plt.title('sig_0 vs SSD')
plt.show()
plt.scatter(data_cleaned[:,1],data_cleaned[:,2],c=clusters_gmm)
plt.xlabel("PP")
plt.ylabel("SSD")
plt.title('PP vs SSD')
plt.show()
```


    
![png](README_files/README_33_0.png)
    



    
![png](README_files/README_33_1.png)
    



    
![png](README_files/README_33_2.png)
    


## Physical Waveform Alignment

To compare waveforms on a common footing we can **align** them using the known orbit geometry. This physically-based approach was developed at the [Alfred Wegener Institute (AWI)](https://gitlab.awi.de/siteo/aligned-waveform-generator).




```python
# ============================================================
# Physical Waveform Alignment (AWI-style)
# ============================================================

from scipy.interpolate import interp1d

# --- Step 1: Read alignment variables from the NetCDF file ---
print("Reading alignment variables...")
tracker_range_all = unpack_gpod('tracker_range_20_ku')
altitude_all      = unpack_gpod('alt_20_ku')
mss_all           = unpack_gpod('mean_sea_surf_sol1_20_ku')

# Sum atmospheric and geophysical range corrections (1 Hz → 20 Hz).
# We use a dedicated interpolation that filters NaN before fitting,
# because unpack_gpod can fail on masked 1 Hz arrays in newer SciPy.
correction_names = [
    'mod_dry_tropo_cor_zero_altitude_01',
    'mod_wet_tropo_cor_zero_altitude_01',
    'iono_cor_gim_01_ku',
    'ocean_tide_sol1_01',
    'load_tide_sol1_01',
    'pole_tide_01',
    'solid_earth_tide_01',
]

def safe_interp_1hz(var_name, n_20hz):
    """Read a 1 Hz variable and interpolate to 20 Hz, handling NaN."""
    vals = np.array(SAR_data.variables[var_name][:], dtype=float)
    if hasattr(vals, 'filled'):
        vals = np.ma.filled(vals, np.nan)
    time_1hz  = np.array(SAR_data.variables['time_01'][:], dtype=float)
    time_20hz = np.array(SAR_data.variables['time_20_ku'][:], dtype=float)
    valid = ~np.isnan(vals)
    if np.sum(valid) < 2:
        return np.zeros(n_20hz)
    f = interp1d(time_1hz[valid], vals[valid], kind='linear',
                 fill_value='extrapolate')
    return f(time_20hz)

n_20hz = len(tracker_range_all)
total_correction_all = np.zeros(n_20hz)
for name in correction_names:
    try:
        total_correction_all += safe_interp_1hz(name, n_20hz)
    except Exception as e:
        print(f"  Skipping {name}: {e}")

# --- Step 2: Apply the same filters used for waves_cleaned ---
# (find → NaN mask → sea-ice/lead flag filter)
_flag_filt = np.array(flag)[find][mask]
ice_lead_filt = (_flag_filt == 1) | (_flag_filt == 2)

tracker_range_c = tracker_range_all[find][mask][ice_lead_filt]
altitude_c      = altitude_all[find][mask][ice_lead_filt]
mss_c           = mss_all[find][mask][ice_lead_filt]
correction_c    = total_correction_all[find][mask][ice_lead_filt]

# --- Step 3: Compute the alignment shift for each waveform ---
alignment_m = altitude_c - tracker_range_c - correction_c - mss_c

print(f"\nRaw alignment shifts ({len(alignment_m)} waveforms):")
print(f"  Mean:  {np.nanmean(alignment_m):.3f} m")
print(f"  Std:   {np.nanstd(alignment_m):.3f} m")
print(f"  Range: [{np.nanmin(alignment_m):.3f}, {np.nanmax(alignment_m):.3f}] m")

# Clip outliers: shifts far from the bulk arise from poor MSS or
# correction data at high latitudes.  Keep the central 98 %.
finite = np.isfinite(alignment_m)
p1, p99 = np.nanpercentile(alignment_m[finite], [1, 99])
outlier = (~finite) | (alignment_m < p1) | (alignment_m > p99)
alignment_m[outlier] = 0.0

# Remove the mean offset so we only correct for *differential* alignment.
# The mean shift is dominated by average freeboard / MSS bias, not by
# waveform-to-waveform tracker variation.
nonzero = alignment_m != 0
alignment_m[nonzero] -= np.mean(alignment_m[nonzero])

print(f"After clipping + de-meaning ({np.sum(outlier)} outliers zeroed):")
print(f"  Std:   {np.nanstd(alignment_m):.3f} m")

# --- Step 4: Define alignment helper functions ---
RANGE_GATE_RES = 0.2342   # metres per range bin (Sentinel-3 Ku-band)

def fft_oversample(waveform, factor):
    """Oversample a waveform using FFT zero-padding."""
    n = len(waveform)
    n_os = n * factor
    ft = np.fft.fftshift(np.fft.fft(np.nan_to_num(waveform)))
    pad = int(np.floor(n_os / 2 - n / 2))
    ft_padded = np.concatenate([np.zeros(pad), ft, np.zeros(pad)])
    return np.real(np.fft.ifft(np.fft.fftshift(ft_padded))) * n_os / n

def align_single_waveform(waveform, shift_m, n_bins, resolution, os_factor):
    """Shift a single waveform by shift_m metres via FFT oversampling."""
    wf_os = fft_oversample(waveform, os_factor)
    x_m = np.linspace(0, n_bins * resolution, len(wf_os), endpoint=False)
    shifted = np.interp(x_m + shift_m, x_m, wf_os)
    return shifted[::os_factor]          # decimate back to original bins

# --- Step 5: Normalise and align all cleaned waveforms ---
n_bins = waves_cleaned.shape[1]
os_factor = int(np.ceil(RANGE_GATE_RES * 100))   # ~24x for ~1 cm resolution

# Per-waveform normalisation to [0, 1]
wf_max = np.nanmax(waves_cleaned, axis=1, keepdims=True).astype(float)
wf_max[wf_max == 0] = 1
waves_norm = waves_cleaned / wf_max

print(f"\nAligning {len(waves_norm)} waveforms (oversample x{os_factor}) ...")
waves_aligned = np.zeros_like(waves_norm)
for i in range(len(waves_norm)):
    shift = alignment_m[i]
    if np.isnan(shift):
        shift = 0.0
    waves_aligned[i] = align_single_waveform(
        waves_norm[i], shift, n_bins, RANGE_GATE_RES, os_factor)

# Quick summary: peak-position improvement
peaks_before = np.argmax(waves_norm, axis=1)
peaks_after  = np.argmax(waves_aligned, axis=1)
print(f"\nPeak position std:   {np.std(peaks_before):.2f}  ->  {np.std(peaks_after):.2f} bins")
```

    Reading alignment variables...
    
    Raw alignment shifts (12195 waveforms):
      Mean:  4.169 m
      Std:   3.739 m
      Range: [-27.545, 76.854] m
    After clipping + de-meaning (2934 outliers zeroed):
      Std:   1.897 m
    
    Aligning 12195 waveforms (oversample x24) ...
    
    Peak position std:   10.77  ->  8.19 bins


### Effect of alignment on individual waveforms




```python
# ============================================================
# Individual waveform comparison: before vs after alignment
# ============================================================

abs_shifts = np.abs(alignment_m)
has_shift  = np.isfinite(alignment_m) & (abs_shifts > 0)

# Pick the 3 sea-ice and 3 lead waveforms with the largest *finite* shifts
ice_valid  = np.where((clusters_gmm == 0) & has_shift)[0]
lead_valid = np.where((clusters_gmm == 1) & has_shift)[0]

ice_top3  = ice_valid[np.argsort(abs_shifts[ice_valid])[::-1][:3]]
lead_top3 = lead_valid[np.argsort(abs_shifts[lead_valid])[::-1][:3]]
show_idx  = np.concatenate([ice_top3, lead_top3])

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
x = np.arange(n_bins)

for i, idx in enumerate(show_idx):
    ax = axes[i // 3, i % 3]
    ax.plot(x, waves_norm[idx],    'b-',  alpha=0.7, linewidth=1.2, label='Original')
    ax.plot(x, waves_aligned[idx], 'r--', alpha=0.9, linewidth=1.2, label='Aligned')

    # Mark peak positions
    pk_orig  = np.argmax(waves_norm[idx])
    pk_align = np.argmax(waves_aligned[idx])
    ax.axvline(pk_orig,  color='blue', alpha=0.3, linewidth=0.8, linestyle=':')
    ax.axvline(pk_align, color='red',  alpha=0.3, linewidth=0.8, linestyle=':')

    cls_name  = 'Sea Ice' if clusters_gmm[idx] == 0 else 'Lead'
    shift_val = alignment_m[idx]
    shift_bins = shift_val / RANGE_GATE_RES
    ax.set_title(f'{cls_name}  ($\delta$ = {shift_val:.2f} m / {shift_bins:.1f} bins)',
                 fontsize=10)
    ax.set_xlabel('Range Bin')
    ax.set_ylabel('Normalised Power')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)

    # Zoom to the leading-edge region for clarity
    ax.set_xlim(max(0, pk_orig - 40), min(n_bins, pk_orig + 80))

plt.suptitle('Individual Waveform Alignment: Original (blue) vs Aligned (red)',
             fontsize=13)
plt.tight_layout()
plt.show()
```

    <>:33: SyntaxWarning: invalid escape sequence '\d'
    <>:33: SyntaxWarning: invalid escape sequence '\d'
    /tmp/ipython-input-3333233364.py:33: SyntaxWarning: invalid escape sequence '\d'
      ax.set_title(f'{cls_name}  ($\delta$ = {shift_val:.2f} m / {shift_bins:.1f} bins)',



    
![png](README_files/README_37_1.png)
    


### Aggregate alignment comparison

The figure below summarises the alignment effect across **all** waveforms:

- **Top row** — Echogram before and after alignment. A tighter bright band indicates better alignment.
- **Bottom left** — histogram of peak positions. The aligned distribution (red) should be narrower.
- **Bottom right** — mean waveform per class. After alignment the mean leading edge becomes sharper because individual waveforms are better registered.


```python
# ============================================================
# Aggregate alignment comparison
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
n_img = min(1000, len(waves_norm))

# --- Top left: original waveform image ---
ax = axes[0, 0]
im = ax.imshow(waves_norm[:n_img].T, origin='lower', aspect='auto',
               cmap='viridis', vmin=0, vmax=1)
ax.set_title('Original Waveforms')
ax.set_xlabel('Waveform Index')
ax.set_ylabel('Range Bin')
plt.colorbar(im, ax=ax, label='Normalised Power')

# --- Top right: aligned waveform image ---
ax = axes[0, 1]
im = ax.imshow(waves_aligned[:n_img].T, origin='lower', aspect='auto',
               cmap='viridis', vmin=0, vmax=1)
ax.set_title('Aligned Waveforms')
ax.set_xlabel('Waveform Index')
ax.set_ylabel('Range Bin')
plt.colorbar(im, ax=ax, label='Normalised Power')

# --- Bottom left: peak position histograms ---
ax = axes[1, 0]
ax.hist(peaks_before, bins=50, alpha=0.6, color='steelblue',
        label=f'Before ($\sigma$={np.std(peaks_before):.1f} bins)')
ax.hist(peaks_after,  bins=50, alpha=0.6, color='tomato',
        label=f'After ($\sigma$={np.std(peaks_after):.1f} bins)')
ax.set_xlabel('Peak Bin')
ax.set_ylabel('Count')
ax.set_title('Peak Position Distribution')
ax.legend()

# --- Bottom right: mean waveform by class, before vs after ---
ax = axes[1, 1]
xr = np.arange(n_bins)
for cls, name, color in [(0, 'Sea Ice', 'green'), (1, 'Lead', 'blue')]:
    cls_mask = clusters_gmm == cls
    if np.sum(cls_mask) < 5:
        continue
    ax.plot(xr, np.nanmean(waves_norm[cls_mask], axis=0),
            color=color, linestyle='-', alpha=0.5, label=f'{name} (original)')
    ax.plot(xr, np.nanmean(waves_aligned[cls_mask], axis=0),
            color=color, linestyle='--', linewidth=2, label=f'{name} (aligned)')
ax.set_xlabel('Range Bin')
ax.set_ylabel('Normalised Power')
ax.set_title('Mean Waveform by Class')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle('Alignment Effect: Aggregate Comparison', fontsize=14)
plt.tight_layout()
plt.show()
```

    <>:29: SyntaxWarning: invalid escape sequence '\s'
    <>:31: SyntaxWarning: invalid escape sequence '\s'
    <>:29: SyntaxWarning: invalid escape sequence '\s'
    <>:31: SyntaxWarning: invalid escape sequence '\s'
    /tmp/ipython-input-479174154.py:29: SyntaxWarning: invalid escape sequence '\s'
      label=f'Before ($\sigma$={np.std(peaks_before):.1f} bins)')
    /tmp/ipython-input-479174154.py:31: SyntaxWarning: invalid escape sequence '\s'
      label=f'After ($\sigma$={np.std(peaks_after):.1f} bins)')



    
![png](README_files/README_39_1.png)
    


## Compare with ESA data
In the ESA dataset, sea ice = 1 and lead = 2. Therefore, we need to subtract 1 from it so our predicted labels are comparable with the official product labels.


```python
flag_cleaned_modified = flag_cleaned - 1
```


```python
from sklearn.metrics import confusion_matrix, classification_report

true_labels = flag_cleaned_modified   # true labels from the ESA dataset
predicted_gmm = clusters_gmm          # predicted labels from GMM method

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_gmm)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Compute classification report
class_report = classification_report(true_labels, predicted_gmm)

# Print classification report
print("\nClassification Report:")
print(class_report)
```

    Confusion Matrix:
    [[8856   22]
     [  24 3293]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
             0.0       1.00      1.00      1.00      8878
             1.0       0.99      0.99      0.99      3317
    
        accuracy                           1.00     12195
       macro avg       1.00      1.00      1.00     12195
    weighted avg       1.00      1.00      1.00     12195
    



```python
!pip -q install nbconvert
```


```python
!ls

```

    drive  figures	sample_data


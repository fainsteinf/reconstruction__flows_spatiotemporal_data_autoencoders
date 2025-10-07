#Code to generate the figures of  "The reconstruction of flows from spatiotemporal data by autoencoders"

# Facundo Fainstein (1,2), Josefina Catoni (1), Coen Elemans (3) and Gabriel B. Mindlin (1,2,4)* 

# (1) Universidad de Buenos Aires, Facultad de Ciencias Exactas y Naturales, Departamento de Física, Ciudad Universitaria, 1428 Buenos Aires, Argentina.

# (2) CONICET - Universidad de Buenos Aires, Instituto de Física Interdisciplinaria y Aplicada (INFINA), Ciudad Universitaria, 1428 Buenos Aires, Argentina.

# (3) Department of Biology, University of Southern Denmark, 5230 Odense M, Denmark.

# (4) Universidad Rey Juan Carlos, Departamento de Matemática Aplicada, Madrid, Spain. 

# *Gabriel B. Mindlin (corresponding author)
# Email: gabo@df.uba.ar
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc 
#%%
#path to folder where results are saved ("chaotic_spatiotemporal_pattern")
folder_results_170 = '.../'

#path to folder where figures will be saved
save_folder = folder_results_170
#%% Code to generate Figure 1 - Integrate Lorenz63 dyn system

#RK4  
def rk4(dxdt, x, t, dt, *args, **kwargs):
    x = np.asarray(x)
    k1 = np.asarray(dxdt(x, t, *args, **kwargs))*dt
    k2 = np.asarray(dxdt(x + k1*0.5, t, *args, **kwargs))*dt
    k3 = np.asarray(dxdt(x + k2*0.5, t, *args, **kwargs))*dt
    k4 = np.asarray(dxdt(x + k3, t, *args, **kwargs))*dt
    return x + (k1 + 2*k2 + 2*k3 + k4)/6

#Vector field
def f(v, t):
    sigma, beta, rho =10, 8/3 , 28

    x,y, z = v[0],v[1],v[2]

    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x*y - beta * z 

    return [dxdt, dydt, dzdt]

dt = 0.01
time = np.arange(0, 400, dt)
x = np.zeros_like(time)
y = np.zeros_like(time)
z = np.zeros_like(time)

#initial condition
x_0, y_0, z_0 = 0.1, 0.1, 0.1
x[0] = x_0
y[0] = y_0
z[0] = z_0

#integrate
for ix, tt in enumerate(time[:-1]):
    x[ix+1], y[ix+1], z[ix+1] = rk4(f, [x[ix], y[ix], z[ix]], tt, dt)

#remove transient 
x = x[int(10/dt):]
y = y[int(10/dt):]
z = z[int(10/dt):]
time = time[int(10/dt):] - time[int(10/dt)]

#validation data set dynamics
z_lorenz = np.array([x,y,z]).T[-9000:]

#create modes - 200 pixels for visualization
X =  np.linspace(0, 1, 200)
Z = np.linspace(0, 1, 200)
X, Z = np.meshgrid(X, Z)

a = 1

Mode_1 = 1 * np.cos(np.pi * a * X) * np.sin(np.pi * Z) 
Mode_2 = 1 * np.sin(2 * np.pi * Z)

#Create frames for figure

#Select time points to generate the frames
vs = np.linspace(660,730,6)
        
#create the frames
ims = []
for k in vs:
    k = int(k)
    ims.append( (y[k]/20) * Mode_1 - (z[k]/40) * Mode_2  + np.random.normal(0, .01, (200, 200)) ) 

#%% Figure 1 - panel (a)

fsize = 6 
#arial
font = 'Arial'
rc('font',**{'family':'sans-serif','sans-serif':[font]})

plt.rcdefaults()

fig = plt.figure(figsize=(3.37,1.2))  #6.69, 3
gs = fig.add_gridspec(ncols=6, nrows=5,width_ratios=[1, 1, 1, 1, 1, 1], height_ratios=[.35,.3,.3, .3,.35]) #,width_ratios = [1,1],height_ratios=[0.7,0.7,1,1])


plt.subplots_adjust(left=0.01,
                    bottom=0.0, 
                    right=0.99,
                    hspace=0.8,
                    wspace=0.8,
                    top=1)

labelpad_modos = 12.5
#plot first mode
ax1 = plt.subplot(gs[:, 0:2], projection='3d')
surf1 = ax1.plot_surface(X, Z, Mode_1, alpha=1, rstride=1, cstride=1, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
ax1.view_init(elev=15., azim=-60)

ax1.set_xlabel(r"$x$", fontsize=fsize, labelpad=-labelpad_modos)
ax1.set_ylabel(r"$z$", fontsize=fsize, labelpad=-labelpad_modos)
ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
ax1.set_zlabel(r"$\psi_{1}$", rotation=0, fontsize=fsize, labelpad=-labelpad_modos-5)

ax1.tick_params(axis='both', which='both', labelsize=fsize, pad=-5)
ax1.set_xticks([0,1])
ax1.set_yticks([0,1])
ax1.set_zticks([])
ax1.set_yticklabels([0,1],rotation=0, va='center', ha='left')

ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.grid(False)

#plot second mode
ax2 = plt.subplot(gs[:, 2:4], projection='3d')
surf2 = ax2.plot_surface(X, Z, Mode_2, alpha=1, rstride=1, cstride=1, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)

ax2.view_init(elev=15., azim=-60)

ax2.set_xlabel(r"$x$", fontsize=fsize, labelpad=-labelpad_modos)
ax2.set_ylabel(r"$z$", fontsize=fsize, labelpad=-labelpad_modos)

ax2.set_xticks([0,1])
ax2.set_yticks([0,1])
ax2.set_zticks([])

ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax2.grid(False)
ax2.zaxis.set_rotate_label(False)  
ax2.set_zlabel(r"$\psi_{2}$", rotation=0, fontsize=fsize, labelpad=-labelpad_modos-5)
ax2.tick_params(axis='both', which='major', labelsize=fsize, pad=-5)

#plot temporal evolution
lwith = 0.25

ax3 = plt.subplot(gs[1, 4:])
ax3.plot(x[-9000:], lw=lwith, color='gray')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.spines['bottom'].set_visible(False)

ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_ylabel(r"$X$", rotation=0, labelpad=0, fontsize=fsize)

ax4 = plt.subplot(gs[2, 4:])
ax4.plot(y[-9000:], lw=lwith)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_ylabel(r"$Y$", rotation=0, labelpad=0, fontsize=fsize)

ax5 = plt.subplot(gs[3, 4:])
ax5.plot(z[-9000:], lw=lwith)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['left'].set_visible(False)
ax5.spines['bottom'].set_visible(False)
ax5.set_xticks([])
ax5.set_yticks([])
ax5.set_ylabel(r"$Z$", rotation=0, labelpad=0, fontsize=fsize)

fig.tight_layout()
plt.savefig(save_folder+"/Fig_1_part1.png", dpi=600)
#%% Fig 1 - panel (b) 

fig, ax = plt.subplots(1, 6, figsize=(3.37,0.57))  #6.69, 3

plt.subplots_adjust(wspace=0.25, hspace=0, left=0.05, right=.95)

ax = ax.ravel()
for k in range(6):
    ax6 = ax[k]
    sc = ax6.imshow(-ims[k], origin='lower', cmap=cm.coolwarm)
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.spines['left'].set_visible(False)
    ax6.spines['bottom'].set_visible(False)
    ax6.set_aspect('equal')

fig.tight_layout()
plt.savefig(save_folder+"/Fig_1_part2.png", dpi=600)
#%% Get data for Figure 2 - Load latent space of selected network

#Training epochs
epochs = np.arange(10, 610, 10)
#Frames which approximate periodic orbits
index_L =  [4274, 4338]
index_R =  [7954, 8020]
index_LR =  [8676, 8833]

#Selected training to show in Fig 2
ntrain = 123 
latent_space =np.load(folder_results_170+"/latent_space_best_epoch/"+str(ntrain)+"-ls_best.npy")
z_ls = latent_space[-9000:, :].copy()

#Load test loss for different latent space dimensions
folder_valmses =folder_results_170 + 'dimLS_test_mse_best_epoch_numpy_array/'

val_mse1 = np.load(folder_valmses+"best_val_mse1.npy")
val_mse2 = np.load(folder_valmses+"best_val_mse2.npy")
val_mse3 = np.load(folder_valmses+"best_val_mse3.npy")
val_mse4 = np.load(folder_valmses+"best_val_mse4.npy")
val_mse5 = np.load(folder_valmses+"best_val_mse5.npy")

#Get evolution of MSEs for 3-dimensional latent space
folder = folder_results_170+'results_in_numpy_arrays/'
#Train and test MSE latent space dimension 3 
train_mse, val_mse = [], []
for entrenamiento in range(170):    
    train_mse.append( np.load(folder+str(entrenamiento)+'-train_mse.npy') )
    val_mse.append( np.load(folder+str(entrenamiento)+'-val_mse.npy') )
train_mse = np.array(train_mse)
val_mse = np.array(val_mse)

train_mse_mean = np.mean(train_mse, axis=0)
train_mse_error = np.std(train_mse, axis=0) / np.sqrt(len(train_mse))

val_mse_mean = np.mean(val_mse, axis=0)
val_mse_error = np.std(val_mse, axis=0) / np.sqrt(len(val_mse))

#Load topological analysis - Linking numbers
folder_links = folder_results_170+'linking_numbers/'
linking_numbers_aes = []
for entrenamiento in range(170):
    linking_numbers_aes.append( 
        np.load(folder_links+str(entrenamiento)+'-linking_numbers.npy') )
linking_numbers_aes = np.array(linking_numbers_aes)

# Check topology for every trained network, for every epoch
acierto = []
for k in range(len(linking_numbers_aes)):
    acierto_entrenamiento = []
    for i in range(len(linking_numbers_aes[k])):
        if np.sum(linking_numbers_aes[k][i]) == 0:
            acierto_entrenamiento.append(True)
        else:
            acierto_entrenamiento.append(False)
    acierto.append(acierto_entrenamiento)
acierto = np.array(acierto) # Ntrains x Nepochs shape

correct_topology = 100 * np.mean(acierto, axis=0)
error_correct_topology = 100 * np.std(acierto, axis=0) / np.sqrt(len(acierto))
#%% Figure 2 - panel (a)

fsize = 6 
#arial
font = 'Arial'
rc('font',**{'family':'sans-serif','sans-serif':[font]})

plt.rcdefaults()

fig = plt.figure(constrained_layout=True,figsize=(3.37,1.2))  #6.69, 3
gs = fig.add_gridspec(ncols=6, nrows=5,width_ratios=[1, 1, 1, 1, 1, 1], height_ratios=[.35,.3,.3,.3, .35]) #,width_ratios = [1,1],height_ratios=[0.7,0.7,1,1])

plt.subplots_adjust(left=0.02,
                    bottom=0.0, 
                    right=0.95,
                    hspace=0.8,
                    wspace=0.8,
                    top=1)

labelpad_modos = 12.5

ax1 = plt.subplot(gs[:, 0:2], projection='3d')
ax = fig.add_subplot(1,1,1, projection='3d')
ax1.plot(z_lorenz[:,0], z_lorenz[:,1], z_lorenz[:,2], lw=.1)

col = ['b', 'darkred', 'k']
for k, index in enumerate([index_L, index_R, index_LR]):
    ax1.plot(z_lorenz[:,0][index[0]:index[1]], z_lorenz[:,1][index[0]:index[1]],
            z_lorenz[:,2][index[0]:index[1]], lw=.7, c=col[k])
    

ax1.view_init(elev=15., azim=130+180)

ax1.set_xlabel(r"$X$", fontsize=fsize, labelpad=-labelpad_modos)
ax1.set_ylabel(r"$Y$", fontsize=fsize, labelpad=-labelpad_modos)

ax1.set_xlim([-30,30])
ax1.set_ylim([-30,30])
ax1.set_zlim([0,50])


ax1.set_xticks([-20,20])
ax1.set_yticks([-20,20])
ax1.set_zticks([0, 50])

ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax1.grid(False)
ax1.zaxis.set_rotate_label(False)  # disable automatic rotation
ax1.set_zlabel(r"$Z$", rotation=0, fontsize=fsize, labelpad=-labelpad_modos-5)
ax1.tick_params(axis='both', which='both', labelsize=fsize, pad=-5)


ax2 = plt.subplot(gs[:, 4:], projection='3d')
ax2.plot(z_ls[:,0], z_ls[:,1], z_ls[:,2], lw=.1, c='#ff7f0e')

col = ['b', 'darkred', 'k']
for k, index in enumerate([index_L, index_R, index_LR]):
    ax2.plot(z_ls[:,0][index[0]:index[1]], z_ls[:,1][index[0]:index[1]],
            z_ls[:,2][index[0]:index[1]], lw=.7, c=col[k])
    
ax2.view_init(elev=6., azim=160+180)

ax2.set_xlabel(r"$z_{1}$", fontsize=fsize, labelpad=-labelpad_modos)
ax2.set_ylabel(r"$z_{2}$", fontsize=fsize, labelpad=-labelpad_modos)

ax2.set_xlim([-12,5])
ax2.set_ylim([-2,12])
ax2.set_zlim([-5,15])

ax2.set_xticks([-10,0])
ax2.set_yticks([0,10])
ax2.set_zticks([-5,15])

ax2.tick_params(axis='both', which='both', labelsize=fsize, pad=-5)

ax2.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax2.grid(False)
ax2.zaxis.set_rotate_label(False)  # disable automatic rotation
ax2.set_zlabel(r"$z_{3}$", rotation=0, fontsize=fsize, labelpad=-labelpad_modos-5)

plt.savefig(save_folder+"/Fig_2_part1.png", dpi=600)

#%% Fig2 - panels (c) and (d)

#Auxiliary functions for whisker plots
def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')
    
    
fsize = 6 
#arial
font = 'Arial'
rc('font',**{'family':'sans-serif','sans-serif':[font]})

plt.rcdefaults()

fig = plt.figure(constrained_layout=True,figsize=(3.37,1)) 
gs = fig.add_gridspec(ncols=6, nrows=1,width_ratios=[1, 1, 1, 1, 1, 1])
ax1 = plt.subplot(gs[:, 0:2])

ndims = 5
viridis = cm.get_cmap('Greys', ndims*2)
colors = viridis(np.linspace(0.2,0.5,ndims))

data = np.array([val_mse1, val_mse2, val_mse3,
                           val_mse4, val_mse5], dtype=object)
parts = ax1.violinplot(data,positions=[1,2,3,4,5],showmeans=False, 
                       showmedians=False,showextrema=False)

for pc in parts['bodies']:
    pc.set_facecolor('gray')
    pc.set_edgecolor('black')
    pc.set_alpha(.4)

quartile1, medians, quartile3 = [],[],[]
for k in range(len(data)):
    q1, m, q3 =  np.percentile(data[k], [25, 50, 75])
    quartile1.append(q1)
    quartile3.append(q3)
    medians.append(m)
    
whiskers = np.array([adjacent_values(sorted_array, q1, q3) for sorted_array, q1, q3 in zip(data, quartile1, quartile3)]) 
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)

ax1.scatter(inds, medians, marker='o', color='k', s=.2, zorder=3)
ax1.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=1)
ax1.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

ax1.set_xlim(0,6)
ax1.set_xticks([1,2,3,4,5])
ax1.set_ylim([0.8*1e-4,1e-2])
ax1.set_yticks([1e-4,1e-3,1e-2])
ax1.set_yscale('log')
ax1.yaxis.offsetText.set_fontsize(fsize)
ax1.set_xlabel("Latent Space Dimension", fontsize=fsize, labelpad=0);
ax1.set_ylabel("MSE", fontsize=fsize);
ax1.tick_params(axis='both', which='major', labelsize=fsize)

msize = 3
step = 3

viridis = cm.get_cmap('Blues', ndims*2)
colors = viridis(np.linspace(0.3,0.5,ndims))

ax2 = plt.subplot(gs[:, 2:])

ax2.errorbar(epochs[::step], train_mse_mean[9::10][::step], yerr=train_mse_error[9::10][::step], 
             marker='.', linestyle='-', label="Train", markersize=msize, color=colors[2])
ax2.errorbar(epochs[::step], val_mse_mean[9::10][::step], yerr=val_mse_error[9::10][::step], 
             marker='.', linestyle='-', label="Test", markersize=msize, color='gray')
ax2.set_yscale('log')
ax2.set_ylabel("MSE", fontsize=fsize)
ax2.set_xlabel("Epoch", fontsize=fsize)
ax2.set_xticks(np.arange(0, 610, 100))
ax2.set_ylim([10**-4,10**-1])
ax2.legend(loc=2)

ax3 = ax2.twinx()
ylabels_top = []
for k in np.arange(0, 120, 20):
    ylabels_top.append(str(k)+"%")

error = np.std(acierto, axis=0) / np.sqrt(len(acierto))
ax3.errorbar(epochs[::step], correct_topology[::step], yerr=error_correct_topology[::step], 
             marker='.', linestyle='-', color='g', ms=msize)
ax3.set_yticks(np.arange(0, 120, 20), labels=ylabels_top)
ax3.set_ylabel("Correct topology", rotation=-90, labelpad=10, fontsize=fsize)
ax3.set_ylim([-5,105])
ax2.tick_params(axis='both', which='major', labelsize=fsize)
ax3.tick_params(axis='both', which='major', labelsize=fsize)
ax2.legend(fontsize=fsize,loc=(0.6,0.4),ncol=1,frameon=False)
plt.savefig(save_folder+"/Fig_2_part2.png", dpi=600)

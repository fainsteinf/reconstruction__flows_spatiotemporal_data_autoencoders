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
import glob
from skimage import io
from matplotlib import cm
import seaborn as sns
from matplotlib import rc 
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#%% Load data for figures 3 and 4

print("getting ready for figures 3 and 4")


#path to folder of data availability for section 3
#that is path to "labial_oscillations_in_avian_phonation"
path_data = '.../labial_oscillations_in_avian_phonation/'

#path to figures_data
folder_figs_data = path_data + "/figures_data/"

#path to saving folder
save_folder = path_data

#path to autoencoder_results_100_fittings
folder_ae = path_data+'/autoencoder_results_100_fittings/'

#path to autoencoder_results_latent_space_dimension
folder_ae_dim = path_data+'/autoencoder_results_latent_space_dimension/'

#Distance graphical projection 
dg = np.loadtxt(folder_figs_data+'dg.txt')

print("distance graphical projection: loaded")

#%% Load data for figure 3 - Autoencoders latent space representation and losses

#number of trainings
N = 100

loss_ae, val_loss_ae= [], []
z1_ae_train, z2_ae_train, z1_ae_test, z2_ae_test = [], [], [], []

for i in range(N):
  folder_it = folder_ae+'it{}/'.format(i)
  loss_ae.append(np.loadtxt(folder_it+'train_mse.txt'))
  val_loss_ae.append(np.loadtxt(folder_it+'val_mse.txt'))
  z_train=np.loadtxt(folder_it+'layer_output_train.txt')
  z_test=np.loadtxt(folder_it+'layer_output_test.txt')
  z1_ae_train.append(z_train[:,0])
  z2_ae_train.append(z_train[:,1])
  z1_ae_test.append(z_test[:,0])
  z2_ae_test.append(z_test[:,1])

z1_ae_train = np.array(z1_ae_train)
z2_ae_train = np.array(z2_ae_train)
z1_ae_test = np.array(z1_ae_test)
z2_ae_test = np.array(z2_ae_test)

print("autoencoders 100 fittings results: loaded")

#%% Load latent space dimensionality results

arquis = ['LS1','LS2','LS3','LS4']
for i, arquitectura in enumerate(arquis):
    
    parte_1 = 'dimLS_'+str(i+1)+'/dimLS_'+str(i+1)
    actual_path = folder_ae_dim+parte_1
    N = 100
    
    
    loss_aux, val_loss_aux= [], []
    rec_error_aux=[]
    layer_output_aux=[]
    
    for i in range(N):
      folder_it = actual_path+'_it{}/'.format(i)
      loss=np.loadtxt(folder_it+'train_mse.txt')
      val_loss=np.loadtxt(folder_it+'val_mse.txt')
      loss_aux.append(loss)
      val_loss_aux.append(val_loss)
      opt_epoch=np.argmin(loss)
      rec_error_aux=rec_error_aux+list(np.loadtxt(folder_it+'mse_test_epoch_'+str(opt_epoch)+".txt") / 9900)
    
    #loss (MSE) and val_loss (MSE) 
    vars()['loss_{}'.format(arquitectura)] = loss_aux
    vars()['val_loss_{}'.format(arquitectura)] = val_loss_aux
    #Output error of frames in the best epoch
    vars()['test_rec_error{}'.format(arquitectura)] = rec_error_aux

print("autoencoders 100 fittings different latent space dimensions: loaded")
#%% Getting ready for figure 3

#Define iterations with self-intersections in the trajectory
self_intersections = np.array([30,41,44,59,71,92])
no_self_intersections = np.setdiff1d(np.arange(100),self_intersections)

#Average loss during training for the trainings with no self-intersections
loss_ae = np.array(loss_ae)
val_loss_ae = np.array(val_loss_ae)

mean_train_loss = np.mean(loss_ae[no_self_intersections], axis=0)
std_train_loss = np.std(loss_ae[no_self_intersections], axis=0)  / np.sqrt(len(no_self_intersections))

mean_val_loss = np.mean(val_loss_ae[no_self_intersections], axis=0)
std_val_loss = np.std(val_loss_ae[no_self_intersections], axis=0)  / np.sqrt(len(no_self_intersections))

#Average loss during training for the trainings with self-intersections
mean_train_loss_si = np.mean(loss_ae[self_intersections], axis=0)
std_train_loss_si = np.std(loss_ae[self_intersections], axis=0) / np.sqrt(len(self_intersections))
mean_val_loss_si = np.mean(val_loss_ae[self_intersections], axis=0)
std_val_loss_si = np.std(val_loss_ae[self_intersections], axis=0) / np.sqrt(len(self_intersections))

# Remove outliers mse for visualization of different latent space dimensions
n = 0
menores = []
for k in range(len(test_rec_errorLS1)):
    if test_rec_errorLS1[k]>1/9900:
        n+=1
    else:
        menores.append(k)
print("Outliers percentage: ", n/len(test_rec_errorLS1)*100)
test_rec_errorLS1 = np.array(test_rec_errorLS1)

n = 0
menores_ls2 = []
for k in range(len(test_rec_errorLS2)):
    if test_rec_errorLS2[k]>1/9900:
        n+=1
    else:
        menores_ls2.append(k)
print("Outliers percentage: ", n/len(test_rec_errorLS2)*100)
test_rec_errorLS2 = np.array(test_rec_errorLS2)

n = 0
menores_ls3 = []
for k in range(len(test_rec_errorLS3)):
    if test_rec_errorLS3[k]>1/9900:
        n+=1
    else:
        menores_ls3.append(k)
print("Outliers percentage: ", n/len(test_rec_errorLS3)*100)
test_rec_errorLS3 = np.array(test_rec_errorLS3)

n = 0
menores_ls4 = []
for k in range(len(test_rec_errorLS4)):
    if test_rec_errorLS4[k]>1/9900:
        n+=1
    else:
        menores_ls4.append(k)
        
print("Outliers percentage: ", n/len(test_rec_errorLS4)*100)
test_rec_errorLS4 = np.array(test_rec_errorLS4)


#%% Fig 3 - panels (b) and (c)
nframes = 4
viridis = cm.get_cmap('Greys', nframes*2)
colors = viridis(np.linspace(0.15,0.15,nframes))

def normalize(x):
    return x/np.mean(x)

fsize = 6 
#arial
font = 'Arial'
rc('font',**{'family':'sans-serif','sans-serif':[font]})

fig = plt.figure(constrained_layout=True,figsize=(3.37,1.8))  #6.69, 3
gs = fig.add_gridspec(ncols=6, nrows=3,width_ratios=[1, 1,1, 1, 1, 1], height_ratios=[.01, .25, .25])


ax0 = plt.subplot(gs[1:, 0:3])

ax0 = sns.violinplot(data=[test_rec_errorLS1[menores],test_rec_errorLS2[menores_ls2],
                     test_rec_errorLS3[menores_ls3],test_rec_errorLS4],
               zorder=0,cut=0,palette=colors, 
               width=0.8, linewidth=.5)

ax0.set_xlim(-.8,3.8)
ax0.set_xticklabels(labels=['1', '2', '3', '4'])
ax0.set_ylim([2e-5,10e-5])
ax0.set_yticks([1e-5,6e-5,11e-5])
ax0.ticklabel_format(axis='y', style="sci", scilimits=(0,0))
ax0.yaxis.offsetText.set_fontsize(fsize)
ax0.set_xlabel("Latent Space Dimension", fontsize=fsize, labelpad=0);
ax0.set_ylabel("MSE - Test Frames", fontsize=fsize);
ax0.tick_params(axis='both', which='major', labelsize=fsize)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)

viridis = cm.get_cmap('Blues', nframes*2)
colors = viridis(np.linspace(0.3,0.5,nframes))

ax1 = plt.subplot(gs[1, 3:])
ax1.errorbar(np.arange(0, len(mean_val_loss), 10),mean_train_loss[::10],
             yerr=std_train_loss[::10], fmt='.-', lw=2, ms=5, label="Train", color=colors[2])

ax1.errorbar(np.arange(0, len(mean_val_loss), 10),mean_val_loss[::10],
             yerr=std_val_loss[::10], fmt='.-', lw=2, ms=5, label="Test", color='gray')

ax1.legend(fontsize=fsize,loc=(0.6,0.5),ncol=1,frameon=False)
ax1.set_ylim([2e-5,10e-5])
ax1.set_yticks([2e-5,6e-5,10e-5])
ax1.set_ylabel("MSE - Movie", fontsize=fsize)
ax1.set_xlim([-5, 205])
ax1.set_xticks([0,200])
ax1.tick_params(axis='both', which='major', labelsize=fsize)
ax1.tick_params(axis='x', which='major', labelsize=fsize)

ax1.ticklabel_format(axis='y', style="sci", scilimits=(0,0))
ax1.yaxis.offsetText.set_fontsize(fsize)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_title("No self-intersections", fontsize=fsize, pad=10)


ax1 = plt.subplot(gs[2, 3:])

ax1.errorbar(np.arange(0, len(mean_val_loss_si), 10),mean_train_loss_si[::10],
              yerr=std_train_loss_si[::10],fmt='.-', lw=2, ms=5, label="Train", color='darkred',alpha=0.3)
              
ax1.errorbar(np.arange(0, len(mean_val_loss_si), 10),mean_val_loss_si[::10],
              yerr=std_val_loss_si[::10], fmt='.-',lw=2, ms=5, label="Test", color='r', alpha=0.3)

ax1.legend(fontsize=fsize,loc=(0.6,0.5),ncol=1,frameon=False)

ax1.set_ylim([2e-5,10e-5])
ax1.set_yticks([2e-5,6e-5,10e-5])
ax1.set_ylabel("MSE - Movie", fontsize=fsize)
ax1.set_xlabel("Epoch", fontsize=fsize, labelpad=0)
ax1.set_xlim([-5, 205])
ax1.set_xticks([0,200])
ax1.tick_params(axis='both', which='major', labelsize=fsize)
ax1.ticklabel_format(axis='y', style="sci", scilimits=(0,0))
ax1.yaxis.offsetText.set_fontsize(fsize)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_title("With self-intersections", fontsize=fsize, pad=10)

plt.savefig(save_folder+"/Fig3.png", dpi=600)
plt.show()
#%% Computations for figure 4

print("getting ready for figure 4")

#Singular value decomposition analysis
s1vt1 = np.loadtxt(folder_figs_data+'svd_s1vt1.txt')
s2vt2 = np.loadtxt(folder_figs_data+'svd_s2vt2.txt')

print("singular value decomposition results: loaded")

#compute latent space velocity
N = 100
d_lat_test = []
for k in range(N):
    d_lat_test.append( np.sqrt( np.diff(z1_ae_test[k])**2 + np.diff(z2_ae_test[k])**2 ) )
 
#compute velocity in singular value decomposition 2 modes projection
d_svd = np.sqrt( np.diff(s1vt1)**2 + np.diff(s2vt2)**2 )                  
 
#Compute pearson correlation for singular value decomposition              
pearson_svd = stats.pearsonr(dg[300:], d_svd[300:])[0]

pearson_ae = []
for i in range(N):
    pearson_ae.append( stats.pearsonr(dg[300:], d_lat_test[i])[0] )    
pearson_ae = np.array(pearson_ae)
#%% Plot figure 4
fsize=6
fig = plt.figure(figsize=(3.37,3))  #1.685
plt.subplots_adjust(left= 0.12 , bottom=0.08,
                    right=0.98, top=0.99,
                    wspace=0.4, hspace=0.55)

gs = fig.add_gridspec(ncols=2, nrows=5,width_ratios = [.6, .4], height_ratios = [.3, .3, .4, .1, .1]) 

ax1 = plt.subplot(gs[0, :])
ax1.plot(normalize(dg[300:]), 'k', zorder=100, label="Frame's space", lw=1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='both', which='major', labelsize=fsize)
ax1.tick_params(axis='x', which='major', labelsize=0)
ax1.set_ylim([0.5,1.55])
ax1.set_yticks([.5,1.5])
ax1.set_ylabel(r"$\frac{d_{g}}{<d_{g}>}$", fontsize=fsize+2, rotation=0, 
           labelpad=25)
ax1.yaxis.set_label_coords(-0.075,0.27)
ax1.set_xticks([0,100])

ax12 = plt.subplot(gs[1, :])
[ax12.plot(normalize(d_lat_test[j]), c='gray', alpha=.1, lw=.5) for j in no_self_intersections[:-1]]
j = no_self_intersections[-1]
ax12.plot(normalize(d_lat_test[j]), c='gray', alpha=.1,lw=.5, label="AE(2)")

#Index of best autoencoder 
indice = 17 
ax12.plot(normalize(d_lat_test[indice]),'royalblue',label='Best AE(2)', lw=1)
ax12.set_ylim([0,2.65])
ax12.set_yticks([0,2.5])
ax12.set_xticks([0,100])

ax12.set_ylabel(r"$\frac{d_{lat}}{<d_{lat}>}$", fontsize=fsize+2, rotation=0, 
           labelpad=20)
ax12.set_xlabel("Time", fontsize=fsize, labelpad=-7)
ax12.tick_params(axis='both', which='major', labelsize=fsize)
ax12.tick_params(axis='x', which='major', labelsize=fsize)
ax12.yaxis.set_label_coords(-0.075,0.27) 
ax12.spines['top'].set_visible(False)
ax12.spines['right'].set_visible(False)
ax12.legend(fontsize=fsize,loc=(0.52,.9),ncol=2,frameon=False)

#################################### PEARSON #################################
ax2 = plt.subplot(gs[2:, 0])

sns.violinplot(data=[pearson_ae[no_self_intersections]],color=".95",zorder=1, size=4,linewidth=1)
sns.stripplot(data=[pearson_ae[no_self_intersections]],zorder=1,size=2.5, color='gray', 
              alpha=.5)

ax2.scatter(-1.45,pearson_svd,s=15,color='blueviolet',marker="o",zorder=2,label='SVD two modes')
ax2.scatter([0],[pearson_ae[no_self_intersections][indice]],s=15,color='royalblue',marker="o",zorder=5,label='mean AE training')
ax2.set_xticks([-1.45,0])
ax2.set_xticklabels(['SVD(2)', 'AE(2)'], rotation=0, fontsize=fsize)
ax2.set_xlim([-2.5,1])
ax2.set_ylim([0,1.1])
ax2.set_ylabel('Pearson Correlation', fontsize=fsize, labelpad=1)
ax2.tick_params(axis='both', which='major', labelsize=fsize)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

################################# SVD #########################################


axins = inset_axes(ax2, width="100%", height="100%",
                   bbox_to_anchor=(0.15, 0.12, .3, .3),
                   bbox_transform=ax2.transAxes, loc=2, borderpad=0)
axins.tick_params(left=False, right=True, labelleft=False, labelright=True)
axins.plot(s1vt1[300:], s2vt2[300:], '.', ms=.75, color='blueviolet')

axins.set_xlabel(r"$\sigma_{1}^{2}V^{t}_{1}$", fontsize=fsize, labelpad=1)
axins.set_ylabel(r"$\sigma_{2}^{2}V^{t}_{2}$", fontsize=fsize, rotation=90, labelpad=1)
axins.set_aspect('equal')
axins.set_xticks([])
axins.set_yticks([])
axins.set_xlim([-6,5])
axins.set_ylim([-5,5])


############################# AE #########################################

axi = plt.subplot(gs[2, 1])
sc = axi.scatter(z1_ae_test[indice][:-1], z2_ae_test[indice][:-1],s=1.25, 
                 c=normalize(d_lat_test[indice]), cmap='Blues_r', 
            vmax = 2.5, vmin=0.4)
axi.set_xlabel(r"$z_{1}$", fontsize=fsize+2)
axi.set_ylabel(r"$z_{2}$", fontsize=fsize+2, rotation=0, labelpad=10)
axi.yaxis.set_label_coords(-0.15,0.375) 
axi.set_aspect('equal')
cbar = plt.colorbar(sc)
cbar.ax.set_ylabel(r'$d_{lat}$', rotation=270, fontsize=fsize, 
                   labelpad=8)
axi.set_xticks([])
axi.set_yticks([])
cbar.ax.get_yaxis().set_ticks([])
axi.set_xlim([-9,7])
axi.set_ylim([-3.5,12])

############################# AE  MODES #########################################
axi = plt.subplot(gs[3, 1])
axi.plot(z1_ae_test[indice], '.-', ms=.25, lw=1, color='royalblue')
axi.set_ylabel(r"$z_{1}$", fontsize=fsize+2, rotation=0)
axi.yaxis.set_label_coords(-0.15,0.3) 
axi.set_xlabel("Time",fontsize=fsize, labelpad=5)

axi.spines['top'].set_visible(False)
axi.spines['right'].set_visible(False)
axi.set_xticks([])
axi.set_yticks([])
axi.set_ylim([-9,7])


axi = plt.subplot(gs[4, 1])
axi.plot(z2_ae_test[indice], '.-', ms=.25, lw=1, color='royalblue')
axi.set_ylabel(r"$z_{2}$", fontsize=fsize+2, rotation=0)
axi.yaxis.set_label_coords(-0.15,0.3) 

axi.spines['top'].set_visible(False)
axi.spines['right'].set_visible(False)
axi.set_xticks([])
axi.set_yticks([])
# axi.set_ylim([-9,7])
axi.set_ylim([-3.5,12])
axi.set_xlabel("Time",fontsize=fsize, labelpad=5)

plt.savefig(save_folder+"/Fig_4.png", dpi=600)

plt.show()
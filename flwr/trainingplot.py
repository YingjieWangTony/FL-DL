import numpy as np
import matplotlib.pyplot as plt
import pathlib

STRATEGY = 'FL'
SUBSTRATEGY = 'None'
MODEL = 'CNN'
MODEL = 'DNN'
# MODEL = 'WAVENET'

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 28


FLdata = np.load('Results/losses/loss_FL_None_'+str(MODEL)+'_.npy')
DRINGdata = np.load('Results/losses/loss_DL_Ring_'+str(MODEL)+'_.npy')
DFULLdata = np.load('Results/losses/loss_DL_Full_'+str(MODEL)+'_.npy')
DMSdata = np.load('Results/losses/loss_DL_MS_'+str(MODEL)+'_.npy')

DMSforecast = np.load('Results/losses/forecast_DL_MS_'+str(MODEL)+'_.npy.npz')
FLforecast = np.load('Results/losses/forecast_FL_None_'+str(MODEL)+'_.npy.npz')
real = DMSforecast['arr_0']
DMS = DMSforecast['arr_1']
FL = FLforecast['arr_1']
print(len(FLdata),len(DRINGdata),len(DFULLdata),len(DMSdata))
x = np.arange(len(DMS))
plt.figure(figsize=(16,9))
plt.plot(x, real, label = "Real",linewidth=3)
plt.plot(x, DMS, label = "DMS",linewidth=3)
plt.plot(x, FL, label = "FL",linewidth=3)
plt.savefig('Results/losses/forecastfig'+str(MODEL)+'.png', bbox_inches='tight')

# FL=[]
# DR=[]
# DF=[]
# DM=[]


# for i in range(len(FLdata)):
#     FL.append(FLdata[i][1])
#     DR.append(DRINGdata[i][1])
#     DF.append(DFULLdata[i][1])
#     DM.append(DMSdata[i][1])








# x = np.arange(len(FLdata))
# plt.figure(figsize=(16,9))
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.plot(x, FL, label = "FedAvg",linewidth=3)
# plt.plot(x, DR, label = "DRING",linewidth=3)
# plt.plot(x, DF, label = "DFC",linewidth=3)
# plt.plot(x, DM, label = "DMS",linewidth=3)
# plt.xlabel('Training Epoches')
# plt.ylabel('Mean Squared Errors')
# plt.legend()



# plt.savefig('Results/losses/testfig'+str(MODEL)+'.png', bbox_inches='tight')
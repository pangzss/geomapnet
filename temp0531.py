import matplotlib.pyplot as plt
import numpy as np
hue_seq = np.arange(-0.5,0.6,0.1)
print(hue_seq)
bright_seq_1 = np.arange(0.1,1,0.1)
bright_seq_2 = np.arange(1,6.0,0.5)
bright_seq = np.concatenate([bright_seq_1,bright_seq_2])

hue_pos_0 = [7.45,7.25,7.2,7.4,7.35,7.19,7.06,7.11,7.12,6.95,7.42]
hue_ori_0 = [10.14,9.69,9.39,9.64,9.18,10.04,9.85,10.37,10.16,9.88,10.09]

hue_pos_4 = [5.71,5.76,5.82,5.65,5.78,5.66,5.54,5.67,5.73,5.73,5.73]
hue_ori_4 = [7.37,7.7,7.2,7.19,7.06,6.95,7.04,6.94,6.8,7.09,7.53]

hue_pos_8 = [6.37,6.38,6.4,6.37,6.4,6.39,6.48,6.29,6.43,6.47,6.41]
hue_ori_8 = [8.49,7.99,8.18,8.48,8.48,8.15,8.17,8.6,7.99,7.99,8.37]

hue_pos_16 = [7.64,7.93,7.79,7.7,7.92,7.6,8.28,8.07,8.09,7.84,7.67]
hue_ori_16 = [9.86,9.56,9.64,10.04,9.85,9.47,9.91,10.35,10.24,10.2,9.8]

bright_pos_0 = [7.16,7.06,7.33,7.3,7.51,7.55,7.54,7.87,9.22]
bright_pos_0 = bright_pos_0[::-1]
bright_pos_0 = bright_pos_0 + \
               [7.19,7.26,8.63,11.76,17.4,24.68,31.7,41.66,52.09,53.96]

bright_ori_0  = [9.86,9.6,9.66,9.55,9.64,9.73,9.9,10.4,12.43]
bright_ori_0 = bright_ori_0[::-1]
bright_ori_0 = bright_ori_0 + \
                [10.04,9.68,11.11,16.05,20.24,28.86,37.94,49.59,54.92,61.04]

bright_pos_4 = [5.69,5.5,5.41,5.48 ,5.46,5.56,5.43,6.11,6.74]
bright_pos_4 = bright_pos_4[::-1]
bright_pos_4 = bright_pos_4 + \
                [5.66,5.87,6.64,6.68,8.7,9.64,12.97,17.73,20.78,26.03]

bright_ori_4 = [6.96,7.03,7.09,7.04,7.1,7.16,7.8,7.81,9.61]
bright_ori_4 = bright_ori_4[::-1]
bright_ori_4 = bright_ori_4 + \
                [6.95,7.32,7.93,8.03,10.01,12.98,16.58,21.96,26.78,39.5]

bright_pos_8 = [6.51,6.48,6.63,6.58,7.06,6.58,6.67,6.73,7.28]
bright_pos_8 = bright_pos_8[::-1]
bright_pos_8 = bright_pos_8 + \
                [6.39,6.89,6.74,7.36,8.61,11.5,15.63,18.31,25.04,32.73]

bright_ori_8 = [8.22,8.03,8.36,9.03,9.84,9.04,8.89,8.08,9.06]
bright_ori_8 = bright_ori_8[::-1]
bright_ori_8 = bright_ori_8 + \
               [8.15,8.29,8.99,9.88,11.38,13.48,17.18,20.84,25.74,33.17]

bright_pos_16 = [7.76,7.93,8.04,8.14,8.31,8.7,9.79,8.62,9.1]
bright_pos_16 = bright_pos_16[::-1]
bright_pos_16 = bright_pos_16 + \
                [7.6,8.05,8.34,9.46,10.87,12.63,15.37,20.45,24.86,31.28]

bright_ori_16 = [9.53,9.66,9.45,9.34,9.18,9.64,10.48,10.42,11.2]
bright_ori_16 = bright_ori_16[::-1]
bright_ori_16 = bright_ori_16 + \
                [9.47,9.58,10.38,12.02,13.61,17.68,18.9,30.12,33.91,38.98]


fig,ax = plt.subplots(1,2)

ax[0].plot(hue_seq,np.array(hue_pos_0)-hue_pos_0[5],label='0 styles')
ax[0].plot(hue_seq,np.array(hue_pos_4)-hue_pos_4[5],label='4 styles')
ax[0].plot(hue_seq,np.array(hue_pos_8)-hue_pos_8[5],label='8 styles')
ax[0].plot(hue_seq,np.array(hue_pos_16)-hue_pos_16[5],label='16 styles')
ax[0].plot(0,0,'*',markersize=12)
ax[0].legend()
ax[0].set_xlabel('Hue factor')
ax[0].set_ylabel('Val error variation [m]')
ax[0].set_title('position')

ax[1].plot(hue_seq,np.array(hue_ori_0)-hue_ori_0[5],label='0 styles')
ax[1].plot(hue_seq,np.array(hue_ori_4)-hue_ori_4[5],label='4 styles')
ax[1].plot(hue_seq,np.array(hue_ori_8)-hue_ori_8[5],label='8 styles')
ax[1].plot(hue_seq,np.array(hue_ori_16)-hue_ori_16[5],label='16 styles')
ax[1].plot(0,0,'*',markersize=12)
ax[1].legend()
ax[1].set_xlabel('Hue factor')
ax[1].set_ylabel('Val error variation [deg]')
ax[1].set_title('orientation')
plt.subplots_adjust(wspace=0.5)

fig_1,ax_1 = plt.subplots(1,2)

ax_1[0].plot(bright_seq,np.array(bright_pos_0)-bright_pos_0[9],label='0 styles')
ax_1[0].plot(bright_seq,np.array(bright_pos_4)-bright_pos_4[9],label='4 styles')
ax_1[0].plot(bright_seq,np.array(bright_pos_8)-bright_pos_8[9],label='8 styles')
ax_1[0].plot(bright_seq,np.array(bright_pos_16)-bright_pos_16[9],label='16 styles')
ax_1[0].plot(1,0,'*',markersize=12)
ax_1[0].legend()
ax_1[0].set_xlabel('Brightness factor')
ax_1[0].set_ylabel('Val error variation [m]')
ax_1[0].set_title('position')

ax_1[1].plot(bright_seq,np.array(bright_ori_0)-bright_ori_0[9],label='0 styles')
ax_1[1].plot(bright_seq,np.array(bright_ori_4)-bright_ori_4[9],label='4 styles')
ax_1[1].plot(bright_seq,np.array(bright_ori_8)-bright_ori_8[9],label='8 styles')
ax_1[1].plot(bright_seq,np.array(bright_ori_16)-bright_ori_16[9],label='16 styles')
ax_1[1].plot(1,0,'*',markersize=12)
ax_1[1].legend()
ax_1[1].set_xlabel('Brightness factor')
ax_1[1].set_ylabel('Val error variation [deg]')
ax_1[1].set_title('orientation')
plt.subplots_adjust(wspace=0.5)
plt.show()

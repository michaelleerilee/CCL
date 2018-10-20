#!/usr/bin/env python

import cv2
import numpy as np

from Krige import DataField as df

obj = df.DataField(\
         datafilename='MYD08_D3.A2015304.061.2018054061429.hdf'\
         ,datafieldname='Atmospheric_Water_Vapor_Mean'\
         ,srcdirname='/home/mrilee/data/NOGGIN/MODIS-61-MYD08_D3/'\
)

print('data-mnmx: ',np.nanmin(obj.data),np.nanmax(obj.data))
if False:
    cv2.imshow('data',obj.data); cv2.waitKey(0); cv2.destroyAllWindows()

data = np.zeros(obj.data.shape,dtype=np.uint8)
data[:,:] = 255*(obj.data[:,:]/np.nanmax(obj.data))
d_lo = int(255*(7.25/np.nanmax(obj.data)))
d_hi = int(255*(7.74/np.nanmax(obj.data)))
print('selection data mnmx: ',d_lo,d_hi)

print('type:  ',type(data))
print('dtype: ',data.dtype)
print('shape: ',data.shape)

ret, thresh = cv2.threshold(data, d_lo, d_hi, cv2.THRESH_BINARY)
print ('thresh-ret: ',ret)
# ret, thresh = cv2.threshold(data, 7, 7.74, cv2.THRESH_BINARY_INV)

data0 = thresh
print('type(thresh): ',type(thresh))
print('type(thresh[0,0]): ',type(thresh[0,0]))
print('thresh.dtype: ',thresh.dtype)
print('thresh mnmx: ',np.nanmin(thresh),np.nanmax(thresh))
print('thresh uniq: ',np.unique(thresh))
print('thresh[0:12,0:12]: ',thresh[0:12,0:12])
print('thresh[167:179,0:12]: ',thresh[167:179,0:12])
if False:
    cv2.imshow('data0',data0); cv2.waitKey(0); cv2.destroyAllWindows()

ret, markers = cv2.connectedComponents(data0)
print ('markers-ret: ',ret)
print('markers: ',np.amax(markers))

data1 = markers.astype(np.float)/np.amax(markers)

if False:
    cv2.imshow('markers',data1); cv2.waitKey(0); cv2.destroyAllWindows()

print('bot box: ',markers[   0:12,0:12])
print('top box: ',markers[167:179,0:12])

bot_unique = np.unique(markers[  0,:])
bot_label  = bot_unique[1]
for i in range(1,bot_unique.size):
    markers[np.where(markers == bot_unique[i])] = bot_label

top_unique = np.unique(markers[  179,:])
top_label  = top_unique[1]
for i in range(1,top_unique.size):
    markers[np.where(markers == top_unique[i])] = top_label

print('bot box: ',markers[   0:12,0:12])
print('top box: ',markers[167:179,0:12])


dateline_check_thresh_idx_0 = \
    np.where(
        (thresh[ :  ,0] == thresh[ :  ,359]) &
        (thresh[:,359] == 255)
    )[0]
dateline_check_thresh_idx_p = \
    np.where(
        (thresh[1:  ,0] == thresh[ :-1,359]) &
        (thresh[:-1,359] == 255)
    )[0]
dateline_check_thresh_idx_m = \
    np.where(
        (thresh[ :-1,0] == thresh[1:  ,359]) &
        (thresh[1:,359] == 255)
    )[0]

# print 'dateline_check_thresh_idx_0: ',dateline_check_thresh_idx_0
id_0 = []
for i in dateline_check_thresh_idx_0:
    # print 'i: ',i
    id_0.append([markers[i,0],markers[i,359]])
# print 'id_0: ',id_0
    
id_p = []
for i in dateline_check_thresh_idx_p:
    id_p.append([markers[i+1,0],markers[i,359]])
# print('id_p: ',id_p)

id_m = []
for i in dateline_check_thresh_idx_m:
    id_m.append([markers[i,0],markers[i+1,359]])
# print('id_m: ',id_m)

id_all = id_0 + id_p + id_m
id_all_fixed = []
for i in id_all:
    if (min(i) != max(i)):
        id_all_fixed.append([min(i),max(i)])
        
id_all_uniq = []
for i in id_all_fixed:
    if (i not in id_all_uniq):
        id_all_uniq.append(i)
for i in range(len(id_all_uniq)):
    r = id_all_uniq[i]
    for j in range(i+1,len(id_all_uniq)):
        s = id_all_uniq[j]
        if r[1] == s[0]:
            id_all_uniq[j][0]=r[0]

print "id_all:      ",id_all
print "id_all_uniq: ",id_all_uniq

print '100: ',np.unique(markers)
for i in id_all_uniq:
    print 'i: ',i
    markers[np.where(markers == i[1])] = i[0]
markers_unique=np.unique(markers)
print '110: ',markers_unique

for i in range(len(markers_unique)):
    markers[np.where(markers == markers_unique[i])] = i
markers_unique_1=np.unique(markers)
print '120: ',markers_unique_1
    
cm = np.zeros((256,1,3),np.uint8)
cm[:,0,0] = 255-np.arange(256)
cm[:,0,1] = (np.arange(256)*(255-np.arange(256)))/255
cm[:,0,2] = np.arange(256)
cm[0,0,:] = 0

data2 = np.zeros(markers.shape,dtype=np.uint8)
data2[:,:] = 255*(markers.astype(np.float)/np.amax(markers))
# data2 = cv2.applyColorMap(data2,cv2.COLORMAP_RAINBOW)
data2 = cv2.applyColorMap(data2,cm)

if False:
    cv2.imshow('markers',data2)
    cv2.waitKey(0); cv2.destroyAllWindows()

# print('type(cm): ',type(cv2.COLORMAP_RAINBOW))
# print('cm:       ',cv2.COLORMAP_RAINBOW)



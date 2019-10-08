#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Simplest usage:

  labels = ccl_marker_stack().make_labels_from(data_slices,data_threshold_mnmx)

where data_slices is a list of 2D numpy arrays.


Note: data_threshold_mnmx is a misnomer. To clarify.

  data_threshold_mnmx = (trigger, data_value_out)

The trigger is the data value that the thresholding routine cv2.threshold uses
to determine what is "in" vs. what is excluded. Default operation is that data
values below trigger (the threshold) are accepted. Acceptance of a pixel is 
noted in the output image array by setting the output pixel to data_value_out.
Hence the output array of cv2.threshold has a range of two values {0,mx}
where all values of the input array greater than trigger (the mn) are 
sent to mx, where mx is data_value_out. 

Our options have expanded as we have added thresh_inverse and perform_threshold, 
which invert the thresholding described above. That is, acceptance is still 
denoted by data_value_out, but triggered by data values below trigger.

--

Copyright © 2018-2019 Michael Lee Rilee, mike@rilee.net, Rilee Systems Technologies LLC

ccl_marker_stack. Label a stack of 2d data slices using ccl2d.

For license information see the file LICENSE that should have accompanied this source.

"""
import unittest

import cv2
from dask.distributed import Client
import numpy as np
import os
from ccl2d import ccl2d

from stopwatch import sw_timer

def ccl_backsub(m,translations):
    "Apply the translations to the markers m."
    if translations is None or translations == []:
        # print( 'ccl_backsub: copy' )
        mx = m.copy()
    else:
        # print( 'ccl_backsub: translate' )
        # mx = np.zeros(m.shape)
        mx = m.copy()
        m_unique = np.unique(m)
        for im in m_unique:
            for x in translations:
                if im in x[0]:
                    # print( 'ccl_backsub: translate subs',x )
                    #if im == 20:
                    #    print( mx[40:50,320:330] )
                    mx[np.where(m == im)] = x[1]
                    #if im == 20:
                    #    print( mx[40:50,320:330] )
    #print( 'cc100' )
    #print( mx[40:50,320:330] )
    return mx.copy()

def ccl_compose_translations(x0,x1):
    # print( 'b99: x0:  ',x0 )
    # print( 'b99: x1:  ',x1 )
    x01 = []
    # i in x0 lands on j in x1, gotta relabel
    for j in x1:
        r  = set([])
        rj = j[0]
        sj = j[1]
        for i in x0:
            if i[1] in rj:
                r = r | set(i[0])
        if r != set([]):
            x01.append([r,sj])
            
    for i in x0:
        i_found = False
        for ielt in i[0]:
            for ij in x01:
                i_found = i_found or ielt in ij[0]
        if not i_found:
            # i doesn't map to an ij, so add directly
            x01.append(i)
        
    for j in x1:
        j_found = False
        for jelt in j[0]:
            for ij in x01:
                j_found = j_found or jelt in ij[0]
        if not j_found:
            # j doesn't map to an ij, so add directly
            x01.append(j)
           
    # print( 'b99: x01: ',x01 )
    return x01
                

def ccl_relabel2(m0,m1,verbose=False,marker_base=None,global_latlon_grid=True):
    """Identify structures in common between the two 2D numpy arrays m0 and m1. Structures merging and splitting from one array to the next cause labels to be degenerate, i.e. you have multiple labels for the same 2D+1 structure. This routine returns a 5-tuple (m0n,m1n,m0eol,translation01,translation11). m0n and m1n are relabeled. Coalesced labels are given a new number and the old numbers are 'lost,' which is why we return the translation lists translation01 and translation01. m0eol is a list of structures in m0 that are not found in m1, hence eol, end-of-life. translation01: m0 -> m0_new and translation11: m1 -> m1_new. If m0 and m1 have completely different labeling schemes, one might say the two translations are two separate functions from two different domains to the same range. t01: 0->1, while arguably t11: 1->1.
"""

    sw_timer.stamp("cms:relabel2 start")
    ## m0[np.where(m0 > 0)] += marker_base
    ## marker_base += np.max(m0)
    if marker_base is None:
        marker_base = np.max(m0)
    marker_base_0 = marker_base
    
    # print( '00-markers-shape: ',m0.shape )
    # print( '00-markers-type:  ',type(m0) )
    # print( '00-markers-dtype: ',m0.dtype )
    # print( '00-markers-mnmx:  ',np.min(m0[np.where(m0 > 0)])\ )
    #     ,np.max(m0[np.where(m0 > 0)])
    # print( '00-markers:       ',m0 )
    # print( '00-markers-1:     ',m0.astype(np.float)/np.amax(m0) )

    m1_save = m1.copy()
    m1[np.where(m1 > 0)] += marker_base
    marker_base += np.max(m1)
    # print( '01-markers-shape: ',m1.shape )
    # print( '01-markers-type:  ',type(m1) )
    # print( '01-markers-dtype: ',m1.dtype )
    # print( '01-markers-mnmx:  ',np.min(m1[np.where(m1 > 0)])\ )
    #   ,np.max(m1[np.where(m1 > 0)])
    # print( '01-markers:       ',m1 )
    # print( '01-markers-1:     ',m1.astype(np.float)/np.amax(m1) )

    # m1[:,:] = m0[:,:]
    # markers_tmp = np.zeros(m0.shape,dtype=np.float)
    # markers_tmp[:,:] = markers

    # When starting use 01 as the base, later, just carry it through.
    # marker_max = np.max(m1)
    
    markers_sum = np.zeros(m0.shape,dtype=np.uint8)
    # tmp = np.zeros(m0.shape,dtype=np.float)
    tmp = np.zeros(m0.shape,dtype=np.float)

    # tmp[:,:] = m0+m1
    tmp[:,:] = m1
    tmp_mx = np.max(tmp)
    if tmp_mx == 0:
        tmp_mx = 1
    tmp[:,:] = tmp/tmp_mx
    # print( 'max(tmp): ',np.max(tmp) )
    itmp = np.zeros(m0.shape,dtype=np.uint8)
    itmp[np.where(tmp > 0)] = 255
    # print( 'max(itmp): ',np.max(itmp) )
    # cv2.imshow('itmp',itmp); cv2.waitKey(0); cv2.destroyAllWindows()
    # tmp = m0
    # print( 'max(tmp): ',np.max(tmp) )
    # markers_sum[:,:] = 255*(tmp/np.max(tmp))
    markers_sum[:,:] = itmp
    # print( 'max(markers_sum): ',np.max(markers_sum) )

    # cv2.imshow('markers sum',markers_sum); cv2.waitKey(0); cv2.destroyAllWindows()
    
    ret, thresh = cv2.threshold(markers_sum\
                                ,0,np.max(markers_sum)\
                                ,cv2.THRESH_BINARY)

#                                ,1,np.max(markers_sum)\

    # cv2.imshow('thresh',thresh.astype(np.float)/np.amax(thresh)); cv2.waitKey(0); cv2.destroyAllWindows()

    # print(1000)
    # ret, markers01 = cv2.connectedComponents(thresh)
    markers01 = ccl2d(thresh,(254,255)
                      ,global_latlon_grid = global_latlon_grid
                  )
    # print(2000)
    # cv2.imshow('markers01',markers01.astype(np.float)/np.amax(markers01)); cv2.waitKey(0); cv2.destroyAllWindows()

    idx_00_01 = np.where((thresh == 255) & (m0 != 0))
    # print( 'idx_00_01: ',idx_00_01 )

    def map_slice_to_combined(m,markers,thresh):
        idx = np.where((thresh == 255) & (m != 0))
        id  = []
        for ij in range(len(idx[0])):
            i=idx[0][ij]
            j=idx[1][ij]
            r = m[i,j]; s = markers[i,j]
            rs = [r,s]
            id.append(rs)
        return id

    id_00_01 = map_slice_to_combined(m0,markers01,thresh)
    id_01_01 = map_slice_to_combined(m1,markers01,thresh)

    # print(3000)
    
    # if verbose:
    if False:
        print( 'id_00_01: ',id_00_01 )
            
    # if verbose:
    if False:
        print( 'id_01_01: ',id_01_01 )
        print( 'np.unique(m0): ',np.unique(m0) )

    # Detect forking
    marker_fork_equivalence_sets = []
    # Go over the map from 0 to 01

    if False:
        print( 'len(id_00_01): ',len(id_00_01) )

    # r0 = id_00_01[i,1]

    sw_timer.stamp("cms:relabel2 translation tables A")

    markers_unique = np.unique(markers01)

    rs=[]
    i = 0
    while i < len(markers_unique):
        r = []
        s = []
        r0 = markers_unique[i]
        ir = 0
        while ir < len(id_00_01):
            if (r0 == id_00_01[ir][1]):
                if id_00_01[ir][0] not in r:
                    r.append(id_00_01[ir][0])                
            ir += 1
        it = 0
        while it < len(id_01_01):
            if (r0 == id_01_01[it][1]):
                if id_01_01[it][0] not in s:
                    s.append(id_01_01[it][0])
            it += 1
        if (r != []) or (s != []):
            rs.append([set(r),set(s)])
        i += 1

    sw_timer.stamp("cms:relabel2 translation tables B")

    done = False
    while not done:
        # print( 'rs: ',rs )
        rs_tmp = []
        done = True
        for i in range(len(rs)):
            # print( '0 rs_tmp: ',rs_tmp )
            ri = rs[i][0]
            si = rs[i][1]
            for j in range(i+1,len(rs)):
                rj = rs[j][0]
                sj = rs[j][1]
                # print( '100: ',ri,rj,ri & rj,ri & rj != set([]) )
                if ri & rj != set([]):
                    # print( '101:' )
                    done = False
                    ri = ri | rj
                    si = si | sj
            for j in range(i+1,len(rs)):
                rj = rs[j][0]
                sj = rs[j][1]
                # print( '110: ',si,sj,si & sj,si & sj != set([]) )
                if si & sj != set([]):
                    # print( '111:' )
                    done = False
                    ri = ri | rj
                    si = si | sj
            overlap_found = False; j = 0
            while not overlap_found and j < len(rs_tmp):
                rtj = rs_tmp[j][0]
                stj = rs_tmp[j][1]
                if rtj & ri != set([]):
                    rn = rs_tmp[j][0] | ri
                    sn = rs_tmp[j][1] | si
                    if (rn != set([])) and (sn != set([])):
                        rs_tmp[j][0] = rn
                        rs_tmp[j][1] = sn
                    overlap_found = True
                j += 1
            if not overlap_found:
                if (ri != set([])) and (si != set([])):
                    rs_tmp.append([ri,si])
            # print( '2 rs_tmp: ',rs_tmp )
        rs = rs_tmp

    marker_fork_equivalence_sets = rs

    sw_timer.stamp("cms:relabel2 translation tables C")
                    
    # i = 0
    # while i < len(id_00_01):
    #     print( 'i:  ',i )
    #     # Get a target from 01
    #     r0 = id_00_01[i][1]
    #     # print( 'r0: ',r0 )
    #     r = []
    #     for j in range(i,len(id_00_01)):
    #         # print( 's0 comparing: ',id_00_01[j][1],' to r0=',r0 )
    #         # Have we
    #         if (r0 == id_00_01[j][1]):
    #             print( 'appending r: ',id_00_01[j][0] )
    #             if id_00_01[j][0] not in r:
    #                 r.append(id_00_01[j][0])
    #     if r != []:
    #         s=[]
    #         for j in range(len(id_01_01)):
    #             if r0 == id_01_01[j][1]:
    #                 print( 'appending s: ',id_01_01[j][0] )
    #                 if id_01_01[j][0] not in s:
    #                     s.append(id_01_01[j][0])
    #             
    #         if(s != []):
    #             r.sort(); s.sort();
    #             marker_fork_equivalence_sets.append([r,s])
    #     i += 1

    if False:
        print( 'marker_fork_equiv: ',marker_fork_equivalence_sets )

    marker_current = marker_base
    m0_new = np.zeros(m0.shape,dtype=np.int)
    m1_new = np.zeros(m1.shape,dtype=np.int)
    relabeled4 = []
    relabeled5 = []
    translation01 = []
    for i in range(len(marker_fork_equivalence_sets)):
        
        m0s = marker_fork_equivalence_sets[i][0]
        relabeled4.extend(m0s)
        for j in m0s:
            m0_new[np.where(m0 == j)] = marker_current
        translation01.append([m0s,marker_current])
        
        m1s = marker_fork_equivalence_sets[i][1]
        relabeled5.extend(m1s)
        for j in m1s:
            m1_new[np.where(m1 == j)] = marker_current

        # print( 'i,m0s,m1s,marker: ',i,m0s,m1s,marker_current )
        
        marker_current += 1

    marker_base = marker_current

    sw_timer.stamp("cms:relabel2 translation tables D")
    
    # print( 'equivalence sets k= ',len(marker_fork_equivalence_sets) )

    k = 0
    for i in np.unique(m1):
        if i not in relabeled5:
            m1_new[np.where(m1 == i)] = i
            k = k + 1
    # print( 'independent sets in m1 k= ',k )

    k = 0
    for i in np.unique(m0):
        if i not in relabeled4:
            m0_new[np.where(m0 == i)] = i
            k = k + 1
    # print( 'independent sets in m0 k= ',k )
    
    m0_new_unique = np.unique(m0_new)
    # print( 'unique(m0_new): ',m0_new_unique )
    
    m1_new_unique = np.unique(m1_new)
    # print( 'unique(m1_new): ',m1_new_unique )

    # Compress labels above marker_base_0
    old_labels_to_replace            = m1_new_unique[np.where(m1_new_unique > marker_base_0)]
    # new_compressed_labels = np.arange(len(new_lables))
    for i in range(len(old_labels_to_replace)):
        m0_new[np.where(m0_new == old_labels_to_replace[i])] = i + marker_base_0 + 1
        m1_new[np.where(m1_new == old_labels_to_replace[i])] = i + marker_base_0 + 1

    if len(old_labels_to_replace) > 0:
        for j in range(len(translation01)):
            labels0 = translation01[j][1]
            if labels0 in old_labels_to_replace:
                k = np.where(old_labels_to_replace == labels0)
                k = k[0][0]
                labels1 = k + marker_base_0 + 1
            else:
                labels1 = labels0
            translation01[j][1] = labels1
        
    marker_base = np.amax(m1_new)

    m0_new_unique = np.unique(m0_new)
    # print( 'unique(m0_new): ',m0_new_unique )
    
    m1_new_unique = np.unique(m1_new)
    # print( 'unique(m1_new): ',m1_new_unique )
    
    m0_eol = []
    for i in range(1,len(m0_new_unique)):
        if m0_new_unique[i] not in m1_new_unique:
            m0_eol.append(m0_new_unique[i])

    sw_timer.stamp("cms:relabel2 translation tables E")
    
    # eol means that it has no entry in the translation01
    # For cases where we are tracking more than one previous history item,
    # we can remove translation01 entries that are eol at t-1.
    # We can use a translation01 table to go back in time to relabel the
    # histories if needed.

    # print( 'm0_eol: ',m0_eol )
    # print( 'xlat:   ',translation01 )

    # translation01 maps 4's ids to the reconciled 5. Both 4_new and 5_new have the
    # New ids. I guess we could leave 4 alone, and just populate the translation01 table--
    # but then figuring out what's eol would be a little more involved.
    
    #while True:
    #    delta = np.abs(m1 - m0)
    #    m1a = np.full_like(m1,np.nan)
    #    id = []
    #    for i in np.where(delta > 0):
    #        id.append([m0[i],m1[i]])

    # return translation_1old_to_1new

    # m1_unique = np.unique(m1)

    translation11 = []

    m1_new_unique = np.unique(m1_new)

    if False:
        print( 'm0:            \n',m0 )
        print( 'm1_save:       \n',m1_save )
        print( 'm1_new:        \n',m1_new )
        print( 'n1_new_unique: \n',m1_new_unique )
        print( '----' )

    sw_timer.stamp("cms:relabel2 translation tables F")
    
    for i in range(1,len(m1_new_unique)):
        x_11n = m1_save[np.where(m1_new == m1_new_unique[i])]
        if False:
            print( 'i,x_11n: ',i,x_11n,', bool: ',type(x_11n),', size= ',x_11n.size )
        # if x_11n != []:
        if x_11n.size > 0:
            if False:
                print( '-:' )
            translation11.append([set(x_11n),m1_new_unique[i]])
        else:
            if False:
                print( '+:',[set(x_11n),m1_new_unique[i]] )


    sw_timer.stamp("cms:relabel2 end")
    return (m0_new,m1_new,m0_eol,translation01,translation11)

class ccl_marker_stack(object):
    def __init__(self,global_latlon_grid = True):
        self.m_results            = []
        self.m_results_translated = []
        self.m_ages               = {}
        self.translations         = []
        self.marker_base          = 0
        self.global_latlon_grid   = global_latlon_grid

# To merge two stacks, we need to identify the mapping between the two
# sets of labels at the interface. The labels at the bottom start
# at '1,' so they will need to be relabeled to avoid collisions.
# This will engender relabeling all of the second stack.
#
# The natural relabeling would be to add the max label of the
# first stack to the labels of the second stack. This would
# include both in the marker (label) stacks and the translation
# stack. Simply adding to every label should work for the second
# stack. If the second stack has already been "resolved," no
# further work needs be done for the second stack just from
# the shift.

# It gets more complicated when we actually interface the first and
# second stacks. For that, we have to do a relabel2 to determine
# coincidence, merge, and branch between the two slices. This
# will drive a relabeling wave up and down away from the interface
# for a full reconciliation of the labelling schemes of the two
# stacks. The wave going up does not have to reflect back down
# since each stack has already been reconciled separately. The
# new interfacing drives the new constraint.

# Step 1. Get max of first stack/slice. Shift labels in second stack.
# Step 2. relabel2 the interface slices.
# Step 3. backsub the translations down through the first stack.
# Step 4. forward-sub the translations through the second stack.
# Goal.   A single stack of the form self.m_results
# i.e.  [...,[slice,translations],[slice,translations],...]

# Note: we're trying to reuse the work previously done to each stack,
# stitching together the results.

    def shift_labels(self,id_delta):
        # Shift all the labels by id_delta
        for i in range(len(self.m_results)):
            # Change the markers
            self.m_results[i][0][np.where(self.m_results[i][0] > 0)] += id_delta

            # If translations are empty, should it remain empty?
            # Merging is a case that should change this... But not here.
            new_translations = []
            if self.m_results[i][1] != []:
                for j in range(len(self.m_results[i][1])):
                    # print( '---\ni: ',i )
                    # print( 'shift_labels: ',self.m_results[i],'the marker,translation pair from a set of ids to the current id' )
                    # print( 'shift_labels: ',self.m_results[i][1], 'the translations component, a list of set(l0),[l1] pairs' )
                    # print( 'shift_labels: ',self.m_results[i][1][0], 'the 0th translation' )
                    # for x in self.m_results[i][1]:
                    #     print( 'shift_labels:  ',x, 'a translation' )
                    # for x in self.m_results[i][1][0]:
                    #     print( 'shift_labels:  ',x,'A piece of a translation' )
                    # print( 'shift_labels: ',self.m_results[i][1][0][1],'range of the 0th' )
                    # print( '+--' )
                    # print( 'domain:  ',[ x    for x in self.m_results[i][1]] )
                    # print( 'domain:  ',[ x[0] for x in self.m_results[i][1]] )
                    # print( 'domain:  ',[ r for r in x[0] for x in self.m_results[i][1]] )
                    # print( 'domain:  ',[ r + id_delta if r > 0 else r for r in x[0] for x in self.m_results[i][1]] )
                    # print( 'range:   ',[ s + id_delta if s > 0 else s for s in [x[1]] for x in self.m_results[i][1]] )
                    # x = []
                    # id_domain = [ r + id_delta if r > 0 else r for r in x[0] for x in self.m_results[i][1] ]
                    x = self.m_results[i][1][j]
                    id_domain = []
                    for r in x[0]:
                        if r > 0:
                            id_domain.append(r + id_delta)
                        else:
                            id_domain.append(r)
                    id_range  = [ s + id_delta if s > 0 else s for s in [x[1]] ]
                    if len(id_range) == 1:
                        id_range = id_range[0]
                    else:
                        print( 'WARNING shift_labels detected multiple-element range' )
                    new_translations.append([ id_domain, id_range ])
                self.m_results[i][1] = new_translations
# prefer this...                        [ s + id_delta if s > 0 else s for s in [x[1]] for x in self.m_results[i][1]] ]

                # print( 'result0:   \n',self.m_results[i][0] )
                # print( 'result1:   \n',self.m_results[i][1] )
                # print( 'range:   ',[s + id_delta if s > 0 else s for s in x[1] for x in self.m_results[i][1][0] ] )
                
        # print( '+++' )
        # revalidate the applied translations and ages
        # for i in self.m_results:
        #     print( 'm_result\n',i )
        self.resolve_labels_across_stack()

#segmented    def compare_slices_at_interface(stack0,stack1):
#segmented        # Compare the two slices at the interface.
#segmented        m0 = stack0.m_results[-1][0]
#segmented        m1 = stack1.m_results[ 0][0]
#segmented        m0_new,m1_new,m0_eol,translation01\
#segmented                = ccl_relabel2(m0,m1,marker_base=0)
#segmented        
#segmented        # Just sketching below
#segmented        stack0.m_results[-1][0] = m0_new
#segmented        stack0.m_results[-1][1] = [] # Translation01 already applied to m0_new
#segmented
#segmented        stack1.m_results[ 0][0] = 
#segmented        stack1.m_results[ 0][1] = 
        
    # When a slice is added, it is ccl'd, which has labels starting at 1. 
    # To avoid collisions, the maximum label of the previous slice is added to
    # the labels of the new slice. In relabel2, We identify coincidences
    # between the labels between the two slices, and handle any merges
    # or splits. In general, merges eliminate labels, so we rename the
    # labels so there are no gaps in the labels used in the new slice. A
    # list of translations between adjacent slices is maintained.
    #
    def make_slice_from(self,data,data_threshold_mnmx,graph=False
                        ,thresh_inverse=False
                        ,norm_data=True
                        ,perform_threshold=True):

        ### There's a bug here. Some blobs are not correctly renamed.
        m1 = ccl2d(data,data_threshold_mnmx,graph=graph
                   ,thresh_inverse=thresh_inverse
                   ,global_latlon_grid = self.global_latlon_grid
                   ,norm_data=norm_data
                   ,perform_threshold=perform_threshold)

        # If m1 has structures, then the min label # is 1. A label of 0 is no structure.
        # We need m1's labels to be distinct, so we need to add the max label of all the m0s.
        # Because if the current m0 has no structure, then it won't keep track of the max label.
        if self.m_results == []:
            self.m_results = [[m1[:,:],[]]]
            m0_new        = []
            m1_new        = m1[:,:]
            m0_eol        = []
            translation01 = []
            marker_base   = np.amax(m1)
        else:
            m0 = self.m_results[-1][0]
            if self.marker_base <= np.amax(m0): # Update max label if needed.
                self.marker_base = np.amax(m0)
            # print(100)
            m0_new,m1_new,m0_eol,translation01,translation11\
                = ccl_relabel2(m0,m1,marker_base=self.marker_base,global_latlon_grid=self.global_latlon_grid)
            # print(200)
            self.m_results.append([m1_new[:,:],translation01[:]])
        return (m0_new,m1_new,m0_eol,translation01)
    
    def make_labels_from(self,data_slices,data_threshold_mnmx,graph=False):
        for d in data_slices:
            self.make_slice_from(d,data_threshold_mnmx,graph=graph)
        return self.resolve_labels_across_stack()
    
    def resolve_labels_across_stack(self):
        m                    = self.m_results[-1][0].copy()
        m_unique             = np.unique(m[np.where(m>0)])
        for i in m_unique:
            self.m_ages[i] = 1
        self.m_results_translated = [m]
        self.translations = [[]]
        x                         = self.m_results[-1][1][:]
        for i in range(len(self.m_results)-2,-1,-1):
            self.translations.append(x)
            m0n        = ccl_backsub(self.m_results[i][0],x)
            m0n_unique = np.unique(m0n[np.where(m0n>0)])
            for im in m0n_unique:
                if im in self.m_ages.keys():
                    self.m_ages[im] = self.m_ages[im] + 1
                else:
                    self.m_ages[im] = 1

            x = ccl_compose_translations(self.m_results[i][1],x)[:]
            self.m_results_translated.append(m0n.copy())
        self.m_results_translated.reverse()
        self.translations.reverse()
        return self.m_results_translated

    def apply_translations(self,translations_in):
        for i in range(len(self.m_results_translated)):
            self.m_results_translated[i] = ccl_backsub(self.m_results_translated[i],translations_in)
        return 

    def ids_resolved(self):
        ids = np.array([],dtype=np.int)
        for i in self.m_results_translated:
            ids = np.unique(np.concatenate((ids,np.unique(i))))
        return ids

    def ids_min_nonzero(self):
        tmp = self.ids_resolved()
        return np.amin(tmp[np.where(tmp > 0)])
        
    def ids_max(self):
        return np.amax(self.ids_resolved())
    
    def copy_of_translations(self):
        return self.translations

    def copy_of_translations_at(self,idx):
        return self.translations[idx]
    
    def copy_of_ages(self):
        return self.m_ages.copy()

    def copy_of_results(self):
        return self.m_results.copy()

    def slice_at(self,idx):
        return self.m_results[idx][0]

    def copy_of_translated_results(self):
        return self.m_results_translated

    def copy_of_translated_slice_at(self,idx):
        return self.m_results_translated[idx].copy()

    def copy_of_ages_at(self,idx):
        imr = np.zeros(self.m_results_translated[idx].shape)
        for k in self.m_ages.keys():
            imr[np.where(self.m_results_translated[idx] == k)] = self.m_ages[k]
        return imr

    def len(self):
        return len(self.m_results)
        
    def shape(self):
        if self.len() > 0:
            return len(self.m_results[0].shape)
        else:
            return ()

    def len_translated(self):
        return len(self.m_results_translated)
    
    def shape_translated(self):
        if self.len_translated() > 0:
            return len(self.m_results_translated[0].shape)
        else:
            return ()

def load_a_stack(fname):
    f_handle = open(fname,'rb')
    seg = np.load(f_handle)
    f_handle.close()
    return seg

def make_a_stack(d,thresh_mnmx,global_latlon_grid=True):
    ccl_stack = ccl_marker_stack(global_latlon_grid = global_latlon_grid)
    ccl_stack.make_labels_from(d,thresh_mnmx)
    return ccl_stack

def shift_labels(stack_seg0,stack_seg1):
    delta = stack_seg0.ids_max()            
    stack_seg1.shift_labels(delta)
    return stack_seg1

def make_translations(i_if,stack_seg0,stack_seg1):
    m0 = stack_seg0.copy_of_translated_slice_at(-1)
    m1 = stack_seg1.copy_of_translated_slice_at(0)
    
    m0n,m1n,m0_eol,trans01,trans11 = ccl_relabel2(m0
                                                  ,m1
                                                  ,marker_base = stack_seg0.ids_max()
                                                  ,global_latlon_grid = stack_seg0.global_latlon_grid)
    return (trans01,trans11)

def apply_interface_translation0(xab,ccl_stack):
    for x in xab:
        x_domain = x[0]
        if len(x_domain) > 1:
            x_single = max(x_domain)
            for i in x_domain:
                for im in range(len(ccl_stack.m_results_translated)):
                    ccl_stack.m_results_translated[im][np.where(ccl_stack.m_results_translated[im] == i)] = x_single
    return ccl_stack

def apply_translations(translations,ccl_stack):
    for im in range(len(ccl_stack.m_results_translated)):
        for xt in translations:
            ccl_stack.m_results_translated[im][np.where(ccl_stack.m_results_translated[im] == xt[0])] \
                = xt[1]
    return ccl_stack

class ccl_dask(object):
    def __init__(self,global_latlon_grid=True):
        self.client = Client()
        self.ccl_stacks           = []
        self.ccl_stacks_relabeled = []
        self.data_segs  = []
        self.nseg       = 0
        self.global_latlon_grid = global_latlon_grid

    def load_data_segments(self,file_list):
        self.nseg = len(file_list)
        for fn in file_list:
            self.data_segs.append(self.client.submit(load_a_stack,fn))

    def make_stacks(self,thresh_mnmx):
        self.thresh_mnmx = thresh_mnmx
        for i in range(self.nseg):
            self.ccl_stacks.append(self.client.submit(make_a_stack\
                                                      ,self.data_segs[i]\
                                                      ,self.thresh_mnmx\
                                                      ,self.global_latlon_grid))

    def shift_labels(self):
        self.ccl_stacks_relabeled = [self.ccl_stacks[0]]
        for i_interface in range(self.nseg-1):
            self.ccl_stacks_relabeled\
                .append(self.client.submit(shift_labels\
                                      ,self.ccl_stacks_relabeled[i_interface]\
                                      ,self.ccl_stacks[i_interface+1]))

    def make_translations(self):
        self.interface_translationsXX = []
        for i_interface in range(self.nseg-1):
            self.interface_translationsXX.append(self.client.submit(make_translations\
                                                               ,i_interface\
                                                               ,self.ccl_stacks_relabeled[i_interface]\
                                                               ,self.ccl_stacks_relabeled[i_interface+1]))
            
    def apply_translations(self):
        ccl_stacks_a             = []
        self.global_translations = []
        # ccl_stack_last = ccl_stacks_z[-1] # a future
        ccl_stack_last = self.ccl_stacks_relabeled[-1] # a future
        
        for i_interface in range(self.nseg-2,-1,-1):
            new_interface_translations = []
            ccl_stack1 = ccl_stack_last # a future
            ccl_stack0 = self.ccl_stacks_relabeled[i_interface] # a future
            XX = self.interface_translationsXX[i_interface] # a future
            xx = XX.result() # is (x01,x11)
            # Save the top for futuring global relabeling
            ccl_stacks_a.append(self.client.submit(apply_interface_translation0,xx[1],ccl_stack1))
            # The bottom is the top in the next iteration
            ccl_stack_last = self.client.submit(apply_interface_translation0,xx[0],ccl_stack0)

            # Propagate labels. Note this is essentially serial as currently written.
            x11 = xx[1]
            x01 = xx[0]
            for x1 in x11:
                x1_domain = max(x1[0])
                x1_fict   = x1[1]

                i0 = 0
                done = not( i0 < len(x01) )
                while not done:
                    x0 = x01[i0]
                    x0_domain = max(x0[0])
                    x0_fict = x0[1]
                    if x0_fict == x1_fict:
                        if len(self.global_translations) > 0:
                            for old_x in self.global_translations[-1]:
                                if old_x[0] == x1_domain:
                                    x1_domain = old_x[1]
                                    break
                        new_x = [x0_domain,x1_domain]
                        new_interface_translations.append(list(new_x))
                        done = True
                    else:
                        i0 = i0 + 1
                        done = i0 > len(x01)-1
            # if len(new_interface_translations) > 0:
            self.global_translations.append(list(new_interface_translations))
            # print( '080 global_translations: ',global_translations )

        # print( '099 len ccl_stacks_a = ',len(ccl_stacks_a) )
        # print( '100 global_translations: ',global_translations )
        
        self.global_translations.reverse()
        ccl_stacks_a.append(ccl_stack_last)
        ccl_stacks_a.reverse()
        
        # print( '110 len(ccl_stacks_a) =       ',len(ccl_stacks_a) )
        # print( '110 global_translations:      ',global_translations )
        # print( '110 len(global_translations): ',len(global_translations) )

        # Apply the translations globally
        self.ccl_stacks_b = []
        for i_if in range(self.nseg-1):
            iseg = i_if
            translations = self.global_translations[i_if]
            self.ccl_stacks_b.append(self.client.submit(apply_translations,translations,ccl_stacks_a[i_if]))
        self.ccl_stacks_b.append(ccl_stacks_a[-1])

        self.ccl_results = []
        for i_st in range(len(self.ccl_stacks_b)):
            self.ccl_results.append(self.ccl_stacks_b[i_st].result())
        
        # for i_st in range(len(self.ccl_stacks_b)):
        #     print( 'i_st: ',i_st )
        #     print( self.ccl_stacks_b[i_st].result().m_results_translated )
        #     print( '--' )
            
        # # Gather results here -- order of arrival? Maybe save order information and sort after gather is done...
        # # E.g. add [iseg,future] to ccl_stacks_b instead of just the future.
        # self.ccl_results = self.client.gather(self.ccl_stacks_b)
            
        self.client.close()
            
class Tests(unittest.TestCase):

    def test_diagonals(self):
        d=[]
        d.append(np.zeros((5,6)))
        d[0][1,1] = 2
        d[0][2,1] = 2
        d[0][3,1] = 2
        d[0][4,1] = 2
        d[0][0,3] = 2
        d[0][1,3] = 0
        d[0][2,3] = 2
        d[0][3,3] = 0
        d[0][1,5] = 2
        d[0][2,5] = 0
        d[0][3,5] = 2
        d.append(np.zeros((5,6)))
        d[1][1,1] = 2
        d[1][2,1] = 0
        d[1][3,1] = 2
        d[1][4,1] = 2
        d[1][0,3] = 2
        d[1][1,3] = 0
        d[1][2,3] = 2
        d[1][3,3] = 2
        d[1][1,5] = 2
        d[1][2,5] = 2
        d[1][3,5] = 2
        d.append(np.zeros((5,6)))
        d[2][1,1] = 2
        d[2][2,1] = 0
        d[2][3,1] = 0
        d[2][4,1] = 2
        d[2][0,3] = 2
        d[2][1,3] = 2
        d[2][2,3] = 2
        d[2][3,3] = 2
        d[2][1,5] = 2
        d[2][2,5] = 0
        d[2][3,5] = 2
        d.append(np.zeros((5,6)))
        d.append(np.zeros((5,6)))
        d[-1][2,3] = 2
        d.append(np.zeros((5,6)))
        d.append(np.zeros((5,6)))
        d[-1][0,0] = 2
        d[-1][0,4] = 2
        d[-1][1,1] = 2
        d[-1][2,0] = 2
        # d[-1][2,5] = 2
        d[-1][3,5] = 2
        d[-1][4,2] = 2
        d[-1][4,4] = 2
        expected_results = \
        [
            np.array(
                [[ 0,  0,  0, 11,  0,  0],
                 [ 0, 10,  0,  0,  0, 12],
                 [ 0, 10,  0, 11,  0,  0],
                 [ 0, 10,  0,  0,  0, 12],
                 [ 0, 10,  0,  0,  0,  0]]),
            np.array(
                [[ 0,  0,  0, 11, 0,  0],
                 [ 0, 10,  0,  0, 0, 12],
                 [ 0,  0,  0, 11, 0, 12],
                 [ 0, 10,  0, 11, 0, 12],
                 [ 0, 10,  0,  0, 0,  0]]),
            np.array(
                [[ 0,  0,  0, 11, 0,  0],
                 [ 0, 10,  0, 11, 0, 12],
                 [ 0,  0,  0, 11, 0,  0],
                 [ 0,  0,  0, 11, 0, 12],
                 [ 0, 10,  0,  0, 0,  0]]),
            np.array(
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]),
            np.array(
                [[ 0,  0,  0,  0,  0,  0],
                 [ 0,  0,  0,  0,  0,  0],
                 [ 0,  0,  0, 13,  0,  0],
                 [ 0,  0,  0,  0,  0,  0],
                 [ 0,  0,  0,  0,  0,  0]]),
            np.array(
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]),
            np.array(
                [[14,  0,  0,  0, 14,  0],
                 [ 0, 14,  0,  0,  0,  0],
                 [14,  0,  0,  0,  0,  0],
                 [ 0,  0,  0,  0,  0, 14],
                 [ 0,  0, 14,  0, 14,  0]])
        ]

        if False:
            k = 0
            for i in d:
                print( '**** d of i' )
                print( 'k = ',k )
                k += 1
                print( i )
                print( '***' )
        # marker_stack = ccl_marker_stack()
        thresh_mnmx = (1,2)
        markers = ccl_marker_stack().make_labels_from(d,thresh_mnmx)

        if False:
            k = 0
            for i in markers:
                print( "**** marker's k" )
                print( 'k = ',k )
                print( i )
                # print( 'delta' )
                # print( i-expected_results[k] )
                k += 1

        k = 0
        for i in markers:
            self.assertTrue(np.allclose(expected_results[k],i, rtol=1e-05, atol=1e-08))
            k += 1

        expected_ages =np.array(\
                                [
                                    [[0., 0., 0., 3., 0., 0.],
                                     [0., 3., 0., 0., 0., 3.],
                                     [0., 3., 0., 3., 0., 0.],
                                     [0., 3., 0., 0., 0., 3.],
                                     [0., 3., 0., 0., 0., 0.]],
                                    [[0., 0., 0., 3., 0., 0.],
                                     [0., 3., 0., 0., 0., 3.],
                                     [0., 0., 0., 3., 0., 3.],
                                     [0., 3., 0., 3., 0., 3.],
                                     [0., 3., 0., 0., 0., 0.]],
                                    [[0., 0., 0., 3., 0., 0.],
                                     [0., 3., 0., 3., 0., 3.],
                                     [0., 0., 0., 3., 0., 0.],
                                     [0., 0., 0., 3., 0., 3.],
                                     [0., 3., 0., 0., 0., 0.]],
                                    [[0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0., 0.]],
                                    [[0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., 1., 0., 0.],
                                     [0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0., 0.]],
                                    [[0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0., 0.]],
                                    [[1., 0., 0., 0., 1., 0.],
                                     [0., 1., 0., 0., 0., 0.],
                                     [1., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0., 1.],
                                     [0., 0., 1., 0., 1., 0.]]
                                ])
            
        marker_stack = ccl_marker_stack()
        marker_stack.make_labels_from(d,thresh_mnmx)
        for i in range(marker_stack.len()):
            # print( 'i,ages(i): ',i,'\n',marker_stack.copy_of_ages_at(i) )
            self.assertTrue(np.allclose(expected_ages[i],marker_stack.copy_of_ages_at(i), rtol=1e-05, atol=1e-08))

    def test_segmented_ccl(self):
        if False:
            print( '---test segmented ccl---' )
        
        d    = []
        nseg    = 5
        nstride = 5
        nd   = nseg*nstride
        nshape = (4,5)
        thresh_mnmx = (1,2)
        for i in range(nd):
            # d.append( np.random.randint(3,size=nshape,dtype=np.int) )
            d.append( np.full(nshape,0,dtype=np.int) )
            d[-1][2,2] = 2
            d[-1][2,4] = 2
            # d.append( np.full(nshape,i,dtype=np.int) )
        # print( 'len(d): ',len(d) )
        d[-nstride-1][0,2] = 2
        d[-nstride-1][1,2] = 2
        d[-nstride][0,2] = 2

        d[2*nstride]  [0,0] = 2
        d[2*nstride+1][0,0] = 2
        
        ccl_stack = ccl_marker_stack()
        markers = ccl_stack.make_labels_from(d,thresh_mnmx)

        dseg = []
        for i in range(nseg):
            # print( 'i,idx: ',i,i*nstride,((i+1)*nstride)-1 )
            dseg.append(d[i*nstride:(i+1)*nstride])
        # print( 'len(dseg): ',len(dseg) )
        
        ccl_stacks = []
        seg_markers = []
        for i in range(nseg):
            ccl_stacks.append(ccl_marker_stack())
            seg_markers.append(ccl_stacks[i].make_labels_from(dseg[i],thresh_mnmx))




        # for i in range(nseg):
        #     print( dseg[i] )
        #     # print( ccl_stacks[i].copy_of_translated_slice_at(-1) )

        id_delta = []
        # for i_interface in range(nseg-1):
        # if True:
        #    i_interface = 0
        interface_translations01 = []
        interface_translations11 = []
        for i_interface in range(nseg-1):
            stack_seg0 = ccl_stacks[i_interface]
            delta = stack_seg0.ids_max()
            id_delta.append(delta)
            stack_seg1 = ccl_stacks[i_interface+1]
            #?
            stack_seg1.shift_labels(delta)
            # Do we need to merge any labels?

            m0 = stack_seg0.copy_of_translated_slice_at(-1)
            # m1 = stack_seg1.slice_at(0)
            m1 = stack_seg1.copy_of_translated_slice_at(0)

            if False:
                print( 'i_if: ',i_interface )
                print( 'm0: \n',m0 )
                print( 'm1: \n',m1 )
            
            #m0n,m1n,m0_eol,trans01 = ccl_relabel2(stack_seg0.copy_of_translated_slice_at(-1)
            #                                     ,stack_seg1.slice_at(0)
            #                                     ,marker_base = stack_seg0.ids_max())

            m0n,m1n,m0_eol,trans01,trans11 = ccl_relabel2(m0
                                                          ,m1
                                                          ,marker_base = stack_seg0.ids_max()
                                                          ,global_latlon_grid = stack_seg0.global_latlon_grid)
            
#### review here ####
            interface_translations01.append(trans01)
            interface_translations11.append(trans11)
            # The interface also contains coalescence information.
#### review here ####

            if False:
                print( 'm0n: \n',m0n )
                print( 'm1n: \n',m1n )
                print( 'trans01: \n',trans01 )
                print( 'trans11: \n',trans11 )
                print( '---------\n' )

        #??? # Now that the interface_translation??s have been constructed, now we go through
        #??? # and reconcile the whole data set. Start at the top and go back to the bottom.

        global_translations = []
        
        for i_interface in range(nseg-2,-1,-1):

            new_interface_translations = []
            
            x11 = interface_translations11[i_interface]
            # 100
            # iterate over translations
            # - replace multiplicitous labels with a single label for both top and bottom of interface
            # top of interface
            for x in x11:
                # x is a translation
                x_domain = x[0]
                if len(x_domain) > 1:                    
                    x_single = max(x_domain)
                    # print( 'x_single: ',x_single )
                    iseg = i_interface + 1
                    # print( 'iseg: ',iseg )
                    for i in x_domain:
                        # print( 'i: ',i )
                        # print( 'len(ccl stacks m_results): ',len(ccl_stacks[iseg].m_results_translated) )
                        for im in range(len(ccl_stacks[iseg].m_results_translated)):
                            ccl_stacks[iseg].m_results_translated[im][\
                                                                      np.where(ccl_stacks[iseg].m_results_translated[im] == i)]\
                                                                      = x_single

            # bottom of interface
            x01 = interface_translations01[i_interface]
            for x in x01:
                x_domain = x[0]
                if len(x_domain) > 1:
                    x_single = max(x_domain)
                    iseg = i_interface
                    for i in x_domain:
                        for im in range(len(ccl_stacks[iseg].m_results_translated)):                        
                            ccl_stacks[iseg].m_results_translated[im][\
                                                              np.where(ccl_stacks[iseg].m_results_translated[im] == i)]\
                                                              = x_single

            # Propagate labels from the upper to the lower translations for this interface
            # Once we traverse all the segments, we'll have globally reconciled labels.
            # Use the fictitious labels to cross the interface.
            for x1 in x11:
                x1_domain = max(x1[0])
                x1_fict   = x1[1]
                # print( 'x1_fict: ',x1_fict )

                i0 = 0
                done = not( i0 < len(x01) )
                while not done:
                    x0 = x01[i0]
                    x0_domain = max(x0[0])
                    x0_fict   = x0[1]
                    # print( 'x0_fict: ',x0_fict )
                    if x0_fict == x1_fict:
                        if len(global_translations) > 0:
                            for old_x in global_translations[-1]:
                                # print( 'old_x,x0_domain,x1_domain: ',old_x, x0_domain, x1_domain )
                                if old_x[0] == x1_domain:
                                    x1_domain = old_x[1]
                                    break
                        new_x = [x0_domain,x1_domain] # from real-0 to real-1
                        new_interface_translations.append(list(new_x))
                        # global_translations.append(new_x)
                        done = True
                    else:
                        i0 = i0 + 1
                        done = i0 > len(x01)-1

            # if len(new_interface_translations) > 0:
            global_translations.append(list(new_interface_translations))

        # Reverse so that the interfaces are in order.
        global_translations.reverse()
        # print( 'global_translations\n',global_translations )

        # Apply the translations applicable to each segment, so that each is now globally reconciled.
        for i_if in range(nseg-1):
            iseg = i_if
            translations = global_translations[i_if]
            for im in range(len(ccl_stacks[iseg].m_results_translated)):
                for xt in translations:
                    ccl_stacks[iseg].m_results_translated[im][\
                                                              np.where(ccl_stacks[iseg].m_results_translated[im] == xt[0])]\
                                                              = xt[1]

        if False:
            print
            for iseg in range(nseg):
                print( 'iseg, id min,max: ',iseg,ccl_stacks[iseg].ids_min_nonzero(),ccl_stacks[iseg].ids_max() )

            print
            for iseg in range(nseg-1):
                print( 'iseg, id_delta: ',iseg,id_delta[iseg] )

            print
            for iseg in range(nseg-1):
                print( 'iseg, if_tran01s: ',iseg,interface_translations01[iseg] )
                print( 'iseg, if_tran11s: ',iseg,interface_translations11[iseg] )
            print

            #print
            #for iseg in range(nseg-1):
            #    print( 'iseg, if_tran11s: ',iseg,interface_translations11[iseg] )
            
            print
            for iseg in range(nseg):
                print( 'iseg',iseg )
                print( ccl_stacks[iseg].copy_of_translated_results() )
            print

        expected_labelling = []
        for i in range(nd):
            expected_labelling.append(np.full(nshape,0,dtype=np.int))
            expected_labelling[i]      [2,2] = 52
            expected_labelling[i]      [2,4] = 53
        expected_labelling[-nstride-1] [0,2] = 52
        expected_labelling[-nstride-1] [1,2] = 52
        expected_labelling[-nstride]   [0,2] = 52
        expected_labelling[2*nstride]  [0,0] = 24
        expected_labelling[2*nstride+1][0,0] = 24
        
        for iseg in range(nseg):
            for islice in range(nstride):
                self.assertTrue(np.allclose(expected_labelling[iseg*nstride+islice]\
                                            ,ccl_stacks[iseg].m_results_translated[islice]))

        # broken            
            
        # k = 0
        # for i in range(nseg):
        #     print( 'i: ',i,'\n' )
        #     for j in dseg[i]:
        #         print( '---' )
        #         print( j )
        #         print( d[k] )
        #         k += 1

        # Notes for parallelization
        if False:
            print( 'ms.ids-mx: ',np.amax(marker_stack.ids_resolved()) )
            
            split_id = 4
            print( 'split_id: ',split_id )
            d1 = d[0:split_id]
            d2 = d[split_id:]
    
            ms1 = ccl_marker_stack()
            labels1 = ms1.make_labels_from(d1,thresh_mnmx)
            print( 'ms1: \n',ms1.copy_of_translations() )
            print( 'ms1-mx: ',np.amax(ms1.ids_resolved()) )
            print( 'ms1-mx: ',ms1.ids_max() )
            print( 'ms1: \n',ms1.copy_of_translated_slice_at(-1) )
            
            ms2 = ccl_marker_stack()
            labels2 = ms2.make_labels_from(d2,thresh_mnmx)
            print( 'ms2: \n',ms2.copy_of_translations() )
            print( 'ms2-mx: ',np.amax(ms2.ids_resolved()) )
            print( 'ms2: \n',ms2.copy_of_translated_slice_at(0) )
    
            m1n,m2n,m1eol,trans01,trans11 = ccl_relabel2(ms1.copy_of_translated_slice_at(-1)
                                                         ,ms2.copy_of_translated_slice_at(0)
                                                         ,marker_base = ms1.ids_max()
                                                         ,global_latlon_grid = ms1.global_latlon_grid)
            print( 'trans01\n',trans01 )
            print( 'trans11\n',trans11 )
            print( 'm1n:    \n',m1n )
            print( 'm2n-mx: ',np.amax(m2n) )
            print( 'm2n:    \n',m2n )
            print( 'ms :    \n',marker_stack.copy_of_translated_slice_at(split_id) )
    
            # Okay, for a parallel computation.
            # 1. Split the data stack at a bunch of split_id's.
            # 2. In parallel, construct marker_stacks for each data stack. 
            # 3. In serial, copy interface data and run relabel2 to match at interface and rename ids.
            # 4. Propagate renaming to upper boundary of the 2nd stack, and repeat 3 at the next node.
            # 5  At the top of the super stack you now know the top id. Repeat steps 3 and 4
            #    in reverse to propagate the top id info back down to the bottom of the superstack.
            #    Maybe we do backsub on our way up and down.


    def test_ccl_dask_object(self):
        ##################################################
        # Construct the test data
        #
        d    = []
        nseg    = 5
        nstride = 5
        nd   = nseg*nstride
        nshape = (4,5)
        thresh_mnmx = (1,2)
        for i in range(nd):
            # d.append( np.random.randint(3,size=nshape,dtype=np.int) )
            d.append( np.full(nshape,0,dtype=np.int) )
            d[-1][2,2] = 2
            d[-1][2,4] = 2
            # d.append( np.full(nshape,i,dtype=np.int) )
        # print( 'len(d): ',len(d) )
        d[-nstride-1][0,2] = 2
        d[-nstride-1][1,2] = 2
        d[-nstride][0,2] = 2
        d[2*nstride]  [0,0] = 2
        d[2*nstride+1][0,0] = 2

        dseg = []
        file_list = []
        for i in range(nseg):        
            dseg.append(d[i*nstride:(i+1)*nstride])
            fname=str(i)+'.npy'
            file_list.append(fname)
            with open(fname,'wb') as f_handle:
                np.save(f_handle,dseg[i])

        ##################################################
        # Set up expectations
        expected_labelling = []
        for i in range(nd):
            expected_labelling.append(np.full(nshape,0,dtype=np.int))
            expected_labelling[i]      [2,2] = 52
            expected_labelling[i]      [2,4] = 53
        expected_labelling[-nstride-1] [0,2] = 52
        expected_labelling[-nstride-1] [1,2] = 52
        expected_labelling[-nstride]   [0,2] = 52
        expected_labelling[2*nstride]  [0,0] = 24
        expected_labelling[2*nstride+1][0,0] = 24

        ##################################################
        # The calculation
        ccl_dask_object = ccl_dask()
        ccl_dask_object.load_data_segments(file_list)
        ccl_dask_object.make_stacks(thresh_mnmx)
        ccl_dask_object.shift_labels()
        ccl_dask_object.make_translations()
        ccl_dask_object.apply_translations()
        
        ##################################################
        # Check results
        for iseg in range(nseg):
            for islice in range(nstride):
                self.assertTrue(np.allclose(expected_labelling[iseg*nstride+islice]\
                                            ,ccl_dask_object.ccl_results[iseg].m_results_translated[islice]))
        
        ##################################################
        # Clean up
        for i in range(nseg):
            fname=str(i)+'.npy'
            os.remove(fname)
        
    def test_dask_ccl(self):

        client = Client()

        ##################################################
        # Construct the test data
        #
        d    = []
        nseg    = 5
        nstride = 5
        nd   = nseg*nstride
        nshape = (4,5)
        thresh_mnmx = (1,2)
        for i in range(nd):
            # d.append( np.random.randint(3,size=nshape,dtype=np.int) )
            d.append( np.full(nshape,0,dtype=np.int) )
            d[-1][2,2] = 2
            d[-1][2,4] = 2
            # d.append( np.full(nshape,i,dtype=np.int) )
        # print( 'len(d): ',len(d) )
        d[-nstride-1][0,2] = 2
        d[-nstride-1][1,2] = 2
        d[-nstride][0,2] = 2
        d[2*nstride]  [0,0] = 2
        d[2*nstride+1][0,0] = 2

        dseg = []
        for i in range(nseg):        
            dseg.append(d[i*nstride:(i+1)*nstride])
            fname=str(i)+'.npy'
            with open(fname,'wb') as f_handle:
                np.save(f_handle,dseg[i])

        ##################################################
            
        expected_labelling = []
        for i in range(nd):
            expected_labelling.append(np.full(nshape,0,dtype=np.int))
            expected_labelling[i]      [2,2] = 52
            expected_labelling[i]      [2,4] = 53
        expected_labelling[-nstride-1] [0,2] = 52
        expected_labelling[-nstride-1] [1,2] = 52
        expected_labelling[-nstride]   [0,2] = 52
        expected_labelling[2*nstride]  [0,0] = 24
        expected_labelling[2*nstride+1][0,0] = 24
        
        ##################################################
        def load_a_stack(fname):
            f_handle = open(fname,'rb')
            seg = np.load(f_handle)
            f_handle.close()
            return seg

        data_segs = []
        for i in range(nseg):
            fname=str(i)+'.npy'            
            data_segs.append(client.submit(load_a_stack,fname))
            
        ##################################################
        # Construct stacks of futures
        #
        def make_a_stack(d,thresh_mnmx):
            ccl_stack = ccl_marker_stack()
            ccl_stack.make_labels_from(d,thresh_mnmx)
            return ccl_stack

        ccl_stacks = []
        for i in range(nseg):
            ccl_stacks.append(client.submit(make_a_stack,data_segs[i],thresh_mnmx))
            # ccl_stacks.append(client.submit(make_a_stack,dseg[i],thresh_mnmx))

        ##################################################
        # Make translations
        #
        def shift_labels(stack_seg0,stack_seg1):
            delta = stack_seg0.ids_max()            
            stack_seg1.shift_labels(delta)
            return stack_seg1

        ccl_stacks_z = [ccl_stacks[0]]
        for i_interface in range(nseg-1):
            # Note i_interface is like -1 below.
            ccl_stacks_z.append(client.submit(shift_labels,ccl_stacks_z[i_interface],ccl_stacks[i_interface+1]))

        # print( 'sl-000 len(ccl_stacks_z) = ',len(ccl_stacks_z) )
        
        def make_translations(i_if,stack_seg0,stack_seg1):
            m0 = stack_seg0.copy_of_translated_slice_at(-1)
            m1 = stack_seg1.copy_of_translated_slice_at(0)
            
            m0n,m1n,m0_eol,trans01,trans11 = ccl_relabel2(m0
                                                          ,m1
                                                          ,marker_base = stack_seg0.ids_max()
                                                          ,global_latlon_grid = stack_seg0.global_latlon_grid)
            return (trans01,trans11)
            
        interface_translationsXX = []
        for i_interface in range(nseg-1):
            # print( 'make_trans i_interface = ',i_interface )
            interface_translationsXX.append(client.submit(make_translations\
                                                          ,i_interface\
                                                          ,ccl_stacks_z[i_interface]\
                                                          ,ccl_stacks_z[i_interface+1]))


        ##################################################
        # Global translation

        def apply_interface_translation0(xab,ccl_stack):
            for x in xab:
                x_domain = x[0]
                if len(x_domain) > 1:
                    x_single = max(x_domain)
                    for i in x_domain:
                        for im in range(len(ccl_stack.m_results_translated)):
                            ccl_stack.m_results_translated[im][np.where(ccl_stack.m_results_translated[im] == i)] = x_single
            return ccl_stack

        def apply_translations(translations,ccl_stack):
            for im in range(len(ccl_stack.m_results_translated)):
                for xt in translations:
                    ccl_stack.m_results_translated[im][np.where(ccl_stack.m_results_translated[im] == xt[0])] \
                        = xt[1]
            return ccl_stack
        
        ccl_stacks_a        = []
        global_translations = []
        ccl_stack_last = ccl_stacks_z[-1] # a future
        
        # print( '000 nseg:               ',nseg )
        # print( '000 range(nseg-2,-1-1): ',range(nseg-2,-1,-1) )
        for i_interface in range(nseg-2,-1,-1):
            # print( '010 i_interface: ',i_interface )
            new_interface_translations = []
            ccl_stack1 = ccl_stack_last # a future
            ccl_stack0 = ccl_stacks_z[i_interface] # a future
            XX = interface_translationsXX[i_interface] # a future
            xx = XX.result() # is (x01,x11)
            # Save the top for futuring global relabeling
            ccl_stacks_a.append(client.submit(apply_interface_translation0,xx[1],ccl_stack1))
            # The bottom is the top in the next iteration
            ccl_stack_last = client.submit(apply_interface_translation0,xx[0],ccl_stack0)

            # Propagate labels. Note this is essentially serial as currently written.
            x11 = xx[1]
            x01 = xx[0]
            for x1 in x11:
                x1_domain = max(x1[0])
                x1_fict   = x1[1]

                i0 = 0
                done = not( i0 < len(x01) )
                while not done:
                    x0 = x01[i0]
                    x0_domain = max(x0[0])
                    x0_fict = x0[1]
                    if x0_fict == x1_fict:
                        if len(global_translations) > 0:
                            for old_x in global_translations[-1]:
                                if old_x[0] == x1_domain:
                                    x1_domain = old_x[1]
                                    break
                        new_x = [x0_domain,x1_domain]
                        new_interface_translations.append(list(new_x))
                        done = True
                    else:
                        i0 = i0 + 1
                        done = i0 > len(x01)-1
            # if len(new_interface_translations) > 0:
            global_translations.append(list(new_interface_translations))
            # print( '080 global_translations: ',global_translations )

        # print( '099 len ccl_stacks_a = ',len(ccl_stacks_a) )
        # print( '100 global_translations: ',global_translations )
        
        global_translations.reverse()
        ccl_stacks_a.append(ccl_stack_last)
        ccl_stacks_a.reverse()
        
        # print( '110 len(ccl_stacks_a) =       ',len(ccl_stacks_a) )
        # print( '110 global_translations:      ',global_translations )
        # print( '110 len(global_translations): ',len(global_translations) )

        # Apply the translations globally
        ccl_stacks_b = []
        for i_if in range(nseg-1):
            iseg = i_if
            translations = global_translations[i_if]
            ccl_stack_f = ccl_stacks_a[i_if]
            ccl_stacks_b.append(client.submit(apply_translations,translations,ccl_stack_f))
        ccl_stacks_b.append(ccl_stacks_a[-1])
        
        if False:
            print( 'ccl_stacks_b, len = ',len(ccl_stacks_b) )
        if False:
            for i_st in range(len(ccl_stacks_b)):
                print( 'i_st: ',i_st )
                print( ccl_stacks_b[i_st].result().m_results_translated )
                print( '--' )
        
        for iseg in range(nseg):
            for islice in range(nstride):
                self.assertTrue(np.allclose(expected_labelling[iseg*nstride+islice]\
                                            ,ccl_stacks_b[iseg].result().m_results_translated[islice]))

        for i in range(nseg):
            fname=str(i)+'.npy'
            os.remove(fname)
            
        client.close()

    def test_relabel2(self):
        if True:
            verbose = False
            d0 = np.zeros((5,6))
            d0[0,0] = 0
            d0[:,2] = 2
            
            d1 = np.zeros((5,6))
            d1[0,0] = 0
            d1[0,2] = 2
            d1[2,2] = 2
            
            d1[0,5] = 0
            d1[1,5] = 2
            d1[2,5] = 2
            d1[3,5] = 2
            d1[4,5] = 2
            
            d2 = np.zeros((5,6))
            d2[0,0] = 0
            d2[0,2] = 2
            
            d2[0,5] = 0
            d2[1,5] = 2
            d2[2,5] = 0
            d2[3,5] = 2
            d2[4,5] = 2

            if verbose:
                print( '***' )
            m0 = ccl2d(d0,(1,2))
            m0_orig = m0.copy()
            if verbose:
                print( 'm0:     \n',m0 )
            m1 = ccl2d(d1,(1,2))
            if verbose:
                print( 'm1:     \n',m1 )
                print( '***' )
            m0_new,m1_new,m0_eol,translation01,translation11\
                = ccl_relabel2(m0,m1)
            translation01_0 = translation01
            m0_new_0 = m0_new
            if verbose:
                print( '***' )
                print( 'm0:     \n',m0 )
                print( 'm1:     \n',m1 )
                print( 'm0_new: \n',m0_new )
                print( 'm1_new: \n',m1_new )
                print( 'm0_eol: \n',m0_eol )
                print( 'xl_   : \n',translation01 )
                print( '******' )

            self.assertTrue(\
                            np.allclose(\
                                        m1_new
                                        ,np.array(\
                                                  [[0, 0, 3, 0, 0, 0],
                                                   [0, 0, 0, 0, 0, 2],
                                                   [0, 0, 3, 0, 0, 2],
                                                   [0, 0, 0, 0, 0, 2],
                                                   [0, 0, 0, 0, 0, 2]])))

            if verbose:
                print( '***' )
            m0 = m1_new.copy()
            if verbose:
                print( 'm0:     \n',m0 )
            m1 = ccl2d(d2,(1,2)); # m1[np.where(m1 > 0)] += marker_base; marker_base = np.max(m1)
            if verbose:
                print( 'm1:     \n',m1 )
                print( '***' )
            m0_new,m1_new,m0_eol,translation01,translation11\
                = ccl_relabel2(m0,m1)
            translation01_1 = translation01
            if verbose:
                print( 'm0:     \n',m0 )
                print( 'm1:     \n',m1 )
                print( 'm0_new: \n',m0_new )
                print( 'm1_new: \n',m1_new )
                print( 'm0_eol: \n',m0_eol )
                print( 'xl_01 : \n',translation01 )
                print( 'xl_11 : \n',translation11 )
                print( '******' )
            xl_012 = ccl_compose_translations(translation01_0,translation01_1)
            if verbose:
                print( 'xl_012   : \n',xl_012 )
            m0_orig_x = ccl_backsub(m0_orig,xl_012)
            if verbose:
                print( 'm0_orig_x:     \n',m0_orig_x )
                print( '******' )

            self.assertTrue(np.allclose(\
                                        np.array(\
                                                  [[0, 0, 4, 0, 0, 0],
                                                   [0, 0, 4, 0, 0, 0],
                                                   [0, 0, 4, 0, 0, 0],
                                                   [0, 0, 4, 0, 0, 0],
                                                   [0, 0, 4, 0, 0, 0]])
                                        ,m0_orig_x))

# https://stackoverflow.com/questions/46441893/connected-component-labeling-in-python
# For fun viz.
def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    # cv2.imshow('labeled.png', labeled_img); cv2.waitKey()
    return labeled_img

            
if __name__ == '__main__':

    if True:
        unittest.main()

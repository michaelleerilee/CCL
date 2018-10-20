#!/usr/bin/env python

import cv2
import numpy as np
from ccl2d import ccl2d

def ccl_backsub(m,translations):
    if translations is None:
        # print 'ccl_backsub: copy'
        mx = m.copy()
    else:
        # print 'ccl_backsub: translate'
        # mx = np.zeros(m.shape)
        mx = m.copy()
        m_unique = np.unique(m)
        for im in m_unique:
            for x in translations:
                if im in x[0]:
                    # print 'ccl_backsub: translate subs',x
                    #if im == 20:
                    #    print mx[40:50,320:330]
                    mx[np.where(m == im)] = x[1]
                    #if im == 20:
                    #    print mx[40:50,320:330]
    #print 'cc100'
    #print mx[40:50,320:330]
    return mx.copy()

def ccl_compose_translations(x0,x1):
    print 'b99: x0:  ',x0
    print 'b99: x1:  ',x1
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
            # i doesn't map, so add directly
            x01.append(i)
        
    for j in x1:
        j_found = False
        for jelt in j[0]:
            for ij in x01:
                j_found = j_found or jelt in ij[0]
        if not j_found:
            # i doesn't map, so add directly
            x01.append(j)
           
    print 'b99: x01: ',x01
    return x01
                

def ccl_relabel2(m0,m1,verbose=False):

    ## m0[np.where(m0 > 0)] += marker_base
    ## marker_base += np.max(m0)
    marker_base = np.max(m0)
    marker_base_0 = marker_base
    
    # print '00-markers-shape: ',m0.shape
    # print '00-markers-type:  ',type(m0)
    # print '00-markers-dtype: ',m0.dtype
    # print '00-markers-mnmx:  ',np.min(m0[np.where(m0 > 0)])\
    #     ,np.max(m0[np.where(m0 > 0)])
    # print '00-markers:       ',m0
    # print '00-markers-1:     ',m0.astype(np.float)/np.amax(m0)

    m1[np.where(m1 > 0)] += marker_base
    marker_base += np.max(m1)
    # print '01-markers-shape: ',m1.shape
    # print '01-markers-type:  ',type(m1)
    # print '01-markers-dtype: ',m1.dtype
    # print '01-markers-mnmx:  ',np.min(m1[np.where(m1 > 0)])\
    #   ,np.max(m1[np.where(m1 > 0)])
    # print '01-markers:       ',m1
    # print '01-markers-1:     ',m1.astype(np.float)/np.amax(m1)

    # m1[:,:] = m0[:,:]
    # markers_tmp = np.zeros(m0.shape,dtype=np.float)
    # markers_tmp[:,:] = markers

    # When starting use 01 as the base, later, just carry it through.
    # marker_max = np.max(m1)
    
    markers_sum = np.zeros(m0.shape,dtype=np.uint8)
    tmp = np.zeros(m0.shape,dtype=np.float)

    # tmp[:,:] = m0+m1
    tmp[:,:] = m1
    tmp_mx = np.max(tmp)
    if tmp_mx == 0:
        tmp_mx = 1
    tmp[:,:] = tmp/tmp_mx
    # print 'max(tmp): ',np.max(tmp)
    itmp = np.zeros(m0.shape,dtype=np.uint8)
    itmp[np.where(tmp > 0)] = 255
    # print 'max(itmp): ',np.max(itmp)
    # cv2.imshow('itmp',itmp); cv2.waitKey(0); cv2.destroyAllWindows()
    # tmp = m0
    # print 'max(tmp): ',np.max(tmp)
    # markers_sum[:,:] = 255*(tmp/np.max(tmp))
    markers_sum[:,:] = itmp
    # print 'max(markers_sum): ',np.max(markers_sum)

    # cv2.imshow('markers sum',markers_sum); cv2.waitKey(0); cv2.destroyAllWindows()
    
    ret, thresh = cv2.threshold(markers_sum\
                                ,0,np.max(markers_sum)\
                                ,cv2.THRESH_BINARY)

#                                ,1,np.max(markers_sum)\

    # cv2.imshow('thresh',thresh.astype(np.float)/np.amax(thresh)); cv2.waitKey(0); cv2.destroyAllWindows()
    
    # ret, markers01 = cv2.connectedComponents(thresh)
    markers01 = ccl2d(thresh,(254,255))

    # cv2.imshow('markers01',markers01.astype(np.float)/np.amax(markers01)); cv2.waitKey(0); cv2.destroyAllWindows()

    idx_00_01 = np.where((thresh == 255) & (m0 != 0))
    # print 'idx_00_01: ',idx_00_01

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
    
    # if verbose:
    if False:
        print 'id_00_01: ',id_00_01
            
    # if verbose:
    if False:
        print 'id_01_01: ',id_01_01
        print 'np.unique(m0): ',np.unique(m0)

    # Detect forking
    marker_fork_equivalence_sets = []
    # Go over the map from 0 to 01

    if False:
        print 'len(id_00_01): ',len(id_00_01)

    # r0 = id_00_01[i,1]

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

    done = False
    while not done:
        # print 'rs: ',rs
        rs_tmp = []
        done = True
        for i in range(len(rs)):
            # print '0 rs_tmp: ',rs_tmp
            ri = rs[i][0]
            si = rs[i][1]
            for j in range(i+1,len(rs)):
                rj = rs[j][0]
                sj = rs[j][1]
                # print '100: ',ri,rj,ri & rj,ri & rj != set([])
                if ri & rj != set([]):
                    # print '101:'
                    done = False
                    ri = ri | rj
                    si = si | sj
            for j in range(i+1,len(rs)):
                rj = rs[j][0]
                sj = rs[j][1]
                # print '110: ',si,sj,si & sj,si & sj != set([])
                if si & sj != set([]):
                    # print '111:'
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
            # print '2 rs_tmp: ',rs_tmp
        rs = rs_tmp

    marker_fork_equivalence_sets = rs
                    
    # i = 0
    # while i < len(id_00_01):
    #     print 'i:  ',i
    #     # Get a target from 01
    #     r0 = id_00_01[i][1]
    #     # print 'r0: ',r0
    #     r = []
    #     for j in range(i,len(id_00_01)):
    #         # print 's0 comparing: ',id_00_01[j][1],' to r0=',r0
    #         # Have we
    #         if (r0 == id_00_01[j][1]):
    #             print 'appending r: ',id_00_01[j][0]
    #             if id_00_01[j][0] not in r:
    #                 r.append(id_00_01[j][0])
    #     if r != []:
    #         s=[]
    #         for j in range(len(id_01_01)):
    #             if r0 == id_01_01[j][1]:
    #                 print 'appending s: ',id_01_01[j][0]
    #                 if id_01_01[j][0] not in s:
    #                     s.append(id_01_01[j][0])
    #             
    #         if(s != []):
    #             r.sort(); s.sort();
    #             marker_fork_equivalence_sets.append([r,s])
    #     i += 1

    if False:
        print 'marker_fork_equiv: ',marker_fork_equivalence_sets

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

        # print 'i,m0s,m1s,marker: ',i,m0s,m1s,marker_current
        
        marker_current += 1

    marker_base = marker_current
    
    # print 'equivalence sets k= ',len(marker_fork_equivalence_sets)

    k = 0
    for i in np.unique(m1):
        if i not in relabeled5:
            m1_new[np.where(m1 == i)] = i
            k = k + 1
    # print 'independent sets in m1 k= ',k

    k = 0
    for i in np.unique(m0):
        if i not in relabeled4:
            m0_new[np.where(m0 == i)] = i
            k = k + 1
    # print 'independent sets in m0 k= ',k
    
    m0_new_unique = np.unique(m0_new)
    # print 'unique(m0_new): ',m0_new_unique
    
    m1_new_unique = np.unique(m1_new)
    # print 'unique(m1_new): ',m1_new_unique

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
    # print 'unique(m0_new): ',m0_new_unique
    
    m1_new_unique = np.unique(m1_new)
    # print 'unique(m1_new): ',m1_new_unique
    
    m0_eol = []
    for i in range(1,len(m0_new_unique)):
        if m0_new_unique[i] not in m1_new_unique:
            m0_eol.append(m0_new_unique[i])

    # eol means that it has no entry in the translation01
    # For cases where we are tracking more than one previous history item,
    # we can remove translation01 entries that are eol at t-1.
    # We can use a translation01 table to go back in time to relabel the
    # histories if needed.

    # print 'm0_eol: ',m0_eol
    # print 'xlat:   ',translation01

    # translation01 maps 4's ids to the reconciled 5. Both 4_new and 5_new have the
    # New ids. I guess we could leave 4 alone, and just populate the translation01 table--
    # but then figuring out what's eol would be a little more involved.
    
    #while True:
    #    delta = np.abs(m1 - m0)
    #    m1a = np.full_like(m1,np.nan)
    #    id = []
    #    for i in np.where(delta > 0):
    #        id.append([m0[i],m1[i]])
    
    return (m0_new,m1_new,m0_eol,translation01)

class ccl_marker_stack(object):
    def __init__(self):
        self.m_results            = []
        self.m_results_translated = []
        self.m_ages               = {}
        self.translations         = []

    def make_slice_from(self,data,data_threshold_mnmx,graph=False):
        ### There's a bug here. Some blobs are not correctly renamed.
        m1 = ccl2d(data,data_threshold_mnmx,graph=graph)
        if self.m_results == []:
            self.m_results = [[m1[:,:],[]]]
            m0_new        = []
            m1_new        = m1[:,:]
            m0_eol        = []
            translation01 = []
        else:
            m0 = self.m_results[-1][0]
            m0_new,m1_new,m0_eol,translation01\
                = ccl_relabel2(m0,m1)
            self.m_results.append([m1_new[:,:],translation01[:]])

        return (m0_new,m1_new,m0_eol,translation01)

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

    def copy_of_translations(self):
        return self.translations

    def copy_of_translations_at(self,idx):
        return self.translations[idx]
    
    def copy_of_ages(self):
        return self.m_ages.copy()

    def copy_of_results(self):
        return self.m_results.copy()

    def copy_of_translated_results(self):
        return self.m_results_translated.copy()

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
    

if __name__ == '__main__':
    print 'ccl_relabel2.py start'


    if True:
        cm        = np.zeros((256,1,3),np.uint8)
        cm[:,0,0] = 255-np.arange(256)
        cm[:,0,1] = (np.arange(256)*(255-np.arange(256)))/255
        cm[:,0,2] = np.arange(256)
        cm[0,0,:] = 0
        
    if True:
        from Krige import DataField as df

        if False:
            src_files=[ 'MOD08_D3.A2015304.061.2017323113710.hdf'\
                        ,'MOD08_D3.A2015305.061.2017323113224.hdf'\
                        ,'MOD08_D3.A2015306.061.2017323115110.hdf'\
            ]

        if False:
            src_files=[ 'MOD08_D3.A2015306.061.2017323115110.hdf'\
                        ,'MOD08_D3.A2015305.061.2017323113224.hdf'\
                        ,'MOD08_D3.A2015304.061.2017323113710.hdf'\
            ]

        if False:
            src_files=[ 'MOD08_D3.A2015304.061.2017323113710.hdf'\
                        ,'MOD08_D3.A2015305.061.2017323113224.hdf'\
                        ,'MOD08_D3.A2015306.061.2017323115110.hdf'\
                        ,'MOD08_D3.A2015306.061.2017323115110.hdf'\
                        ,'MOD08_D3.A2015305.061.2017323113224.hdf'\
                        ,'MOD08_D3.A2015304.061.2017323113710.hdf'\
        ]

        if False:
            src_files=[ 'MOD08_D3.A2015304.061.2017323113710.hdf'\
                        ,'MOD08_D3.A2015305.061.2017323113224.hdf'\
                        ,'MOD08_D3.A2015306.061.2017323115110.hdf'\
                        ,'MOD08_D3.A2015306.061.2017323115110.hdf'\
                        ,'MOD08_D3.A2015305.061.2017323113224.hdf'\
                        ,'MOD08_D3.A2015304.061.2017323113710.hdf'\
        ]

        if True:
            src_files=[ \
                        'MOD08_D3.A2015305.061.2017323113224.hdf'\
                        ,'MOD08_D3.A2015306.061.2017323115110.hdf'\
                        ,'MOD08_D3.A2015306.061.2017323115110.hdf'\
                        ,'MOD08_D3.A2015306.061.2017323115110.hdf'\
                        ,'MOD08_D3.A2015306.061.2017323115110.hdf'\
                        ,'MOD08_D3.A2015306.061.2017323115110.hdf'\
                        ,'MOD08_D3.A2015305.061.2017323113224.hdf'\
                        ,'MOD08_D3.A2015304.061.2017323113710.hdf'\
        ]
            
            
        # src_files=[ 'MOD08_D3.A2015304.061.2017323113710.hdf'\
        #             ,'MOD08_D3.A2015305.061.2017323113224.hdf'\
        # ]
        
        src_dir    = '/home/mrilee/data/NOGGIN/MODIS-61/'
        field_name = 'Atmospheric_Water_Vapor_Mean'
    
        data_thresh_mnmx = (6.892,8)

        marker_stack = ccl_marker_stack()
        
        i = 0
        obj = df.DataField(\
                           datafilename=src_files[i]\
                           ,datafieldname=field_name\
                           ,srcdirname=src_dir\
        )
        obj.data[:,0:269] = 0
        # m0 = ccl2d(obj.data,data_thresh_mnmx,graph=False)
        print 'data-mnmx: ',np.amin(obj.data),np.amax(obj.data)
        i += 1

        marker_stack.make_slice_from(obj.data,data_thresh_mnmx)

        # m_results = [[m0[:,:],[]]]
        done = False
        while not done:
            obj = df.DataField(\
                                datafilename=src_files[i]\
                                ,datafieldname=field_name\
                                ,srcdirname=src_dir\
            )
            obj.data[:,0:269] = 0
            print 'data-mnmx: ',np.amin(obj.data),np.amax(obj.data)
            # m1 = ccl2d(obj.data,data_thresh_mnmx,graph=False)
            i += 1

            m0_new,m1_new,m0_eol,translation01\
                = marker_stack.make_slice_from(obj.data,data_thresh_mnmx)
            
            # m0_new,m1_new,m0_eol,translation01\
            #    = ccl_relabel2(m0,m1)

            # m_results.append([m1_new[:,:],translation01[:]])

            # m0 = m1_new[:,:]
            
            done = (i >= len(src_files))
            
        print 'len(m_results): ',marker_stack.len()

        if False:
            for i in range(len(m_results)):
                print 'i,x: ',i,m_results[i][1]

        # # age...
        # m=m_results[-1][0].copy()
        # m_unique = np.unique(m[np.where(m > 0)])
        # m_ages = {}
        # for i in m_unique:
        #     m_ages[i]=1
        # m_results_translated = [m]
        # x                    =  m_results[-1][1][:]
        # print 'x: ',x
        # for i in range(len(m_results)-2,-1,-1):
        #     # print 'i: ',i
        #     # # print 'mx: ',m_results[i][1]
        #     # print 'x0: ',x
        #     # print 'm_results[i][0].shape:       ',m_results[i][0].shape
        #     # print 'm_results[i][0][...]:        \n',m_results[i][0][40:50,320:330]
        #     m0n = ccl_backsub(             m_results[i][0],x)
        #     # print 'm0n.shape:                   ',m0n.shape
        #     # print 'm0n[...]:                    \n',m0n[40:50,320:330]
        # 
        #     #for i in x:
        #     #    for j in x[0]:
        #     #        m_ages[-1]...
        # 
        #     
        #     m0n_unique = np.unique(m0n[np.where(m0n>0)])
        #     for im in m0n_unique:
        #         if im in m_ages.keys():
        #             m_ages[im] = m_ages[im] + 1
        #         else:
        #             m_ages[im] = 1
        #                     
        #     if False:
        #         a = m_results[i][0].copy()
        #         a_mx = np.amax(a)
        #         b = m0n.copy()
        #         b_mx = np.amax(b)
        #         norm = max(a_mx,b_mx)
        #         ai = np.zeros(a.shape,dtype=np.uint8)
        #         ai[:,:] = 255*(a.astype(np.float)/norm)
        #         bi = np.zeros(b.shape,dtype=np.uint8)
        #         bi[:,:] = 255*(b.astype(np.float)/norm)
        #         ai = cv2.applyColorMap(ai,cm)
        #         bi = cv2.applyColorMap(bi,cm)
        #         cv2.imshow('a',ai)
        #         cv2.imshow('b',bi)
        #         cv2.waitKey(0); cv2.destroyAllWindows()
        #         
        #     x   = ccl_compose_translations(m_results[i][1],x)[:]
        #     # print 'x1: ',x
        #     # print 'delta: ',np.amin(m0n-m_results[i][0]),np.amax(m0n-m_results[i][0])
        #     m_results_translated.append(m0n.copy())
        # m_results_translated.reverse()

        m_results_translated = marker_stack.resolve_labels_across_stack()
        
        i = 0
        for imr in marker_stack.m_results_translated:
            # print 'i.shape: ',i[0].shape
            print 'i.mnmx:  ',i,np.amin(imr),np.amax(imr)

        print 'len(m_results_translated): ',marker_stack.len_translated()

        if True:
            print 'a900: ',len(marker_stack.copy_of_translations())
            for k in range(marker_stack.len()):
                print 'k,x: ',k,marker_stack.copy_of_translations_at(k)
            
        if True:
            np.set_printoptions(threshold=4000,linewidth=400 )
            k = 0
            # for imr in m_results_translated:
            for imr in marker_stack.m_results_translated:
                dlat =  60
                dlon = dlat
                lat0 = 180 - dlat
                lon0 = 360 - dlon
                print 'k, imr: ',k,'\n',np.array_str(imr[lat0:lat0+dlat,lon0:lon0+dlon])
                k += 1
            np.set_printoptions(threshold=1000,linewidth=75)

        # Print ages
        if True:
            np.set_printoptions(threshold=4000,linewidth=400 )
            im = 0
            for im in range(marker_stack.len_translated()):
                # imr = np.zeros(m_results_translated[im].shape)
                # imr = np.zeros(marker_stack.shape_translated())
                # for k in m_ages.keys():
                #     imr[np.where(m_results_translated[im] == k)] = m_ages[k]
                imr = marker_stack.copy_of_ages_at(im)
                dlat =  60
                dlon = dlat
                lat0 = 180 - dlat
                lon0 = 360 - dlon
                print 'i,m_age: ',im,'\n',np.array_str(111*imr[lat0:lat0+dlat,lon0:lon0+dlon])
                im += 1
            np.set_printoptions(threshold=1000,linewidth=75)
            
        if True:
            norms = []
            for imr in marker_stack.m_results_translated:
                norms.append(np.amax(imr))
            norm = max(norms)
            if norm == 0:
                norm = 1
            norm = norm/255.0
            # norm = 1
            i=0
            for imr in marker_stack.m_results_translated:
                mi = np.zeros(imr.shape,dtype=np.uint8)
                mi[:,:] = (imr.astype(np.float)/norm)
                mi = cv2.applyColorMap(mi,cm)
                cv2.imshow(str(i),mi)
                print 'im i mnmx: ',i,np.amin(mi[np.where(mi > 0)]),np.amax(mi)
                i += 1
            cv2.waitKey(0); cv2.destroyAllWindows()

            
        if False:
            m0_new = marker_stack.m_results_translated[0]
            m1_new = marker_stack.m_results_translated[1]
        if False:
            m0i       = np.zeros(m0_new.shape,dtype=np.uint8)
            m0_norm   = np.amax(m0_new)
            m1i       = np.zeros(m1_new.shape,dtype=np.uint8)
            m1_norm   = np.amax(m0_new)
            norm      = max([m0_norm,m1_norm])
            if norm == 0:
                norm = 1
            m0i[:,:]  = 255*(m0_new.astype(np.float)/norm)
            m1i[:,:]  = 255*(m1_new.astype(np.float)/norm)
            
            m0i = cv2.applyColorMap(m0i,cm)
            m1i = cv2.applyColorMap(m1i,cm)

            print 'm0  mnmx: ',np.amin(m0[np.where(m0>0)]),np.amax(m0)
            print 'm1  mnmx: ',np.amin(m1[np.where(m1>0)]),np.amax(m1)
            print 'm0n mnmx: ',np.amin(m0_new[np.where(m0_new>0)]),np.amax(m0_new)
            print 'm1n mnmx: ',np.amin(m1_new[np.where(m1_new>0)]),np.amax(m1_new)
            
            if False:
                cv2.imshow('m0',m0_new.astype(np.float)/np.amax(m0_new)); cv2.waitKey(0); cv2.destroyAllWindows()
                cv2.imshow('m1',m1_new.astype(np.float)/np.amax(m1_new)); cv2.waitKey(0); cv2.destroyAllWindows()
            if True:
                print 'translation01: ',translation01
                cv2.imshow('m0',m0i); # cv2.waitKey(0); cv2.destroyAllWindows()
                cv2.imshow('m1',m1i); cv2.waitKey(0); cv2.destroyAllWindows()


    if False:
        print 100
        d0 = np.zeros((5,6))
        d0[0,1] = 0
        d0[0,2] = 0
        d0[1,2] = 2
        d0[2,0] = 0
        d0[2,5] = 0
        d0[0,4] = 2
        d0[4,0] = 0
        d0[4,5] = 0

        d1 = np.zeros((5,6))
        d1[0,2] = 0
        d1[1,2] = 2
        d1[3,3] = 0

        d2 = np.zeros((5,6))
        d2[1,2] = 2

        print '***'
        m0 = ccl2d(d0,(1,2))
        print 'm0:     \n',m0
        m1 = ccl2d(d1,(1,2))
        print 'm1:     \n',m1
        print '***'
        m0_new,m1_new,m0_eol,translation01\
            = ccl_relabel2(m0,m1)
        print 'm0:     \n',m0
        print 'm1:     \n',m1
        print 'm0_new: \n',m0_new
        print 'm1_new: \n',m1_new
        print 'm0_eol: \n',m0_eol
        print 'xl_   : \n',translation01

        print '***'
        m0 = m1_new.copy()
        print 'm0:     \n',m0
        m1 = ccl2d(d2,(1,2)); # m1[np.where(m1 > 0)] += marker_base; marker_base = np.max(m1)
        print 'm1:     \n',m1
        print '***'
        m0_new,m1_new,m0_eol,translation01\
            = ccl_relabel2(m0,m1)
        print 'm0:     \n',m0
        print 'm1:     \n',m1
        print 'm0_new: \n',m0_new
        print 'm1_new: \n',m1_new
        print 'm0_eol: \n',m0_eol
        print 'xl_   : \n',translation01

    if False:
        print 200
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

        print '***'
        m0 = ccl2d(d0,(1,2))
        m0_orig = m0.copy()
        print 'm0:     \n',m0
        m1 = ccl2d(d1,(1,2))
        print 'm1:     \n',m1
        print '***'
        m0_new,m1_new,m0_eol,translation01\
            = ccl_relabel2(m0,m1)
        translation01_0 = translation01
        m0_new_0 = m0_new
        print '***'
        print 'm0:     \n',m0
        print 'm1:     \n',m1
        print 'm0_new: \n',m0_new
        print 'm1_new: \n',m1_new
        print 'm0_eol: \n',m0_eol
        print 'xl_   : \n',translation01
        print '******'

        print '***'
        m0 = m1_new.copy()
        print 'm0:     \n',m0
        m1 = ccl2d(d2,(1,2)); # m1[np.where(m1 > 0)] += marker_base; marker_base = np.max(m1)
        print 'm1:     \n',m1
        print '***'
        m0_new,m1_new,m0_eol,translation01\
            = ccl_relabel2(m0,m1)
        translation01_1 = translation01
        print 'm0:     \n',m0
        print 'm1:     \n',m1
        print 'm0_new: \n',m0_new
        print 'm1_new: \n',m1_new
        print 'm0_eol: \n',m0_eol
        print 'xl_   : \n',translation01
        print '******'
        xl_012 = ccl_compose_translations(translation01_0,translation01_1)
        print 'xl_012   : \n',xl_012
        m0_orig_x = ccl_backsub(m0_orig,xl_012)
        print 'm0_orig_x:     \n',m0_orig_x
        print '******'



    if False:
        print 300
        d0 = np.zeros((5,6))
        
        d0[1,1] = 2
        d0[2,1] = 2
        d0[3,1] = 2
        d0[4,1] = 2
        d0[0,3] = 2
        d0[1,3] = 0
        d0[2,3] = 2
        d0[3,3] = 0
        d0[1,5] = 2
        d0[2,5] = 0
        d0[3,5] = 2
        
        d1 = np.zeros((5,6))
        d1[1,1] = 2
        d1[2,1] = 0
        d1[3,1] = 2
        d1[4,1] = 2
        d1[0,3] = 2
        d1[1,3] = 0
        d1[2,3] = 2
        d1[3,3] = 2
        d1[1,5] = 2
        d1[2,5] = 2
        d1[3,5] = 2

        d2 = np.zeros((5,6))
        d2[1,1] = 2
        d2[2,1] = 0
        d2[3,1] = 0
        d2[4,1] = 2
        d2[0,3] = 2
        d2[1,3] = 2
        d2[2,3] = 2
        d2[3,3] = 2
        d2[1,5] = 2
        d2[2,5] = 0
        d2[3,5] = 2
        

        print '***'
        m0 = ccl2d(d0,(1,2))
        m0_orig = m0.copy()
        print 'm0:     \n',m0
        m1 = ccl2d(d1,(1,2))
        print 'm1:     \n',m1
        print '***'
        m0_new,m1_new,m0_eol,translation01\
            = ccl_relabel2(m0,m1)
        translation01_0 = translation01
        m0_new_0 = m0_new
        print '***'
        print 'm0:     \n',m0
        print 'm1:     \n',m1
        print 'm0_new: \n',m0_new
        print 'm1_new: \n',m1_new
        print 'm0_eol: \n',m0_eol
        print 'xl_   : \n',translation01
        print '******'

        print '***'
        m0 = m1_new.copy()
        print 'm0:     \n',m0
        m1 = ccl2d(d2,(1,2)); # m1[np.where(m1 > 0)] += marker_base; marker_base = np.max(m1)
        print 'm1:     \n',m1
        print '***'
        m0_new,m1_new,m0_eol,translation01\
            = ccl_relabel2(m0,m1)
        translation01_1 = translation01
        print 'm0:     \n',m0
        print 'm1:     \n',m1
        print 'm0_new: \n',m0_new
        print 'm1_new: \n',m1_new
        print 'm0_eol: \n',m0_eol
        print 'xl_   : \n',translation01
        print '******'
        xl_012 = ccl_compose_translations(translation01_0,translation01_1)
        print 'xl_012   : \n',xl_012
        m0_orig_x = ccl_backsub(m0_orig,xl_012)
        print 'm0_orig_x:     \n',m0_orig_x
        print '******'

        
        
    print 'ccl_relabel2.py done'


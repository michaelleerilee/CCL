#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright © 2018 Michael Lee Rilee, mike@rilee.net, Rilee Systems Technologies LLC

Test using MODIS data.

For license information see the file LICENSE that should have accompanied this source.

"""

import cv2
import numpy as np
from ccl_marker_stack import ccl_marker_stack

if __name__ == '__main__':
    print 'ccl_test_on_modis.py'

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



    print 'done\nccl_test_on_modis.py'


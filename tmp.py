    m0[np.where(m0 > 0)] += marker_base
    marker_base += np.max(m0)
    print '00-markers-shape: ',m0.shape
    print '00-markers-type:  ',type(m0)
    print '00-markers-dtype: ',m0.dtype
    print '00-markers-mnmx:  ',np.min(m0[np.where(m0 > 0)])\
        ,np.max(m0[np.where(m0 > 0)])
    # print '00-markers:       ',m0
    # print '00-markers-1:     ',m0.astype(np.float)/np.amax(m0)

    m1[np.where(m1 > 0)] += marker_base
    marker_base += np.max(m1)
    print '01-markers-shape: ',m1.shape
    print '01-markers-type:  ',type(m1)
    print '01-markers-dtype: ',m1.dtype
    print '01-markers-mnmx:  ',np.min(m1[np.where(m1 > 0)])\
        ,np.max(m1[np.where(m1 > 0)])
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
    tmp[:,:] = tmp/np.max(tmp)
    print 'max(tmp): ',np.max(tmp)
    itmp = np.zeros(m0.shape,dtype=np.uint8)
    itmp[np.where(tmp > 0)] = 255
    print 'max(itmp): ',np.max(itmp)
    # cv2.imshow('itmp',itmp); cv2.waitKey(0); cv2.destroyAllWindows()
    # tmp = m0
    # print 'max(tmp): ',np.max(tmp)
    # markers_sum[:,:] = 255*(tmp/np.max(tmp))
    markers_sum[:,:] = itmp
    print 'max(markers_sum): ',np.max(markers_sum)

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
    
    id_00_01 = []
    for ij in range(len(idx_00_01[0])):
        i=idx_00_01[0][ij]
        j=idx_00_01[1][ij]
        r = m0[i,j]; s = markers01[i,j]
        rs = [r,s]
        # print '00 rs: ',rs
        if rs not in id_00_01:
            id_00_01.append(rs)
    print 'id_00_01: ',id_00_01
            
    idx_01_01 = np.where((thresh == 255) & (m1 != 0))
    id_01_01 = []
    for ij in range(len(idx_01_01[0])):
        i=idx_01_01[0][ij]
        j=idx_01_01[1][ij]
        r = m1[i,j]; s = markers01[i,j]
        rs = [r,s] 
        # print '01 rs: ',rs
        if rs not in id_01_01:
            id_01_01.append(rs)
    print 'id_01_01: ',id_01_01

    print 'np.unique(m0): ',np.unique(m0)

    # Detect forking
    marker_fork_equivalence_sets = []
    r_checked = []
    # Go over the map from 4 to 01
    for i in range(len(id_00_01)):
    # for i in [225]:
    # for i in [0]:
    # for i in range(200):
        # print 'i:  ',i
        # Get a target from 01
        r0 = id_00_01[i][1]
        # print 'r0: ',r0
        r = []
        for j in range(i,len(id_00_01)):
            # print 's0 comparing: ',id_00_01[j][1],' to r0=',r0
            # Have we
            if (r0 == id_00_01[j][1]):
                # print 'appending r: ',id_00_01[j][0]
                if id_00_01[j][0] not in r_checked:
                    r_checked.append(id_00_01[j][0])
                    r.append(id_00_01[j][0])
        if r != []:
            s = []
            for j in range(len(id_01_01)):
                if r0 == id_01_01[j][1]:
                    s.append(id_01_01[j][0])
            if(s != []):
                marker_fork_equivalence_sets.append([r,s])
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
        marker_current += 1

    marker_base = marker_current
    
    print 'equivalence sets k= ',len(marker_fork_equivalence_sets)

    k = 0
    for i in np.unique(m1):
        if i not in relabeled5:
            m1_new[np.where(m1 == i)] = i
            k = k + 1
    print 'independent sets in m1 k= ',k

    k = 0
    for i in np.unique(m0):
        if i not in relabeled4:
            m0_new[np.where(m0 == i)] = i
            k = k + 1
    print 'independent sets in m0 k= ',k
    
    m0_new_unique = np.unique(m0_new)
    print 'unique(m0_new): ',m0_new_unique
    
    m1_new_unique = np.unique(m1_new)
    print 'unique(m1_new): ',m1_new_unique

    m0_eol = []
    for i in range(1,len(m0_new_unique)):
        if m0_new_unique[i] not in m1_new_unique:
            m0_eol.append(m0_new_unique[i])

    # eol means that it has no entry in the translation01
    # For cases where we are tracking more than one previous history item,
    # we can remove translation01 entries that are eol at t-1.
    # We can use a translation01 table to go back in time to relabel the
    # histories if needed.

    print 'm0_eol: ',m0_eol
    print 'xlat:   ',translation01

    # translation01 maps 4's ids to the reconciled 5. Both 4_new and 5_new have the
    # New ids. I guess we could leave 4 alone, and just populate the translation01 table--
    # but then figuring out what's eol would be a little more involved.
    
    #while True:
    #    delta = np.abs(m1 - m0)
    #    m1a = np.full_like(m1,np.nan)
    #    id = []
    #    for i in np.where(delta > 0):
    #        id.append([m0[i],m1[i]])

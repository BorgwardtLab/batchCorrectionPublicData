from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
import anndata as an
import os
from collections import Counter


# Data loading and annotation
def getZHoverview(metadata_cat,obs_eval):
    'Return a summary of the metadata categories for each ZH'

    n_samples = []
    n_batch = []
    n_cluster = len(np.unique(metadata_cat[obs_eval]))
    for i in range(0, n_cluster):
        zh = str(i)
        metasub = metadata_cat[metadata_cat['ZeroHop'] == zh]
        n_batch.append(len(np.unique(metasub['gse'])))
        n_samples.append(len(metasub))

        if i == 0:
            df_meta = metasub.iloc[0, :].to_frame().transpose()
        else:
            df_meta = pd.concat([df_meta, metasub.iloc[0, :].to_frame().transpose()], axis=0)

    df_meta.set_index('ZeroHop', drop=True, inplace=True)
    df_meta.drop('gse', axis=1, inplace=True)
    df_meta['n_samples'] = n_samples
    df_meta['n_gse'] = n_batch
    return df_meta
def getZeroHops_andCheck(df_PAO1, metadata_samples_pao1):

    # Remove gse and gsm
    metadata_hops_PAO1 = metadata_samples_pao1.drop(['gse'],axis = 1)
    zeroHops_PAO1 = getZeroHops(metadata_hops_PAO1)
    metadata_samples_pao1['ZeroHop'] = zeroHops_PAO1


    # count gses and exclude clusters with less than 2 gses
    Counter(zeroHops_PAO1)
    excl_ZH_PAO1 = []
    excl_gsm_ZH = []
    for zh in np.unique(zeroHops_PAO1):
        data_sub = metadata_samples_pao1[metadata_samples_pao1['ZeroHop']==zh]
        n_gse = len(data_sub.gse.unique())
        if n_gse<2:
            excl_ZH_PAO1.append(zh)
            excl_gsm_ZH += list(data_sub.index)
    df_PAO1.drop(excl_gsm_ZH, inplace = True)
    metadata_samples_pao1.drop(excl_gsm_ZH, inplace = True)

    zh_num = np.unique(metadata_samples_pao1['ZeroHop'])
    zh_str =[str(i) for i,zh in enumerate(zh_num)]
    zip_iterator = zip(zh_num, zh_str)
    dict_sh_PAO1 = dict(zip_iterator)
    zh_str_PAO1 = [dict_sh_PAO1[this_zh_num] for this_zh_num in metadata_samples_pao1['ZeroHop']]
    metadata_samples_pao1['ZeroHop'] = zh_str_PAO1
    return metadata_samples_pao1, df_PAO1
def getZeroHops(metadata_hops):
    if 'gse' in metadata_hops.columns:
        metadata_hops.drop('gse', axis=1, inplace=True)

    if 'gsm' in metadata_hops.columns:
        metadata_hops.drop('gsm', axis=1, inplace=True)

    metadata_hops_cat = metadata_hops.copy()

    for c in metadata_hops.columns:
        column = metadata_hops[c]
        col_dict = dict(zip(column.unique(), list(range(len(column.unique())))))
        column_cat = [col_dict[item] for item in column]
        metadata_hops_cat[c] = column_cat

    distances = (pairwise_distances(metadata_hops_cat, metadata_hops_cat, metric='hamming') *
                 metadata_hops_cat.shape[
                     1]).astype(int)

    # new zero hops (79)
    zero_hops = []
    for d in distances:
        zero_hops.append(tuple(metadata_hops.loc[d == 0].index.values))
    zero_hops = set(zero_hops)
    zero_hops = [list(z) for z in zero_hops]

    # Add as new metadata information
    thisZerohop = []
    for s in metadata_hops.index:
        for iz in range(0, len(zero_hops)):
            if s in zero_hops[iz]:
                thisZerohop.append(iz)
    return thisZerohop
def checkAndRefineZeroHops(data_path, metadata, metadata_cat, df_meta,obs_eval,media_filename):

    # load medium definitions
    media_df = pd.read_csv(os.path.join(data_path, media_filename))
    media_df = media_df[:92]
    new_ind = []
    for m in media_df['Medium in data base']:
        new_ind.append(m.split(':')[0])
    media_df['medCombi'] = new_ind
    media_df.set_index('medCombi', inplace=True)

    # Get the combined names of media and consistent supplements
    combi_med = []
    for i in metadata.index:
        this_medium = metadata.loc[i, 'medium']
        this_suppl = metadata.loc[i, 'consistant_supplements']

        try:
            np.isnan(this_suppl)
            combi_med.append(this_medium)
        except:
            combi_med.append(this_medium + this_suppl)
    metadata['combiMed'] = combi_med


    # get those zeroHops with more than nine gses and try to subset
    zh_large = df_meta[df_meta.n_gse>6].index
    counter_zh_new = len(df_meta)
    for zh in zh_large:
        df_sub = metadata[metadata['ZeroHop']==zh]

        if df_meta.loc[zh, 'MediumCoarse'] == 'rich':

            richness = []
            for s in df_sub.index:
                med = df_sub.loc[s,'combiMed']
                if 'AGSY\xa0' == med:
                    med = 'AGSYxa0'
                if med == 'LB1\u2009mM glutamine':
                    med = 'LB1u2009mM glutamine'
                if med == 'ABT\xa0minimal medium0.5':
                    med = 'ABTxa0minimal medium0.5'
                if med == 'ABT\xa0minimal medium0.5% glucose':
                    med = 'ABTxa0minimal medium0.5% glucose'

                if media_df.loc[med,'Richness score']<0:
                    sc = 'lessRich'
                elif media_df.loc[med,'Richness score']>0:
                    sc = 'moreRich'
                else:
                    sc = 'rich0'
                richness.append(sc)
            df_sub.loc[:,'richness'] = richness

            # Check that 3 gse remain:
            n_gse = []
            for sc in np.unique(richness):
                df_sub_sub = df_sub[df_sub.richness == sc]
                n_gse.append(len(np.unique(df_sub_sub.gse)))

            if any(np.array(n_gse) < 2):

                # coarsen further
                print('Coarsen rich for zh'+zh)
            else:

                # assign new zh
                for i_sc, sc in enumerate(np.unique(richness)):
                    df_sub_sub = df_sub[df_sub.richness == sc]
                    metadata.loc[df_sub_sub.index, 'MediumCoarse'] = sc
                    metadata_cat.loc[df_sub_sub.index, 'MediumCoarse'] = sc

                    if i_sc > 0:
                        metadata_cat.loc[df_sub_sub.index, 'ZeroHop'] = str(counter_zh_new)
                        metadata.loc[df_sub_sub.index, 'ZeroHop'] = str(counter_zh_new)
                        counter_zh_new += 1

        elif df_meta.loc[zh, 'MediumCoarse'] == 'defined':
            definedness = []
            for s in df_sub.index:
                med = df_sub.loc[s, 'combiMed']
                if 'AGSY\xa0' == med:
                    med = 'AGSYxa0'
                if med == 'LB1\u2009mM glutamine':
                    med = 'LB1u2009mM glutamine'
                if med == 'ABT\xa0minimal medium0.5':
                    med = 'ABTxa0minimal medium0.5'
                if med == 'ABT\xa0minimal medium0.5% glucose':
                    med = 'ABTxa0minimal medium0.5% glucose'

                if media_df.loc[med, 'Gluconeogenic'] =='yes':
                    sc = 'Gluconeogenic'
                elif media_df.loc[med, 'Gluconeogenic'] =='no':
                    sc = 'notGluconeogenic'
                elif media_df.loc[med, 'Gluconeogenic'] == 'both':
                    sc = 'both'
                else:
                    sc = 'notDefined'
                definedness.append(sc)
            df_sub.loc[:,'definedness'] = definedness

            # Check that 3 gse remain:
            n_gse = []
            for sc in np.unique(definedness):
                df_sub_sub = df_sub[df_sub.definedness== sc]
                n_gse.append(len(np.unique(df_sub_sub.gse)))

            if any(np.array(n_gse) < 2):
                # coarsen further
                print('Coarsen defined')
            else:

                # assign new zh
                for i_sc,sc in enumerate(np.unique(definedness)):
                    df_sub_sub = df_sub[df_sub.definedness == sc]
                    metadata.loc[df_sub_sub.index, 'MediumCoarse'] = sc
                    metadata_cat.loc[df_sub_sub.index, 'MediumCoarse'] = sc
                    if i_sc > 0:
                        metadata_cat.loc[df_sub_sub.index, 'ZeroHop'] = str(counter_zh_new)
                        metadata.loc[df_sub_sub.index, 'ZeroHop'] = str(counter_zh_new)
                        counter_zh_new += 1

    n_batch = []
    for zh in metadata['ZeroHop'].unique():
        metasub = metadata[metadata['ZeroHop']==zh]
        n_batch.append(len(np.unique(metasub['gse'])))
    n_obs =  len(np.unique(metadata[obs_eval]))
    n_cluster = n_obs


    # Annotate zero hop clusters again and redefine df_meta
    annot = []
    n_samples = []
    n_batch = []
    for i in range(0,n_cluster):
        zh = str(i)
        metasub = metadata_cat[metadata_cat['ZeroHop']==zh]
        n_batch.append(len(np.unique(metasub['gse'])))
        n_samples.append(len(metasub))

        if i ==0:
            df_meta = metasub.iloc[0,:].to_frame().transpose()
        else:
            df_meta =pd.concat([df_meta,metasub.iloc[0,:].to_frame().transpose()], axis = 0)

    df_meta.set_index('ZeroHop', drop=True, inplace = True)
    df_meta.drop('gse', axis = 1, inplace = True)
    df_meta['n_samples'] = n_samples
    df_meta['n_gse'] = n_batch

    return metadata_cat, df_meta
def coarsenMetadata(metadata):


    # More general meta data annotations for growth phase
    phase = []
    lag_phase = ['OD 0.080–0.090', 'OD 0.2', 'OD 0.15']
    exp_phase = ['OD 0.25', 'OD 0.3', 'mid-exponential-phase', 'OD 1.5', 'OD 0.5', '5h', 'exponential',
                 'OD 1.0', '4h', 'OD 0.4-0.45', 'OD 0.4',
                 'OD 0.7-0.8',
                 '2.5h', '3h', 'OD 0.6', 'OD 0.4-0.6', 'OD 2.0', 'OD 0.5-0.6', '6h',
                 'OD 0.8', '3-4h',
                 'OD  0.7-0.8',
                 'OD 1.4', 'OD 1.1', '2h',
                 'OD 0.4-0.5', 'OD 1.2', 'OD 1.6',
                 'late exponential', 'mid exponential', 'OD 0.4-0.5', 'exponential-phase, 5h', 'exponential-phase',
                 'OD 0.7']
    plateau_phase = [
        '18h', 'OD 3.0',
        '12h',
        'stationary phase', '72h', '28d', '48h', '20h',
        '7h', '8h', '24h', '9h', '16h', 'OD 3.5',
        'OD 9.0', '15h', '40h', '2 x 24h', '96h',
        'OD 2.5', 'OD 2.8', '84h', '5d', '3d', '6d',
        '52h', 'stationary-phase']

    # More general meta data annotations for culture type
    culture = []
    liquid = ['Liquid culture', 'Liquid culture, Reversed Osmosis']
    film = ['Biofilm culture, tube',
            'Biofilm culture, mouse', 'Biofilm culture, on cells', 'Biofilm culture, Reversed Osmosis',
            'Biofilm culture, on glass wool', 'Biofilm culture, plate', 'Biofilm culture, on Teflon',
            'Biofilm culture', 'Biofilm culture, drip flow', 'Biofilm culture, on slides',
            'Biofilm culture, in tube', 'Biofilm culture, flow-through']
    plate = ['Plate culture', 'Plate culture, for twitching']
    invivo = ['in-vivo, Mouse', 'in-vivo, mouse', 'in-vivo, lettuce', 'in-vivo, human', 'Plant culture']

    # More general metadata for temperature
    temperature = []
    rt = ['RT', '25', '22', '23']
    body_temp = ['Human Body temp', 'Mouse Body temp', '37', '35']
    surface = ['28', '30', '37/RT', '30/37']

    # More general metadata for oxygenation
    oxygenation = []
    arobic = [np.nan, '70%-80% humidity', 'aerobic', '20% oxygen', '5% CO2', 'humid']
    hypoxic = ['hermetically closed', 'high to low oxygen tension',
               'low to high oxygen tension', 'microaerobic', '0% oxygen',
               '0.4% oxygen', '2% oxygen', '<1% oxygen', 'anaerobic']

    # More general metadata for medium
    medium     = []
    rich = ['LB', 'PIA', 'PB', 'MH', 'BHI', '2YT', 'TSB', 'Nutrient Broth no. 2', 'MEM', 'LB dil', 'PY', 'R2B dil',
            'Mueller Hinton', 'serum-RPMI', 'TB', 'RPMI 1640', 'NY', 'Nematode Growth Medium', 'MHB',
            '0.1 x LB', 'King’s A', 'TSA', 'TY', '0.1 x TY', 'Medium C', 'LANS','AGSY\xa0']
    defined = ['MOPS', 'M9', 'M63', 'ABT', 'PPGAS', 'Minimal medium P', 'MMC', 'ABT\xa0minimal medium', 'PBM', 'BBM',
               'Minimal medium', 'AB minimal medium', 'Pseudomonas Basal Mineral media',
               'QSM minimal medium', 'M63 minimal media', 'Minimal Medium', '0.1 x TBS', 'BSM', 'CAA', 'PBS']


    in_vivo_like = ['SCFM', 'Synthetic CF sputum medium', 'CWE-mimic medium', 'ASMDM', 'artificial urine medium']
    in_vivo = ['Mouse', 'Human, CF Sputum', 'Plant, Xanthi', 'Plant, Samsun', 'Human, Burn Wound', 'Lettuce',
               'Human, burn wound', 'Human, CF sputum']
    poor = ['Pond Water', 'Tap Water', 'Water']
    # missing: 'Medium C','TY', 'LANS','AGSY\xa0', '0.1 x TY' - were assigned by sarah (pretty randomly...)


    ABs = ['0.1% DMF', '0.5 µg/mL AZM', '2 µg/mL AZM', '2 µg/ml AZM','1 ug/mL  BF8',
           'PAA', 'cefoxitin',
           'ceftazidime'
            , '1.0 μg/ml Ciprofloxacin',
           '10 μg/ml Tobramycin',
           '0.5 mM 7HI', '0.1% DMF',
            '8 µg/mL azithromycin',
           '2 µg/mL azithromycin', '0.25 µg/mL ceftamazidime',
           '0.04 µg/mL ciprofloxacin','200 µA DC',
           '500 µg/mL tobramycin', '5 µg/mL tobramycin',
            '0.5 µg/mL AZM', '2 µg/mL AZM',
           '0.15 µg/mL colistin',
           '1 mg/L ciprofloxacin', '2 µg/ml AZM',
           ' 0.1 mg/ml IAN',
           '2-AA',
           '125 μM of Protoanemonin', '12.5 µg/ml gentamicin',
           '12.5 µg/ml gentamicin, acoustic waves',
           'streptomyces 230 supernatant','100 µg/ml penicillin', '1 ug/mL  BF8',
           '30 ug/ml gentamicin', '25µg/ml piperacillin', '100 µM tetracycline', '1500 Units penicillin G/mL',
           '50 µM C30','150 µg/ml gentamicin']
    # Check for AB
    treatment = []



    for i in metadata.index:

        # Get growth phase
        try:
            exp_phase.index(metadata.loc[i, 'growth_phase_time_OD'])
            phase.append('Exp')
        except:
            try:
                lag_phase.index(metadata.loc[i, 'growth_phase_time_OD'])
                phase.append('Lag')
            except:
                try:
                    plateau_phase.index(metadata.loc[i, 'growth_phase_time_OD'])
                    phase.append('Plat')
                except:
                    phase.append('Exp')

        # get culture type
        try:
            liquid.index(metadata.loc[i, 'culture_type'])
            culture.append('liquid')
        except:
            try:
                film.index(metadata.loc[i, 'culture_type'])
                culture.append('film')
            except:
                try:
                    plate.index(metadata.loc[i, 'culture_type'])
                    culture.append('plate')
                except:
                    try:
                        invivo.index(metadata.loc[i, 'culture_type'])
                        culture.append('in vivo')
                    except:
                        culture.append('liquid')

        # get temperature
        try:
            rt.index(metadata.loc[i, 'temperature'])
            temperature.append('RT')
        except:
            try:
                body_temp.index(metadata.loc[i, 'temperature'])
                temperature.append('body_temp')
            except:
                try:
                    surface.index(metadata.loc[i, 'temperature'])
                    temperature.append('surface')
                except:
                    temperature.append('body_temp')

        # Get oxygenation
        try:
            hypoxic.index(metadata.loc[i, 'special_oxigenation_environment'])
            oxygenation.append('hypoxic')
        except:
            oxygenation.append('aerobic')


        # Get detailed Medium


        # Get medium
        try:
            defined.index(metadata.loc[i, 'medium'])
            # defined_add = str(media_df.loc[metadata.loc[i, 'combiMed']]['Gluconeogenic'])
            # if defined_add == 'no':
            #     defined_add_use = 'no'
            # else:
            #     defined_add_use = 'yes'



            medium.append('defined')

            # Check if consistent supplements make it rich
            # this_supplement = metadata.loc[i, 'consistant_supplements']
            # if this_supplement in makesRich:
            #     medium.append('rich')
            # else:
            #     medium.append('defined')


        except:
            try:
                rich.index(metadata.loc[i, 'medium'])
                # rich_add = str(media_df.loc[metadata.loc[i, 'combiMed']]['Richness score'])
                #
                # if float(rich_add)<0:
                #     rich_add_use = 'neg'
                # else:
                #     rich_add_use = 'pos'


                medium.append('rich')
            except:
                try:
                    in_vivo_like.index(metadata.loc[i, 'medium'])
                    medium.append('in_vivo_like')
                except:
                    try:
                        in_vivo.index(metadata.loc[i, 'medium'])
                        medium.append('in_vivo')
                    except:
                        try:
                            poor.index(metadata.loc[i, 'medium'])
                            medium.append('poor')
                        except:
                            medium.append(np.nan)

        # Get AB
        this_treatment  = metadata.loc[i, 'treatment_supplement']
        this_consistent = metadata.loc[i, 'consistant_supplements']
        if this_treatment in ABs:
            treatment.append('treat')
        else:
            if this_consistent in ABs:
                treatment.append('consistent')
            else:
                treatment.append('noABX')

    metadata['GrowthPhase'] = phase
    metadata['Culture_Coarse'] = culture
    metadata['Temperature_Coarse'] = temperature
    metadata['Oxygenation'] = oxygenation
    metadata['MediumCoarse'] = medium
    metadata['Antibiotic'] = treatment

    # Drop patients with missing information
    metadata.drop(list(metadata.index[metadata['MediumCoarse'].isna().values]), axis=0, inplace=True)

    # Some more corrections
    metadata.medium[metadata.medium == 'Human, Burn Wound'] = 'Human, burn wound'
    metadata.medium[metadata.medium == 'Minimal Medium'] = 'Minimal medium'
    metadata.temperature[metadata.temperature == 'Mouse  Body temp'] = 'Mouse Body temp'

    # Now check for replicas
    metadata_hops = metadata.copy()
    metadata_hops = metadata_hops.drop(columns=['title',
                                                'Replicas',
                                                'growth_phase_time_OD',
                                                'culture_type',
                                                'Reference',
                                                'To_plot_by',
                                                'Grouping_by_Replicas',
                                                'treatment duration',
                                                'Comments_Characteristic',
                                                'control',
                                                'Normalized_by',
                                                'Replicas',
                                                'temperature',
                                                'medium',
                                                'consistant_supplements',
                                                'treatment_supplement',
                                                'special_oxigenation_environment',
                                                'genotype', 'plasmids', 'gse'
                                                ])
    return metadata, metadata_hops
def removeRepeatedSamples(batchfield, data_use, metadata_hops, metadata_all):
    # Define cutoff for being identical
    diagonal = []
    cdist = pairwise_distances(data_use)
    for i in range(0, len(data_use)):
        diagonal.append(cdist[i, i])
    thresh = 1.05 * np.max(diagonal)

    drop_gsm = []
    data_use_standard = ((data_use.T - data_use.T.mean()) / data_use.T.std()).T
    genes = data_use_standard.columns
    samples = data_use_standard.index
    adata_new = an.AnnData(X=data_use_standard, obs=metadata_hops)
    data_use_standard = pd.DataFrame(data=adata_new.X, columns=genes, index=samples)

    cdist_stand = pairwise_distances(data_use_standard)
    cdist = pairwise_distances(data_use)
    for i, s in enumerate(list(metadata_hops.index)):

        this_batch = metadata_hops.loc[s, batchfield]
        ind_diff_samples = np.where(metadata_hops.index != s)[0]

        # check for zeros
        dist_diff_samples_stand = cdist_stand[i, ind_diff_samples]
        dist_diff_samples_normal = cdist[i, ind_diff_samples]

        ind_identical_samples_stand = ind_diff_samples[np.where(dist_diff_samples_stand < thresh)[0]]
        ind_identical_samples_normal = ind_diff_samples[np.where(dist_diff_samples_normal < thresh)[0]]
        ind_identical_samples = np.unique(list(ind_identical_samples_normal) + list(ind_identical_samples_stand))

        if len(ind_identical_samples) > 0:
            rep_gsms = list(metadata_hops.index[ind_identical_samples])
            for this_rep_gsm in rep_gsms:

                # Rename the gse and drop
                gse_2 = metadata_hops.loc[this_rep_gsm, batchfield]

                if gse_2 != this_batch:
                    all_gses = list(metadata_hops.gse.values)
                    all_gse_new = []
                    for gi in all_gses:
                        if gi != gse_2:
                            all_gse_new.append(gi)
                        else:
                            all_gse_new.append(this_batch)

                    metadata_hops.gse = all_gse_new
                    metadata_all.gse = all_gse_new

                # Drop replica if neither the gsm or repeat gsm is in list
                if len(ind_identical_samples) == 1:
                    if s not in drop_gsm:
                        drop_gsm.append(this_rep_gsm)
                else:
                    if s not in drop_gsm:
                        drop_gsm.append(this_rep_gsm)

    # Drop replica
    metadata_hops.drop(drop_gsm, inplace=True)
    metadata_all.drop(drop_gsm, inplace=True)
    data_use.drop(drop_gsm, inplace=True)
    return data_use, metadata_hops, metadata_all
def removeBatchesWithLimSamples(data_use, metadata_hops, metadata_all, n_min=2):

    excl_gsm = []
    excl_gse = []
    for this_gse in metadata_hops.gse.unique():
        data_sub = metadata_hops[metadata_hops.gse == this_gse]
        if data_sub.shape[0] < n_min:
            excl_gse.append(this_gse)
            excl_gsm += list(data_sub.index)

    metadata_hops.drop(excl_gsm, inplace=True)
    metadata_all.drop(excl_gsm, inplace=True)
    data_use.drop(excl_gsm, inplace=True)
    return data_use, metadata_hops, metadata_all
def removeZHwithFewBatches(data_use,metadata_hops,metadata_all):
    excl_gsm = []
    excl_zh = []
    for this_zh in metadata_hops.ZeroHop.unique():
        data_sub = metadata_hops[metadata_hops.ZeroHop == this_zh]
        n_gse = len(data_sub.gse.unique())
        if n_gse < 2:
            excl_zh.append(this_zh)
            excl_gsm += list(data_sub.index)
    metadata_hops.drop(excl_gsm, inplace=True)
    metadata_all.drop(excl_gsm, inplace=True)
    data_use.drop(excl_gsm, inplace=True)
    return data_use, metadata_hops, metadata_all
def renameZH(metadata_hops, metadata_all):
    zh_num = np.unique(metadata_hops['ZeroHop'])
    zh_str = [str(i) for i, zh in enumerate(zh_num)]
    zip_iterator = zip(zh_num, zh_str)
    dict_sh = dict(zip_iterator)
    zh_str_assign = [dict_sh[this_zh_num] for this_zh_num in metadata_hops['ZeroHop']]
    metadata_hops['ZeroHop'] = zh_str_assign
    metadata_all['ZeroHop'] = zh_str_assign
    return metadata_hops, metadata_all
def getZeroHopsArray(data, metadata, metadata_hops):
    metadata_hops_cat = metadata_hops.copy()
    for c in metadata_hops.columns:
        column = metadata_hops[c]
        col_dict = dict(zip(column.unique(), list(range(len(column.unique())))))
        column_cat = [col_dict[item] for item in column]
        metadata_hops_cat[c] = column_cat

    distances = (pairwise_distances(metadata_hops_cat, metadata_hops_cat, metric='hamming') * metadata_hops_cat.shape[
        1]).astype(int)

    # new zero hops (79)
    zero_hops = []
    for d in distances:
        zero_hops.append(tuple(metadata_hops.loc[d == 0].index.values))
    zero_hops = set(zero_hops)
    zero_hops = [list(z) for z in zero_hops]


    # Add as new metadata information
    thisZerohop = []
    for s in metadata_hops.index:
        for iz in range(0,len(zero_hops)):
            if s in zero_hops[iz]:
                thisZerohop.append(iz)
    metadata_hops['ZeroHop'] = thisZerohop


    # add back information on gses
    metadata_hops['gse'] = metadata.loc[metadata_hops.index]['gse'].values

    # get all metadata
    data_use     = data.loc[metadata_hops.index]
    metadata_all = metadata.loc[metadata_hops.index]
    metadata_all['ZeroHop'] = metadata_hops['ZeroHop']
    metadata_all['ZeroHop'] = metadata_all['ZeroHop'].astype(str)
    return data_use, metadata_all, metadata_hops
def getArrayData(data_path,excl2SampleBatches, data_filename, metadata_filename):

    data      = pd.read_csv(os.path.join(data_path,data_filename),sep='\t',header=0,index_col=0).transpose()
    metadata  = pd.read_csv(os.path.join(data_path,metadata_filename),header=0,index_col=0)
    data.sort_index(inplace=True)
    metadata.sort_index(inplace=True)
    assert np.array(data.index == metadata.index).all()

    # Coarsen metadata
    metadata, metadata_hops = coarsenMetadata(metadata)


    # Define Zero Hops
    data_use, metadata_all, metadata_hops = getZeroHopsArray(data, metadata, metadata_hops)


    # Check for identical samples uploaded multiple times:'GSE25945'='GSE23007';  'GSM637594'='GSM567672'
    batchfield = 'gse'
    data_use, metadata_hops, metadata_all = removeRepeatedSamples(batchfield, data_use,metadata_hops,metadata_all)


    # Filter out gse with less than X gsms:
    if excl2SampleBatches:
        data_use, metadata_hops, metadata_all = removeBatchesWithLimSamples(data_use, metadata_hops,metadata_all, n_min = 2)


    # Check Zerohops to cantain 2 batches
    data_use, metadata_hops, metadata_all = removeZHwithFewBatches(data_use,metadata_hops,metadata_all)

    # Check the number of ZHs - subdivide large Zerohops
    n_samples = []
    n_batch = []
    for i,zh in enumerate(metadata_hops['ZeroHop'].unique()):
        metasub = metadata_hops[metadata_hops['ZeroHop'] == zh]
        n_batch.append(len(np.unique(metasub['gse'])))
        n_samples.append(len(metasub))

        if i == 0:
            df_meta = metasub.iloc[0, :].to_frame().transpose()
        else:
            df_meta = pd.concat([df_meta, metasub.iloc[0, :].to_frame().transpose()], axis=0)

    df_meta.set_index('ZeroHop', drop=True, inplace=True)
    df_meta.drop('gse', axis=1, inplace=True)
    df_meta['n_samples'] = n_samples
    df_meta['n_gse'] = n_batch

    df_meta['n_samples']


    # Get ZH with 10 or more batches
    zh_highBatch = df_meta.index[df_meta['n_gse']>9].values
    for zh in zh_highBatch:
        metasub = metadata_all[(metadata_hops['ZeroHop'] == 22) | (metadata_hops['ZeroHop'] == 89)  ]


    # Resassign ZH to str
    metadata_hops, metadata_all = renameZH(metadata_hops, metadata_all)



    # Create adata
    allgenes = list(data.keys().values)
    adata = an.AnnData(X=pd.DataFrame(data=data_use, columns=allgenes), obs=metadata_all)

    return adata, metadata_all, data_use, metadata_hops


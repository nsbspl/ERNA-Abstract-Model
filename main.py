try:
    if INIT:
        print("Initialized.")
except:
    from pyNN.nest import *
    from pyNN.random import RandomDistribution
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import find_peaks
    import pickle

import pandas as pd

INIT=True

def OUProcess(tau,tstop,dt):
    rng = np.random.random
    ETA     = np.zeros(int(tstop/dt))
    inp_sig = np.copy(ETA)
    N_tau   = np.sqrt(2/tau)
    for i,E in enumerate(ETA[:-2]):
        inp_sig[i] = rng()
        Einf = tau*N_tau*inp_sig[i]/np.sqrt(dt)
        ETA[i + 1] = Einf + (ETA[0] - Einf)*np.exp(-dt/tau);
    t = np.arange(0,tstop,dt)
    return ETA, inp_sig, t

def runSim(F=100,reciprocity=(28,38),plot=False,bias=2.3,static=True):
    DBS_Freq = F
    DBS_strt = 2000

    DBS_arr  = np.arange(1000,2000,1000/DBS_Freq)
    DBS_times = np.repeat(DBS_arr,2)
    DBS_times += np.random.normal(0.1,1e-3,np.shape(DBS_times))
    DBS_times[1::2]+=0.1
    DBS_amp = np.zeros(np.shape(DBS_times))
    DBS_amp[::2]  = 300


    STN_freq = 39.9 # 27.6 # Luka 2021

    setup()
    DBS_pulses = SpikeSourceArray(spike_times=DBS_arr+2)
    DBS_pulse  = Population(1,DBS_pulses)
    DBS_anti   = StepCurrentSource(times=DBS_times,
                                   amplitudes=DBS_amp)
    weight_distr = RandomDistribution('normal', [1.6, 1e-4])
    delay_distr = RandomDistribution('uniform',(0.7,1.2))
    DBS_ortho=None
    if static:
        DBS_ortho  = StaticSynapse(weight=0.6,
                                   delay=0.5)
    else:
        DBS_ortho  = TsodyksMarkramSynapse(U=0.5,
                                           tau_rec=671,
                                           tau_facil=17,
                                           weight=1.2,
                                           delay=0.5)

    weight_distr = RandomDistribution('normal', [1, 1e-2])

    STN_type = SpikeSourcePoisson(duration=1000 if F>=80 else 2000
                                  , rate=STN_freq, start=0)
    STN_syn  = TsodyksMarkramSynapse(U=0.04, tau_rec=10.0,
                                tau_facil=10,
                                weight=weight_distr,
                                delay=RandomDistribution('normal',(2,1e-2)))

    GPe_type = IF_cond_exp(i_offset=bias
                           ,v_rest=-60
                           ,v_thresh=-40
                           ,tau_m=10
                           ,tau_refrac=1)

    weight_distr = RandomDistribution('normal', [0.1, 1e-4])
    GPe_syn = TsodyksMarkramSynapse(U=0.4159,
                                    tau_rec=30.0646,
                                    tau_facil=10.1047,
                                    weight=weight_distr,
                                    delay=RandomDistribution('normal',(0.8,1e-2)))
    GPe_anti = TsodyksMarkramSynapse(U=0.4159,
                                     tau_rec=30.0646,
                                     tau_facil=10.1047,
                                     weight=1.5,
                                     delay=0.5)

    GPe_out_syn = TsodyksMarkramSynapse(U=0.4159, tau_rec=30.0646, tau_facil=10.1047,
                                       weight=1,
                                       delay=RandomDistribution('normal',(1.25,1e-2)))

    DBS_conn = FixedProbabilityConnector(0.8)

    STN_pop = Population(1000, STN_type)
    GPe_pop = Population(100, GPe_type)

    Dummy_type = IF_cond_exp(tau_syn_E=3
                            ,tau_syn_I=5)

    GPe_out = Population(100, Dummy_type)

    noise, _, tt = OUProcess(5,2000,0.1)
    inj = StepCurrentSource(times=tt,
                            amplitudes=noise*10)
    inj.inject_into(GPe_pop)

    GPe_GPe = FixedNumberPreConnector(RandomDistribution('uniform_int',\
                                                         reciprocity))
    GPe_out_con = OneToOneConnector()

    STN_GPe = FixedNumberPreConnector(
        RandomDistribution('uniform_int',(18,22))
        ,with_replacement=False
    )

    ortho_connections = Projection(
        STN_pop, GPe_pop
        ,connector=STN_GPe
        ,synapse_type=STN_syn
        ,receptor_type='excitatory'
        ,space=Space('xy')
        ,label='Orthodromic Connections'
    )

    recip_connections = Projection(
        GPe_pop, GPe_pop
        ,connector=GPe_GPe
        ,synapse_type=GPe_syn
        ,receptor_type='inhibitory'
        ,space=Space('xy')
        ,label='Reciprocal Connections'
    )

    dbs_conn_orth = Projection(
        DBS_pulse, GPe_pop
        ,connector=DBS_conn
        ,synapse_type=DBS_ortho
        ,receptor_type='excitatory'
        ,space=Space('xy')
        ,label='Orthodromic DBS'
    )
    dbs_conn_anti = Projection(
        DBS_pulse, GPe_pop
        ,connector=DBS_conn
        ,synapse_type=GPe_anti
        ,receptor_type='inhibitory'
        ,space=Space('xy')
        ,label='Antidromic DBS'
    )

    dummy_monitor = Projection(
        GPe_pop, GPe_out
        ,connector=GPe_out_con
        ,synapse_type=GPe_out_syn
        ,receptor_type='inhibitory'
        ,space=Space('xy')
        ,label='GPe Out'
    )

    # Set recordings
    GPe_pop.record(['v','spikes','gsyn_exc','gsyn_inh'])
    GPe_out.record(['v','spikes','gsyn_inh'])

    ## Start STN DBS Impact Settings based on Brain Stim Paper
    ##  https://www.biorxiv.org/content/10.1101/2020.11.30.404269v1
    w_=0.8
    d=RandomDistribution('normal',(3,0.2))
    syn_exc_dep = TsodyksMarkramSynapse(U=0.5,
                tau_rec=671/10,
                tau_facil=17,
                weight=w_,
                delay=d)

    syn_inh_dep = TsodyksMarkramSynapse(U=0.4159,
                                        tau_rec=30.0646,
                                        tau_facil=10.1047,
                                        weight=w_*2,
                                        delay=d)
    AllConn = AllToAllConnector()

    num_exc = 225
    num_inh = 275

    DBS_exc_dep  = Population(int(1.0*num_exc),DBS_pulses)

    DBS_inh_dep  = Population(int(1.0*num_inh),DBS_pulses)

    DBS_inh_gpe  = Population(1,DBS_pulses)

    STN_dum_pop  = Population(1, Dummy_type)
    STN_dum_pop2 = Population(1, Dummy_type)

    def projSyn(pop,syn,excInh):
        return Projection(
            pop, STN_dum_pop
            ,connector=AllConn
            ,synapse_type=syn
            ,receptor_type=excInh
            ,space=Space('xy')
            ,label=f'STN DBS {excInh}'
        )
    stn_exc_dep = projSyn(DBS_exc_dep,syn_exc_dep,'excitatory')

    stn_inh_dep = projSyn(DBS_inh_dep,syn_inh_dep,'inhibitory')

    stn_inh_gpe = Projection(
            DBS_inh_gpe, STN_dum_pop2
            ,connector=AllConn
            ,synapse_type=GPe_out_syn
            ,receptor_type='inhibitory'
            ,space=Space('xy')
            ,label=f'STN DBS GPE'
        )

    STN_dum_pop.record(['gsyn_exc','gsyn_inh'])
    STN_dum_pop2.record(['gsyn_inh'])

    ## End STN DBS Impact Settings

    run(2000)
    data = GPe_pop.get_data()
    seg_out = GPe_out.get_data().segments[0]
    v_out    = np.array(seg_out.filter(name='v')[0])
    s_out    = np.array([np.array(s) for s in seg_out.spiketrains])
    ginh_out = np.array(seg_out.filter(name='gsyn_inh')[0])
    seg  = data.segments[0]
    Vs   = np.array(seg.filter(name="v")[0])
    gexc = np.array(seg.filter(name="gsyn_exc")[0])
    ginh = np.array(seg.filter(name="gsyn_inh")[0])
    spikes = np.array([np.array(s) for s in seg.spiketrains])
    stn_dat = STN_dum_pop.get_data().segments[0]
    stngexc = np.array(stn_dat.filter(name="gsyn_exc")[0])
    stnginh = np.array(stn_dat.filter(name="gsyn_inh")[0])
    stngpeinh = np.array(STN_dum_pop2.get_data().segments[0].filter(name="gsyn_inh")[0])
    end()

    return DBS_arr, ginh_out, gexc, ginh, stngexc,stnginh, stngpeinh

def dataGen(freq=100,recip=(38,48),bias=2.3,static=True):
    t = np.arange(0,2000.1,0.1)

    Os   = []
    gExcs = []
    gInhs = []
    stngExcs = []
    stngInhs = []
    stngpeinh = []
    DBS_arr = None
    for i in range(10):
        DBS_arr_t,o,gpeInExc,gpeInInh,stngExc,stngInh,sg = runSim(freq,recip,bias=bias,static=static)
        DBS_arr=DBS_arr_t
        Os.append(np.mean(o.T,axis=0))
        gExcs.extend(gpeInExc.T)
        gInhs.extend(gpeInInh.T)
        stngExcs.append(stngExc)
        stngInhs.append(stngInh)
        stngpeinh.append(sg)
    return Os, gExcs, gInhs, stngExcs, stngInhs, stngpeinh, DBS_arr

def figGen(Os,gExcs,gInhs,stngExcs,stngInhs,stngpeinh, DBS_arr, scale=1/500 ,save=False,name="DATA.csv",p=20,lines=False):
    plt.figure()
    t = np.arange(0,2000.1,0.1)
    def fillRange(x,std,c,tt):
        plt.plot(tt,x,color=c)
        plt.fill_between(tt,x+std,x-std,color=c,alpha=0.5)

    plt.subplot(141)
    plt.xlim(1000-(DBS_arr[1]-DBS_arr[0]),DBS_arr[9])
    plt.title("STN DBS Induced Conductances")
    plt.xlabel("Time (ms)")
    plt.ylabel("Conductance (μS)")
    stngExc = np.mean(stngExcs,axis=0).ravel()*scale*0.5
    stngExcStd = np.std(stngExcs,axis=0).ravel()*scale
    stngInh = np.mean(stngInhs,axis=0).ravel()*scale
    stngInhStd = np.std(stngInhs,axis=0).ravel()*scale
    stngTot = stngExc-stngInh

    fillRange(stngExc,stngExcStd,'g',t)
    fillRange(-stngInh,stngInhStd,'r',t)
    fillRange(stngTot,stngExcStd+stngInhStd,'k',t)

    plt.subplot(142)
    plt.xlim(1000-(DBS_arr[1]-DBS_arr[0]),DBS_arr[9])
    plt.title("GPe Inward Conductances")
    plt.xlabel("Time (ms)")

    gExc = np.mean(gExcs,axis=0).ravel()
    gExcStd = np.std(gExcs,axis=0).ravel()
    gInh = np.mean(gInhs,axis=0).ravel()
    gInhStd = np.std(gInhs,axis=0).ravel()
    Omean = np.mean(Os,axis=0).ravel()
    Ostd = np.std(Os,axis=0).ravel()
    fillRange(gExc,gExcStd,'g',t)
    fillRange(-gInh,gInhStd,'r',t)
    fillRange(gExc-gInh,gExcStd+gInhStd,'k',t)

    plt.subplot(143)
    plt.xlim(1000-(DBS_arr[1]-DBS_arr[0]),DBS_arr[9])
    plt.title("Total Inward STN Conductances")
    plt.xlabel("Time (ms)")
    plt.fill_between(t,0,-stngTot,color='g',alpha=0.5)
    plt.fill_between(t,-stngTot,Omean-stngTot,color='r',alpha=0.5)
    plt.plot(t,Omean-stngTot,color='k')
    if (lines):
        plt.vlines(np.arange(0,2001,1),plt.ylim()[0],plt.ylim()[1],color='#777',linestyle='--')
        plt.vlines(DBS_arr,plt.ylim()[0],plt.ylim()[1],color='k')

    plt.subplot(144)
    O__ = Omean-stngTot
    peakss=[]
    fps=[]
    fpstd=[]
    sps=[]
    spstd=[]
    for d in DBS_arr[:p]:
        s = (t>d)*(t<d+1000/100)
        curve = O__[s]
        cstd  = Ostd[s]
        peaks = find_peaks(curve,prominence=0.01)[0]

        if len(peaks)==1:
            peakss.append([peaks[0],90 if d==1000 else 80])
        elif len(peaks)==3:
            peakss.append(peaks[1:])
        elif len(peaks)!=2 or peaks[0]<=45:
            peakss.append([55,80])
        else:
            peakss.append(peaks)
        fps.append(curve[peakss[-1][0]])
        fpstd.append(cstd[peakss[-1][0]])
        sps.append(curve[peakss[-1][1]])
        spstd.append(cstd[peakss[-1][1]])
    l=np.arange(1,p+1)
    fillRange(np.array(fps)/fps[0],np.array(fpstd)/fps[0],'b',tt=l)
    fillRange(np.array(sps)/sps[0],np.array(spstd)/sps[0],'r',tt=l)

    if save:
        data = [stngExc,stngExcStd,stngInh,stngInhStd,stngTot,gExc,gExcStd,gInh,gInhStd,Omean,Ostd,fps,fpstd,sps,spstd,DBS_arr,t]
        header = ["stngExc","stngExcStd","stngInh","stngInhStd","stngTot","gExc","gExcStd","gInh","gInhStd","Omean","Ostd","fps","fpstd","sps","spstd","DBS_arr","t"]
        df = pd.DataFrame(data)
        df.T.to_csv(name,header=header,index=False)

    return fps,sps

DBS_100_recip = dataGen(100,(20,25))
DBS_100_norecip = dataGen(100,(1,2))
DBS_20_recip = dataGen(20,(20,25))
DBS_100_nonstatic = dataGen(100,(20,25),static=False)

s=False
figGen(*DBS_100_recip,lines=False,save=s,name="./data/DBS_100_recip.csv")
figGen(*DBS_100_norecip,lines=False,save=s,name="./data/DBS_100_norecip.csv")
figGen(*DBS_20_recip,lines=False,save=s,name="./data/DBS_20_recip.csv")
figGen(*DBS_100_nonstatic,lines=False,save=s,name="./data/DBS_100_nonstatic.csv")
plt.show()

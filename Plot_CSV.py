try:
    if INIT:
        print("Initialized.")
except:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import find_peaks
    import pandas as pd

INIT=True
def figGen(name="DATA.csv",i=0,T=False):
    df = pd.read_csv(name,header=0)
    stngExc = df['stngExc'].to_numpy()
    stngExcStd = df['stngExcStd'].to_numpy()
    stngInh = df['stngInh'].to_numpy()
    stngInhStd = df['stngInhStd'].to_numpy()
    stngTot = df['stngTot'].to_numpy()
    gExc = df['gExc'].to_numpy()
    gExcStd = df['gExcStd'].to_numpy()
    gInh = df['gInh'].to_numpy()
    gInhStd = df['gInhStd'].to_numpy()
    Omean = df['Omean'].to_numpy()
    Ostd = df['Ostd'].to_numpy()
    fps = df['fps'].to_numpy()
    fps = fps[~np.isnan(fps)]
    fpstd = df['fpstd'].to_numpy()
    fpstd = fpstd[~np.isnan(fpstd)]
    p=len(fpstd)
    sps = df['sps'].to_numpy()
    sps = sps[~np.isnan(sps)]
    spstd = df['spstd'].to_numpy()
    spstd = spstd[~np.isnan(spstd)]
    DBS_arr = df['DBS_arr'].to_numpy()
    t = df['t'].to_numpy()
    db = "#1a6ca4"  # First peak colour
    lb = "#06adea"  # Second peak colour

    def fillRange(x,std,c,tt,shade=False):
        plt.plot(tt,x,color=c)
        if shade:
            plt.fill_between(tt,x+std,x-std,color=c,alpha=0.5)
    
    plt.subplot(4,4,(i-1)*4+1 if T else i)
    plt.xlim(1000-(DBS_arr[1]-DBS_arr[0]),DBS_arr[9])
    plt.ylim(-0.2,0.4)
    plt.title("STN DBS Afferent Conductance")
    plt.xlabel("Time (ms)")
    plt.ylabel("Conductance (μS)")
    fillRange(-stngExc,stngExcStd,'g',t)
    fillRange(stngInh,stngInhStd,'r',t)
    fillRange(-stngTot,stngExcStd+stngInhStd,'k',t)

    plt.subplot(4,4,(i-1)*4+2 if T else i+4)
    plt.xlim(1000-(DBS_arr[1]-DBS_arr[0]),DBS_arr[9])
    plt.ylim(-1,1)
    plt.title("GPe Inward Conductances")
    plt.xlabel("Time (ms)")

    fillRange(-gExc,gExcStd,'g',t)
    fillRange(gInh,gInhStd,'r',t)
    fillRange(-gExc+gInh,gExcStd+gInhStd,'k',t)

    plt.subplot(4,4,(i-1)*4+3 if T else i+8)
    plt.xlim(1000-(DBS_arr[1]-DBS_arr[0]),DBS_arr[9])
    plt.title("STN Total Conductance")
    plt.xlabel("Time (ms)")
    plt.fill_between(t,0,-stngTot,color='g',alpha=0.5)
    plt.fill_between(t,-stngTot,Omean-stngTot,color='r',alpha=0.5)
    plt.plot(t,Omean-stngTot,color='k')
    plt.ylim(0,plt.ylim()[1])
    plt.subplot(4,4,(i-1)*4+4 if T else i+12)
    plt.ylim(0.4,1.6)
    l=np.arange(1,p+1)
    fc = db # First peak colour
    sc = lb # Second peak colour
    fillRange(np.array(fps)/fps[0],np.array(fpstd)/fps[0],fc,tt=l,shade=True)
    fillRange(np.array(sps)/sps[0],np.array(spstd)/sps[0],sc,tt=l,shade=True)
    plt.xlabel("Number of Spikes")
    plt.title("Peak Change over Time")

plt.figure(figsize=(14.5,14.5))
figGen("./data/DBS_100_nonstatic.csv",4)
figGen("./data/DBS_100_norecip.csv",2)
figGen("./data/DBS_100_recip.csv",1)
figGen("./data/DBS_20_recip.csv",3)
plt.tight_layout()
plt.savefig("ERNA_Combined_Plots.pdf")
plt.figure(figsize=(14.5,14.5))
figGen("./data/DBS_100_nonstatic.csv",4,T=True)
figGen("./data/DBS_100_norecip.csv",2,T=True)
figGen("./data/DBS_100_recip.csv",1,T=True)
figGen("./data/DBS_20_recip.csv",3,T=True)
plt.tight_layout()
plt.savefig("ERNA_Combined_Plots_Transpose.pdf")
plt.show()



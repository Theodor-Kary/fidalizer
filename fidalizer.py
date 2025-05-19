import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
from lmfit import Parameters, Model, minimize
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
import os
import re
from IPython.display import clear_output
import pandas as pd


# Import libraries and functionalities to interact with NN environment
import novopy as novopy
import novopy.services.eln as eln
import requests
from iidp.models.datacube import Run, Datacube
from iidp.models.entry import IDSEntry
from iidp.wrapper import Connection, Search
from iidp.query import Query
from novopy.services.sciworm.connect_to_sciworm import return_headers_for_calling_sciworm





    
    
    
class Data():
    def __init__(self, time, signal, fname = None):
        #Data
        self.t = np.asarray(time)*60   #tTime measurments... convert from minute units, to second units upon init
        self.s = np.asarray(signal)    #Sigmal measurments
    

        #Sample Details
        self.fname = fname
        self.conc = None
        self.name = None
        self.date = None
        self.fname_parse()
        
        self.conc_eff = None
        
        
        
        #Fitting parameters
        self.tr = None    #resident time
        self.sd = None    #standard deviation
        self.rh = None    #Hydrodynamic radius (nm)
        self.d = None     #diffusion coefficient
        self.x_fit = None
        self.y_fit = None
        self.r2 = None
        self.auc = None
        self.snr = None
        self.taylor_conditions = None
        
        return

    
    def __str__(self):
        return self.fname

    def __repr__(self):
        return self.fname

    
    def fname_parse(self):
        if self.fname != None:
            input_string = self.fname
            regex_pattern = r'\[(\d+(?:\.\d+)?)\s*(.+)\]\s*(.+)-(\d{2}-\d{2}-\d{4}-\d{2}-\d{2})'
            match = re.match(regex_pattern, input_string)
            if match:
                self.conc = (float(match[1]), match[2])
                self.name = match[3]
                self.date = match[4]
            else:
                print(f'File name \'{fname}\' cannot be parsed')
        return

    
    @classmethod
    def get_from_iidp(cls, IID_reg):
        conn = Connection() # read files in from IIDP
        session, query = Search(conn), Query() # intilize a session and query
        barcode_iidp = IID_reg.lower() # Input your registration barcode
        query.labware.barcode = barcode_iidp
        entry: IDSEntry = session.execute(query)
        
        data_set = []
        files = [i for i in entry]
        for file_path in files:
            file = file_path.file.rpartition('/')[0].rpartition('/')[-1]
            file_name, file_extension = os.path.splitext(file)
            if file_extension != '.txt':
                continue
    
            content = file_path.raw.content
            data = content.decode('utf-8')
            rows = data.strip().split('\n')
            time = []
            signal = []
            for row in rows:
                values = row.split('\t')
                time.append(float(values[0]))
                signal.append(float(values[1]))
            
            instance = cls(time, signal, fname = file_name)
            data_set.append(instance)
            print('Retrieved {0}/{1} data objects from {2}'.format(len(data_set),len(files), IID_reg))
            clear_output(wait=True)
        return data_set
    
    @classmethod
    def get_from_txt(cls, dire):
        data_set = []

        fs_dir = os.fsencode(dire)
        for file in os.listdir(fs_dir):
            file_name, file_extension = os.path.splitext(file)
            file_name = os.fsdecode(file_name)
            file_extension = os.fsdecode(file_extension)
            if file_extension != '.txt':
                continue
            #print('FOUND:', file_name)
            
            file_path = (os.fsdecode(os.path.join(fs_dir, file)))
            data = np.genfromtxt(file_path, delimiter='\t')
            time = []
            signal = []
            for line in data:
                time.append(line[0])
                signal.append(line[1])

            instance = cls(time, signal, fname = file_name)
            data_set.append(instance)
            print('Retrieved {0}/{1} data objects from {2}'.format(len(data_set),len(os.listdir(fs_dir)), dire))
            clear_output(wait=True)
        return data_set


            

    def cal_conc_eff(self):
        if self.r2 == None:
            return
        
        b = 2 * np.sqrt(self.rc**2/(24*self.d)) * (self.p_mob / (self.p_inj*self.t_inj))    
        #equation 29 of dilution factor paper
        
        self.df = (2/(b*np.sqrt(self.tr))) - (2/(b**2*self.tr)) * np.log(1+b*np.sqrt(self.tr))
        self.conc_eff = self.conc[0] * self.df
        return
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def fit(self, baseline_drift = False, coverage = 75, smoothing = False, gate_start = None, gate_end = None, debug_plot = False, mute = False):
        def gaussian_function(x, amp, mean, sd):
            return amp * np.exp(-((x - mean) / sd) ** 2 / 2)
        
        #Taking the region of only the data inteded for fitting
        if gate_start == None:
            gate_start = np.min(self.t)
        if gate_end == None:
            gate_end = np.max(self.t)
        i_data = np.where((self.t >= gate_start) & (self.t <= gate_end))
        s = np.array(self.s)[i_data]
        t = np.array(self.t)[i_data]
        
        x = t
        y = s 
        
        #simple baseline subtraction
        y = y - np.min(y)
        
        #estimate average elution (tr)
        i_tr = np.where(np.cumsum(y)/np.sum(y) >= 0.5)[0][0]
        tr0 = x[i_tr]
        if abs(self.tr_expected - tr0) > 5 and mute == False:  #check if calculated residence time is around expected
            print(f'Error with data {self.fname}: tr = {tr0:.0f}, excpected tr = {self.tr_expected:.0f}.')
        
        #amplitude normalizeation
        ymax = np.max(y[i_tr-5:i_tr+5])
        y=y/ymax
        
        #for i in range(len(y)):
        #    itot = itot+y[i]
        #    if itot/ytot >= 0.5:
        #        tr0 = x[i]
        #        
        #        break
        
        
        #calculate standard deviation
        def guess_sd(self, size, mean):
            d = stokes_einstein(self.temp, self.viscosity, d = None, rh = size * 1e-9)
            return taylor_diffusion(self.rc, mean, d = d, sd = None)
        sd0 = guess_sd(self, 5, tr0)
        sd0l = guess_sd(self, 0.3, tr0)
        sd0u = guess_sd(self, 100, tr0)
        
        
        #round 1 fitting
        bounds = ([0.8,min(x),sd0l], [1.2,max(x),sd0u])
        try:
            param1, _ = curve_fit(gaussian_function, x, y, p0=[1, tr0, sd0], bounds=bounds)
        except:
            print(f'Error with data {self.fname}: failed to fit in iteration 1')
            if debug_plot == True:
                plt.scatter(x, y)
                y_fit = gaussian_function(x, 1, tr0, sd0)
                plt.plot(x, y_fit, color='crimson')
                plt.show()
            return
        amp1, mean1, sd1 = param1
        
        if mean1 - 4*sd1 <= 0:
            print(f'Error with data {self.fname}: fit iteration 1 returned unrealistic values, mean = {mean1} or sd = {sd1}')
            return

    
    
    
    
        ### Iteration 2
        x = t
        y = s
        
        #Region isolation
        peak_start = mean1 - 4*sd1
        if peak_start < gate_start:
            print(f'gate start: {gate_start}, peak start: {peak_start}, tr: {mean1}, sd: {sd1}')
            return 
        peak_end = mean1 + 4*sd1
        
        h = (100-coverage)*0.02
        peak_trunc =  mean1 + sd1*np.sqrt(-2*np.log(h))#the x-axis at which y = 0.5*ymax this representing 75%
        tip_start = mean1 - 0.25*sd1  
        tip_end = mean1 + 0.25*sd1 
        
        #Smoothing step
        if smoothing == True: 
            delta = (x[-1]-x[0])/len(x)
            y = savgol_filter(y, window_length=5, polyorder=3, delta=delta)              
        
        #baseline correction
        i_baseline = np.where(x < peak_start)
        x_baseline = x[i_baseline]
        if len(x_baseline) == 0:
            plt.plot(x, y)
            return
        y_baseline = y[i_baseline]
        if baseline_drift == True:
            coef = np.polyfit(x_baseline, y_baseline, 1)
            bl_fn = np.poly1d(coef) 
            self.baseline = bl_fn(x)
        else:
            self.baseline = np.mean(y_baseline)
        y = y - self.baseline #Removing baseline
        

        #amplitude normilization
        i_tip = np.where((x >= tip_start) & (x <= tip_end))
        y_tip = y[i_tip]
        self.ymax = np.max(y[i_tip])
        y = y/self.ymax
        
        
        #Fitting
        i_peak_hw = np.where((x >= peak_start) & (x <= peak_trunc))
        x_peak = x[i_peak_hw]
        y_peak = y[i_peak_hw]
        assert len(x_peak) == len(y_peak) and len(x_peak) > 1, 'region short'  
        
        try:
            param2, _ = curve_fit(gaussian_function, x_peak, y_peak, p0=param1, bounds=bounds)
        except:
            print(f'Error with data {self.fname}:failed to fit in iteration 2')
            plt.plot(x_peak, y_peak)
            plt.show()
            return

        # Calculating params
        self.amp, self.tr, self.sd = param2
        self.x_fit = x_peak
        self.y_fit = y_peak
        
        self.y_pred = gaussian_function(x_peak, *param2)
        self.r2 = r2_score(self.y_pred, self.y_fit)
        self.d = taylor_diffusion(self.rc, self.tr, d = None, sd = self.sd)
        self.rh = stokes_einstein(self.temp, self.viscosity, d = self.d, rh = None) * 1e9 #1e9 m -> nm units

        #Signal to Noise
        baseline = (s - self.baseline)[i_baseline]
        self.noise = np.std(baseline)
        signal = self.amp*self.ymax
        self.snr = signal / self.noise

        #Area under curve
        i_peak = np.where((x >= peak_start) & (x <= peak_end))
        peak = (s - self.baseline)[i_peak]
        self.auc = np.sum(peak)
        
        self.s_trunc = s - self.baseline
        self.t_trunc = t
            
        #if self.r2 > 0.95:
        #    self.good_fit = True
        #print(f'Successful fit of {self}')
        #clear_output(wait=True)
        return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        
        
    
    def plot(self, title = None, xlabel = 'Time [S]', ylabel = 'Signal [a. u]',  fit = False):
        x = self.t
        y = self.s
        if title == None:
            title = 'Taylorgram of '+str(self)

        if fit == True and self.r2 != None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

            ax1.scatter(x, y, s=1)
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(ylabel)

            ax2.scatter(self.x_fit, self.y_fit, s=1)
            ax2.plot(self.x_fit, self.y_pred, label='fit', color='crimson')
            ax2.set_ylim(top=1.1)
            ax2.set_title('')
            ax2.set_xlabel(xlabel)
            ax2.set_ylabel(ylabel)
            props = dict(boxstyle='square, pad=0.4', facecolor='white', alpha=0)
            text = r'$R_H = {:.2f}$, $R^2 = {:.2f}$'.format(self.rh, self.r2)
            ax2.text(0.05, 0.90, text, fontsize=10, transform = ax2.transAxes, verticalalignment='top', bbox=props)
            plt.legend(loc = 'upper left')

            plt.subplots_adjust(wspace=0.2)
            fig.suptitle(title)
        else:
            plt.scatter(x, y, s=1)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
        plt.show()
    
    
    
    
    
    
    def visc_correct(self, tr_ref = None, visc_ref = None):

        
        x = np.array(self.t)
        y = np.array(self.s)
        
        if visc_ref == None:
            visc_ref, _ = lookup_viscosity(self.temp)
        
        if tr_ref == None:
            Q = HagenPoiseuille(self.p_mob, self.rc, visc_ref, self.L)
            u = Q / (np.pi*self.rc**2) #Linear flow rate = volumetric flow rate / crossesctional area
            tr_ref = self.wc / u
        
        #calculate residence time
        self.fit(baseline_drift = True, smoothing=True, mute = True)
        if self.r2 == None:
            return
        tr_exp = self.tr
        
        self.viscosity = tr_exp/tr_ref * visc_ref
        self.cal_flow()
        #self.fit(baseline_drift = True, smoothing=True, mute = True)
        #print(tr_exp, self.tr, self.tr_expected)
        return
    
    
    
    
    
    
    
    
    
    
    def set_conditions(self, temp = 25, cap_radius = 0.000075*0.5, cap_length = 1, length_to_window = 0.84, t_inj = 10, p_inj = 50, p_mob = 400):
        self.temp = temp     #Celcius
        
        self.rc = cap_radius        #Capillary radius m (default: 75 um diameter)
        self.L = cap_length        #Capillary length to window 0.84 m length
        self.wc = length_to_window  #capillary total length 1 m

        self.t_inj = t_inj        #10 second injection time is standard
        self.p_inj = p_inj        #Pressure difference between two ends (mbar)
        self.u_inj = None               # m/s
        self.Q_inj = None               # m^3/s
        self.v_inj = None             # m^3
        
        self.p_mob = p_mob    #Pressure difference between two ends (mbar)
        self.u_mob = None             # m/s
        self.Q_mob = None             # m^3/s
        self.tr_expected = None         # s
             
        self.viscosity, self.density = lookup_viscosity(self.temp) # Pa s      # g/cm³ 
        self.reynolds = None  
        self.cal_flow()

        
    


    def cal_flow(self):
        self.Q_inj = HagenPoiseuille(self.p_inj, self.rc, self.viscosity, self.L)
        self.u_inj = self.Q_inj / (np.pi*self.rc**2) #Linear flow rate = volumetric flow rate / crossesctional area
        self.v_inj = self.Q_inj * self.t_inj
        
        self.Q_mob = HagenPoiseuille(self.p_mob, self.rc, self.viscosity, self.L)
        self.u_mob = self.Q_mob / (np.pi*self.rc**2) #Linear flow rate = volumetric flow rate / crossesctional area
        self.tr_expected = self.wc / self.u_mob 
    
        self.reynolds = (self.density * self.u_mob * self.rc*2) / self.viscosity
        return
    
        
        
        
    
    
    
    






    
def taylor_diffusion(rc, tr, d = None, sd = None):
    if d is not None and sd is not None:
        raise ValueError('Only d or sd should be provided, not both')
    
    if sd is not None:
        d = (rc**2 * tr) / (24 * sd**2)
        return d
    elif d is not None:
        sd = np.sqrt((rc**2 * tr) / (24 * d))
        return sd
    else:
        raise ValueError('Either d or sd should be provided')
    

def stokes_einstein(temp, visc, d = None, rh = None):
    kB = 1.380649e-23   #J/K
    if d is not None and rh is not None:
        raise ValueError('Only d or rh should be provided, not both')
    
    if d is not None:
        rh = (kB * (temp+273.15)) / (6 * np.pi * visc * d)
        return rh
    elif rh is not None:
        d = (kB * (temp+273.15)) / (6 * np.pi * visc * rh)
        return d
    else:
        raise ValueError('Either d or rh should be provided')    

        
    
def HagenPoiseuille(pressure, radius, viscosity, length):
    #Volumetric flow rate
    Q = (pressure * 1e2 * np.pi * radius**4) / (8*viscosity* length) #m^3/s      1 mbar = 1e2 Pa
    return Q
    

    
    
    
    
def lookup_viscosity(temp):
    #https://wiki.anton-paar.com/dk-en/water/
    #temp  : dynamic_viscosity [mPa.s],  density [g/cm³]
    water_dict = {2: [1.6735, 0.9999], 
                  3: [1.619, 1.0], 
                  4: [1.5673, 1.0], 
                  5: [1.5182, 1.0], 
                  6: [1.4715, 0.9999], 
                  7: [1.4271, 0.9999], 
                  8: [1.3847, 0.9999], 
                  9: [1.3444, 0.9998], 
                  10: [1.3059, 0.9997],
                  11: [1.2692, 0.9996], 
                  12: [1.234, 0.9995], 
                  13: [1.2005, 0.9994], 
                  14: [1.1683, 0.9992], 
                  15: [1.1375, 0.9991], 
                  16: [1.1081, 0.9989], 
                  17: [1.0798, 0.9988], 
                  18: [1.0526, 0.9986], 
                  19: [1.0266, 0.9984], 
                  20: [1.0016, 0.9982], 
                  21: [0.9775, 0.998], 
                  22: [0.9544, 0.9978], 
                  23: [0.9321, 0.9975], 
                  24: [0.9107, 0.9973], 
                  25: [0.89, 0.997], 
                  26: [0.8701, 0.9968], 
                  27: [0.8509, 0.9965], 
                  28: [0.8324, 0.9962], 
                  29: [0.8145, 0.9959], 
                  30: [0.7972, 0.9956], 
                  31: [0.7805, 0.9953], 
                  32: [0.7644, 0.995], 
                  33: [0.7488, 0.9947], 
                  34: [0.7337, 0.9944], 
                  35: [0.7191, 0.994], 
                  36: [0.705, 0.9937], 
                  37: [0.6913, 0.9933], 
                  38: [0.678, 0.993], 
                  39: [0.6652, 0.9926], 
                  40: [0.6527, 0.9922], 
                  45: [0.5958, 0.9902], 
                  50: [0.5465, 0.988], 
                  55: [0.5036, 0.9857], 
                  60: [0.466, 0.9832], 
                  65: [0.4329, 0.9806], 
                  70: [0.4035, 0.9778], 
                  75: [0.3774, 0.9748], 
                  80: [0.354, 0.9718]}

    viscosity = water_dict[temp][0] * 1e-3        #from mPa.s to Pa.s
    density = water_dict[temp][1] * 1e3           #from g/cm3 to kg/m3
    return viscosity, density
    
    

    
        













def summary(data_set, r2_cutoff = 0.98, mute = False):
    x = []
    y = []
    c = []
    snr = []
    auc = []
    tr = []
    
    for data in data_set:
        if data.r2 == None or data.r2 < r2_cutoff:
            if mute == False:
                print(data, 'is skipped due to bad fit')
            continue
    
        x.append(data.conc[0])
        y.append(data.rh)
        c.append(data.conc_eff)
        snr.append(data.snr)
        auc.append(data.auc)
        tr.append(data.tr)
        
    x_unique = np.unique(x)  # Get unique values of x
    c_mean = [np.mean([c[i] for i, value in enumerate(x) if value == x_val]) for x_val in x_unique] 
    y_mean = [np.mean([y[i] for i, value in enumerate(x) if value == x_val]) for x_val in x_unique]  # Calculate mean values for each unique x
    std_dev = [np.std([y[i] for i, value in enumerate(x) if value == x_val]) for x_val in x_unique]  # Calculate standard deviation for each unique x
    snrs = [np.mean([snr[i] for i, value in enumerate(x) if value == x_val]) for x_val in x_unique]  # Calculate standard deviation for each unique x
    aucs = [np.mean([auc[i] for i, value in enumerate(x) if value == x_val]) for x_val in x_unique]  # Calculate standard deviation for each unique x
    trs = [np.mean([tr[i] for i, value in enumerate(x) if value == x_val]) for x_val in x_unique]  # Calculate standard deviation for each unique x
    r_v_c = {'conc_inj':np.array(x_unique), 'conc_eff':np.array(c_mean), 'r_avr':np.array(y_mean), 'r_std':np.array(std_dev), 'tr':np.array(trs), 's:n':np.array(snrs), 'auc':np.array(aucs), 'conc_unit':data_set[0].conc[1]}
    return r_v_c

def cal_taylor_conditions(self, display=False):
    
    #Peclet number
    peclet_number = self.con.u_mob * self.con.rc / self.d
    condition1 = peclet_number > 70
    
    tau = self.d * self.tr / self.con.rc**2
    condition2 = tau > 1.4
    
    #Inject %
    cap_vol = self.con.L * np.pi * self.con.rc**2
    inj_percent = self.con.v_inj / cap_vol * 100
    condition3 = inj_percent < 1
    if display:
        print('Peclet={0}>70: {1}, Tau={2}>1.4: {3}, Vol(inj)={4}<1%: {5}'.format(peclet_number, condition1, tau, condition2, inj_percent, condition3))
        
    self.taylor_conditions = condition1 and condition2 and condition3
    
    return









def affinity_fit(c_tot, r_avr, r_std, r1=None, n=2, plot=False, conc_unit='nM', title='Affinity Fit of Size-Concentration Dependency'):
    
    def affinity_function(c_tot, kD, r1, rd, n):
        r2 = r1 + rd #Calculate radius of species 2 (ensures larger as rd > 0)

        #Fits concentration of monomer based on known total concentration and current estimated kD
        def concentration_relation(pars, c_tot, kD, n):
            c1 = pars['c1'].value
            residual = c_tot - (c1 + n * (c1**n / kD))
            return residual
        c1 = []
        for c in c_tot:
            pars = Parameters()
            pars.add('c1', value=1, min=0, max=c)
            result = minimize(concentration_relation, pars, args=(c, kD, n))
            c1.append(result.params['c1'].value)
        c2 = (c_tot - c1) / 2
        r1 = np.asarray(r1)
        r_avr = (c1 * r1 + 2 * c2 * r2) / c_tot
        return r_avr

    model = Model(affinity_function)

    if r1 is None:
        model.set_param_hint('r1', value=np.mean(r_avr), min=0.3, max=max(r_avr), vary=True)
    else:
        model.set_param_hint('r1', value=r1, vary=False)  # Assuming fix_r1 should be r1

    # Ensure c_tot is scalar or handled correctly
    model.set_param_hint('kD', value=np.median(c_tot), min=0, max=0.5 * max(c_tot), vary=True)
    model.set_param_hint('rd', value=0.3, min=0, max=max(r_avr), vary=True)
    model.set_param_hint('n', value=n, vary=False)

    pars = model.make_params()
    result = model.fit(r_avr, pars, r_std, c_tot=c_tot)
    vardict = result.params.valuesdict()
    vardict['chisqr'] = result.chisqr
    
    
    if plot == True:
        fig, ax = plt.subplots()
        plt.errorbar(c_tot, r_avr, r_std, fmt='o', capsize=5, capthick=1)
        x = np.arange(min(c_tot),max(c_tot),(max(c_tot)-min(c_tot))/100)
        y = affinity_function(x, vardict['kD'], vardict['r1'], vardict['rd'], vardict['n'])
        plt.plot(x, y, color='crimson', label='fit')
        plt.ylabel('Hydrodynaic Radius [nm]')
        plt.xlabel(f'Effective Concentration [{conc_unit}]')
        plt.title(title)
        props = dict(boxstyle='square, pad=0.4', facecolor='white', alpha=0)
        text = f'\(K_D= {vardict["kD"]:.2f}\) {conc_unit} \n \(R_h(mono) = {vardict["r1"]:.2f}\) nm \n \(R_h(oligo) = {vardict["r1"]+vardict["rd"]:.2f}\) nm'
        plt.xscale('log')
        ax.text(0.02, 0.95, text, fontsize=10, transform = ax.transAxes, verticalalignment='top', bbox=props)

    return vardict






def find_best_n(c_tot, r_avr, r_std, r1=None, max_n = 9):
    best_chi = np.inf
    best_n = 0
    results_dict = {}
    for n in range(2, max_n):
        results = affinity_fit(c_tot, r_avr, r_std, r1=r1, n=n)
        results_dict[results['n']] = results
        results_dict[results['n']].pop('n')
    for n, res in results_dict.items():
        if res['chisqr'] < best_chi:
            best_n = n
            best_chi = res['chisqr']
    print(f'''
    n\t kD \t r1 \t r2 \t chisqr
    {best_n}\t{results_dict[best_n]['kD']:.2f}\t{results_dict[best_n]['r1']:.2f}\t{results_dict[best_n]['r1']+results_dict[best_n]['rd']:.2f}\t{results_dict[best_n]['chisqr']:.2e}
    ''')
    return results_dict




def raw_display(data_set, normalize=None, title=None, xmin = None, xmax=None, cutoff=0.98):
    conc_dict = {}
    for data in data_set:
        if (data.r2 != None and data.r2 > cutoff) and (data.conc[0] not in conc_dict or data.r2 > conc_dict[data.conc[0]].r2):
            conc_dict[data.conc[0]] = data
    concs = sorted(conc_dict.values(), key=lambda x: x.conc[0])
    
    
    for data in concs:
        s = data.s_trunc
        t = data.t_trunc
        k = 0.25
        if normalize == 'c':
            s = s/data.conc[0]
        elif normalize == 's':
            s = s/data.ymax
        elif normalize == 'e':
            s = s/data.conc_eff
        else:
            s = s
        plt.scatter(t, s, label=f'{data.conc[0]} {data.conc[1]}', s=1)
            
    plt.xlabel('Time [S]')
    if normalize == 'c':
        plt.ylabel('$I/C_{inj}$ [a. u]')
    elif normalize == 'e':
        plt.ylabel('$I/C_{eff}$ [a. u]')
    elif normalize == 's':
        plt.ylabel('$I/I_tr$ [a. u]')
        plt.ylim(-0.1, 1.1)
    else:
        plt.ylabel('$I$ [a. u]')

    if xmin == None:
        xmin = np.min(data.t)
    if xmax == None:
        xmax = np.max(data.t)
    plt.xlim(xmin,xmax)
    plt.title(title)
    lgnd = plt.legend(loc="upper left", scatterpoints=1, fontsize=10)
    for leg in lgnd.legend_handles:
        leg._sizes = [30]
    return




def fit_display(data_set, title = None, cutoff=0.98):
    conc_dict = {}
    for data in data_set:
        if (data.r2 != None and data.r2 > cutoff) and (data.conc[0] not in conc_dict or data.r2 > conc_dict[data.conc[0]].r2):
            conc_dict[data.conc[0]] = data
    concs = sorted(conc_dict.values(), key=lambda x: x.conc)
    


    cols = int(np.ceil(len(concs)/2))

    fig, axs = plt.subplots(2, cols, figsize=(2*cols, 6))
    i=0
    for data in concs:
        col = int(i % cols)
        row = int(np.floor(i/cols))
        axs[row, col].scatter(data.x_fit-data.tr, data.y_fit, label=f'{data.conc[0]} {data.conc[1]}', s=4)
        axs[row, col].plot(data.x_fit-data.tr, data.y_pred, color='crimson', linewidth=1.7)

        axs[row, col].set_xlabel(f'Time -$t_r$ [S]')
        axs[row, col].set_ylabel('Signal [a. u]')
        axs[row, col].set_ylim(-0.02, 1.02)
        axs[row, col].set_xlim(-3.1*data.sd,1)
        #axs[row, col].legend(loc='upper left', shadow=True, ncol=1, fontsize = 8)
        axs[row, col].set_title(f'{data.conc[0]} [{data.conc[1]}]', fontsize = 10)
        text = f'$R_H={data.rh:.2f}$ nm \n $t_r={data.tr:.0f}$ S \n $r^2$={data.r2:.2f}'
        axs[row, col].text(0.1, 0.80, text, fontsize=10, transform=axs[row,col].transAxes)
        i = i + 1

    plt.tight_layout(pad=0.9)
    fig.suptitle(title, fontsize=15, y=1.02)
    

            
        
import matlab
import numpy as np
import matplotlib.pyplot as plt

eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath('/opt/matlab_software/roc_toolbox'), nargout=0)

targf = [96,15,16,17,17,21,5,9,2,2]
luref = [9,13,23,29,22,26,29,25,9,15]
eng.workspace['model'] = 'dpsd'
eng.workspace['nBins'] = float(10)
eng.workspace['nConds'] = float(1)
eng.workspace['parNames'] = ['Ro','F']
x0, lb, ub = eng.eval('gen_pars(model,nBins,nConds,parNames)', nargout=3)
eng.workspace['x0'] = x0
eng.workspace['lb'] = lb    
eng.workspace['ub'] = ub
eng.workspace['fitStat'] = '-LL'
eng.workspace['targf'] = matlab.double(targf)
eng.workspace['luref'] = matlab.double(luref)
roc_data = eng.eval("roc_solver(targf,luref,model,fitStat,x0,lb,ub,'figure',false)", nargout=1)

Ro = roc_data['dpsd_model']['parameters']['Ro']
F = roc_data['dpsd_model']['parameters']['F']
criterion = np.asarray(roc_data['dpsd_model']['parameters']['criterion'])
targ_proportion = np.asarray(roc_data['observed_data']['target']['cumulative'])
targ_proportion = np.concatenate((targ_proportion, targ_proportion+.1), axis=0)
lure_proportion = np.asarray(roc_data['observed_data']['lure']['cumulative'])
lure_proportion = np.concatenate((lure_proportion, lure_proportion), axis=0)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(lure_proportion[0], targ_proportion[0], 'ro')
ax1.plot(lure_proportion[1], targ_proportion[1], 'bx')
plt.show()

eng.quit()
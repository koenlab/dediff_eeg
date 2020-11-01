import matlab.engine
import matlab

eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath('/opt/matlab_software/roc_toolbox'), nargout=0)

x0, lb, ub = eng.gen_pars('dpsd',6,1,['Ro','F'])

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


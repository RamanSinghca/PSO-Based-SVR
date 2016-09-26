#This is a prediction model SVR for City of Edmonton Bus Data 
#118 Street & Jasper Avenue,Stop #:1688

#Original Author to implement SVR @Lu Yin -- https://github.com/koosha/catching_bus/blob/master/SVR_Model/busStop16_SVR.py

#Author @Ramanpreet Singh (b4s79@unb.ca),  Sept 24, 2016 - Re-factored and optimized SVR using PSO

'''
@External Dependencies:  Pandas, pyswarm, sklearn packages.
pyswarm - https://github.com/tisimst/pyswarm/
        - http://pythonhosted.org/pyswarm/
'''
import os
import pandas
from pyswarm import pso
from sklearn.svm import SVR
from sklearn.cross_validation import KFold

''' 
    Variable(s) Initialization 
'''
fold_count=10 # K-Fold Cross validation.
bus_stop_number=16
userhome = os.path.expanduser('~') # Set user home directory


''' (1) Data Files Import : Split the original extraData into Dif_file which contains y(real difference) 
    and Previous which contains the arrival_time of previous_bus_stop (used for MAPE)
    (1.1) Read CSV : Use pandas to read the CSv files  
'''
def load_csvdata():
    global X
    global Y
    
    Data = userhome +r'/Desktop/Data/bustStop_{bsn}.csv'.format(bsn=bus_stop_number)
    Dif = userhome +r'/Desktop/Data/extraData_{bsn}.csv'.format(bsn=bus_stop_number)
    print('Data read from file', Data)
    X = pandas.read_csv(Data,names = ['a','b','c','d','e','f','g','h','i']);
    X = X.values
    Y = pandas.read_csv(Dif,names=['diff','previousarrvial']);
    Y = Y.values


'''(2) PSO based SVR objective function to minimize.
    @param: params[]-- parameters array containing values for  C --> params[0] , epsilon --> params[1]  and other parameters if needed. 
    @return: sMape value for the predictions.
'''
def svrPso(params):
    kf = KFold(len(X), n_folds=fold_count)
    for train, test in kf:
        mapeTotal = 0
        # Prepare Training and Testing data set. Use cross validation output to split the data sets. 
        X_train, X_test, y_train, y_test = X[train], X[test], Y[train,0], Y[test,0]
        previousarrvial = Y[test,1]
        nn = SVR(C=params[0], epsilon = params[1])
        nn.fit(X_train,y_train)
        result = nn.predict(X_test);
        thisMape = calsMAPE(X_test,previousarrvial,y_test, result)
        mapeTotal = mapeTotal + thisMape    
    mapeCV = mapeTotal/fold_count; 
    print('Optimizing the Parameters ..... C = {c}, epsilon={e}, MAPE={m}'.format(c=params[0], e=params[1], m=mapeCV))
    return mapeCV


'''(3) sMAPE Calculation: sMAPE function to calculate error between predicted and actual values. 
    @param: X_test, previousarrvial, Y_test, result
    @return: sMape Value for the predictions.
'''
def calsMAPE(X_test,previousarrvial,Y_test, result):
    sum_up = 0
    n = 0
    size = len(X_test)
    for i in range(size):
        if previousarrvial[i] != 0:
            diff = result[i] - Y_test[i]
            diff = abs(diff)
            n = n+1
            sum_up = sum_up + (diff/abs((X_test[i,:1]*17363+53240) + Y_test[i] - previousarrvial[i]))
    MAPE = sum_up/n
    return MAPE
    
'''(0) Main Run function to execute.
'''
def main_run():
    
        # Load .csv Data from filestore.
        load_csvdata()
    
        #Define lower bound (lb) and upper bound (ub) for parameter(s) to be optimized 
        # To optimize more than one parameter, define lb and up values as lower and upper bound of each parameter = [p1,p2,p3,.....pn]
        #  = [C   , epsilon, ....... other parameters]
        lb = [10.0, 0.08]
        ub = [500.0, 0.3]
        
        # xopt: array of parameter(s) for optimized values --- fopt: optimized objective function value  --- svrPSO: function to be optimized.
        xopt, fopt = pso(svrPso, lb, ub, maxiter=1, debug=True,phip=10, swarmsize=200, minfunc=0.001 )
        
        print(" ")
        print("************ Objective Function optimized *****************")
        print(" ")
        print('ALL Parameters optimized: C = {c}, epsilon={e}, Overall MAPE={m}'.format(c=xopt[0], e=xopt[1], m=fopt))
        print(" ")
        print(" ")

print("************  Initializing PSO based SVR *****************")
main_run()
print("************  Optimization Finished *****************")

#****** Sample Output *****
# .......
# .......
# .......
#Optimizing the Parameters ..... C = 10.0, epsilon=0.1678947834862191, MAPE=[ 0.00349968]
#Optimizing the Parameters ..... C = 296.62710365281237, epsilon=0.2347927613193696, MAPE=[ 0.00262928]
#Optimizing the Parameters ..... C = 500.0, epsilon=0.2472140331130177, MAPE=[ 0.0025515]
#New best for swarm at iteration 1: [  5.00000000e+02   2.47214033e-01] [ 0.0025515]
#Stopping search: Swarm best objective change less than 0.001
 
#************ Objective Function optimized *****************
 
#ALL Parameters optimized: C = 500.0, epsilon=0.2472140331130177, Overall MAPE=[ 0.0025515]
 
 
#************  Optimization Finished *****************



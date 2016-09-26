PSO-Based-SVR to forecast potential delay time of bus arrival. Applied on City of Edmonton real data for 118 Street & Jasper Avenue, Stop :1688. Data can be found at : 
https://github.com/koosha/catching_bus/blob/master/Data/busStop/bustStop_16.csv and 
https://github.com/koosha/catching_bus/blob/master/Data/ExtraData/extraData_16.csv

More details on Orginal SVR work and data can be found on : 

https://github.com/koosha/catching_bus and 

http://ace.edmonton.ca/outreach/open-science/

In this work, 
Improvements are made by  optimizing SVR parameters by utilizing the concepts of Particle swarm optimization. 

1. ~8% improvement in MAPE with 10-fold cross validation 

   ~3% improvement with 5-fold cross validation. 
   
2. Dynamic parameters calculation: The optimal values for C and epsilon, which minimize MAPE(error), were calculated dynamically. 

3. Future improvements: The framework could also be utilized to further optimize Kernel function parameters. 



@External Dependencies:  
Pandas, pyswarm, sklearn packages.
pyswarm - https://github.com/tisimst/pyswarm/
        - http://pythonhosted.org/pyswarm/




****** Sample Output for 10-fold cross validation applied on busStop_16 data*****

 .......
 .......
 .......
 
Optimizing the Parameters ..... C = 10.0, epsilon=0.1678947834862191, MAPE=[ 0.00349968]

Optimizing the Parameters ..... C = 296.62710365281237, epsilon=0.2347927613193696, MAPE=[ 0.00262928]

Optimizing the Parameters ..... C = 500.0, epsilon=0.2472140331130177, MAPE=[ 0.0025515]

New best for swarm at iteration 1: [  5.00000000e+02   2.47214033e-01] [ 0.0025515]
Stopping search: Swarm best objective change less than 0.001

 
************ Objective Function optimized *****************

ALL Parameters optimized: C = 500.0, epsilon=0.2472140331130177, Overall MAPE=[ 0.0025515]

************  Optimization Finished *****************


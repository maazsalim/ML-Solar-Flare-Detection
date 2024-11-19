Built an ML-based binary classification model using the data from the Helioseismic and Magnetic Imager Instrument on NASA’s Solar DynamicsObservatory (SDO) satellite that captures various solar events. 
It is the first instrument that continuously maps the vector magnetic field of the sun. 
The magnetic activity recorded using these instruments will serve as the feature set for the ML model.
A major solar event is defined as a burst of GEOS X-ray flux of peak magnitude above the M1.0 level. 
A positive solar event is defined as an active HARP region that flares with a peak magnitude above the M1.0 level, as reported in the GEOS database. A negative solar event is an active region that does not have such an event (a flare above M1.0 level) within 24 hours.
The goal of the classification model is to train on the given data and predict whether a major solar event will occur in the next 24 hours. 
This means that the classifier predicts whether a given solar event is positive (indicating a flaring active region) or negative (indicating a non-flaring active region).
I have implemented a Support Vector Machine (SVM) classifier that can distinguish between a positive and negative solar flare event. 
Evaluated using accuracymeasures, k-fold cross-validation, and on 2-different datasets.

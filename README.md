# DeepSCFT
DeepSCFT is a deep neural network (DNN) accelerated SCFT simulation program of polymeric materials. 
Polymer SCFT simulation relies on the fast computation of the well known
modified diffusion equation (MDE) and the effective iteration scheme to relax the "self consistent field",aka,chemical potential. This project is aimed at
accelerating the SCFT simulation by directly predict the density profile from
a given input potential field,i.e., omega --> rho, with the blackbox deep
neural network (DNN). A well trained DNN should give a relatively accurate 
approximation of rho for a given input omega,thus greatly decrease the computation time of the iteration,after sufficient long steps, we can swith to numerically solving the MDE in order to produce an accurate result. Given the fact that efficient iteration sheme (e.g., anderson mixing) ususally converges quite fast provided that the initial stateis close to the solution, which we expect to be farely true with the prediction from DNN. 


Any suggestions or commments are deeply appreciated by the author!

Jiuzhou Tang
Senior parallel engineer
National Super Computing Center in Wuxi,China.
E-mail: tangjiuzhou@iccas.ac.cn




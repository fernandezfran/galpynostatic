///------------------------------------------------------------------------------------///
///             Code for simulating a diagram diagnosis  			          ///
///------------------------------------------------------------------------------------///

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
//#include <conio.h>
#include  <time.h>
#include <omp.h> 

const int       Npx=1000;
const int       Npt=100000;
const int       NPOINT=100;             ///Numero de puntos del voltagrama
const int       NMOD=Npt/NPOINT;        ///Modulo para imprimir
const double    NX=Npx;
const double    NT=Npt;


FILE *archivo, *archivo1, *archivo2, *archivo3;

#define salida_en "VProfile-2-1-6-g0.dat"
#define salida_diag_en "Diagrma-TEST-profile.dat"
#define salida_perf_en "CProfile-SOC-0-7.dat"
/* ======================  VARIABLES GLOBALES  =========================  */

int     i,j,k,K;
double  tita0[Npx],tita1[Npx],x[Npx],l,ieqI,ieqII,E0II,titaTot0,titaTot1,ti,Ei,BmasII[Npx],AmasII[Npx],AmenosII[Npx],BmenosII[Npx];
double  I,i0,i1,KK,titT0,titT1,E00I,E00II,tiempo,r[Npx],alfaII[Npx],betaII[Npx],bN[Npx];
double  AII,BII,AkII,BkII,A0II,B0II,Dx,d,lambda,L,Vol,m,S,ic,phi,phi2,logL,ttot,utot,dt,Dt,Dd;
/* ===================   CAJAS   ====================== */

void graba();


/*---------------------------------------------------------------------------*/

int main()
{
  const int NMOD = Npt / NPOINT; /// Printing Module
  const double geo = 2.0;	///geometry parameter 0: plane, 1: cylinder, 2: sphere
  const double F = 96484.5561;   // faraday constant
  const double R = 8.314472;
  const double th = 3600.0;
  const double T=298.0;
  const double Qmax=372.0;
  const double rho=2.26;
  const double f = R * T / F;
  const int NXi = 12;
  const int NL = 12; 
  const double Xi0=4.0;
  const double Xif=-3.3;
  const double L0=-4.0;
  const double Lf=1.2;
  const double deltaXi = (Xif - Xi0) / (NXi - 1);
  const double deltaL = (Lf - L0) / (NL - 1);
  const double    D=1.69e-10;  
  const double    ks=3.07e-7;                      //c<konstante heterogenea de velocidad aparente en cm.s-1
  const double    Mr=72.0;
  const double    m=1.0;
  const double    Eoff=-0.15;
  const double	   g=0.0; 			///frumkin parameter
  
  const double SOCperf=0.7;
  const double paso=1e-4;
     
  // Diagram parameters
  double logXi[NXi];
  double logL[NL];
  double ii = 0.0;
  for (int i = 0; i < NXi; i++) {
    logXi[i] = Xi0 + deltaXi * ii;
    ii++;
  }
  ii = 0.0;
  for (int i = 0; i < NL; i++) {
    logL[i] = L0 + deltaL * ii;
    ii++;
  }
  int pp = 0;
  
  ///Threads define
  int num_threads = omp_get_num_procs();
  omp_set_num_threads(num_threads);
  //if(N_THREADS==0){int num_threads = omp_get_num_procs();omp_set_num_threads(num_threads);}
  //else{omp_set_num_threads(N_THREADS);}  
  
/// DIAGRAM LOOP
/// STEP-----------------------------------------------------------------------------------------
#pragma omp parallel
  {
#pragma omp for collapse(2) firstprivate(logXi, logL)
    for (int EL = 0; EL < NL; EL++) { /// L Loop
      for (int XI = 0; XI < NXi; XI++) {
        int thread_id = omp_get_thread_num();
        // printf("id=%d",thread_id);///Xi Loop
        /// Actualization of the parameters
        double L = pow(10, logL[EL]);
        double Xi = pow(10, logXi[XI]);
        double Cr = (ks / Xi) * (ks / Xi) * (th / D);   /// C-rate
        double d = 2.0 * sqrt((L * (1.0 + geo) * D * th) / Cr); /// particle diameter, cm
        double S = 2.0 * (1.0 + geo) * m / (rho * d); /// Surface area, cm2
        // double	Vol=m/rho; 						      	///Volume of
        // active mass, cm3
        double Rohm=0.0;
        double ic = - Cr * Qmax * m / (1000 * S); /// constant current density, A/cm2
        double iR = Rohm * ic * S;                         /// IR drop, A*ohm=V
        double c1 = rho / Mr;
        double iN = 1.0 / (F * D * c1);
      //  double ttot = 0.5 * 0.5 * d * (rho / Mr) * F / (-ic); /// total time, s CHEQUEAAARRRR
        double ttot = abs(Qmax * m * 3.6 / (ic * S));
        double NT = Npt;
        double NX = Npx;
        double Dt = ttot / (NT - 1.0);                       /// time step, s
        double Dd = 0.5 * d / (NX - 1.0);                    /// space step, cm
        

        // Cleaning vectors
        double betaT[Npx], alfaT[Npx], bN[Npx], tita0[Npx], tita1[Npx];

        for (int i = 0; i < Npx; i++) {
          betaT[i] = alfaT[i] = bN[i] = tita0[i] = tita1[i] = 0.0;
        }
        double ii = 0.0;
        double r[Npx];
        for (int i = 0; i < Npx; i++) {
          r[i] = ii * Dd;
          ii++;
        }

        ////Crank Nicholson parameters and Constant Thomas coefficients
        double Abi = D * Dt / (2.0*Dd * Dd);
        double Bbi = geo * D * Dt / (4.0 * Dd);
        double A0bi = 1.0 + (2.0 * Abi);
        double A0nbi= 1.0 - (2.0 * Abi); ///NUEVO
        alfaT[1] = 2.0 * Abi / A0bi;
        for (int i = 2; i < Npx; i++) {
          alfaT[i] =
              (Abi + (Bbi / (r[i - 1]))) / (A0bi - (Abi - (Bbi / (r[i - 1]))) * alfaT[i - 1]);
        }
        double ABmas[Npx];
        double ABmenos[Npx];
        for (int i = 0; i < Npx; i++) {
          ABmas[i] = Abi+(Bbi/(r[i]));
          ABmenos[i] = Abi-(Bbi/(r[i]));
        }
        /// Initial Point
        double ti = 0.0;
        for (int i = 0; i < Npx; i++) {
          tita1[i] = 1e-4;
        }
        double Ei = Eoff + 1.0; // any value just that Ei>Eoff
        double E0 = 0.0;

        int Npot = 0;
        int TP = 0;
        int out=0;

        /// TIME LOOP------------------------------------------------------------------------

        while(Ei>Eoff){
        ///POTENTIAL CALCULATION STEP
          //Search range of experimental points where superficial concentration (tita1) belongs
         
          	double  i0 = F*c1*ks*sqrt(tita1[Npx-1]*(1.0-tita1[Npx-1]));
          	double  Eg=E0+f*(g*(0.5-tita1[Npx-1])+log((1.0-tita1[Npx-1])/tita1[Npx-1])); ///Frumkin
          	//Potential calculation
          	Ei = Eg + 2.0*f*asinh(ic/(2.0*i0));
          	//printf("dtit=%f Ai=%f Bi=%f Ci=%f Di=%f E0=%f i0=%f Ei=%f",dtitas, Ai, Bi, Ci, Di, E0, i0, Ei);
      
       	 ///PRINT POTENTIAL PROFILE POINT
      	
      		 double SOC=0.0;
      		 for(int i=0;i<Npx;i++){SOC+=tita1[i];}
      		 SOC/=(Npx-1);  
      		 
      		///Potential Profile
      		/* if(TP%NMOD==0){(archivo =fopen (salida_en,"a"));
    			fprintf(archivo,"%f %f %f %f %f %f %f \n",(float)(SOC),(float)(Ei));
   		 fclose(archivo);
   		 }
           	*/
           	/*
           	///Concentration profile
       	if((SOC>SOCperf-paso)&&(SOC<SOCperf+paso)){
       		if(out==0){
       			(archivo1 =fopen (salida_perf_en,"a"));
        			for(int i=0;i<Npx;i++){
            	  			fprintf(archivo1,"%f %f %f\n",(float)(r[i]/(d*0.5)),(float)(tita1[i]),(float)(SOC));
        			}
        		fclose(archivo1);
        		out++;
        		}
        	
          	}*/
          
        ///ACTUALIZATION STEP
          for(int i=0;i<Npx;i++){
            tita0[i]=tita1[i];
            betaT[i]=bN[i]=tita1[i]=0.0;
          }

          //Vector of solutions and Thomas coefficients calculation ESTO CAMBIA CON CN
            bN[0]=A0nbi*tita0[0]+2.0*Abi*tita0[1]; ///CN
            bN[Npx-1]=A0nbi*tita0[Npx-1]+2*Abi*tita0[Npx-2]-ABmas[Npx-1]*4.0*Dd*(ic*iN);///CN
            for(int i=1;i<Npx-1;i++){bN[i]=A0nbi*tita0[i]+ABmas[i]*tita0[i+1]+ABmenos[i]*tita0[i-1];}///CN

            betaT[1]=bN[0]/A0bi;
            for(int i=2;i<Npx;i++){
                betaT[i]=(bN[i-1]+ABmenos[i-1]*betaT[i-1])/(A0bi-ABmenos[i-1]*alfaT[i-1]);
            }

          //Concentration calculation
            tita1[Npx-1]=(bN[Npx-1]+2.0*Abi*betaT[Npx-1])/(A0bi-2.0*Abi*alfaT[Npx-1]);
            for(int i=2;i<Npx+1;i++){
              tita1[Npx-i]=(alfaT[Npx-(i-1)]*tita1[Npx-(i-1)])+betaT[Npx-(i-1)];
            }
        ti+=Dt;TP++;    ///time increment
        }
        //printf("STOP");
        //getchar();


        /// PRINT POTENTIAL PROFILE POINT AFTER WHILE LOOP ENDS
        double SOC = 0.0;
        for (int i = 0; i < Npx; i++) {
          SOC += tita0[i];
        }
        SOC /= (Npx - 1);

   	 (archivo =fopen (salida_diag_en,"a"));
    		fprintf(archivo,"%f %f %f \n",(float)(logL[EL]),(float)(logXi[XI]),(float)(SOC));     
	 fclose(archivo);
	 
      } /// En of Xi loop
    }   /// End of L loop
  }     /// PARALLELIZATION
 }


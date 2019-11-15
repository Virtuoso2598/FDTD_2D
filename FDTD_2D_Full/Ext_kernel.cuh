#ifndef EXT_KERNEL
#define EXT_KERNEL

//#include "FDTD_cuda.cuh"


//  Fundamental Constants (MKS units)
    extern double pi;
    extern double C;
    extern double muO;
    extern double epsO;

	//  Specify Material Relative Permittivity and Conductivity
    extern double epsR;//free space

	//  Specify Number of Time Steps and Grid Size Parameters
    extern int nMax;  // total number of time steps
  

	// grid size corresponding to the number of Ez field components

    extern int Imax;
    extern int Jmax;
    extern int Kmax;

	//  Specify Grid Cell Size in Each Direction and Calculate the
  	//  Resulting Courant-Stable Time Step
	extern double ds;
    extern double dx;
    extern double dy;
    extern double dz; // cell size in each direction
    // time step increment
    extern double dt;

	//  Specify the Impulsive Source (Differentiated Gaussian) parameters
    extern double tw;//pulse width
    extern double tO;//delay
    extern double source;//Differentiated Gaussian source
	extern double amp;// Amplitude

    //Specify the Time Step at which the data has to be saved for Visualization
    extern int save_modulus;

	//  Specify the dipole Boundaries(A cuboidal rode- NOT as a cylinder)
    extern int istart, iend, jstart;
    extern int  jend, kstart, kend;

	//Output recording point
    extern int ksource;

	//  Specify the CPML Thickness in Each Direction (Value of Zero
	//  Corresponds to No PML, and the Grid is Terminated with a PEC)
    // PML thickness in each direction
    extern int nxPML_1, nxPML_2, nyPML_1;
    extern int nyPML_2, nzPML_1, nzPML_2;

	//  Specify the CPML Order and Other Parameters:
    extern int m, ma;

    extern double sig_x_max;
    extern double sig_y_max;
    extern double sig_z_max;
    extern double alpha_x_max;
    extern double alpha_y_max;
    extern double alpha_z_max;
    extern double kappa_x_max;
    extern double kappa_y_max;
    extern double kappa_z_max;
	
	//Loop indices
	extern int i,j,ii,jj,k,kk,n,l;
	
	//Index reculc indicators
	extern int Im, Jm, Km;
	extern int Ip, Jp, Kp; 
	extern int Jindex, Kindex, Iindex;


	// H & E Field components
    extern float *Hx;
    extern float *Hy;
    extern float *Hz;
    extern float *Ex;
    extern float *Ey;
    extern float *Ez;

	extern float *Jx;
    extern float *Jy;
    extern float *Jz;

	extern int *ID1;//medium definition array for Ex
	extern int *ID2;//medium definition array for Ey
	extern int *ID3;//medium definition array for Ez

	extern int *ID1_close;//medium definition array for Ex
	extern int *ID2_close;//medium definition array for Ey
	extern int *ID3_close;//medium definition array for Ez


	//  CPML components (Taflove 3rd Edition, Chapter 7)
    extern float *psi_Ezx_1;
    extern float *psi_Ezx_2;
    extern float *psi_Hyx_1;
    extern float *psi_Hyx_2;
    extern float *psi_Ezy_1;
    extern float *psi_Ezy_2;
    extern float *psi_Hxy_1;
    extern float *psi_Hxy_2;
    extern float *psi_Hxz_1;
    extern float *psi_Hxz_2;
    extern float *psi_Hyz_1;
    extern float *psi_Hyz_2;
    extern float *psi_Exz_1;
    extern float *psi_Exz_2;
    extern float *psi_Eyz_1;
    extern float *psi_Eyz_2;
    extern float *psi_Hzx_1;
    extern float *psi_Eyx_1;
    extern float *psi_Hzx_2;
    extern float *psi_Eyx_2;
    extern float *psi_Hzy_1;
    extern float *psi_Exy_1;
    extern float *psi_Hzy_2;
    extern float *psi_Exy_2;

    extern float *be_x_1, *ce_x_1, *alphae_x_PML_1, *sige_x_PML_1, *kappae_x_PML_1;
    extern float *bh_x_1, *ch_x_1, *alphah_x_PML_1, *sigh_x_PML_1, *kappah_x_PML_1;
    extern float *be_x_2, *ce_x_2, *alphae_x_PML_2, *sige_x_PML_2, *kappae_x_PML_2;
    extern float *bh_x_2, *ch_x_2, *alphah_x_PML_2, *sigh_x_PML_2, *kappah_x_PML_2;
    extern float *be_y_1, *ce_y_1, *alphae_y_PML_1, *sige_y_PML_1, *kappae_y_PML_1;
    extern float *bh_y_1, *ch_y_1, *alphah_y_PML_1, *sigh_y_PML_1, *kappah_y_PML_1;
    extern float *be_y_2, *ce_y_2, *alphae_y_PML_2, *sige_y_PML_2, *kappae_y_PML_2;
    extern float *bh_y_2, *ch_y_2, *alphah_y_PML_2, *sigh_y_PML_2, *kappah_y_PML_2;
    extern float *be_z_1, *ce_z_1, *alphae_z_PML_1, *sige_z_PML_1, *kappae_z_PML_1;
    extern float *bh_z_1, *ch_z_1, *alphah_z_PML_1, *sigh_z_PML_1, *kappah_z_PML_1;
    extern float *be_z_2, *ce_z_2, *alphae_z_PML_2, *sige_z_PML_2, *kappae_z_PML_2;
    extern float *bh_z_2, *ch_z_2, *alphah_z_PML_2, *sigh_z_PML_2, *kappah_z_PML_2;

	// denominators for the update equations
    extern float *den_ex;
    extern float *den_hx;
    extern float *den_ey;
    extern float *den_hy;
    extern float *den_ez;
    extern float *den_hz;

	//Max number of materials allowed
	extern int numMaterials;

    //permittivity, permeability and conductivity of diffrent materials
	extern double *epsilon;
	extern double *mu;
	extern double *sigma;
	extern double *Ro;

	//E field update coefficients
	extern float *CA;
	extern float *CB;

	//H field update coefficients
    extern float *DA;
    extern float *DB;

	//Save field parameters
	extern FILE *pEx, *pEy, *pEz, *pHx, *pHy, *pHz, *deb, *plog, *pLineDet, *pLineDet2, *pDet, *pSLine, *pComDet;
	extern FILE *p_detEx, *p_detEy, *p_detEz, *p_detHx, *p_detHy, *p_detHz, *p_infDetEx, *p_detAll, *p_detP;
	//extern int sieve, save_app=1, save_app_count, save_counter=0, Sindex;
	//extern int to_time, from_time;
	//extern float mag;

	// cubes
	extern float * proc_line;
	extern float ** proc_field;

	//detector
	extern float *detect;
	extern float *infDet;

	extern float *det_Ex, *det_Ey, *det_Ez, *det_Hx, *det_Hy, *det_Hz; 
	extern float *infDet_Ex, *infDet_Ey, *infDet_Ez, *infDet_Hx, *infDet_Hy, *infDet_Hz; 

	// cubes


	//Compensation detector parameters
	extern float *CompDet;

	//line detector
	extern float *line_detect;

	//Power steering on the line var
	extern float *line_medium;
	extern float *line_detect2;

	//runtime calc
	extern clock_t runTime;

	//Source parameter coef
	extern int pulseW;
	extern float fPulseW, alpha, beta, fCoef, fCoef2, Nes, *SourceP;

	extern float *pne;
	//AESA parameters
	extern int *arrayI, *arrayK;
	extern int qantI, qantK;
	//AESA Detector parameters
	extern int *det_arrayI, *det_arrayK;
	extern int det_qantI, det_qantK;

	extern int GPU_device;

	extern int RC_qantI, RC_qantK;
	extern int *RCdetArrayI, *RCdetArrayK;

	extern int *PspdLine;
	extern float *PspdFiltred;

	// ---------------------Struct------------------------------------

	extern struct CenCoord {
		int I;
		int J;
		int K;
	};

	extern struct SaveParam {
		int fromTime;
		int ToTime;
		int FieldSieve;
		int TimeSive;
	};

	extern struct regimes {
		int S2D;
	};

	extern struct PulseForm {
		float Angle;
		float Width;
	};

	extern struct FilterParam {
	int window;
	float level;
	};

	extern CenCoord SourceCen;
	extern CenCoord DetectCen;

	extern SaveParam FieldSaveParam;

	extern regimes useRegime;

	extern PulseForm AESAsourceParam;

	extern FilterParam RCdetFilt;

	extern Analysis Processing;

#endif // EXT_KERNEL
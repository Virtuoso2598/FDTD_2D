#ifndef _FDTD_CUDA_CUH_
#define _FDTD_CUDA_CUH_

//Header files (Libraries to be included)
	#include<stdio.h>
	#include<math.h>
	#include<string.h>
	#include<stdlib.h>
	#include<time.h>
	#include<fstream> 
	#include<iostream>
	#include"Settings.cuh"
	#include"INIlib/INIReader.h"

	#include "processing.cuh"



	//  Fundamental Constants (MKS units)
    double pi = 3.14159265358979;
    double C = 2.99792458E8;
    double muO;
    double epsO;

	//  Specify Material Relative Permittivity and Conductivity
    double epsR = 1.0;//free space

	//  Specify Number of Time Steps and Grid Size Parameters
    int nMax = MAX_TIME_STEP;  // total number of time steps
    
	// grid size corresponding to the number of Ez field components
	int Imax = I_SIZE;
	int Jmax = J_SIZE;
	int Kmax = K_SIZE;

	//  Specify Grid Cell Size in Each Direction and Calculate the
  	//  Resulting Courant-Stable Time Step

	double ds = DS;
	double dt_coef = TIME_COEF;

    double dx = DS*1E-3; //0.5E-3;	//1.0E-3;
    double dy = DS*1E-3;
    double dz = DS*1E-3; // cell size in each direction

    //double dx = 0.75E-3;
    //double dy = 0.75E-3;
    //double dz = 0.75E-3; // cell size in each direction

    // time step increment
	bool Exp_time_key;
    double dt;
	double Exp_time;

	double Plato_ns, Fronts_ns;

	//  Specify the Impulsive Source (Differentiated Gaussian) parameters
    double tw = 53.0E-12;//pulse width
    double tO;//delay
    double source;//Differentiated Gaussian source
	double amp = 1000;// Amplitude

    //Specify the Time Step at which the data has to be saved for Visualization
    int save_modulus = 10;

	//  Specify the dipole Boundaries(A cuboidal rode- NOT as a cylinder)
    int istart, iend, jstart;
    int  jend, kstart, kend;

	//Output recording point
    int ksource = 12;

	//  Specify the CPML Thickness in Each Direction (Value of Zero
	//  Corresponds to No PML, and the Grid is Terminated with a PEC)
    // PML thickness in each direction
    int nxPML_1, nxPML_2, nyPML_1;
    int nyPML_2, nzPML_1, nzPML_2;

	//  Specify the CPML Order and Other Parameters:
    int m = 3, ma = 1;

    double sig_x_max;
    double sig_y_max;
    double sig_z_max;
    double alpha_x_max;
    double alpha_y_max;
    double alpha_z_max;
    double kappa_x_max;
    double kappa_y_max;
    double kappa_z_max;
	
	//Loop indices
	int i,j,ii,jj,k,kk,n,l;
	
	//Index reculc indicators
	int Im, Jm, Km;
	int Ip, Jp, Kp; 
	int Jindex, Kindex, Iindex;


	// H & E Field components
    float *Hx;
    float *Hy;
    float *Hz;
    float *Ex;
    float *Ey;
    float *Ez;

    float *Jx;
    float *Jy;
    float *Jz;


	int *ID1;//medium definition array for Ex
	int *ID2;//medium definition array for Ey
	int *ID3;//medium definition array for Ez

	int *ID1_close;//2nd medium definition array for Ex
	int *ID2_close;//2nd medium definition array for Ey
	int *ID3_close;//2nd medium definition array for Ez

	//  CPML components (Taflove 3rd Edition, Chapter 7)
    float *psi_Ezx_1;
    float *psi_Ezx_2;
    float *psi_Hyx_1;
    float *psi_Hyx_2;
    float *psi_Ezy_1;
    float *psi_Ezy_2;
    float *psi_Hxy_1;
    float *psi_Hxy_2;
    float *psi_Hxz_1;
    float *psi_Hxz_2;
    float *psi_Hyz_1;
    float *psi_Hyz_2;
    float *psi_Exz_1;
    float *psi_Exz_2;
    float *psi_Eyz_1;
    float *psi_Eyz_2;
    float *psi_Hzx_1;
    float *psi_Eyx_1;
    float *psi_Hzx_2;
    float *psi_Eyx_2;
    float *psi_Hzy_1;
    float *psi_Exy_1;
    float *psi_Hzy_2;
    float *psi_Exy_2;

    float *be_x_1, *ce_x_1, *alphae_x_PML_1, *sige_x_PML_1, *kappae_x_PML_1;
    float *bh_x_1, *ch_x_1, *alphah_x_PML_1, *sigh_x_PML_1, *kappah_x_PML_1;
    float *be_x_2, *ce_x_2, *alphae_x_PML_2, *sige_x_PML_2, *kappae_x_PML_2;
    float *bh_x_2, *ch_x_2, *alphah_x_PML_2, *sigh_x_PML_2, *kappah_x_PML_2;
    float *be_y_1, *ce_y_1, *alphae_y_PML_1, *sige_y_PML_1, *kappae_y_PML_1;
    float *bh_y_1, *ch_y_1, *alphah_y_PML_1, *sigh_y_PML_1, *kappah_y_PML_1;
    float *be_y_2, *ce_y_2, *alphae_y_PML_2, *sige_y_PML_2, *kappae_y_PML_2;
    float *bh_y_2, *ch_y_2, *alphah_y_PML_2, *sigh_y_PML_2, *kappah_y_PML_2;
    float *be_z_1, *ce_z_1, *alphae_z_PML_1, *sige_z_PML_1, *kappae_z_PML_1;
    float *bh_z_1, *ch_z_1, *alphah_z_PML_1, *sigh_z_PML_1, *kappah_z_PML_1;
    float *be_z_2, *ce_z_2, *alphae_z_PML_2, *sige_z_PML_2, *kappae_z_PML_2;
    float *bh_z_2, *ch_z_2, *alphah_z_PML_2, *sigh_z_PML_2, *kappah_z_PML_2;

	// denominators for the update equations
    float *den_ex;
    float *den_hx;
    float *den_ey;
    float *den_hy;
    float *den_ez;
    float *den_hz;

	//Max number of materials allowed
	int numMaterials = 150;

    //permittivity, permeability and conductivity of diffrent materials
	double *epsilon;
	double *mu;
	double *sigma;
	double *Ro;

	//E field update coefficients
	float *CA;
	float *CB;

	//H field update coefficients
    float *DA;
    float *DB;

	//Save field parameters
	FILE *pEx, *pEy, *pEz, *pHx, *pHy, *pHz, *deb, *plog, *pLineDet, *pLineDet2, *pDet, *pSLine, *pComDet;
	FILE *p_detEx, *p_detEy, *p_detEz, *p_detHx, *p_detHy, *p_detHz, *p_infDetEx, *p_detAll, *p_detP;
	int sieve, save_app=1, save_app_count, save_counter=0, Sindex;
	int to_time, from_time;
	float mag;
	
	//detector
	float *detect;
	float *infDet;

	float *det_Ex, *det_Ey, *det_Ez, *det_Hx, *det_Hy, *det_Hz;
	float *infDet_Ex, *infDet_Ey, *infDet_Ez, *infDet_Hx, *infDet_Hy, *infDet_Hz;

	//Compensation detector parameters
	float *CompDet;

	//line detector
	float *line_detect;

	// processing 
	float * proc_line;
	float ** proc_field;

	//Power steering on the line var
	float *line_medium;
	float *line_detect2;

	//runtime calc
	clock_t runTime;

	//Source parameter coef
	int pulseW;
	float fPulseW;
	float alpha, beta, fCoef, fCoef2, Nes, *SourceP;
	//AESA parameters
	int *arrayI, *arrayK;
	int qantI, qantK;
	//AESA Detector parameters
	int *det_arrayI, *det_arrayK;
	int det_qantI, det_qantK;
	//Elecro densety profile
	float *pne;

	//RCdetector parameters
	int RC_qantI, RC_qantK;

	float *PspdFiltred;
	int *PspdLine;

	int *RCdetArrayI;
	int *RCdetArrayK;


	
	//Function prototype definitions
	int readINIparam();//read ini parameters from settings.ini
	int logINIread();//log ini parameters to file
	void recuclParametersToMetric();//recalculation of parameters in the metric system
	void initlimits();//init initial setengs
	void initialize();//Memeory initialization
	void setUp();//Coefficients, parameters etc will get computed
	void initializeCPML();//CPML coefficient computation
	void compute();//E & H Field update equation
	void buildObject();//Creates the object geometry
	void yeeCube (int, int,int, int);//Sets material properties to a cell
	void yeeCubeFace (int I, int J,int K, int mType, char face);
	void writeField(int);//Writes output
	void buildSphere();//Builds a spherical object
	void buildDipole();//Builds a dipole

	void addYeeCube (int I, int J,int K, int mType, char face);  //sets difference between existing cube material and new 	
	void ChangeYeeCube (int I, int J,int K, int mType); //Sets material properties to a 2nd material cell
	void CopyMedium();  //Copy medium 1(ID ) to medium 2 (ID _close)
	void buildPlateMedium(); //Make plate wall 
	void buildMedium();//Builds a medium
	void buildSphericalMedium(); //Builds a shrecal medium 
	void buildSphericalMediumFluc(int EqTupe, int I_cen, int J_cen, int K_cen, int P_coef, int Q_coef, int N_coef, float A_coef, float B_coef, float C_coef, float pne0, float ra, float alpha, float beta, float angle_x, float angle_y, float angle_z, int sign); //Add a shrecal medium fluctuations 
	void buildTorusMedium(); //Builds Toroidal medium 
	void buildSourceProfile(); //Build ampl source profile
	void calculate2Darrai(); //Build AESA Source in 2D space
	void calculateDetect2Darrai(); ////Build AESA Detector in 2D space
	void calculateRCdetect2Darrai();//Build RC detect net
	void buildCubikAntenna(int Antena_Tupe); //Build metal cubik antenna around source
	void MakeCloseTheDoor(); //Close antena hole in 2nd medium 
	void buildCubikDetector(int Antena_Tupe); //Build metal cubik antenna around main detector

	void draw_2D_figure();
	void draw_width_line(int x1, int y1, int x2, int y2, int matN); // draw 2D line on 2D field with width 2 pixels, using "draw_line" function twice, second from right side.
	void draw_line(int x1, int y1, int x2, int y2, int matN); // draw 2D line on 2D field
	void draw_Circl(int a, int b, int R, int width, int matN); //draw 2D circle in center '(a,b)' with radius 'R', width 'width' //NB draw 0(vacuum) inside the circle, use cerefully, may erase some important things.
	void Draw_Square(int x1, int y1, int x2, int y2, int matN); //draw 2D parallelogram from '(x1, y1)' to '(x2, y2)'

	int Build_medium=0;
	int Build_fluc_medium=0;

	int Build_Stell_wall=0;

	int GPU_device;

//------------settings parameters structs-----------------

	struct CenCoord {
		int I;
		int J;
		int K;
	};

	struct FloatDec {
		float I;
		float J;
		float K;
	};

	struct AESAparameters {
		int PI;
		int PK;
		int deltaAuto;
		int DI;
		int DK;
	};

	struct SaveParam {
		int fromTime;
		int ToTime;
		int FieldSieve;
		int TimeSive;
	};

	struct FigureCoef {
		float A;
		float B;
		float C;
		int P;
		int Q;
		int N;
	};

	struct NeEqCoef {
		int R;
		float Alpha;
		float Beta;
		float ne0;
		float neA;
	};

	struct MaterialFigureCoef {
		int Draw2D;
		int FigPreset;
		int PatLength;
		int DrawWall;
		int WallPos;
	};

	struct PulseCoef {
		int chirp;
		float freqA;
		float freqB;
		int Plato;
		int Front;
	};

	struct regimes {
		int S2D;
	};

	struct PulseForm {
		float Angle;
		float Width;
	};

	struct FilterParam {
	int window;
	float level;
	};



/*	struct Analysis { // только по Ez
		bool perform;
		// методы прив€зки
		bool sec_der;
		bool pf;
		bool kav;
		// методы филтровани€
		bool medfilt;
		bool floating_average;

		bool save_original;
		bool save_filtred;

		int dim_sieve;
		int time_sieve;

		int begin_after;
		int pos_i;
		int pos_j;
		int pos_k;
		int size_i;
	}; */

	int Torus_big_r;
	int Order;

	int flucSign;

	bool ObjectsEnable;


	CenCoord RCdetectCen;
	AESAparameters RCdetect;
	FilterParam RCdetFilt;

	
	CenCoord SourceCen;
	CenCoord DetectCen;
	CenCoord MediumCen;
	CenCoord MediumFlucCen;

	FloatDec FlucAngle;

	AESAparameters SourceAESA;
	AESAparameters DetectAESA;

	SaveParam FieldSaveParam;

	FigureCoef SphereCoef;
	FigureCoef SphereFlucCoef;

	NeEqCoef EqCoef;
	NeEqCoef FlucEqCoef;

	MaterialFigureCoef Kamera;

	PulseCoef SourcePulse;

	regimes useRegime;

	PulseForm AESAsourceParam;

	Analysis Processing;

#endif // _FDTD_CUDA_CUH_
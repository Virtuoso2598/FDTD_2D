#ifndef KERNEL
#define KERNEL

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> 
#include <fstream> 
#include <iostream> 
#include "Settings.cuh"

#include "processing.cuh"


#ifdef I_SIZE
	__constant__ int DImax;
#endif

#ifdef J_SIZE
	__constant__ int DJmax;
#endif

#ifdef K_SIZE
	__constant__ int DKmax;
#endif

#ifdef PML_SIZE
	__constant__ int DPML_size=PML_SIZE;
#endif


//-------------------------calc kernels------------------------>>>>>>>>>>>>>>>>>>>>
	//__global__ void DevHcalc(float *Hclc, float *H_frst, float *E_frst, float *E_scnd, float *den_frst, float *den_scnd, float DA, float DB, 
	//					 int PLUS_1, int PLUS_2, int PLUS_3, int PLUS_4, int Iminus, int Jminus, int Kminus, int Iplus, int Jplus, int Kplus );

//------------------------------Jx Jy Jz---------------------------

	__global__ void Dev_JzCalc(float *DJz, float *DEz, int *D_ID3, float *DNe, double D_dt);

//--------------------------------Hx----------------------------
	__global__ void Dev_HxCalc(float *DHx, float *DEz, int *D_ID1, float *D_DA, float *D_DB, float *D_den_hy, float *D_den_hz);

	__global__ void Dev_HxPML_Jdir_Calc(float *DHx, float *DEz, float *D_DB, float *D_psi_Hxy_1, float *D_psi_Hxy_2, float *D_bh_y_1, float *D_bh_y_2,
									float *D_ch_y_1, float *D_ch_y_2, int *D_ID1, float dy );
	__global__ void Dev_HxPML_Kdir_Calc(float *DHx, float *DEy, float *D_DB, float *D_psi_Hxz_1, float *D_psi_Hxz_2, float *D_bh_z_1, float *D_bh_z_2,
									float *D_ch_z_1, float *D_ch_z_2, int *D_ID1, float dz );
//--------------------------------Hy----------------------------
	__global__ void Dev_HyCalc(float *DHy, float *DEz, int *D_ID2, float *D_DA, float *D_DB, float *D_den_hx, float *D_den_hz);

	__global__ void Dev_HyPML_Idir_Calc(float *DHy, float *DEz, float *D_DB, float *D_psi_Hyx_1, float *D_psi_Hyx_2, float *D_bh_x_1, float *D_bh_x_2,
									float *D_ch_x_1, float *D_ch_x_2, int *D_ID2, float dx );
	__global__ void Dev_HyPML_Kdir_Calc(float *DHy, float *DEx, float *D_DB, float *D_psi_Hyz_1, float *D_psi_Hyz_2, float *D_bh_z_1, float *D_bh_z_2,
									float *D_ch_z_1, float *D_ch_z_2, int *D_ID2, float dz );
//--------------------------------Hz----------------------------
//--------------------------------Ex---------------------------
//--------------------------------Ey---------------------------
//--------------------------------Ez---------------------------
	__global__ void Dev_EzCalc(float *DEz, float *DHy, float *DHx, float *DJz, int *D_ID3, float *D_CA, float *D_CB, float *D_den_ex, float *D_den_ey);

	__global__ void Dev_EzPML_Xdir_Calc(float *DEz, float *DHy, float *D_CB, float *D_psi_Ezx_1, float *D_psi_Ezx_2, float *D_be_x_1, float *D_be_x_2,
									float *D_ce_x_1, float *D_ce_x_2, int *D_ID3, float dx );
	__global__ void Dev_EzPML_Ydir_Calc(float *DEz, float *DHx, float *D_CB, float *D_psi_Ezy_1, float *D_psi_Ezy_2, float *D_be_y_1, float *D_be_y_2,
									float *D_ce_y_1, float *D_ce_y_2, int *D_ID3, float dy );

	// cubes
	__global__ void Dev_SaveLine(float * DEz, float * D_proc_line, Analysis A);
	// tau

//--------------------------------Source---------------------------
	__global__ void Dev_EzAppSourceCalc(int i, int j, int k, float *DEz, int *D_ID3, float *D_CB, int n, float amp, float dt, float tO, float tw, int SourceSize );
	__global__ void Dev_EzAppSourceCalc2(int i, int j, int k, float *DEz, int *D_ID3, float *D_CB, int n, float *amp, float dt, float tO, float fCoef, int SourceSize );
	__global__ void Dev_UnAppSourceCalc(int i_cen, int j_cen, int k_cen, float *DEx, float *DEy, float *DEz, int *D_ID3, float *D_CB, int n, float *amp, float dt, float tO, float fCoef, float fCoef2, int s_sizeI, int s_sizeJ, int s_sizeK, char Dir );
	__global__ void Dev_AESASourceCalcDev_AESASourceCalc(int i_cen, int j_cen, int k_cen, float *DEz, int *D_ID3, float *D_CB, int n, float *amp, int *D_arrayI, int *D_arrayK,  float dt, float tO, float fCoef, float fCoef2, float PulseW, int s_sizeI, int s_sizeJ, int s_sizeK, float Angle );

//--------------device calculation kernels------------		
	__device__ float Dev_calculateChirp( float f0, float f1, float t1, float p, float t, float phi);
	__device__ float Dev_FindMax(float *PwrMedium, int i_pos ,int k_pos, int counts);

//--------------Addition parameters calculation kernels------------				
	__global__ void Dev_AESAComponentDetector(float *DHx, float *DHy, float *DHz, float *DEx, float *DEy, float *DEz, float *D_det_Ex, float *D_det_Ey, float *D_det_Ez, float *D_det_Hx, float *D_det_Hy, float *D_det_Hz, int *D_arrayI, int *D_arrayK, int d_sizeI, int d_sizeK, int i_det, int j_det, int k_det, int n);
	__global__ void Dev_AESADetectorSim(float *DHx, float *DHy, float *DEz, float *D_detect, int *D_arrayI, int *D_arrayK, int d_sizeI, int d_sizeK, int i_det, int j_det, int k_det, int n);
	__global__ void Dev_AESACompDetectorSim(float *DHx, float *DHy, float *DHz, float *DEx, float *DEy, float *DEz, float *D_detect, int *D_arrayI, int *D_arrayK, float *D_Plust, int d_sizeI, int d_sizeK, int i_det, int j_det, int k_det, int n);
	__global__ void Dev_DetectorSim(float *DHx, float *DHy, float *DHz, float *DEx, float *DEy, float *DEz, float *D_detect, int det_size, int i_det, int j_det, int k_det, int n);
	__global__ void Dev_LineDetectorSim(float *DHx, float *DHy, float *DHz, float *DEx, float *DEy, float *DEz, float *D_line_detect, int det_size, int det_quant, int det_step, int i_det, int j_det, int k_det, int n);
	__global__ void Dev_PowerLineSave(float *DHx, float *DHy, float *DHz, float *DEx, float *DEy, float *DEz, float *D_line_medium, int i_coord, int k_coord, int n);
	__global__ void Dev_RCcalc(float *DEz, int *DSpdLine, int *D_arrayI, int *D_arrayK,  float edgePercent, float *DedgeL, int d_sizeI, int d_sizeK, int n);
	__global__ void DevFilterPostTime(int *mas, float *ans, int counts, int w_size); //isn't use yet... calc on CPU

//-------------------------calc kernels------------------------<<<<<<<<<<<<<<<<<<<<
//-------------------------calc prepare kernels---------------->>>>>>>>>>>>>>>>>>>>
	__global__ void DevCalcPrepare(float *dCoef, int DCoef_size );	
	__global__ void DevCalcPrepare(int *dCoef, int DCoef_size );
	__global__ void DevCalcPrepare(float *dCoef, int DCoef_size , float initial);
//-------------------------calc prepare kernels----------------<<<<<<<<<<<<<<<<<<<<


	__host__ cudaError_t DevMemoryAllocation();
	__host__ cudaError_t DevSourceToDetectInfluenceCalc();
	__host__ cudaError_t DevFieldCalculation();
	__host__ cudaError_t CheckCoefInGPU(float *dCoef, int DCoef_size );	
	__host__ cudaError_t CheckCoefInGPU(int *dCoef, int DCoef_size );
	__host__ cudaError_t DevMemoryFree();
	__host__ void VriteSaveParameters( int SliceSaved );		
	__host__ cudaError_t CloseOpenDoor( int action );	
	__host__ cudaError_t CheckDimConstInGPU( );

//-------------------------------Additional host subsidiary function---------------------

	__host__ void findSpdLine(float *mas);
	__host__ float FindMax(float *mas, int counts);
	__host__ void FilterPostTime(int *mas, float *ans, int counts);

	__host__ void CalculateRCfromPwrLines(int *mas, float *fltr);
	__host__ void CalculateTaufromPwrLines(int *mas, float *fltr);
	__host__ void RClineFilterPostTime(int *mas, float *ans, int w_size);



#endif // KERNEL
#include "Kernel.cuh"
#include "Ext_kernel.cuh"

#include "processing.cuh"

//--------------device calculation kernels------------				


	//  CPML components (Taflove 3rd Edition, Chapter 7) on GPU
    float *D_psi_Ezx_1;
    float *D_psi_Ezx_2;
    float *D_psi_Hyx_1;
    float *D_psi_Hyx_2;
    float *D_psi_Ezy_1;
    float *D_psi_Ezy_2;
    float *D_psi_Hxy_1;
    float *D_psi_Hxy_2;
    float *D_psi_Hxz_1;
    float *D_psi_Hxz_2;
    float *D_psi_Hyz_1;
    float *D_psi_Hyz_2;
    float *D_psi_Exz_1;
    float *D_psi_Exz_2;
    float *D_psi_Eyz_1;
    float *D_psi_Eyz_2;
    float *D_psi_Hzx_1;
    float *D_psi_Eyx_1;
    float *D_psi_Hzx_2;
    float *D_psi_Eyx_2;
    float *D_psi_Hzy_1;
    float *D_psi_Exy_1;
    float *D_psi_Hzy_2;
    float *D_psi_Exy_2;


	float *D_be_x_1, *D_ce_x_1;
    float *D_bh_x_1, *D_ch_x_1;
    float *D_be_x_2, *D_ce_x_2;
    float *D_bh_x_2, *D_ch_x_2;
    float *D_be_y_1, *D_ce_y_1;
    float *D_bh_y_1, *D_ch_y_1;
    float *D_be_y_2, *D_ce_y_2;
    float *D_bh_y_2, *D_ch_y_2;
    float *D_be_z_1, *D_ce_z_1;
    float *D_bh_z_1, *D_ch_z_1;
    float *D_be_z_2, *D_ce_z_2;
    float *D_bh_z_2, *D_ch_z_2;

	// denominators for the update equations
    float *D_den_ex;
    float *D_den_hx;
    float *D_den_ey;
    float *D_den_hy;
    float *D_den_ez;
    float *D_den_hz;


//field	on GPU
	float *DHx;
	float *DHy;
	float *DHz;

	float *DEx;
	float *DEy;
	float *DEz;

	float *DJx;
	float *DJy;
	float *DJz;

	int *D_ID1;
	int *D_ID2;
	int *D_ID3;

	//E field update coefficients
	float *D_CA;
	float *D_CB;

	//H field update coefficients
    float *D_DA;
    float *D_DB;

	float *D_NE;

	//detector
	float *D_detect;
	//int i_det=I_DET;
	//int j_det=J_DET;
	//int k_det=K_DET;

	float *D_det_Ex, *D_det_Ey, *D_det_Ez, *D_det_Hx, *D_det_Hy, *D_det_Hz;

	// processing
	float * D_proc_line;


	//line detector
	float *D_line_detect;
	float *D_line_detect2;

	//Source
	float *D_sourceP;

	//AESA parameters
	int *D_arrayI, *D_arrayK, *Ddet_arrayI, *Ddet_arrayK;
	int *D_RCdetArrayI, *D_RCdetArrayK;
	//RCdet param
	float *DPspdFiltred;
	int *DPspdLine;
	float *DedgeL;

	//Compensation detector parameters
	float *D_Plust;					//вспомогательный массив для хранения мощности в прошлый момент времени.
	float *D_CompDet;

	//Power steering on the line var
	float *D_line_medium;

	__constant__ float D_CJ=2.81794E-8;		//e^2/me	Заряд электрона в квадрате на массу электрона


	//__global__ float edgeL;




__global__ void Dev_JzCalc(float *DJz, float *DEz, int *D_ID3, float *DNe, double D_dt)
{

int Im, Jm, Km, Ip, Jp, Kp, Jindex, Kindex, Iindex, i, j, k, id;
int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;

	Im=DImax-2;
	Jm=DJmax-2;
	Km=1;

	if(l < Im*Jm*Km)
	{

		Ip=1;
		Jp=1;
		Kp=0; 

		Jindex=(l/(Im*Km));
		Kindex=((l-(Jindex*(Im*Km)))/Im);
		Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

		i=Iindex+Ip;
		j=Jindex+Jp;
		k=Kindex+Kp;
//-----------------------------------------------------------------
id = D_ID3[j*(DImax*DKmax)+(k*DImax)+i];
if(DNe[id]>0)
		{
		DJz[j*(DImax*DKmax)+(k*DImax)+i] = DJz[j*(DImax*DKmax)+(k*DImax)+i] +D_CJ * DEz[j*(DImax*DKmax)+(k*DImax)+i] * DNe[id] * D_dt;

				//DEz[j*(DImax*DKmax)+(k*DImax)+i] = D_CA[id] * DEz[j*(DImax*DKmax)+(k*DImax)+i] + D_CB[id]
				//		* ((DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+i] - DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+(i-1)]) * D_den_ex[i] +
				//	(DHx[(j-1)*(DImax*DKmax)+(k*DImax)+i] - DHx[j*(DImax*DKmax)+(k*DImax)+i]) * D_den_ey[j]);
		}
//-----------------------------------------------------------------
}
}


//--------------------------------Hx----------------------------------------------------------
__global__ void Dev_HxCalc(float *DHx, float *DEz, int *D_ID1, float *D_DA, float *D_DB, float *D_den_hy, float *D_den_hz)
{

int Im, Jm, Km, Ip, Jp, Kp, Jindex, Kindex, Iindex, i, j, k, id;
int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;

	Im=DImax-1;
	Jm=DJmax-1;
	Km=1;

	if(l < Im*Jm*Km)
	{
		Ip=0;
		Jp=0;
		Kp=0; 

		Jindex=(l/(Im*Km));
		Kindex=((l-(Jindex*(Im*Km)))/Im);
		Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

		i=Iindex+Ip;
		j=Jindex+Jp;
		k=Kindex+Kp;
//-----------------------------------------------------------------

		id = D_ID1[j*(DImax*DKmax)+(k*DImax)+i];
		DHx[j*(DImax*DKmax)+(k*DImax)+i] = D_DA[id] * DHx[j*(DImax*DKmax)+(k*DImax)+i] + D_DB[id] *
		((DEz[j*(DImax*DKmax)+(k*DImax)+i] - DEz[(j+1)*(DImax*DKmax)+(k*DImax)+i]) * D_den_hy[j] );      
	}
}

__global__ void Dev_HxPML_Jdir_Calc(float *DHx, float *DEz, float *D_DB, float *D_psi_Hxy_1, float *D_psi_Hxy_2, float *D_bh_y_1, float *D_bh_y_2,
									float *D_ch_y_1, float *D_ch_y_2, int *D_ID1, float dy )
{

int Im, Jm, Km, Ip, Jp, Kp, Jindex, Kindex, Iindex, i, j, k, jj, id;
int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;

	Im=DImax-1;
	Jm=DPML_size-1;
	Km=1;

	if(l < Im*Jm*Km)
	{


		Ip=0;
		Jp=0;
		Kp=0; 

		Jindex=(l/(Im*Km));
		Kindex=((l-(Jindex*(Im*Km)))/Im);
		Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

		i=Iindex+Ip;
		j=Jindex+Jp;
		k=Kindex+Kp;

//-----------------------------------------------------------------------------

		id = D_ID1[j*(DImax*DKmax)+(k*DImax)+i];
		D_psi_Hxy_1[j*(DImax*DKmax)+(k*DImax)+i] = D_bh_y_1[j] * D_psi_Hxy_1[j*(DImax*DKmax)+(k*DImax)+i]
			+ D_ch_y_1[j] * (DEz[j*(DImax*DKmax)+(k*DImax)+i] - DEz[(j+1)*(DImax*DKmax)+(k*DImax)+i]) / dy;
		DHx[j*(DImax*DKmax)+(k*DImax)+i] = DHx[j*(DImax*DKmax)+(k*DImax)+i] + D_DB[id] * D_psi_Hxy_1[j*(DImax*DKmax)+(k*DImax)+i];

//----------------------------------------------------------------------------
		jj=(DJmax-2)-j;

		id = D_ID1[jj*(DImax*DKmax)+(k*DImax)+i];
		D_psi_Hxy_2[j*(DImax*DKmax)+(k*DImax)+i] = D_bh_y_2[j] * D_psi_Hxy_2[j*(DImax*DKmax)+(k*DImax)+i]
			+ D_ch_y_2[j] * (DEz[jj*(DImax*DKmax)+(k*DImax)+i] - DEz[(jj+1)*(DImax*DKmax)+(k*DImax)+i]) / dy;
		DHx[jj*(DImax*DKmax)+(k*DImax)+i] = DHx[jj*(DImax*DKmax)+(k*DImax)+i] + D_DB[id] * D_psi_Hxy_2[j*(DImax*DKmax)+(k*DImax)+i];
	}
}


__global__ void Dev_HxPML_Kdir_Calc(float *DHx, float *DEy, float *D_DB, float *D_psi_Hxz_1, float *D_psi_Hxz_2, float *D_bh_z_1, float *D_bh_z_2,
									float *D_ch_z_1, float *D_ch_z_2, int *D_ID1, float dz )
{

	int Im, Jm, Km, Ip, Jp, Kp, Jindex, Kindex, Iindex, i, j, k, kk, id;
	int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;

	Im=DImax-1;
	Jm=DJmax-1;
	Km=DPML_size-1;

	if(l < Im*Jm*Km)
	{

		Ip=0;
		Jp=0;
		Kp=1; 

		Jindex=(l/(Im*Km));
		Kindex=((l-(Jindex*(Im*Km)))/Im);
		Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

		i=Iindex+Ip;
		j=Jindex+Jp;
		k=Kindex+Kp;


//-----------------------------------------------------------------------------
		id = D_ID1[j*(DImax*DKmax)+(k*DImax)+i];

		D_psi_Hxz_1[j*(DImax*Km)+((k-1)*DImax)+i] = D_bh_z_1[k-1] * D_psi_Hxz_1[j*(DImax*Km)+((k-1)*DImax)+i]
			+ D_ch_z_1[k-1] * (DEy[j*(DImax*(DKmax-1))+(k*DImax)+i] - DEy[j*(DImax*(DKmax-1))+((k-1)*DImax)+i]) / dz;
		DHx[j*(DImax*DKmax)+(k*DImax)+i] = DHx[j*(DImax*DKmax)+(k*DImax)+i] + D_DB[id] * D_psi_Hxz_1[j*(DImax*Km)+((k-1)*DImax)+i];

//----------------------------------------------------------------------------
		Kp=0; k=Kindex+Kp; 	
		kk=(DKmax-2)-k;

		id = D_ID1[j*(DImax*DKmax)+(kk*DImax)+i];
		D_psi_Hxz_2[j*(DImax*Km)+(k*DImax)+i] = D_bh_z_2[k] * D_psi_Hxz_2[j*(DImax*Km)+(k*DImax)+i]
			+ D_ch_z_2[k] * (DEy[j*(DImax*(DKmax-1))+(kk*DImax)+i] - DEy[j*(DImax*(DKmax-1))+((kk-1)*DImax)+i]) / dz;
		DHx[j*(DImax*DKmax)+(kk*DImax)+i] = DHx[j*(DImax*DKmax)+(kk*DImax)+i] + D_DB[id] * D_psi_Hxz_2[j*(DImax*Km)+(k*DImax)+i];	
	}
}

__global__ void Dev_HyCalc(float *DHy, float *DEz, int *D_ID2, float *D_DA, float *D_DB, float *D_den_hx, float *D_den_hz)
{

int Im, Jm, Km, Ip, Jp, Kp, Jindex, Kindex, Iindex, i, j, k, id;
int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;

	Im=DImax-1;
	Jm=DJmax-1;
	Km=1;

	if(l < Im*Jm*Km)
	{

		Ip=0;
		Jp=0;
		Kp=0; 

		Jindex=(l/(Im*Km));
		Kindex=((l-(Jindex*(Im*Km)))/Im);
		Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

		i=Iindex+Ip;
		j=Jindex+Jp;
		k=Kindex+Kp;
//-----------------------------------------------------------------

		id = D_ID2[j*(DImax*DKmax)+(k*DImax)+i];
		DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+i] = D_DA[id] * DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+i] + D_DB[id] *
		((DEz[j*(DImax*DKmax)+(k*DImax)+(i+1)] - DEz[j*(DImax*DKmax)+(k*DImax)+i]) * D_den_hx[i]);   
	}
}


__global__ void Dev_HyPML_Idir_Calc(float *DHy, float *DEz, float *D_DB, float *D_psi_Hyx_1, float *D_psi_Hyx_2, float *D_bh_x_1, float *D_bh_x_2,
									float *D_ch_x_1, float *D_ch_x_2, int *D_ID2, float dx )
{

int Im, Jm, Km, Ip, Jp, Kp, Jindex, Kindex, Iindex, i, j, k, ii, id;
int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;

	Im=DPML_size-1;
	Jm=DJmax-1;
	Km=1;

	if(l < Im*Jm*Km)
	{

		Ip=0;
		Jp=0;
		Kp=0; 

		Jindex=(l/(Im*Km));
		Kindex=((l-(Jindex*(Im*Km)))/Im);
		Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

		i=Iindex+Ip;
		j=Jindex+Jp;
		k=Kindex+Kp;

//-----------------------------------------------------------------------------

		id = D_ID2[j*(DImax*DKmax)+(k*DImax)+i];
		D_psi_Hyx_1[j*(Im*DKmax)+(k*Im)+i] = D_bh_x_1[i] * D_psi_Hyx_1[j*(Im*DKmax)+(k*Im)+i]
			+ D_ch_x_1[i] * (DEz[j*(DImax*DKmax)+(k*DImax)+(i+1)] - DEz[j*(DImax*DKmax)+(k*DImax)+i]) / dx;
		DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+i] = DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+i] + D_DB[id] * D_psi_Hyx_1[j*(Im*DKmax)+(k*Im)+i];

//----------------------------------------------------------------------------
	ii=(DImax-2)-i;

		id = D_ID2[j*(DImax*DKmax)+(k*DImax)+ii];
		D_psi_Hyx_2[j*(Im*DKmax)+(k*Im)+i] = D_bh_x_2[i] * D_psi_Hyx_2[j*(Im*DKmax)+(k*Im)+i]
			+ D_ch_x_2[i] * (DEz[j*(DImax*DKmax)+(k*DImax)+(ii+1)] - DEz[j*(DImax*DKmax)+(k*DImax)+ii]) / dx;
		DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+ii] = DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+ii] + D_DB[id] * D_psi_Hyx_2[j*(Im*DKmax)+(k*Im)+i];
	}
}

__global__ void Dev_HyPML_Kdir_Calc(float *DHy, float *DEx, float *D_DB, float *D_psi_Hyz_1, float *D_psi_Hyz_2, float *D_bh_z_1, float *D_bh_z_2,
									float *D_ch_z_1, float *D_ch_z_2, int *D_ID2, float dz )
{

int Im, Jm, Km, Ip, Jp, Kp, Jindex, Kindex, Iindex, i, j, k, kk, id;
int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;

	Im=DImax-1;
	Jm=DJmax-1;
	Km=DPML_size-1;

	if(l < Im*Jm*Km)
	{

		Ip=0;
		Jp=0;
		Kp=1; 

		Jindex=(l/(Im*Km));
		Kindex=((l-(Jindex*(Im*Km)))/Im);
		Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

		i=Iindex+Ip;
		j=Jindex+Jp;
		k=Kindex+Kp;

//-----------------------------------------------------------------------------
		id = D_ID2[j*(DImax*DKmax)+(k*DImax)+i];
		D_psi_Hyz_1[j*((DImax-1)*Km)+((k-1)*(DImax-1))+i] = D_bh_z_1[k-1] * D_psi_Hyz_1[j*((DImax-1)*Km)+((k-1)*(DImax-1))+i]
			+ D_ch_z_1[k-1] * (DEx[j*((DImax-1)*(DKmax-1))+((k-1)*(DImax-1))+i] - DEx[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i]) / dz;
		DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+i] = DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+i] + D_DB[id] * D_psi_Hyz_1[j*((DImax-1)*Km)+((k-1)*(DImax-1))+i];

//----------------------------------------------------------------------------
		//Kp=0; 
		k=Kindex; 	
		kk=(DKmax-2)-k;

		id = D_ID2[j*(DImax*DKmax)+(kk*DImax)+i];
		D_psi_Hyz_2[j*((DImax-1)*Km)+(k*(DImax-1))+i] = D_bh_z_2[k] * D_psi_Hyz_2[j*((DImax-1)*Km)+(k*(DImax-1))+i]
				+ D_ch_z_2[k] * (DEx[j*((DImax-1)*(DKmax-1))+((kk-1)*(DImax-1))+i] - DEx[j*((DImax-1)*(DKmax-1))+(kk*(DImax-1))+i]) / dz;
		DHy[j*((DImax-1)*DKmax)+(kk*(DImax-1))+i] = DHy[j*((DImax-1)*DKmax)+(kk*(DImax-1))+i] + D_DB[id] * D_psi_Hyz_2[j*((DImax-1)*Km)+(k*(DImax-1))+i];
	}
}






__global__ void Dev_EzCalc(float *DEz, float *DHy, float *DHx, float *DJz, int *D_ID3, float *D_CA, float *D_CB, float *D_den_ex, float *D_den_ey)
{

int Im, Jm, Km, Ip, Jp, Kp, Jindex, Kindex, Iindex, i, j, k, id;
int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;

	Im=DImax-2;
	Jm=DJmax-2;
	Km=1;

	if(l < Im*Jm*Km)
	{

		Ip=1;
		Jp=1;
		Kp=0; 

		Jindex=(l/(Im*Km));
		Kindex=((l-(Jindex*(Im*Km)))/Im);
		Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

		i=Iindex+Ip;
		j=Jindex+Jp;
		k=Kindex+Kp;
//-----------------------------------------------------------------

			id = D_ID3[j*(DImax*DKmax)+(k*DImax)+i];
			if(id == 1) { // PEC
				
				DEz[j*(DImax*DKmax)+(k*DImax)+i] = 0;

			} else {

				DEz[j*(DImax*DKmax)+(k*DImax)+i] = D_CA[id] * DEz[j*(DImax*DKmax)+(k*DImax)+i] + D_CB[id]
						* ((DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+i] - DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+(i-1)]) * D_den_ex[i] +
					(DHx[(j-1)*(DImax*DKmax)+(k*DImax)+i] - DHx[j*(DImax*DKmax)+(k*DImax)+i]) * D_den_ey[j] - DJz[j*(DImax*DKmax)+(k*DImax)+i]);
			}
	}
}

__global__ void Dev_EzPML_Idir_Calc(float *DEz, float *DHy, float *D_CB, float *D_psi_Ezx_1, float *D_psi_Ezx_2, float *D_be_x_1, float *D_be_x_2,
									float *D_ce_x_1, float *D_ce_x_2, int *D_ID3, float dx )
{

	int Im, Jm, Km, Ip, Jp, Kp, Jindex, Kindex, Iindex, i, j, k, id, ii;
	int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;			  

	Im=DPML_size-1;
	Jm=DJmax-2;
	Km=1;

	if(l < Im*Jm*Km)
	{

		Ip=1;
		Jp=1;
		Kp=0; 

		Jindex=(l/(Im*Km));
		Kindex=((l-(Jindex*(Im*Km)))/Im);
		Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

		i=Iindex+Ip;
		j=Jindex+Jp;
		k=Kindex+Kp;

//----------------------------------------------------------------------------

					  id = D_ID3[j*(DImax*DKmax)+(k*DImax)+i];
					  D_psi_Ezx_1[j*(DPML_size*DKmax)+(k*DPML_size)+i] = D_be_x_1[i] * D_psi_Ezx_1[j*(DPML_size*DKmax)+(k*DPML_size)+i]
							+ D_ce_x_1[i] * (DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+i] - DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+(i-1)]) / dx;
					  DEz[j*(DImax*DKmax)+(k*DImax)+i] = DEz[j*(DImax*DKmax)+(k*DImax)+i] + D_CB[id] * D_psi_Ezx_1[j*(DPML_size*DKmax)+(k*DPML_size)+i];

//----------------------------------------------------------------------------
		ii=(DImax-1)-i;

					  id = D_ID3[j*(DImax*DKmax)+(k*DImax)+ii];
					  D_psi_Ezx_2[j*(DPML_size*DKmax)+(k*DPML_size)+i] = D_be_x_2[i] * D_psi_Ezx_2[j*(DPML_size*DKmax)+(k*DPML_size)+i]
							+ D_ce_x_2[i] * (DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+ii] - DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+(ii-1)]) / dx;
					  DEz[j*(DImax*DKmax)+(k*DImax)+ii] = DEz[j*(DImax*DKmax)+(k*DImax)+ii] + D_CB[id] * D_psi_Ezx_2[j*(DPML_size*DKmax)+(k*DPML_size)+i];
	}
}

__global__ void Dev_EzPML_Jdir_Calc(float *DEz, float *DHx, float *D_CB, float *D_psi_Ezy_1, float *D_psi_Ezy_2, float *D_be_y_1, float *D_be_y_2,
									float *D_ce_y_1, float *D_ce_y_2, int *D_ID3, float dy )
{

	int Im, Jm, Km, Ip, Jp, Kp, Jindex, Kindex, Iindex, i, j, k, id, jj;
	int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;			  

	Im=DImax-2;
	Jm=DPML_size-1;
	Km=1;

	if(l < Im*Jm*Km)
	{

		Ip=1;
		Jp=1;
		Kp=0; 

		Jindex=(l/(Im*Km));
		Kindex=((l-(Jindex*(Im*Km)))/Im);
		Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

		i=Iindex+Ip;
		j=Jindex+Jp;
		k=Kindex+Kp;

//----------------------------------------------------------------------------

		id = D_ID3[j*(DImax*DKmax)+(k*DImax)+i];
		D_psi_Ezy_1[j*(DImax*DKmax)+(k*DImax)+i] = D_be_y_1[j] * D_psi_Ezy_1[j*(DImax*DKmax)+(k*DImax)+i]
			+ D_ce_y_1[j] * (DHx[(j-1)*(DImax*DKmax)+(k*DImax)+i] - DHx[j*(DImax*DKmax)+(k*DImax)+i]) / dy;
		DEz[j*(DImax*DKmax)+(k*DImax)+i] = DEz[j*(DImax*DKmax)+(k*DImax)+i] + D_CB[id] * D_psi_Ezy_1[j*(DImax*DKmax)+(k*DImax)+i];

//----------------------------------------------------------------------------
		jj=(DJmax-1)-j;

		id = D_ID3[jj*(DImax*DKmax)+(k*DImax)+i];
		D_psi_Ezy_2[j*(DImax*DKmax)+(k*DImax)+i] = D_be_y_2[j] * D_psi_Ezy_2[j*(DImax*DKmax)+(k*DImax)+i]
			+ D_ce_y_2[j] * (DHx[(jj-1)*(DImax*DKmax)+(k*DImax)+i] - DHx[jj*(DImax*DKmax)+(k*DImax)+i]) / dy;
		DEz[jj*(DImax*DKmax)+(k*DImax)+i] = DEz[jj*(DImax*DKmax)+(k*DImax)+i] + D_CB[id] * D_psi_Ezy_2[j*(DImax*DKmax)+(k*DImax)+i];
	}
}

__global__ void Dev_EzAppSourceCalc(int i, int j, int k, float *DEz, int *D_ID3, float *D_CB, int n, float amp, float dt, float tO, float tw, int SourceSize )
{

int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;

	if(l<SourceSize)
	{
	   float source = amp * -2.0 * ((n * dt - tO) / tw)
	   				 * exp(-pow(((n * dt - tO) / tw), 2));//Differentiated Gaussian pulse
	 
	   DEz[j*(DImax*DKmax)+(k*DImax)+i] = DEz[j*(DImax*DKmax)+(k*DImax)+i] - D_CB[D_ID3[j*(DImax*DKmax)+(k*DImax)+i]] * source;
	}
}


__global__ void Dev_EzAppSourceCalc2(int i, int j, int k, float *DEz, int *D_ID3, float *D_CB, int n, float *amp, float dt, float tO, float fCoef, int SourceSize )
{

int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;

	if(l<SourceSize)
	{
	   float source = amp[n]*sin(fCoef*n);//sin 
	 
	   //DEz[j*(DImax*DKmax)+(k*DImax)+i] = DEz[j*(DImax*DKmax)+(k*DImax)+i] - D_CB[D_ID3[j*(DImax*DKmax)+(k*DImax)+i]] * source;
	   DEz[j*(DImax*DKmax)+(k*DImax)+i] = DEz[j*(DImax*DKmax)+(k*DImax)+i] - source;
	}
}

__global__ void Dev_UnAppSourceCalc(int i_cen, int j_cen, int k_cen, float *DEx, float *DEy, float *DEz, int *D_ID3, float *D_CB, int n, float *amp, float dt, float tO, float fCoef, float fCoef2, float PulseW, int s_sizeI, int s_sizeJ, int s_sizeK, char Dir )
{

int Im, Jm, Km, Ip, Jp, Kp, Jindex, Kindex, Iindex, i, j, k;
int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;

	Im=s_sizeI;
	Jm=s_sizeJ;
	Km=s_sizeK;

	if(l<Im*Jm*Km)
	{
		Ip=i_cen-(s_sizeI/2);
		Jp=j_cen-(s_sizeJ/2); 
		Kp=k_cen-(s_sizeK/2); 

		Jindex=(l/(Im*Km));
		Kindex=((l-(Jindex*(Im*Km)))/Im);
		Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

		i=Iindex+Ip;
		j=Jindex+Jp;
		k=Kindex+Kp;

//f1=30e9;
//% f0=f1/10;
//f0=10e9;
//
//f=f0*dt*2*pi;
//t1=n*dt;
//p=1;
//phi=90;

//	   float source = amp[n]*sin(fCoef*n);//sin 
	 float source = amp[n]*Dev_calculateChirp(fCoef,fCoef2,PulseW,1,n*dt,90); //cos 
	
	   switch (Dir)
	   {
	   case 'x':
	   DEz[j*(DImax*DKmax)+(k*DImax)+i] = DEz[j*(DImax*DKmax)+(k*DImax)+i] - source;
	   break;

	   case 'y':
	   DEz[j*(DImax*DKmax)+(k*DImax)+i] = DEz[j*(DImax*DKmax)+(k*DImax)+i] - source;
	   break;

	   case 'z':
	   DEz[j*(DImax*DKmax)+(k*DImax)+i] = DEz[j*(DImax*DKmax)+(k*DImax)+i] - source;
	   break;
	   }
	}
}

__global__ void Dev_AESASourceCalc(int i_cen, int j_cen, int k_cen, float *DEz, int *D_ID3, float *D_CB, int n, float *amp, int *D_arrayI, int *D_arrayK,  float dt, float tO, float fCoef, float fCoef2, float PulseW, int s_sizeI, int s_sizeJ, int s_sizeK, float Angle )
{

int Im, Jm, Km, Ip, Jp, Kp, Jindex, Kindex, Iindex, i, j, k;
int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;
//int ix = threadIdx.x;
//int ik = blockIdx.x;

int ix = blockIdx.x;
int ik = threadIdx.x;

//D_arrayI
//D_arrayK

	Im=s_sizeI;
	Jm=s_sizeJ;
	Km=s_sizeK;

	if(l<Im*Jm*Km)
	{
		Ip=D_arrayI[ix];
		Jp=j_cen; 
		Kp=D_arrayK[ik]; 

		Jindex=(l/(Im*Km));
		Kindex=((l-(Jindex*(Im*Km)))/Im);
		Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

		i=Iindex+Ip;
		j=Jindex+Jp;
		k=Kindex+Kp;

//	   float source = amp[n]*sin(fCoef*n);//sin 
	 float source = (amp[n]/(Km*Im))*Dev_calculateChirp(fCoef,fCoef2,PulseW,1,n*dt,90+(Angle*ix)); //cos 

	   DEz[j*(DImax*DKmax)+(k*DImax)+i] = DEz[j*(DImax*DKmax)+(k*DImax)+i] - source;
		//DEz[j*(DImax*DKmax)+(k*DImax)+i] = DEz[j*(DImax*DKmax)+(k*DImax)+i] - source + (D_detect[n*(Im*Km)+l]-D_detect[(n-1)*(Im*Km)+l])*100;
	   }
}


// ---cubics ---

// tau ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

__global__ void Dev_SaveLine(float * DEz, float * D_proc_line, Analysis A) {
	int Im, Jm, Km, ImKm, i, j, k, inBlock;
	int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;

	// Ez
	Im = DImax - 2;
	Jm = DJmax - 2;
	Km = 1;
	ImKm = Im*Km;

	j = l / ImKm;
	k = (l - j*ImKm) / Im;
	i = (l - j*ImKm) - (k*Im) + 1;
	j += 1;

	inBlock = (i >= A.pos_i && i < A.pos_i + A.size_i) && (j == A.pos_j) && (k == A.pos_k);
	if (l < A.size_i && inBlock) {
		//if (A.dim_sieve <= 1 || (i - A.pos_i) % A.dim_sieve == 0) {
			//D_proc_line[int((i - A.pos_i) / A.dim_sieve)] = DEz[j*(DImax*DKmax) + (k*DImax) + i];
			D_proc_line[i - A.pos_i] = DEz[j*(DImax*DKmax) + (k*DImax) + i];
		//}
	}
}

// tau ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//------------------------------------------------>>>>>>>>>>>>>>>>>>>>>>>>>
//					Device related function
//------------------------------------------------>>>>>>>>>>>>>>>>>>>>>>>>>
__device__ float Dev_calculateChirp( float f0, float f1, float t1, float p, float t, float phi)
{
	float beta = (f1-f0)*(pow(t1,-p));
	return cos(2*3.14 * ( beta/(1+p)*(pow(t,1+p)) + f0*t + phi/360));
}

__device__ float Dev_FindMax(float *PwrMedium, int i_pos ,int k_pos, int J_start, int counts)
{

	int i=i_pos;
	int k=k_pos;
	int j=0;
	float temp=PwrMedium[j*(DImax*DKmax)+(k*DImax)+i];
	for( j= J_start;j<counts;j++)
		{
			if(PwrMedium[j*(DImax*DKmax)+(k*DImax)+i]>temp)
			temp=PwrMedium[j*(DImax*DKmax)+(k*DImax)+i];
		}
	return temp;
}

//------------------------------------------------------------------------------>>>>>>>>>>>>>>>>>>>>
//				Addition parameters calculation kernels  
//------------------------------------------------------------------------------>>>>>>>>>>>>>>>>>>>>

__global__ void Dev_AESAComponentDetector(float *DHx, float *DHy, float *DEz, float *D_det_Ex, float *D_det_Ey, float *D_det_Ez, float *D_det_Hx, float *D_det_Hy, float *D_det_Hz, int *D_arrayI, int *D_arrayK, int d_sizeI, int d_sizeK, int i_det, int j_det, int k_det, int n)
{

int Im, Jm, Km, Ip, Jp, Kp, Jindex, Kindex, Iindex, i, j, k;
float EH_i,EH_j,EH_k;
int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;
//int ix = threadIdx.x;
//int ik = blockIdx.x;

int ix = blockIdx.x;
int ik = threadIdx.x;

	Im=d_sizeI;
	Jm=1;
	Km=d_sizeK;

	if(l < Im*Jm*Km)
	{

		Ip=D_arrayI[ix];
		Jp=j_det;
		Kp=D_arrayK[ik]; 

		Jindex=(l/(Im*Km));
		Kindex=((l-(Jindex*(Im*Km)))/Im);
		Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

		i=Iindex+Ip;
		j=Jindex+Jp;
		k=Kindex+Kp;

//		D_det_Ex[n*(Im*Km)+l]=DEx[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i];
//		D_det_Ey[n*(Im*Km)+l]=DEy[j*(DImax*(DKmax-1))+(k*DImax)+i];
		D_det_Ez[n*(Im*Km)+l]=DEz[j*(DImax*DKmax)+(k*DImax)+i];
		D_det_Hx[n*(Im*Km)+l]=DHx[j*(DImax*DKmax)+(k*DImax)+i];
		D_det_Hy[n*(Im*Km)+l]=DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+i];
//		D_det_Hz[n*(Im*Km)+l]=DHz[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i];

	}

}


__global__ void Dev_AESADetectorSim(float *DHx, float *DHy, float *DEz, float *D_detect, int *D_arrayI, int *D_arrayK, int d_sizeI, int d_sizeK, int i_det, int j_det, int k_det, int n)
{

int Im, Jm, Km, Ip, Jp, Kp, Jindex, Kindex, Iindex, i, j, k;
float EH_i,EH_j,EH_k;
int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;
//int ix = threadIdx.x;
//int ik = blockIdx.x;

int ix = blockIdx.x;
int ik = threadIdx.x;


	Im=d_sizeI;
	Jm=1;
	Km=d_sizeK;

	if(l < Im*Jm*Km)
	{

		Ip=D_arrayI[ix];
		Jp=j_det;
		Kp=D_arrayK[ik]; 

		Jindex=(l/(Im*Km));
		Kindex=((l-(Jindex*(Im*Km)))/Im);
		Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

		i=Iindex+Ip;
		j=Jindex+Jp;
		k=Kindex+Kp;

		//EH_i=(DEy[j*(DImax*(DKmax-1))+(k*DImax)+i]*DHz[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i]-DEz[j*(DImax*DKmax)+(k*DImax)+i]*DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+i]);
		//EH_j=-(DEx[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i]*DHz[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i]-DEz[j*(DImax*DKmax)+(k*DImax)+i]*DHx[j*(DImax*DKmax)+(k*DImax)+i]);
		//EH_k=(DEx[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i]*DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+i]-DEy[j*(DImax*(DKmax-1))+(k*DImax)+i]*DHx[j*(DImax*DKmax)+(k*DImax)+i]);

		//EH_k=0;
		//EH_i=0;

		//D_detect[n*(Im*Km)+l]=sqrt(pow(EH_i, 2)+pow(EH_j, 2)+pow(EH_k, 2));
		D_detect[n*(Im*Km)+l]=DEz[j*(DImax*DKmax)+(k*DImax)+i];
	}

}

__global__ void Dev_AESACompDetectorSim(float *DHx, float *DHy, float *DHz, float *DEx, float *DEy, float *DEz, float *D_detect, int *D_arrayI, int *D_arrayK, float *D_Plust, int d_sizeI, int d_sizeK, int i_det, int j_det, int k_det, int n)
{

int Im, Jm, Km, Ip, Jp, Kp, Jindex, Kindex, Iindex, i, j, k;
float EH_i,EH_j,EH_k;
int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;
//int ix = threadIdx.x;
//int ik = blockIdx.x;

int ix = blockIdx.x;
int ik = threadIdx.x;

	Im=d_sizeI;
	Jm=1;
	Km=d_sizeK;

	if(l < Im*Jm*Km)
	{

		Ip=D_arrayI[ix];
		Jp=j_det;
		Kp=D_arrayK[ik]; 

		Jindex=(l/(Im*Km));
		Kindex=((l-(Jindex*(Im*Km)))/Im);
		Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

		i=Iindex+Ip;
		j=Jindex+Jp;
		k=Kindex+Kp;

		EH_i=(DEy[j*(DImax*(DKmax-1))+(k*DImax)+i]*DHz[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i]-DEz[j*(DImax*DKmax)+(k*DImax)+i]*DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+i]);
		EH_j=-(DEx[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i]*DHz[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i]-DEz[j*(DImax*DKmax)+(k*DImax)+i]*DHx[j*(DImax*DKmax)+(k*DImax)+i]);
		EH_k=(DEx[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i]*DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+i]-DEy[j*(DImax*(DKmax-1))+(k*DImax)+i]*DHx[j*(DImax*DKmax)+(k*DImax)+i]);

		//EH_k=0;
		//EH_i=0;
		//D_detect[n*(Im*Km)+l]=D_detect[(n-1)*(Im*Km)+l]+(sqrt(pow(EH_i, 2)+pow(EH_j, 2)+pow(EH_k, 2))-D_Plust[(ik*d_sizeI)+ix]);
			//D_detect[n*(Im*Km)+l]=(sqrt(pow(EH_i, 2)+pow(EH_j, 2)+pow(EH_k, 2))-D_Plust[(ik*d_sizeI)+ix]);
			//D_detect[n*(det_size*det_size)+l]=DEz[j*(DImax*DKmax)+(k*DImax)+i];
		//D_Plust[(ik*d_sizeI)+ix]=sqrt(pow(EH_i, 2)+pow(EH_j, 2)+pow(EH_k, 2));

		//D_detect[n*(Im*Km)+l]=sqrt(pow(EH_i, 2)+pow(EH_j, 2)+pow(EH_k, 2));

		D_detect[n*(Im*Km)+l]=DEz[j*(DImax*DKmax)+(k*DImax)+i];
	}

}

__global__ void Dev_DetectorSim(float *DHx, float *DHy, float *DHz, float *DEx, float *DEy, float *DEz, float *D_detect, int det_size, int i_det, int j_det, int k_det, int n)
{
int Im, Jm, Km, Ip, Jp, Kp, Jindex, Kindex, Iindex, i, j, k;
float EH_i,EH_j,EH_k;
int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;

	Im=det_size;
	Jm=1;
	Km=det_size;

	if(l < Im*Jm*Km)
	{

		Ip=i_det-(det_size/2);
		Jp=j_det;
		Kp=k_det-(det_size/2); 

		Jindex=(l/(Im*Km));
		Kindex=((l-(Jindex*(Im*Km)))/Im);
		Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

		i=Iindex+Ip;
		j=Jindex+Jp;
		k=Kindex+Kp;

		EH_i=(DEy[j*(DImax*(DKmax-1))+(k*DImax)+i]*DHz[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i]-DEz[j*(DImax*DKmax)+(k*DImax)+i]*DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+i]);
		EH_j=-(DEx[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i]*DHz[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i]-DEz[j*(DImax*DKmax)+(k*DImax)+i]*DHx[j*(DImax*DKmax)+(k*DImax)+i]);
		EH_k=(DEx[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i]*DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+i]-DEy[j*(DImax*(DKmax-1))+(k*DImax)+i]*DHx[j*(DImax*DKmax)+(k*DImax)+i]);

		//EH_k=0;
		//EH_i=0;
		D_detect[n*(det_size*det_size)+l]=sqrt(pow(EH_i, 2)+pow(EH_j, 2)+pow(EH_k, 2));
		//D_detect[n*(det_size*det_size)+l]=DEz[j*(DImax*DKmax)+(k*DImax)+i];

	}
}

__global__ void Dev_LineDetectorSim(float *DHx, float *DHy, float *DHz, float *DEx, float *DEy, float *DEz, float *D_line_detect, int det_size, int det_quant, int det_step, int i_det, int j_det, int k_det, int n)
{

int Im, Jm, Km, Ip, Jp, Kp, Jindex, Kindex, Iindex, i, j, k;
float EH_i,EH_j,EH_k;
//int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;

int det_num=blockIdx.x;
int idx=threadIdx.x;

	Im=det_size;
	Jm=1;
	Km=det_size;

	if(idx < (Im*Jm*Km))
	{

		Ip=i_det-(det_size/2);
		Jp=j_det+(det_num*det_step);
		Kp=k_det-(det_size/2); 

		Jindex=(idx/(Im*Km));
		Kindex=((idx-(Jindex*(Im*Km)))/Im);
		Iindex=((idx-(Jindex*(Im*Km)))-(Kindex*Im));

		i=Iindex+Ip;
		j=Jindex+Jp;
		k=Kindex+Kp;


		EH_i=(DEy[j*(DImax*(DKmax-1))+(k*DImax)+i]*DHz[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i]-DEz[j*(DImax*DKmax)+(k*DImax)+i]*DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+i]);
		EH_j=-(DEx[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i]*DHz[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i]-DEz[j*(DImax*DKmax)+(k*DImax)+i]*DHx[j*(DImax*DKmax)+(k*DImax)+i]);
		EH_k=(DEx[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i]*DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+i]-DEy[j*(DImax*(DKmax-1))+(k*DImax)+i]*DHx[j*(DImax*DKmax)+(k*DImax)+i]);

		D_line_detect[n*(blockDim.x*det_quant)+det_num*(det_size*det_size)+idx]=sqrt(pow(EH_i, 2)+pow(EH_j, 2)+pow(EH_k, 2));

	}      
}

__global__ void Dev_PowerLineSave(float *DHx, float *DHy, float *DHz, float *DEx, float *DEy, float *DEz, float *D_line_medium, int i_coord, int k_coord, int n)
{
int Im, Jm, Km, Ip, Jp, Kp, Jindex, Kindex, Iindex, i, j, k;
float EH_i,EH_j,EH_k;
int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;

	Im=1;
	Jm=DJmax-2;
	Km=1;

	if(l < Im*Jm*Km)
	{

		Ip=i_coord;
		Jp=1;
		Kp=k_coord; 

		Jindex=(l/(Im*Km));
		Kindex=((l-(Jindex*(Im*Km)))/Im);
		Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

		i=Iindex+Ip;
		j=Jindex+Jp;
		k=Kindex+Kp;


		//EH_i=(DEy[j*(DImax*(DKmax-1))+(k*DImax)+i]*DHz[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i]-DEz[j*(DImax*DKmax)+(k*DImax)+i]*DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+i]);
		//EH_j=-(DEx[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i]*DHz[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i]-DEz[j*(DImax*DKmax)+(k*DImax)+i]*DHx[j*(DImax*DKmax)+(k*DImax)+i]);
		//EH_k=(DEx[j*((DImax-1)*(DKmax-1))+(k*(DImax-1))+i]*DHy[j*((DImax-1)*DKmax)+(k*(DImax-1))+i]-DEy[j*(DImax*(DKmax-1))+(k*DImax)+i]*DHx[j*(DImax*DKmax)+(k*DImax)+i]);

		//D_line_medium[n*(Jm)+l]=sqrt(pow(EH_i, 2)+pow(EH_j, 2)+pow(EH_k, 2));
		D_line_medium[n*(Jm)+l]=DEz[j*(DImax*DKmax)+(k*DImax)+i];

	}      
}

__global__ void Dev_RCcalc(float *DEz, int *DSpdLine, int *D_arrayI, int *D_arrayK,  float edgePercent, float*DedgeL, int d_sizeI, int d_sizeK, int j_cut, int n)
{

int Im, Jm, Km, Ip, Jp, Kp, Jindex, Kindex, Iindex, i, j, k;
float EH_i,EH_j,EH_k;
int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;
//int ix = threadIdx.x;
//int ik = blockIdx.x;

int ix = blockIdx.x;
int ik = threadIdx.x;


	Im=d_sizeI;
	Jm=1;
	Km=d_sizeK;

	if(l < Im*Jm*Km)
	{

	//__shared__ float edgeL;

	Ip=D_arrayI[ix];
	Jp=0;
	Kp=D_arrayK[ik]; 

	Jindex=(l/(Im*Km));
	Kindex=((l-(Jindex*(Im*Km)))/Im);
	Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

	i=Iindex+Ip;
	j=Jindex+Jp;
	k=Kindex+Kp;

	float edgeLTemp;
	float temp;		
	int coordTemp = j_cut;
	int coordPrev = 0;
	
	


	temp=Dev_FindMax(DEz, D_arrayI[ix], D_arrayK[ik], j_cut, DJmax-2);

	edgeLTemp = temp*edgePercent;
	if (edgeLTemp > DedgeL[ik*d_sizeI+ix])
		DedgeL[ik*d_sizeI+ix] = edgeLTemp;


//			DSpdLine[n*(Im*Km)+l]=0;
			j=(DJmax-2);
			do
			{
				j--;
				//if(abs(DEz[j*(DImax*DKmax)+(k*DImax)+i])<DedgeL[ik*d_sizeI+ix])
				//	coordTemp=j;
				
			}
			while(j>0 && (abs(DEz[j*(DImax*DKmax)+(k*DImax)+i]))<DedgeL[ik*d_sizeI+ix]);

			DSpdLine[n*(Im*Km) + l] = j;

			//if (n > 1)
			//{
			//	coordPrev = DSpdLine[(n - 1)*(Im*Km) + l];

			//	if (coordPrev < coordTemp)
			//		DSpdLine[n*(Im*Km) + l] = coordTemp;
			//	else
			//		DSpdLine[n*(Im*Km) + l] = coordPrev;
			//}


			//if(DSpdLine[n]<DSpdLine[n-1])
			//	DSpdLine[n]=DSpdLine[n-1];			
	}
}

__global__ void DevFilterPostTime(int *mas, float *ans, int counts, int w_size)
{
	int l = (((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x);

	int w_beg=l,w_end=l-w_size;
	float weight=1.0/abs(w_size);
	//int counts = DJmax-2;

		ans[l]=0;
		for(int j=w_beg;j>w_end;j--)
			if(j>0 && j<counts)
			ans[l]+=(float)mas[j]*weight;
}


//------------------------------------------------------------------------------>>>>>>>>>>>>>>>>>>>>
//				calc prepare kernels
//------------------------------------------------------------------------------>>>>>>>>>>>>>>>>>>>>

__global__ void DevCalcPrepare(float *dCoef, int DCoef_size ) {
	int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;
	if(l<DCoef_size) {
		dCoef[l]=0;
	}
}

__global__ void DevCalcPrepare(int *dCoef, int DCoef_size ) {
	int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;
	if(l<DCoef_size) {
		dCoef[l]=0;
	}
}

__global__ void DevCalcPrepare(float *dCoef, int DCoef_size , float initial) {
	int l = ((blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x) + blockIdx.x* blockDim.x + threadIdx.x;
	if(l<DCoef_size) {
		dCoef[l]=initial;
	}
}


__host__ cudaError_t DevMemoryAllocation()
{
    cudaError_t cudaStatus;
	cudaDeviceProp devProp;
	cudaGetDeviceProperties ( &devProp, 0 );

	//-----------init Net size on GPU---------------------
	cudaStatus=cudaMemcpyToSymbol ( DImax, &Imax, sizeof ( int ), 0, cudaMemcpyHostToDevice );    
	cudaStatus=cudaMemcpyToSymbol ( DJmax, &Jmax, sizeof ( int ), 0, cudaMemcpyHostToDevice ); 
	cudaStatus=cudaMemcpyToSymbol ( DKmax, &Kmax, sizeof ( int ), 0, cudaMemcpyHostToDevice );    
    if (cudaStatus != cudaSuccess) {
        printf("hostNetSize copy failed!, %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
	//----------------------------------------------------

	    // Allocate GPU buffers for E and H fields vectors
		cudaStatus = cudaMalloc((void**)&DHx, (Imax*(Jmax-1)*Kmax) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("Hx cudaMalloc failed!");
        return cudaStatus;
    }

		cudaStatus = cudaMalloc((void**)&DHy, ((Imax-1)*Jmax*Kmax) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("Hy cudaMalloc failed!");
        return cudaStatus;
    }


//-------------------------------------------------------------------------------------------------

		cudaStatus = cudaMalloc((void**)&DEz, (Imax*Jmax*Kmax) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("Ez cudaMalloc failed!");
        return cudaStatus;
    }

//-------------------------------------------------------------------------------------------------

		cudaStatus = cudaMalloc((void**)&DJz, (Imax*Jmax*Kmax) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("Jz cudaMalloc failed!");
        return cudaStatus;
    }

//-------------------------------------------------------------------------------------------------

		cudaStatus = cudaMalloc((void**)&D_ID1, (Imax*Jmax*Kmax) * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("ID1 cudaMalloc failed!");
        return cudaStatus;
    }

		cudaStatus = cudaMalloc((void**)&D_ID2, (Imax*Jmax*Kmax) * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("ID2 cudaMalloc failed!");
        return cudaStatus;
    }

		cudaStatus = cudaMalloc((void**)&D_ID3, (Imax*Jmax*Kmax) * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        printf("ID3 cudaMalloc failed!");
        return cudaStatus;
    }

//------------------DEN----------------------------------------------------------------------

		cudaStatus = cudaMalloc((void**)&D_den_ex, (Imax-1) * sizeof(float));		    
		cudaStatus = cudaMalloc((void**)&D_den_hx, (Imax-1) * sizeof(float));

		cudaStatus = cudaMalloc((void**)&D_den_ey, (Jmax-1) * sizeof(float));
		cudaStatus = cudaMalloc((void**)&D_den_hy, (Jmax-1) * sizeof(float));



		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("den coeff cudaMalloc failed!");
        return cudaStatus;
    }

//------------------PML coef---------------------------------------------------------------------

    //float *D_psi_Ezx_1;	    
	cudaStatus = cudaMalloc((void**)&D_psi_Ezx_1, (PML_SIZE*Jmax*Kmax) * sizeof(float)); //!
    if (cudaStatus != cudaSuccess) {
        printf("psi_Ezx_1 cudaMalloc failed!");
        return cudaStatus;
    }
    //float *D_psi_Ezx_2;
	cudaStatus = cudaMalloc((void**)&D_psi_Ezx_2, (PML_SIZE*Jmax*Kmax) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("psi_Ezx_2 cudaMalloc failed!");
        return cudaStatus;
	}
    //float *D_psi_Hyx_1;
	cudaStatus = cudaMalloc((void**)&D_psi_Hyx_1, ((PML_SIZE-1)*Jmax*Kmax) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("psi_Hyx_1 cudaMalloc failed!");
        return cudaStatus;
	}
    //float *D_psi_Hyx_2;  ((nxPML_2-1) * Jmax * Kmax)
	cudaStatus = cudaMalloc((void**)&D_psi_Hyx_2, ((PML_SIZE-1)*Jmax*Kmax) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("psi_Hyx_2 cudaMalloc failed!");
        return cudaStatus;
	}
    //float *D_psi_Ezy_1; (Imax * nyPML_1 * Kmax)
	cudaStatus = cudaMalloc((void**)&D_psi_Ezy_1, (Imax*PML_SIZE*Kmax) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("psi_Ezy_1 cudaMalloc failed!");
        return cudaStatus;
	}
    //float *D_psi_Ezy_2; (Imax * nyPML_2 * Kmax)
	cudaStatus = cudaMalloc((void**)&D_psi_Ezy_2, (Imax*PML_SIZE*Kmax) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("psi_Ezy_2 cudaMalloc failed!");
        return cudaStatus;
	}
    //float *D_psi_Hxy_1;
	cudaStatus = cudaMalloc((void**)&D_psi_Hxy_1, (Imax*(PML_SIZE-1)*Kmax) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("psi_Hxy_1 cudaMalloc failed!");
        return cudaStatus;
    }
    //float *D_psi_Hxy_2;
	cudaStatus = cudaMalloc((void**)&D_psi_Hxy_2, (Imax*(PML_SIZE-1)*Kmax) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("psi_Hxy_2 cudaMalloc failed!");
        return cudaStatus;
    }
    //float *D_psi_Hxz_1;
	cudaStatus = cudaMalloc((void**)&D_psi_Hxz_1, (Imax*(Jmax-1)*(PML_SIZE-1)) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("psi_Hxz_1 cudaMalloc failed!");
        return cudaStatus;
    }
    //float *D_psi_Hxz_2;
	cudaStatus = cudaMalloc((void**)&D_psi_Hxz_2, (Imax*(Jmax-1)*(PML_SIZE-1)) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("psi_Hxz_2 cudaMalloc failed!");
        return cudaStatus;
    }
    //float *D_psi_Hyz_1; (Imax-1) * Jmax * (nzPML_1-1)
	cudaStatus = cudaMalloc((void**)&D_psi_Hyz_1, ((Imax-1)*Jmax*(PML_SIZE-1)) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("psi_Hyz_1 cudaMalloc failed!");
        return cudaStatus;
    }
    //float *D_psi_Hyz_2; (Imax-1) * Jmax * (nzPML_2-1)
	cudaStatus = cudaMalloc((void**)&D_psi_Hyz_2, ((Imax-1)*Jmax*(PML_SIZE-1)) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("psi_Hyz_2 cudaMalloc failed!");
        return cudaStatus;
    }
    //float *D_psi_Exz_1; (Imax-1) * Jmax * nzPML_1)
	cudaStatus = cudaMalloc((void**)&D_psi_Exz_1, ((Imax-1)*Jmax*PML_SIZE) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("psi_Exz_1 cudaMalloc failed!");
        return cudaStatus;
    }
    //float *D_psi_Exz_2; (Imax-1) * Jmax * nzPML_2
	cudaStatus = cudaMalloc((void**)&D_psi_Exz_2, ((Imax-1)*Jmax*PML_SIZE) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("psi_Exz_2 cudaMalloc failed!");
        return cudaStatus;
    }
    //float *D_psi_Eyz_1; (Imax-1) * (Jmax-1) * nzPML_1
	cudaStatus = cudaMalloc((void**)&D_psi_Eyz_1, ((Imax-1)*(Jmax-1)*PML_SIZE) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("psi_Eyz_1 cudaMalloc failed!");
        return cudaStatus;
    }
    //float *D_psi_Eyz_2; (Imax-1) * (Jmax-1) * nzPML_2
	cudaStatus = cudaMalloc((void**)&D_psi_Eyz_2, ((Imax-1)*(Jmax-1)*PML_SIZE) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("psi_Eyz_2 cudaMalloc failed!");
        return cudaStatus;
    }




//---------------------1D PML coeff------------------------------------------------------------------------


		cudaStatus = cudaMalloc((void**)&D_be_x_1, (PML_SIZE) * sizeof(float));
		cudaStatus = cudaMalloc((void**)&D_ce_x_1, (PML_SIZE) * sizeof(float));

		cudaStatus = cudaMalloc((void**)&D_bh_x_1, (PML_SIZE-1) * sizeof(float));
		cudaStatus = cudaMalloc((void**)&D_ch_x_1, (PML_SIZE-1) * sizeof(float));

		cudaStatus = cudaMalloc((void**)&D_be_x_2, (PML_SIZE) * sizeof(float));
		cudaStatus = cudaMalloc((void**)&D_ce_x_2, (PML_SIZE) * sizeof(float));
	    
		cudaStatus = cudaMalloc((void**)&D_bh_x_2, (PML_SIZE-1) * sizeof(float));
		cudaStatus = cudaMalloc((void**)&D_ch_x_2, (PML_SIZE-1) * sizeof(float));

		cudaStatus = cudaMalloc((void**)&D_be_y_1, (PML_SIZE) * sizeof(float));
		cudaStatus = cudaMalloc((void**)&D_ce_y_1, (PML_SIZE) * sizeof(float));

		cudaStatus = cudaMalloc((void**)&D_bh_y_1, (PML_SIZE-1) * sizeof(float));
		cudaStatus = cudaMalloc((void**)&D_ch_y_1, (PML_SIZE-1) * sizeof(float));

		cudaStatus = cudaMalloc((void**)&D_be_y_2, (PML_SIZE) * sizeof(float));
		cudaStatus = cudaMalloc((void**)&D_ce_y_2, (PML_SIZE) * sizeof(float));

		cudaStatus = cudaMalloc((void**)&D_bh_y_2, (PML_SIZE-1) * sizeof(float));
		cudaStatus = cudaMalloc((void**)&D_ch_y_2, (PML_SIZE-1) * sizeof(float));

		cudaStatus = cudaMalloc((void**)&D_be_z_1, (PML_SIZE) * sizeof(float));
		cudaStatus = cudaMalloc((void**)&D_ce_z_1, (PML_SIZE) * sizeof(float));

		cudaStatus = cudaMalloc((void**)&D_bh_z_1, (PML_SIZE-1) * sizeof(float));
		cudaStatus = cudaMalloc((void**)&D_ch_z_1, (PML_SIZE-1) * sizeof(float));

		cudaStatus = cudaMalloc((void**)&D_be_z_2, (PML_SIZE) * sizeof(float));
		cudaStatus = cudaMalloc((void**)&D_ce_z_2, (PML_SIZE) * sizeof(float));

		cudaStatus = cudaMalloc((void**)&D_bh_z_2, (PML_SIZE-1) * sizeof(float));
		cudaStatus = cudaMalloc((void**)&D_ch_z_2, (PML_SIZE-1) * sizeof(float));

//---------------------Calculation coeff------------------------------------------------------------
		cudaStatus = cudaMalloc((void**)&D_CA, (numMaterials) * sizeof(float));
		cudaStatus = cudaMalloc((void**)&D_CB, (numMaterials) * sizeof(float));

		cudaStatus = cudaMalloc((void**)&D_DA, (numMaterials) * sizeof(float));
		cudaStatus = cudaMalloc((void**)&D_DB, (numMaterials) * sizeof(float));

		cudaStatus = cudaMalloc((void**)&D_NE, (numMaterials) * sizeof(float));

		cudaStatus = cudaGetLastError();
	    if (cudaStatus != cudaSuccess) {
		   printf("1D PML coeff cudaMalloc failed!");
        return cudaStatus;
    }
//-------------------------------------------------------------------------------------------------


	//Write zeros to allocated memory
	dim3 blockSize = dim3(devProp.maxThreadsPerBlock/2, 2, 1);    //Размер используемого блока
	dim3 gridSize = dim3((((PML_SIZE*Jmax*Kmax) / devProp.maxThreadsPerBlock)+1), 1, 1); //Размер используемого грида

	DevCalcPrepare<<<gridSize, blockSize>>>(D_psi_Ezx_1, (PML_SIZE*Jmax*Kmax) );
	DevCalcPrepare<<<gridSize, blockSize>>>(D_psi_Ezx_2, (PML_SIZE*Jmax*Kmax) );

	gridSize = dim3((((PML_SIZE-1)*Jmax*Kmax) / devProp.maxThreadsPerBlock)+1, 1, 1); //Размер используемого грида
	DevCalcPrepare<<<gridSize, blockSize>>>(D_psi_Hyx_1, ((PML_SIZE-1)*Jmax*Kmax) );
	DevCalcPrepare<<<gridSize, blockSize>>>(D_psi_Hyx_2, ((PML_SIZE-1)*Jmax*Kmax) );

	gridSize = dim3(((Imax*PML_SIZE*Kmax) / devProp.maxThreadsPerBlock)+1, 1, 1); //Размер используемого грида
	DevCalcPrepare<<<gridSize, blockSize>>>(D_psi_Ezy_1, (Imax*PML_SIZE*Kmax) );
	DevCalcPrepare<<<gridSize, blockSize>>>(D_psi_Ezy_2, (Imax*PML_SIZE*Kmax) );

	gridSize = dim3(((Imax*(PML_SIZE-1)*Kmax) / devProp.maxThreadsPerBlock)+1, 1, 1); //Размер используемого грида
	DevCalcPrepare<<<gridSize, blockSize>>>(D_psi_Hxy_1, (Imax*(PML_SIZE-1)*Kmax) );
	DevCalcPrepare<<<gridSize, blockSize>>>(D_psi_Hxy_2, (Imax*(PML_SIZE-1)*Kmax) );

	gridSize = dim3(((Imax*(Jmax-1)*(PML_SIZE-1)) / devProp.maxThreadsPerBlock)+1, 1, 1); //Размер используемого грида
	DevCalcPrepare<<<gridSize, blockSize>>>(D_psi_Hxz_1, (Imax*(Jmax-1)*(PML_SIZE-1)) );
	DevCalcPrepare<<<gridSize, blockSize>>>(D_psi_Hxz_2, (Imax*(Jmax-1)*(PML_SIZE-1)) );

	gridSize = dim3((((Imax-1)*Jmax*(PML_SIZE-1)) / devProp.maxThreadsPerBlock)+1, 1, 1); //Размер используемого грида
	DevCalcPrepare<<<gridSize, blockSize>>>(D_psi_Hyz_1, ((Imax-1)*Jmax*(PML_SIZE-1)) );
	DevCalcPrepare<<<gridSize, blockSize>>>(D_psi_Hyz_2, ((Imax-1)*Jmax*(PML_SIZE-1)) );

	gridSize = dim3((((Imax-1)*Jmax*PML_SIZE) / devProp.maxThreadsPerBlock)+1, 1, 1); //Размер используемого грида
	DevCalcPrepare<<<gridSize, blockSize>>>(D_psi_Exz_1, ((Imax-1)*Jmax*PML_SIZE) );
	DevCalcPrepare<<<gridSize, blockSize>>>(D_psi_Exz_2, ((Imax-1)*Jmax*PML_SIZE) );

	gridSize = dim3((((Imax-1)*(Jmax-1)*PML_SIZE) / devProp.maxThreadsPerBlock)+1, 1, 1); //Размер используемого грида
	DevCalcPrepare<<<gridSize, blockSize>>>(D_psi_Eyz_1, ((Imax-1)*(Jmax-1)*PML_SIZE) );
	DevCalcPrepare<<<gridSize, blockSize>>>(D_psi_Eyz_2, ((Imax-1)*(Jmax-1)*PML_SIZE) );


		cudaStatus = cudaGetLastError();
//----------------------Zeros H ----------------------------------------------------------
	//&DHx, (Imax*(Jmax-1)*Kmax)
	gridSize = dim3((((Imax*(Jmax-1)*Kmax) / devProp.maxThreadsPerBlock))/2, 3, 1); //Размер используемого грида
	DevCalcPrepare<<<gridSize, blockSize>>>(DHx, (Imax*(Jmax-1)*Kmax) );

	//&DHy, ((Imax-1)*Jmax*Kmax)
	gridSize = dim3(((((Imax-1)*Jmax*Kmax) / devProp.maxThreadsPerBlock))/2, 3, 1); //Размер используемого грида
	DevCalcPrepare<<<gridSize, blockSize>>>(DHy, ((Imax-1)*Jmax*Kmax) );


//---------------------Zeros E------------------------------------------------------------

	//&DEz, (Imax*Jmax*Kmax)
	gridSize = dim3((((Imax*Jmax*Kmax) / devProp.maxThreadsPerBlock))/2, 3, 1); //Размер используемого грида
	DevCalcPrepare<<<gridSize, blockSize>>>(DEz, (Imax*Jmax*Kmax) );

//---------------------Zeros J------------------------------------------------------------


	//&DJz, (Imax*Jmax*Kmax)
	gridSize = dim3((((Imax*Jmax*Kmax) / devProp.maxThreadsPerBlock))/2, 3, 1); //Размер используемого грида
	DevCalcPrepare<<<gridSize, blockSize>>>(DJz, (Imax*Jmax*Kmax) );

//------------------------Zeros ID--------------------------------------------------------

	//blockSize = dim3(((Imax*Jmax*Kmax) / devProp.maxThreadsPerBlock)+1, 1, 1); //Размер используемого грида
	//DevCalcPrepare<<<gridSize, blockSize>>>(D_ID1, (Imax*Jmax*Kmax) );

	//blockSize = dim3(((Imax*Jmax*Kmax) / devProp.maxThreadsPerBlock)+1, 1, 1); //Размер используемого грида
	//DevCalcPrepare<<<gridSize, blockSize>>>(D_ID2, (Imax*Jmax*Kmax) );

	//blockSize = dim3(((Imax*Jmax*Kmax) / devProp.maxThreadsPerBlock)+1, 1, 1); //Размер используемого грида
	//DevCalcPrepare<<<gridSize, blockSize>>>(D_ID3, (Imax*Jmax*Kmax) );

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		   printf("3D Zeros failed!");
        return cudaStatus;
	}


	cudaStatus = cudaMemcpy(D_ID1, ID1, (Imax*Jmax*Kmax) * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(D_ID2, ID2, (Imax*Jmax*Kmax) * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(D_ID3, ID3, (Imax*Jmax*Kmax) * sizeof(int), cudaMemcpyHostToDevice);	
			
	cudaStatus = cudaMemcpy(D_CA, CA, (numMaterials) * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(D_CB, CB, (numMaterials) * sizeof(float), cudaMemcpyHostToDevice);	

	cudaStatus = cudaMemcpy(D_DA, DA, (numMaterials) * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(D_DB, DB, (numMaterials) * sizeof(float), cudaMemcpyHostToDevice);

	cudaStatus = cudaMemcpy(D_NE, pne, (numMaterials) * sizeof(float), cudaMemcpyHostToDevice);

//----------------DEBUG--------------------------------->>>>>>>>>>>>>>
	//cudaStatus=CheckCoefInGPU(D_DB, numMaterials );	//((Imax-1)*Jmax*(Kmax-1))
//----------------DEBUG---------------------------------<<<<<<<<<<<<<<

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		   printf("3D Host to Device copy failed!");
        return cudaStatus;
	}

//---------------------1D PML coeff init----------------------------------------------------------


	    cudaStatus = cudaMemcpy(D_den_ex, den_ex, (Imax-1) * sizeof(float), cudaMemcpyHostToDevice);
	    cudaStatus = cudaMemcpy(D_den_hx, den_hx, (Imax-1) * sizeof(float), cudaMemcpyHostToDevice);

	    cudaStatus = cudaMemcpy(D_den_ey, den_ey, (Jmax-1) * sizeof(float), cudaMemcpyHostToDevice);		
	    cudaStatus = cudaMemcpy(D_den_hy, den_hy, (Jmax-1) * sizeof(float), cudaMemcpyHostToDevice);

	    cudaStatus = cudaMemcpy(D_be_x_1, be_x_1, (PML_SIZE) * sizeof(float), cudaMemcpyHostToDevice);		
	    cudaStatus = cudaMemcpy(D_ce_x_1, ce_x_1, (PML_SIZE) * sizeof(float), cudaMemcpyHostToDevice);

	    cudaStatus = cudaMemcpy(D_bh_x_1, bh_x_1, (PML_SIZE-1) * sizeof(float), cudaMemcpyHostToDevice);		
	    cudaStatus = cudaMemcpy(D_ch_x_1, ch_x_1, (PML_SIZE-1) * sizeof(float), cudaMemcpyHostToDevice);

	    cudaStatus = cudaMemcpy(D_be_x_2, be_x_2, (PML_SIZE) * sizeof(float), cudaMemcpyHostToDevice);		
	    cudaStatus = cudaMemcpy(D_ce_x_2, ce_x_2, (PML_SIZE) * sizeof(float), cudaMemcpyHostToDevice);
  
	    cudaStatus = cudaMemcpy(D_bh_x_2, bh_x_2, (PML_SIZE-1) * sizeof(float), cudaMemcpyHostToDevice);		
	    cudaStatus = cudaMemcpy(D_ch_x_2, ch_x_2, (PML_SIZE-1) * sizeof(float), cudaMemcpyHostToDevice);

	    cudaStatus = cudaMemcpy(D_be_y_1, be_y_1, (PML_SIZE) * sizeof(float), cudaMemcpyHostToDevice);		
	    cudaStatus = cudaMemcpy(D_ce_y_1, ce_y_1, (PML_SIZE) * sizeof(float), cudaMemcpyHostToDevice);

	    cudaStatus = cudaMemcpy(D_bh_y_1, bh_y_1, (PML_SIZE-1) * sizeof(float), cudaMemcpyHostToDevice);		
	    cudaStatus = cudaMemcpy(D_ch_y_1, ch_y_1, (PML_SIZE-1) * sizeof(float), cudaMemcpyHostToDevice);

	    cudaStatus = cudaMemcpy(D_be_y_2, be_y_2, (PML_SIZE) * sizeof(float), cudaMemcpyHostToDevice);		
	    cudaStatus = cudaMemcpy(D_ce_y_2, ce_y_2, (PML_SIZE) * sizeof(float), cudaMemcpyHostToDevice);

	    cudaStatus = cudaMemcpy(D_bh_y_2, bh_y_2, (PML_SIZE-1) * sizeof(float), cudaMemcpyHostToDevice);		
	    cudaStatus = cudaMemcpy(D_ch_y_2, ch_y_2, (PML_SIZE-1) * sizeof(float), cudaMemcpyHostToDevice);

	    cudaStatus = cudaMemcpy(D_be_z_1, be_z_1, (PML_SIZE) * sizeof(float), cudaMemcpyHostToDevice);		
	    cudaStatus = cudaMemcpy(D_ce_z_1, ce_z_1, (PML_SIZE) * sizeof(float), cudaMemcpyHostToDevice);

	    cudaStatus = cudaMemcpy(D_bh_z_1, bh_z_1, (PML_SIZE-1) * sizeof(float), cudaMemcpyHostToDevice);		
	    cudaStatus = cudaMemcpy(D_ch_z_1, ch_z_1, (PML_SIZE-1) * sizeof(float), cudaMemcpyHostToDevice);

	    cudaStatus = cudaMemcpy(D_be_z_2, be_z_2, (PML_SIZE) * sizeof(float), cudaMemcpyHostToDevice);		
	    cudaStatus = cudaMemcpy(D_ce_z_2, ce_z_2, (PML_SIZE) * sizeof(float), cudaMemcpyHostToDevice);

	    cudaStatus = cudaMemcpy(D_bh_z_2, bh_z_2, (PML_SIZE-1) * sizeof(float), cudaMemcpyHostToDevice);		
	    cudaStatus = cudaMemcpy(D_ch_z_2, ch_z_2, (PML_SIZE-1) * sizeof(float), cudaMemcpyHostToDevice);

//------------------Additional memory allocations---------------------------------------------------------------------
		if(CALC_ARREY_POS==ENABLE)
		{
		cudaStatus = cudaMalloc((void**)&D_arrayI, (qantI) * sizeof(int));
		cudaStatus = cudaMalloc((void**)&D_arrayK, (qantK) * sizeof(int));
		cudaStatus = cudaMemcpy(D_arrayI, arrayI, (qantI) * sizeof(int), cudaMemcpyHostToDevice);		
	    cudaStatus = cudaMemcpy(D_arrayK, arrayK, (qantK) * sizeof(int), cudaMemcpyHostToDevice);

		cudaStatus = cudaMalloc((void**)&Ddet_arrayI, (det_qantI) * sizeof(int));
		cudaStatus = cudaMalloc((void**)&Ddet_arrayK, (det_qantK) * sizeof(int));
		cudaStatus = cudaMemcpy(Ddet_arrayI, det_arrayI, (det_qantI) * sizeof(int), cudaMemcpyHostToDevice);		
	    cudaStatus = cudaMemcpy(Ddet_arrayK, det_arrayK, (det_qantK) * sizeof(int), cudaMemcpyHostToDevice);
		}

		if(NET2D_SAVE_ENABLE == ENABLE)
		{

		cudaStatus = cudaMalloc((void**)&D_RCdetArrayI, (RC_qantI) * sizeof(int));
		cudaStatus = cudaMalloc((void**)&D_RCdetArrayK, (RC_qantK) * sizeof(int));
		cudaStatus = cudaMemcpy(D_RCdetArrayI, RCdetArrayI, (RC_qantI) * sizeof(int), cudaMemcpyHostToDevice);		
	    cudaStatus = cudaMemcpy(D_RCdetArrayK, RCdetArrayK, (RC_qantK) * sizeof(int), cudaMemcpyHostToDevice);

		cudaStatus = cudaMalloc((void**)&DPspdLine, (nMax * RC_qantI * RC_qantK) * sizeof(int));
			if (cudaStatus != cudaSuccess) {
				printf("DPspdLine cudaMalloc failed!");
				return cudaStatus;
			}
		cudaStatus = cudaMalloc((void**)&DPspdFiltred, (nMax * RC_qantI * RC_qantK) * sizeof(float));
			if (cudaStatus != cudaSuccess) {
				printf("DPspdFiltred cudaMalloc failed!");
				return cudaStatus;
			}			
		cudaStatus = cudaMalloc((void**)&DedgeL, (RC_qantI * RC_qantK) * sizeof(float));
			if (cudaStatus != cudaSuccess) {
				printf("DedgeL cudaMalloc failed!");
				return cudaStatus;
			}
			dim3 AESA_RCdet_gridSize = dim3(RC_qantI, 1, 1); 
			dim3 AESA_RCdet_blockSize = dim3(RC_qantK, 1, 1); //Размер используемого блока
			gridSize = dim3(((nMax * RC_qantI * RC_qantK) / devProp.maxThreadsPerBlock)+1, 1, 1);
				//&DedgeL, (RC_qantI * RC_qantK)
			DevCalcPrepare<<<AESA_RCdet_gridSize, AESA_RCdet_blockSize>>>(DedgeL, (RC_qantI * RC_qantK), 10.0 );
			DevCalcPrepare<<<gridSize, blockSize>>>(DPspdLine, (nMax * RC_qantI * RC_qantK) );
			//DevCalcPrepare<<<gridSize, blockSize>>>(DPspdFiltred, (nMax * RC_qantI * RC_qantK) );

//			CheckCoefInGPU(DPspdLine, (nMax * RC_qantI * RC_qantK));
		}

		int DetNetSize;
		if(CALC_ARREY_POS==ENABLE)
		{
			DetNetSize=det_qantI * det_qantK;	
		}
		else
		{
			DetNetSize=DET_SIZE * DET_SIZE;
		}
		cudaStatus = cudaMalloc((void**)&D_detect, (nMax * DetNetSize) * sizeof(float));

		if(SAVE_COMPONENTS==ENABLE)
		{
			cudaStatus = cudaMalloc((void**)&D_det_Ex, (nMax * DetNetSize) * sizeof(float));
			cudaStatus = cudaMalloc((void**)&D_det_Ey, (nMax * DetNetSize) * sizeof(float));
			cudaStatus = cudaMalloc((void**)&D_det_Ez, (nMax * DetNetSize) * sizeof(float));
			cudaStatus = cudaMalloc((void**)&D_det_Hx, (nMax * DetNetSize) * sizeof(float));
			cudaStatus = cudaMalloc((void**)&D_det_Hy, (nMax * DetNetSize) * sizeof(float));
			cudaStatus = cudaMalloc((void**)&D_det_Hz, (nMax * DetNetSize) * sizeof(float));

			gridSize = dim3(((nMax * DetNetSize) / devProp.maxThreadsPerBlock)+1, 1, 1); //Зануление
			DevCalcPrepare<<<gridSize, blockSize>>>(D_det_Ex, (nMax * DetNetSize));
			DevCalcPrepare<<<gridSize, blockSize>>>(D_det_Ey, (nMax * DetNetSize));
			DevCalcPrepare<<<gridSize, blockSize>>>(D_det_Ez, (nMax * DetNetSize));
			DevCalcPrepare<<<gridSize, blockSize>>>(D_det_Hx, (nMax * DetNetSize));
			DevCalcPrepare<<<gridSize, blockSize>>>(D_det_Hy, (nMax * DetNetSize));
			DevCalcPrepare<<<gridSize, blockSize>>>(D_det_Hz, (nMax * DetNetSize));
		}

		// processing 
		if (Processing.perform) {
			cudaStatus = cudaMalloc((void**)&D_proc_line, (Processing.size_i/* / Processing.dim_sieve*/) * sizeof(float));
		}
		

		//-----DEBUG------------
		//CheckCoefInGPU(D_detect, (nMax *  DetNetSize));

		if(COMP_DETECT==ENABLE)
		{
		cudaStatus = cudaMalloc((void**)&D_Plust, (qantI * qantK) * sizeof(float));			
		cudaStatus = cudaMalloc((void**)&D_CompDet, (nMax * qantI * qantK) * sizeof(float));	
		gridSize = dim3(((nMax * qantI * qantK) / devProp.maxThreadsPerBlock)+1, 1, 1); //Зануление
		DevCalcPrepare<<<gridSize, blockSize>>>(D_detect, (nMax * qantI * qantK));

		gridSize = dim3(((qantI * qantK) / devProp.maxThreadsPerBlock)+1, 1, 1); //Зануление
		DevCalcPrepare<<<gridSize, blockSize>>>(D_Plust, (qantI * qantK));

		}

		if(LINE_D_ENABLE==ENABLE)
		cudaStatus = cudaMalloc((void**)&D_line_detect, (nMax * LINE_DET_A_QUANT * LINE_DET_A_SIZE * LINE_DET_A_SIZE) * sizeof(float));

		if(LINE_D2_ENABLE==ENABLE)
		cudaStatus = cudaMalloc((void**)&D_line_detect2, (nMax * LINE_DET_B_QUANT * LINE_DET_B_SIZE * LINE_DET_B_SIZE) * sizeof(float));

		gridSize = dim3(((nMax * DetNetSize) / devProp.maxThreadsPerBlock)+1, 1, 1); //Зануление
		DevCalcPrepare<<<gridSize, blockSize>>>(D_detect, (nMax * DetNetSize) );
		
		if(LINE_D_ENABLE==ENABLE)
		{
		gridSize = dim3(((nMax * LINE_DET_A_QUANT * LINE_DET_A_SIZE * LINE_DET_A_SIZE) / devProp.maxThreadsPerBlock)+1, 1, 1); //Зануление
		DevCalcPrepare<<<gridSize, blockSize>>>(D_line_detect, (nMax * LINE_DET_A_QUANT * LINE_DET_A_SIZE * LINE_DET_A_SIZE));
		}

		if(LINE_D2_ENABLE==ENABLE)
		{
		gridSize = dim3(((nMax * LINE_DET_B_QUANT * LINE_DET_B_SIZE * LINE_DET_B_SIZE) / devProp.maxThreadsPerBlock)+1, 1, 1); //Зануление
		DevCalcPrepare<<<gridSize, blockSize>>>(D_line_detect2, (nMax * LINE_DET_B_QUANT * LINE_DET_B_SIZE * LINE_DET_B_SIZE) );
		}

		cudaStatus = cudaMalloc((void**)&D_sourceP, (nMax) * sizeof(float));
		cudaStatus = cudaMemcpy(D_sourceP, SourceP, (nMax) * sizeof(float), cudaMemcpyHostToDevice);

		if(SAVE_LINE_ENABLE==ENABLE) //выделяем память под сохранение мощьности вдоль линии
		{
			cudaStatus = cudaMalloc((void**)&D_line_medium, (nMax * (Jmax-2)) * sizeof(float));
			gridSize = dim3(((nMax * (Jmax-2)) / devProp.maxThreadsPerBlock)+1, 1, 1); //Зануление
			DevCalcPrepare<<<gridSize, blockSize>>>(D_line_medium, (nMax * (Jmax-2)) );
		}


	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
	{
		   printf("1D coef init failed!, Cuda ERROR: %s", cudaGetErrorString(cudaStatus) );
        return cudaStatus;
	}


//----------------DEBUG--------------------------------->>>>>>>>>>>>>>
	//cudaStatus=CheckCoefInGPU(D_den_ey, Jmax-1 );	
//----------------DEBUG---------------------------------<<<<<<<<<<<<<<


	cudaStatus = cudaGetLastError();
return cudaStatus;

}

__host__ cudaError_t DevSourceToDetectInfluenceCalc()
{

	
	printf("\n\nStart Source To Detect Influence Calculation\n");

	int save_counter=0;
	int save_app=1;

	double detSum, detSum2, detSum3, detSum4, detSum5, detSum6;		//float

	float EH_i, EH_j, EH_k;


	 int is_size=I_SRC_SIZE;
	 int js_size=J_SRC_SIZE;
	 int ks_size=K_SRC_SIZE;

	 char s_dir=S_DIRECTION;

	fopen_s(&pDet,"Detect_S2D.txt","w");
	fopen_s(&pComDet,"CompDetect_S2D.txt","w");

	fopen_s(&p_detAll,"detAll_S2D.txt","w");

	fopen_s(&p_detEx,"detEx_S2D.txt","w");
	fopen_s(&p_detEy,"detEy_S2D.txt","w");
	fopen_s(&p_detEz,"detEz_S2D.txt","w");
	fopen_s(&p_detHx,"detHx_S2D.txt","w");
	fopen_s(&p_detHy,"detHy_S2D.txt","w");
	fopen_s(&p_detHz,"detHz_S2D.txt","w");

	fopen_s(&p_detP,"detP_S2D.txt","w");

//----------------------------------------------------------------------------------------

    cudaError_t cudaStatus;
	cudaDeviceProp devProp;
	cudaGetDeviceProperties ( &devProp, 0 );

	cudaStatus=DevMemoryAllocation();
	printf("\nGPU last error on start calculating: %s\n", cudaGetErrorString(cudaStatus));

	//dim3 blockSize = dim3(devProp.maxThreadsPerBlock/2, 2, 1);    //Размер используемого блока
	//dim3 gridSize = dim3((((Imax*Jmax*Kmax) / devProp.maxThreadsPerBlock))+1, 1, 1); //Размер используемого грида
	//dim3 PML_I_gridSize = dim3((((PML_SIZE*Jmax*Kmax) / devProp.maxThreadsPerBlock)+1), 1, 1); //Размер используемого грида
	//dim3 PML_J_gridSize = dim3((((Imax*PML_SIZE*Kmax) / devProp.maxThreadsPerBlock)+1), 1, 1); //Размер используемого грида
	//dim3 PML_K_gridSize = dim3((((Imax*Jmax*PML_SIZE) / devProp.maxThreadsPerBlock)+1), 1, 1); //Размер используемого грида

	int Threads=devProp.maxThreadsPerBlock;

	dim3 blockSize = dim3(Threads, 1, 1);    //Размер используемого блока
	dim3 gridSize = dim3((((Imax*Jmax*Kmax) / Threads))+1, 1, 1); //Размер используемого грида
	dim3 PML_I_gridSize = dim3((((PML_SIZE*Jmax*Kmax) / Threads)+1), 1, 1); //Размер используемого грида
	dim3 PML_J_gridSize = dim3((((Imax*PML_SIZE*Kmax) / Threads)+1), 1, 1); //Размер используемого грида
	dim3 PML_K_gridSize = dim3((((Imax*Jmax*PML_SIZE) / Threads)+1), 1, 1); //Размер используемого грида

	//Расчетные сетки для источника
	dim3 Source_gridSize = dim3(32, 1, 1); 
	dim3 Source_blockSize = dim3(32, 1, 1); //Размер используемого блока

	
	dim3 AESA_Source_gridSize = dim3(qantI, 1, 1); 
	dim3 AESA_Source_blockSize = dim3(qantK, 1, 1); //Размер используемого блока

	dim3 AESA_Detect_gridSize = dim3(det_qantI, 1, 1); 
	dim3 AESA_Detect_blockSize = dim3(det_qantK, 1, 1); //Размер используемого блока

	////Расчетные сетки для детекторов
	//dim3 LineDetect_gridSize = dim3(LINE_DET_A_QUANT, 1, 1); 
	//dim3 LineDetect_blockSize = dim3(LINE_DET_A_SIZE*LINE_DET_A_SIZE, 1, 1); //Размер используемого блока

	////Расчетные сетки для детекторов
	//dim3 LineDetect2_gridSize = dim3(LINE_DET_B_QUANT, 1, 1); 
	//dim3 LineDetect2_blockSize = dim3(LINE_DET_B_SIZE*LINE_DET_B_SIZE, 1, 1); //Размер используемого блока

	dim3 Detect_gridSize = dim3(((DET_SIZE*DET_SIZE)/Threads)+1, 1, 1); 
	dim3 Detect_blockSize = dim3(devProp.maxThreadsPerBlock, 1, 1); //Размер используемого блока

	//Вспомогательные расчетные сетки 
	dim3 SaveLine_blockSize = dim3(32, 1, 1); //Размер используемого блока
	dim3 SaveLine_gridSize = dim3((((Jmax) / SaveLine_blockSize.x)+1), 1, 1); //Размер используемого грида
	


	for(n = 1; n <= nMax; ++n) 
		{

			//if(CLOSE_ENABLE==1)
			//{
			//if(n==CLOSE_TIME)
			//	CloseOpenDoor( 1 );	
			//}
//--------------------------------Hx----------------------------
			Dev_HxCalc<<<gridSize, blockSize>>>(DHx, DEz, D_ID1, D_DA, D_DB, D_den_hy, D_den_hz);

			Dev_HxPML_Jdir_Calc<<<PML_J_gridSize, blockSize>>>(DHx, DEz, D_DB, D_psi_Hxy_1, D_psi_Hxy_2, D_bh_y_1, D_bh_y_2,
															D_ch_y_1, D_ch_y_2, D_ID1, dy );
			//Dev_HxPML_Kdir_Calc<<<PML_K_gridSize, blockSize>>>(DHx, DEy, D_DB, D_psi_Hxz_1, D_psi_Hxz_2, D_bh_z_1, D_bh_z_2,
			//												D_ch_z_1, D_ch_z_2, D_ID1, dz );

//--------------------------------Jx----------------------------


//--------------------------------Hy----------------------------
			Dev_HyCalc<<<gridSize, blockSize>>>(DHy, DEz, D_ID2, D_DA, D_DB, D_den_hx, D_den_hz);

			Dev_HyPML_Idir_Calc<<<PML_I_gridSize, blockSize>>>(DHy, DEz, D_DB, D_psi_Hyx_1, D_psi_Hyx_2, D_bh_x_1, D_bh_x_2,
															D_ch_x_1, D_ch_x_2, D_ID2, dx );
			//Dev_HyPML_Kdir_Calc<<<PML_K_gridSize, blockSize>>>(DHy, DEx, D_DB, D_psi_Hyz_1, D_psi_Hyz_2, D_bh_z_1, D_bh_z_2,
			//												D_ch_z_1, D_ch_z_2, D_ID2, dz );

//--------------------------------Jy----------------------------


//--------------------------------Hz----------------------------


//--------------------------------Jz----------------------------
			Dev_JzCalc<<<gridSize, blockSize>>>(DJz, DEz, D_ID3, D_NE, dt);

//--------------------------------Ex---------------------------

//--------------------------------Ey---------------------------

//--------------------------------Ez---------------------------
			Dev_EzCalc<<<gridSize, blockSize>>>(DEz, DHy, DHx, DJz, D_ID3, D_CA, D_CB, D_den_ex, D_den_ey);

			Dev_EzPML_Idir_Calc<<<PML_I_gridSize, blockSize>>>(DEz, DHy, D_CB, D_psi_Ezx_1, D_psi_Ezx_2, D_be_x_1, D_be_x_2,
															D_ce_x_1, D_ce_x_2, D_ID3, dx );
			Dev_EzPML_Jdir_Calc<<<PML_J_gridSize, blockSize>>>(DEz, DHx, D_CB, D_psi_Ezy_1, D_psi_Ezy_2, D_be_y_1, D_be_y_2,
															D_ce_y_1, D_ce_y_2, D_ID3, dy );
//--------------------------------Source---------------------------

			//Dev_EzAppSourceCalc<<<Source_gridSize, Source_blockSize>>>(SourceCen.I, SourceCen.J, SourceCen.K, DEz, D_ID3, D_CB, n, amp, dt, tO, tw, 1); //gaussian pulse
			//Dev_EzAppSourceCalc2<<<Source_gridSize, Source_blockSize>>>(SourceCen.I, SourceCen.J, SourceCen.K, DEz, D_ID3, D_CB, n, D_sourceP, dt, tO, fCoef, 1); //sin packet
			//if(CALC_ARREY_POS==DISABLE)
			//Dev_UnAppSourceCalc<<<Source_gridSize, Source_blockSize>>>(SourceCen.I, SourceCen.J, SourceCen.K, DEx, DEy, DEz, D_ID3, D_CB, n, D_sourceP, dt, tO, fCoef, fCoef2, fPulseW, is_size, js_size, ks_size, s_dir); //sin packet

			if(CALC_ARREY_POS==ENABLE)
			Dev_AESASourceCalc<<<AESA_Source_gridSize, AESA_Source_blockSize>>>(SourceCen.I, SourceCen.J, SourceCen.K, DEz, D_ID3, D_CB, n, D_sourceP, D_arrayI, D_arrayK,  dt, tO, fCoef, fCoef2, fPulseW, qantI, 1, qantK, AESAsourceParam.Angle );

//--------------------------------Detectors-----------------------------
			//if(CALC_ARREY_POS==DISABLE)
			//Dev_DetectorSim<<<Detect_gridSize, Detect_blockSize>>>(DHx, DHy, DHz, DEx, DEy, DEz, D_detect, DET_SIZE, DetectCen.I, DetectCen.J, DetectCen.K, n);

			if(CALC_ARREY_POS==ENABLE)
			Dev_AESADetectorSim<<<AESA_Detect_gridSize, AESA_Detect_blockSize>>>(DHx, DHy, DEz, D_detect, Ddet_arrayI, Ddet_arrayK, det_qantI, det_qantK, DetectCen.I, DetectCen.J, DetectCen.K, n);

			//if((COMP_DETECT==ENABLE))
			//Dev_AESACompDetectorSim<<<AESA_Source_gridSize, AESA_Source_blockSize>>>(DHx, DHy, DHz, DEx, DEy, DEz, D_CompDet, D_arrayI, D_arrayK, D_Plust, qantI, qantK, SourceCen.I, SourceCen.J, SourceCen.K, n);

			if(SAVE_COMPONENTS==ENABLE)
			Dev_AESAComponentDetector<<<AESA_Detect_gridSize, AESA_Detect_blockSize>>>(DHx, DHy, DEz, D_det_Ex, D_det_Ey, D_det_Ez, D_det_Hx, D_det_Hy, D_det_Hz, D_arrayI, D_arrayK, det_qantI, det_qantK, DetectCen.I, DetectCen.J, DetectCen.K, n);

			//if(LINE_D_ENABLE==ENABLE)
			//Dev_LineDetectorSim<<<LineDetect_gridSize, LineDetect_blockSize>>>(DHx, DHy, DHz, DEx, DEy, DEz, D_line_detect, LINE_DET_A_SIZE, LINE_DET_A_QUANT, LINE_DET_A_STEP, I_LINE_DET_A, J_LINE_DET_A, K_LINE_DET_A, n);

			//if(LINE_D2_ENABLE==ENABLE)
			//Dev_LineDetectorSim<<<LineDetect2_gridSize, LineDetect2_blockSize>>>(DHx, DHy, DHz, DEx, DEy, DEz, D_line_detect2, LINE_DET_B_SIZE, LINE_DET_B_QUANT, LINE_DET_B_STEP, I_LINE_DET_B, J_LINE_DET_B, K_LINE_DET_B, n);


	}//Done time stepping

	//Read field thrue detectors
	int NetSize;
	if(CALC_ARREY_POS==ENABLE)
	{
		NetSize= det_qantI * det_qantK;
	}
	else
	{
		NetSize=DET_SIZE * DET_SIZE;
	}

	cudaStatus = cudaMemcpy(infDet, D_detect, (nMax *  NetSize) * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("detector data cudaMemcpy failed! : %s\n\n", cudaGetErrorString(cudaStatus));
//        return cudaStatus;
    }

	if(SAVE_COMPONENTS==ENABLE)
	{
		//cudaStatus = cudaMemcpy(infDet_Ex, D_det_Ex, (nMax *  NetSize) * sizeof(float), cudaMemcpyDeviceToHost);
		//cudaStatus = cudaMemcpy(infDet_Ey, D_det_Ey, (nMax *  NetSize) * sizeof(float), cudaMemcpyDeviceToHost);
		cudaStatus = cudaMemcpy(infDet_Ez, D_det_Ez, (nMax *  NetSize) * sizeof(float), cudaMemcpyDeviceToHost);
		cudaStatus = cudaMemcpy(infDet_Hx, D_det_Hx, (nMax *  NetSize) * sizeof(float), cudaMemcpyDeviceToHost);
		cudaStatus = cudaMemcpy(infDet_Hy, D_det_Hy, (nMax *  NetSize) * sizeof(float), cudaMemcpyDeviceToHost);
		//cudaStatus = cudaMemcpy(infDet_Hz, D_det_Hz, (nMax *  NetSize) * sizeof(float), cudaMemcpyDeviceToHost);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("Components detector data cudaMemcpy failed! : %s\n\n", cudaGetErrorString(cudaStatus));
	//return cudaStatus;    
		}
	}

	if(COMP_DETECT==ENABLE)
	{
	cudaStatus = cudaMemcpy(CompDet, D_CompDet, (nMax * qantI * qantK) * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("Compensation detector data cudaMemcpy failed!: %s\n\n", cudaGetErrorString(cudaStatus));
//        return cudaStatus;    
    }
	}

//------------------------SAVE TO FILE-----------------------------------		

			for(i=0;i<nMax;i++)
			{
				detSum=0;
				for(k=0;k<(NetSize);k++)
					{
						detSum+=infDet[i*(NetSize)+k];
					}
					fprintf(pDet,"%lf\t%12lf\n ",(dt*i)*1000000, detSum*DET_MULT);
			}

//-------------------------------------------------------------------------------
	if(SAVE_COMPONENTS==ENABLE)
	{
			for(i=0;i<nMax;i++)
			{

				detSum=0; detSum2=0; detSum3=0; detSum4=0; detSum5=0; detSum6=0;				

				for(k=0;k<(NetSize);k++)
					{
						//detSum+=infDet_Ex[i*(NetSize)+k];
						//detSum2+=infDet_Ey[i*(NetSize)+k];
						detSum3+=infDet_Ez[i*(NetSize)+k];
						detSum4+=infDet_Hx[i*(NetSize)+k];
						detSum5+=infDet_Hy[i*(NetSize)+k];
						//detSum6+=infDet_Hz[i*(NetSize)+k];
					}

//			EH_i=(detSum2*detSum6-detSum3*detSum5);
//			EH_j=-(detSum*detSum6-detSum3*detSum4);
//			EH_k=(detSum*detSum5-detSum2*detSum4);

			fprintf(p_detAll,"%lf\t%12lf\t%12lf\t%12lf\t%12lf\t%12lf\t%12lf\n",(dt*i)*1000000, detSum*DET_MULT, detSum2*DET_MULT, detSum3*DET_MULT, detSum4*DET_MULT, detSum5*DET_MULT, detSum6*DET_MULT);
			}
//---------------Save components ----------------------------------------------
			for(i=0;i<nMax;i++)
			{
				for(j=0;j<(det_qantI);j++)
				{
					for(k=0;k<(det_qantK);k++)
					{
						//fprintf(p_detEx,"%12lf\t", infDet_Ex[i*(NetSize)+j*det_qantK+k]*DET_MULT);
						//fprintf(p_detEy,"%12lf\t", infDet_Ey[i*(NetSize)+j*det_qantK+k]*DET_MULT);
						fprintf(p_detEz,"%12lf\t", infDet_Ez[i*(NetSize)+j*det_qantK+k]*DET_MULT);
						fprintf(p_detHx,"%12lf\t", infDet_Hx[i*(NetSize)+j*det_qantK+k]*DET_MULT);
						fprintf(p_detHy,"%12lf\t", infDet_Hy[i*(NetSize)+j*det_qantK+k]*DET_MULT);
						//fprintf(p_detHz,"%12lf\t", infDet_Hz[i*(NetSize)+j*det_qantK+k]*DET_MULT);
	//					fprintf(p_detEx,"%d\t", i*(NetSize)+j*det_qantK+k);
					}
					//fprintf(p_detEx,"\n");
					//fprintf(p_detEy,"\n");
					fprintf(p_detEz,"\n");
					fprintf(p_detHx,"\n");
					fprintf(p_detHy,"\n");
					//fprintf(p_detHz,"\n");
				}
				//fprintf(p_detEx,"\n");
				//fprintf(p_detEy,"\n");
				fprintf(p_detEz,"\n");
				fprintf(p_detHx,"\n");
				fprintf(p_detHy,"\n");
				//fprintf(p_detHz,"\n");

			}

	}
//---------------------comp detect save----------------------------------------
			if(COMP_DETECT==ENABLE)
			{
				for(i=0;i<nMax;i++)
				{
					detSum=0;
					for(k=0;k<(qantI * qantK);k++)
						{
							detSum+=CompDet[i*(qantI * qantK)+k];
						}
						fprintf(pComDet,"%lf\t%12lf\n ",(dt*i)*1000000, detSum*DET_MULT);

	//					out<<detSum<<std::endl;
				}
			}

	fclose(pDet);
	fclose(pComDet);

	fclose(p_detAll);

	fclose(p_detEx);
	fclose(p_detEy);
	fclose(p_detEz);
	fclose(p_detHx);
	fclose(p_detHy);
	fclose(p_detHz);

	fclose(p_detP);

	cudaStatus=DevMemoryFree();
	printf("Done Source To Detect Influence Calculation: %s\n\n", cudaGetErrorString(cudaStatus));
	return cudaStatus;
}

__host__ cudaError_t DevFieldCalculation()
{
	using namespace std;

	int save_counter=0;
	int save_app=1;

	double detSum, detSum2, detSum3, detSum4, detSum5, detSum6;		//float
	double detSum1;		//float

	 int is_size=I_SRC_SIZE;
	 int js_size=J_SRC_SIZE;
	 int ks_size=K_SRC_SIZE;

	 char s_dir=S_DIRECTION;

	 float EH_i, EH_j, EH_k;

//----------------FOR DEBUG----------------
	FILE *test;
	fopen_s(&test,"test.txt","w");
//----------------END DEBUG----------------

	std::ofstream out; 
	out.open("SDetect.txt");

	//FILE *pDet;
	fopen_s(&pDet,"Detect.txt","w");
	fopen_s(&pLineDet,"LineDetect.txt","w");
	fopen_s(&pLineDet2,"LineDetect2.txt","w");

//-----------------------------------------------
	fopen_s(&p_detAll,"detAll.txt","w");

	fopen_s(&p_detEx,"detEx.txt","w");
	fopen_s(&p_detEy,"detEy.txt","w");
	fopen_s(&p_detEz,"detEz.txt","w");
	fopen_s(&p_detHx,"detHx.txt","w");
	fopen_s(&p_detHy,"detHy.txt","w");
	fopen_s(&p_detHz,"detHz.txt","w");

	fopen_s(&p_detP,"detP.txt","w");

	fopen_s(&p_infDetEx,"DifDetEx.txt","w");


//-----------------------------------------------

	fopen_s(&pComDet,"CompDetect.txt","w");

				
	//For save power steering on the line
	if(SAVE_LINE_ENABLE==ENABLE)
	fopen_s(&pSLine,"PowerLine.txt","w");

	pEx = fopen ( "Ex.txt" , "wb");
	//pEx = fopen ( "C:\\FDTD_EXP_data\\Ex.txt" , "w");

	pEy = fopen ( "Ey.txt" , "wb");
	pEz = fopen ( "Ez.txt" , "w");

	pHx = fopen ( "Hx.txt" , "w");
	//pHx = fopen ( "C:\\FDTD_EXP_data\\Hx.txt" , "w");

	pHy = fopen ( "Hy.txt" , "w");
	pHz = fopen ( "Hz.txt" , "w");

	//cubics


	//fprintf(pEx,"Experiment \n");
	//fprintf(pEy,"Experiment \n");
	//fprintf(pEz,"Experiment \n");
	//fprintf(pHx,"Experiment \n");
	//fprintf(pHy,"Experiment \n");
	//fprintf(pHz,"Experiment \n");

	//pEx = fopen ( "Ex.txt" , "a");
	////pEx = fopen ( "C:\\FDTD_EXP_data\\Ex.txt" , "w");


	//pEy = fopen ( "Ey.txt" , "a");
	//pEz = fopen ( "Ez.txt" , "a");

	//pHx = fopen ( "Hx.txt" , "a");
	////pHx = fopen ( "C:\\FDTD_EXP_data\\Hx.txt" , "w");

	//pHy = fopen ( "Hy.txt" , "a");
	//pHz = fopen ( "Hz.txt" , "a");



    cudaError_t cudaStatus;
	cudaDeviceProp devProp;
	cudaGetDeviceProperties ( &devProp, 0 );

	cudaStatus=DevMemoryAllocation();
	printf("\nGPU last error on start calculating: %s\n", cudaGetErrorString(cudaStatus));

	dim3 blockSize = dim3(devProp.maxThreadsPerBlock/2, 2, 1);    //Размер используемого блока
	dim3 gridSize = dim3((((Imax*Jmax*Kmax) / devProp.maxThreadsPerBlock))/2, 3, 1); //Размер используемого грида
	dim3 PML_I_gridSize = dim3((((PML_SIZE*Jmax*Kmax) / devProp.maxThreadsPerBlock)), 1, 1); //Размер используемого грида
	dim3 PML_J_gridSize = dim3((((Imax*PML_SIZE*Kmax) / devProp.maxThreadsPerBlock)), 1, 1); //Размер используемого грида
	dim3 PML_K_gridSize = dim3((((Imax*Jmax*PML_SIZE) / devProp.maxThreadsPerBlock)), 1, 1); //Размер используемого грида

	//Расчетные сетки для источника
	dim3 Source_gridSize = dim3(32, 1, 1); 
	dim3 Source_blockSize = dim3(32, 1, 1); //Размер используемого блока

	
	dim3 AESA_Source_gridSize = dim3(qantI, 1, 1); 
	dim3 AESA_Source_blockSize = dim3(qantK, 1, 1); //Размер используемого блока

	dim3 AESA_Detect_gridSize = dim3(det_qantI, 1, 1); 
	dim3 AESA_Detect_blockSize = dim3(det_qantK, 1, 1); //Размер используемого блока

	//Расчетные сетки для детекторов
	dim3 LineDetect_gridSize = dim3(LINE_DET_A_QUANT, 1, 1); 
	dim3 LineDetect_blockSize = dim3(LINE_DET_A_SIZE*LINE_DET_A_SIZE, 1, 1); //Размер используемого блока

	//Расчетные сетки для детекторов
	dim3 LineDetect2_gridSize = dim3(LINE_DET_B_QUANT, 1, 1); 
	dim3 LineDetect2_blockSize = dim3(LINE_DET_B_SIZE*LINE_DET_B_SIZE, 1, 1); //Размер используемого блока

	dim3 Detect_gridSize = dim3(((DET_SIZE*DET_SIZE)/devProp.maxThreadsPerBlock)+1, 1, 1); 
	dim3 Detect_blockSize = dim3(devProp.maxThreadsPerBlock, 1, 1); //Размер используемого блока

	//расчетные сетки для расчета RC
	dim3 AESA_RCdet_gridSize = dim3(RC_qantI, 1, 1); 
	dim3 AESA_RCdet_blockSize = dim3(RC_qantK, 1, 1); //Размер используемого блока

	//Вспомогательные расчетные сетки 
	dim3 SaveLine_blockSize = dim3(32, 1, 1); //Размер используемого блока
	dim3 SaveLine_gridSize = dim3((((Jmax) / SaveLine_blockSize.x)+1), 1, 1); //Размер используемого грида
	
	Processing.times = 0;

	for(n = 1; n <= nMax; ++n) {

			//if(CLOSE_ENABLE==1)
			//{
			//if(n==CLOSE_TIME)
			//	CloseOpenDoor( 1 );	
			//}
//--------------------------------Hx----------------------------
			Dev_HxCalc<<<gridSize, blockSize>>>(DHx, DEz, D_ID1, D_DA, D_DB, D_den_hy, D_den_hz);

			cudaStatus = cudaGetLastError();

			Dev_HxPML_Jdir_Calc<<<PML_J_gridSize, blockSize>>>(DHx, DEz, D_DB, D_psi_Hxy_1, D_psi_Hxy_2, D_bh_y_1, D_bh_y_2,
															D_ch_y_1, D_ch_y_2, D_ID1, dy );
			//Dev_HxPML_Kdir_Calc<<<PML_K_gridSize, blockSize>>>(DHx, DEy, D_DB, D_psi_Hxz_1, D_psi_Hxz_2, D_bh_z_1, D_bh_z_2,
			//												D_ch_z_1, D_ch_z_2, D_ID1, dz );

						cudaStatus = cudaGetLastError();
//--------------------------------Jx----------------------------


//--------------------------------Hy----------------------------
			Dev_HyCalc<<<gridSize, blockSize>>>(DHy, DEz, D_ID2, D_DA, D_DB, D_den_hx, D_den_hz);

			Dev_HyPML_Idir_Calc<<<PML_I_gridSize, blockSize>>>(DHy, DEz, D_DB, D_psi_Hyx_1, D_psi_Hyx_2, D_bh_x_1, D_bh_x_2,
															D_ch_x_1, D_ch_x_2, D_ID2, dx );
			//Dev_HyPML_Kdir_Calc<<<PML_K_gridSize, blockSize>>>(DHy, DEx, D_DB, D_psi_Hyz_1, D_psi_Hyz_2, D_bh_z_1, D_bh_z_2,
			//												D_ch_z_1, D_ch_z_2, D_ID2, dz );

						cudaStatus = cudaGetLastError();
//--------------------------------Jy----------------------------
	

//--------------------------------Hz----------------------------


//--------------------------------Jz----------------------------
			Dev_JzCalc<<<gridSize, blockSize>>>(DJz, DEz, D_ID3, D_NE, dt);

						cudaStatus = cudaGetLastError();
//--------------------------------Ex---------------------------

//--------------------------------Ey---------------------------

//--------------------------------Ez---------------------------
			Dev_EzCalc<<<gridSize, blockSize>>>(DEz, DHy, DHx, DJz, D_ID3, D_CA, D_CB, D_den_ex, D_den_ey);

			cudaStatus = cudaGetLastError();

			Dev_EzPML_Idir_Calc<<<PML_I_gridSize, blockSize>>>(DEz, DHy, D_CB, D_psi_Ezx_1, D_psi_Ezx_2, D_be_x_1, D_be_x_2,
															D_ce_x_1, D_ce_x_2, D_ID3, dx );

			cudaStatus = cudaGetLastError();

			Dev_EzPML_Jdir_Calc<<<PML_J_gridSize, blockSize>>>(DEz, DHx, D_CB, D_psi_Ezy_1, D_psi_Ezy_2, D_be_y_1, D_be_y_2,
															D_ce_y_1, D_ce_y_2, D_ID3, dy );

						cudaStatus = cudaGetLastError();
//--------------------------------Source---------------------------

			
//				printf("%d\n", n);
				//Dev_EzAppSourceCalc<<<Source_gridSize, Source_blockSize>>>(SourceCen.I, SourceCen.J, SourceCen.K, DEz, D_ID3, D_CB, n, amp, dt, tO, tw, 1); //gaussian pulse
				//Dev_EzAppSourceCalc2<<<Source_gridSize, Source_blockSize>>>(SourceCen.I, SourceCen.J, SourceCen.K, DEz, D_ID3, D_CB, n, D_sourceP, dt, tO, fCoef, 1); //sin packet

				//if(CALC_ARREY_POS==DISABLE)
				//Dev_UnAppSourceCalc<<<Source_gridSize, Source_blockSize>>>(SourceCen.I, SourceCen.J, SourceCen.K, DEx, DEy, DEz, D_ID3, D_CB, n, D_sourceP, dt, tO, fCoef, fCoef2, fPulseW, is_size, js_size, ks_size, s_dir); //sin packet

				if(CALC_ARREY_POS==ENABLE)
				Dev_AESASourceCalc<<<AESA_Source_gridSize, AESA_Source_blockSize>>>(SourceCen.I, SourceCen.J, SourceCen.K, DEz, D_ID3, D_CB, n, D_sourceP, D_arrayI, D_arrayK,  dt, tO, fCoef, fCoef2, fPulseW, qantI, 1, qantK, AESAsourceParam.Angle );

				//if(CALC_ARREY_POS==ENABLE)
				//Dev_AESASourceCalc<<<AESA_Source_gridSize, AESA_Source_blockSize>>>(DetectCen.I, DetectCen.J, DetectCen.K, DEz, D_ID3, D_CB, n, D_sourceP, D_arrayI, D_arrayK,  dt, tO, fCoef, fCoef2, fPulseW, qantI, 1, qantK );
			if ( Processing.perform ) {
				if (/*(Processing.time_sieve <= 1 || n % Processing.time_sieve == 0) &&*/ n > Processing.begin_after && n <= Processing.finish_at) {
					Processing.times++;
					Dev_SaveLine <<< gridSize, blockSize >>>(DEz, D_proc_line, Processing);
					//cudaStatus = cudaMemcpy(proc_field[int((n - Processing.begin_after) / Processing.time_sieve) - 1], D_proc_line, int(Processing.size_i / Processing.dim_sieve) * sizeof(float), cudaMemcpyDeviceToHost);
					cudaStatus = cudaMemcpy(proc_field[n - Processing.begin_after - 1], D_proc_line, Processing.size_i * sizeof(float), cudaMemcpyDeviceToHost);
				}
			}
//--------------------------------Detectors-----------------------------
			//if(CALC_ARREY_POS==DISABLE)
			//Dev_DetectorSim<<<Detect_gridSize, Detect_blockSize>>>(DHx, DHy, DHz, DEx, DEy, DEz, D_detect, DET_SIZE, DetectCen.I, DetectCen.J, DetectCen.K, n);

			if(CALC_ARREY_POS==ENABLE)
			Dev_AESADetectorSim<<<AESA_Detect_gridSize, AESA_Detect_blockSize>>>(DHx, DHy, DEz, D_detect, Ddet_arrayI, Ddet_arrayK, det_qantI, det_qantK, DetectCen.I, DetectCen.J, DetectCen.K, n);

			//if((COMP_DETECT==ENABLE))
			//Dev_AESACompDetectorSim<<<AESA_Source_gridSize, AESA_Source_blockSize>>>(DHx, DHy, DHz, DEx, DEy, DEz, D_CompDet, D_arrayI, D_arrayK, D_Plust, qantI, qantK, SourceCen.I, SourceCen.J, SourceCen.K, n);

			if(SAVE_COMPONENTS==ENABLE)
			Dev_AESAComponentDetector<<<AESA_Detect_gridSize, AESA_Detect_blockSize>>>(DHx, DHy, DEz, D_det_Ex, D_det_Ey, D_det_Ez, D_det_Hx, D_det_Hy, D_det_Hz, D_arrayI, D_arrayK, det_qantI, det_qantK, DetectCen.I, DetectCen.J, DetectCen.K, n);

			//if(LINE_D_ENABLE==ENABLE)
			//Dev_LineDetectorSim<<<LineDetect_gridSize, LineDetect_blockSize>>>(DHx, DHy, DHz, DEx, DEy, DEz, D_line_detect, LINE_DET_A_SIZE, LINE_DET_A_QUANT, LINE_DET_A_STEP, I_LINE_DET_A, J_LINE_DET_A, K_LINE_DET_A, n);

			//if(LINE_D2_ENABLE==ENABLE)
			//Dev_LineDetectorSim<<<LineDetect2_gridSize, LineDetect2_blockSize>>>(DHx, DHy, DHz, DEx, DEy, DEz, D_line_detect2, LINE_DET_B_SIZE, LINE_DET_B_QUANT, LINE_DET_B_STEP, I_LINE_DET_B, J_LINE_DET_B, K_LINE_DET_B, n);

//------------------------Pover line calc-------------------------------
			if(SAVE_LINE_ENABLE==ENABLE)
			Dev_PowerLineSave<<<SaveLine_gridSize, SaveLine_blockSize>>>(DHx, DHy, DHz, DEx, DEy, DEz, D_line_medium, I_LINE, K_LINE, n);

			if(NET2D_SAVE_ENABLE==ENABLE)
			Dev_RCcalc<<<AESA_RCdet_gridSize, AESA_RCdet_blockSize>>>(DEz, DPspdLine, D_RCdetArrayI, D_RCdetArrayK,  RCdetFilt.level, DedgeL, RC_qantI, RC_qantK, SourceCen.J+150, n);

//------------------------SAVE TO FILE----------------------------------
		if (n>=FieldSaveParam.fromTime && n<=FieldSaveParam.ToTime)
		{
		if(save_app==FieldSaveParam.TimeSive)
		{

			//cudaStatus = cudaMemcpy(Hx, DHx, (Imax*(Jmax-1)*Kmax) * sizeof(float), cudaMemcpyDeviceToHost);
			//cudaStatus = cudaMemcpy(Hy, DHy, ((Imax-1)*Jmax*Kmax) * sizeof(float), cudaMemcpyDeviceToHost);
			//cudaStatus = cudaMemcpy(Hz, DHz, ((Imax-1)*(Jmax-1)*(Kmax-1)) * sizeof(float), cudaMemcpyDeviceToHost);

//			cudaStatus = cudaMemcpy(Ex, DEx, ((Imax-1)*Jmax*(Kmax-1)) * sizeof(float), cudaMemcpyDeviceToHost);
//			cudaStatus = cudaMemcpy(Ey, DEy, (Imax*(Jmax-1)*(Kmax-1)) * sizeof(float), cudaMemcpyDeviceToHost);
			cudaStatus = cudaMemcpy(Ez, DEz, (Imax*Jmax*Kmax) * sizeof(float), cudaMemcpyDeviceToHost);

			//cudaStatus = cudaMemcpy(Jx, DJx, ((Imax-1)*Jmax*(Kmax-1)) * sizeof(float), cudaMemcpyDeviceToHost);
			//cudaStatus = cudaMemcpy(Jy, DJy, (Imax*(Jmax-1)*(Kmax-1)) * sizeof(float), cudaMemcpyDeviceToHost);
			//cudaStatus = cudaMemcpy(Jz, DJz, (Imax*Jmax*Kmax) * sizeof(float), cudaMemcpyDeviceToHost);

			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				   printf("Read fields from device failed! %s\n", cudaGetErrorString(cudaStatus));
				return cudaStatus;
			}


			//fprintf(test, "(Ex Ey Ez) (Hx Hy Hz) at time step %d at (25, 63, 12) :  %f  :  %f  :  %f  :  %f  :  %f  :  %f  \n", n, Ex[63*((Imax-1)*(Kmax-1))+(12*(Imax-1))+25], Ey[63*(Imax*(Kmax-1))+(12*Imax)+25], Ez[63*(Imax*Kmax)+(12*Imax)+25], Hx[63*(Imax*Kmax)+(12*Imax)+25], Hy[63*((Imax-1)*Kmax)+(12*(Imax-1))+25], Hz[63*((Imax-1)*(Kmax-1))+(12*(Imax-1))+25]);


			save_counter++;
			save_app=1;

			if(MATLAB_SAVE==DISABLE)
			{
			fprintf(pEx,"Time is=%d \n",n);

			//fprintf(pEy,"Time is=%d \n",n);
			//fprintf(pEz,"Time is=%d \n",n);

			fprintf(pHx,"Time is=%d \n",n);

			//fprintf(pHy,"Time is=%d \n",n);
			//fprintf(pHz,"Time is=%d \n",n);
			}

			k=Kmax/2;
			//k=12;
			j=SourceCen.J+200;
			{

			if(MATLAB_SAVE==DISABLE)
			{
			fprintf(pEx,"slice num = %d \n",k);

			//fprintf(pEy,"slice num = %d \n",k);
			//fprintf(pEz,"slice num = %d \n",k);

			fprintf(pHx,"slice num = %d \n",k);

			//fprintf(pHy,"slice num = %d \n",k);
			//fprintf(pHz,"slice num = %d \n",k);
			}

			for(i=1;i<Imax-1;i+=FieldSaveParam.FieldSieve)
			//for(k=1;k<Kmax-1;k+=SIEVE)
				{
					for(j=1;j<Jmax-1;j+=FieldSaveParam.FieldSieve)
					//for(k=1;k<Kmax-1;k+=SIEVE)
						{
//--------------------------J-SLISE----------------------------------------------------------------------
							//fprintf(pEx,"%12f ", Ex[j*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i]);
							//fprintf(pEy,"%12f ", Ey[j*((Imax)*(Kmax-1))+(k*Imax)+i]);
							fprintf(pEz,"%12f ", Ez[j*((Imax)*(Kmax))+(k*(Imax))+i]);
							//fprintf(pHx,"%12f ", Hx[j*(Imax*Kmax)+(k*Imax)+i]);
							//fprintf(pHy,"%12f ", Hy[j*((Imax-1)*Kmax)+(k*(Imax-1))+i]);
							//fprintf(pHz,"%12f ", Hz[j*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i]);

							//fprintf(pEx,"%12f ", sqrt(pow(Ex[j*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i], 2)+pow(Ey[j*((Imax)*(Kmax-1))+(k*Imax)+i], 2)+pow( Ez[j*((Imax)*(Kmax))+(k*(Imax))+i], 2)));
							//fprintf(pHx,"%12f ", sqrt(pow(Hx[j*(Imax*Kmax)+(k*Imax)+i], 2)+pow(Hy[j*((Imax-1)*Kmax)+(k*(Imax-1))+i], 2)+pow( Hz[j*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i], 2)));
							
							//For J testing

							//fprintf(pHx,"%12f ", Jx[j*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i]);
							//fprintf(pHy,"%12f ", Jy[j*((Imax)*(Kmax-1))+(k*Imax)+i]);
							//fprintf(pHz,"%12f ", Jz[j*((Imax)*(Kmax))+(k*(Imax))+i]);

							//sqrt(pow(Ex[j*(Imax*Kmax)+(ksource*Imax)+i], 2)+pow(Ey[j*(Imax*Kmax)+(ksource*Imax)+i], 2)+pow( Ez[j*(Imax*Kmax)+(ksource*Imax)+i], 2)));
//--------------------------------------------------------------------------------------------------
						}
						fprintf(pEx,"\n");

						//fprintf(pEy,"\n");
						fprintf(pEz,"\n");

						fprintf(pHx,"\n");

						fprintf(pHy,"\n");
						fprintf(pHz,"\n");
				}

				if(MATLAB_SAVE==DISABLE)
				{
				  fprintf(pEx,"\n");

				  //fprintf(pEy,"\n");
				  fprintf(pEz,"\n");

				  fprintf(pHx,"\n");

				  fprintf(pHy,"\n");
				  fprintf(pHz,"\n");
				}
		  }

		  if(MATLAB_SAVE==DISABLE)
		  {
		  fprintf(pEx,"\n");

		  //fprintf(pEy,"\n");
		  fprintf(pEz,"\n");

		  fprintf(pHx,"\n");

		  fprintf(pHy,"\n");
		  fprintf(pHz,"\n");
		  }

		}
		else
		{
			save_app++;
		}
		} 

//------------------------END SAVE TO FILE-----------------------------------

		// cubics 

	}//Done time stepping

	printf("Done GPU calculations\n");

	if (Processing.perform) {
		processing_core(proc_field, Processing);
	}

	VriteSaveParameters( save_counter );


	//Read field thrue detectors
	int NetSize;
	if(CALC_ARREY_POS==ENABLE)
	{
		NetSize= det_qantI * det_qantK;
	}
	else
	{
		NetSize=DET_SIZE * DET_SIZE;
	}

	cudaStatus = cudaMemcpy(detect, D_detect, (nMax *  NetSize) * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("detector data cudaMemcpy failed!");
//        return cudaStatus;    
    }

	if(SAVE_COMPONENTS==ENABLE)
	{
//		cudaStatus = cudaMemcpy(det_Ex, D_det_Ex, (nMax *  NetSize) * sizeof(float), cudaMemcpyDeviceToHost);
//		cudaStatus = cudaMemcpy(det_Ey, D_det_Ey, (nMax *  NetSize) * sizeof(float), cudaMemcpyDeviceToHost);
		cudaStatus = cudaMemcpy(det_Ez, D_det_Ez, (nMax *  NetSize) * sizeof(float), cudaMemcpyDeviceToHost);
		cudaStatus = cudaMemcpy(det_Hx, D_det_Hx, (nMax *  NetSize) * sizeof(float), cudaMemcpyDeviceToHost);
		cudaStatus = cudaMemcpy(det_Hy, D_det_Hy, (nMax *  NetSize) * sizeof(float), cudaMemcpyDeviceToHost);
//		cudaStatus = cudaMemcpy(det_Hz, D_det_Hz, (nMax *  NetSize) * sizeof(float), cudaMemcpyDeviceToHost);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("Components detector data cudaMemcpy failed! : %s\n\n", cudaGetErrorString(cudaStatus));
	//        return cudaStatus;    
		}
	}


	if(COMP_DETECT==ENABLE)
	{
	cudaStatus = cudaMemcpy(CompDet, D_CompDet, (nMax * qantI * qantK) * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("Compensation detector data cudaMemcpy failed!");
//        return cudaStatus;    
    }
	}


	if(LINE_D_ENABLE==ENABLE)
	{
		cudaStatus = cudaMemcpy(line_detect, D_line_detect, (nMax * LINE_DET_A_QUANT * LINE_DET_A_SIZE * LINE_DET_A_SIZE) * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			printf("line detector data cudaMemcpy failed!");
	//        return cudaStatus;    
		}
	}

	if(LINE_D2_ENABLE==ENABLE)
	{
		cudaStatus = cudaMemcpy(line_detect2, D_line_detect2, (nMax * LINE_DET_B_QUANT * LINE_DET_B_SIZE * LINE_DET_B_SIZE) * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			printf("line detector2 data cudaMemcpy failed!");
	//        return cudaStatus;    
		}
	}


	//Read field on line
	if(SAVE_LINE_ENABLE==ENABLE)
	{
		cudaStatus = cudaMemcpy(line_medium, D_line_medium, (nMax * (Jmax-2)) * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			printf("Power on line data cudaMemcpy failed!");
//			return cudaStatus;    
		}
	}

	if(NET2D_SAVE_ENABLE==ENABLE)
	{
		cudaStatus=cudaMemcpy( PspdLine, DPspdLine, (nMax*(RC_qantI * RC_qantK))*sizeof(int), cudaMemcpyDeviceToHost );
		if (cudaStatus != cudaSuccess) {
				printf("DPspdLine cudaMemcpy failed!, %s", cudaGetErrorString);
			return cudaStatus;
		}
	}

//------------------------SAVE TO FILE-----------------------------------		
			//for(k=0;k<(nMax*NetSize);k++)
			//{				
			//	detSum1=detect[k]-infDet[k];
			//	infDet[k]=detSum1;
			//}


			for(i=0;i<nMax;i++)
			{
				detSum=0;
				detSum1=0;
				for(k=0;k<(NetSize);k++)
					{
						detSum+=detect[i*(NetSize)+k];
						//detSum1+=infDet[i*(NetSize)+k];
					}
					fprintf(pDet,"%lf\t%12lf\t%12lf\n ",(dt*i)*TIME_PRINT_COEF, detSum*DET_MULT);

//					out<<detSum<<std::endl;
			}

			//for(i=0;i<(nMax * LINE_DET_QUANT * LINE_DET_SIZE * LINE_DET_SIZE);i++)
			//{
			//	fprintf(pDet,"%d \t%12f mw \n ",i , line_detect[i]*1000);
			//}


//-------------------------------------------------------------------------------
	if(SAVE_COMPONENTS==ENABLE)
	{
			for(i=0;i<nMax;i++)
			{

				detSum=0; detSum2=0; detSum3=0; detSum4=0; detSum5=0; detSum6=0;

				for(k=0;k<(NetSize);k++)
					{
//						detSum+=det_Ex[i*(NetSize)+k];
//						detSum2+=det_Ey[i*(NetSize)+k];
						detSum3+=det_Ez[i*(NetSize)+k];
						detSum4+=det_Hx[i*(NetSize)+k];
						detSum5+=det_Hy[i*(NetSize)+k];
//						detSum6+=det_Hz[i*(NetSize)+k];
					}
					
//			EH_i=(detSum2*detSum6-detSum3*detSum5);
//			EH_j=-(detSum*detSum6-detSum3*detSum4);
//			EH_k=(detSum*detSum5-detSum2*detSum4);

			fprintf(p_detAll,"%lf\t%12lf\t%12lf\t%12lf\t%12lf\t%12lf\t%12lf\n ",(dt*i)*1000000, detSum*DET_MULT, detSum2*DET_MULT, detSum3*DET_MULT, detSum4*DET_MULT, detSum5*DET_MULT, detSum6*DET_MULT);


			}

			for(i=0;i<nMax;i++)
			{
				for(j=0;j<(det_qantI);j++)
				{
					for(k=0;k<(det_qantK);k++)
					{
						//fprintf(p_detEx,"%12lf\t", det_Ex[i*(NetSize)+j*det_qantK+k]*DET_MULT);
						//fprintf(p_detEy,"%12lf\t", det_Ey[i*(NetSize)+j*det_qantK+k]*DET_MULT);
						fprintf(p_detEz,"%12lf\t", det_Ez[i*(NetSize)+j*det_qantK+k]*DET_MULT);
						fprintf(p_detHx,"%12lf\t", det_Hx[i*(NetSize)+j*det_qantK+k]*DET_MULT);
						fprintf(p_detHy,"%12lf\t", det_Hy[i*(NetSize)+j*det_qantK+k]*DET_MULT);
						//fprintf(p_detHz,"%12lf\t", det_Hz[i*(NetSize)+j*det_qantK+k]*DET_MULT);

						//EH_i=(det_Ey[i*(NetSize)+j*det_qantK+k]*det_Hz[i*(NetSize)+j*det_qantK+k]-det_Ez[i*(NetSize)+j*det_qantK+k]*det_Hy[i*(NetSize)+j*det_qantK+k]);
						//EH_j=-(det_Ex[i*(NetSize)+j*det_qantK+k]*det_Hz[i*(NetSize)+j*det_qantK+k]-det_Ez[i*(NetSize)+j*det_qantK+k]*det_Hx[i*(NetSize)+j*det_qantK+k]);
						//EH_k=(det_Ex[i*(NetSize)+j*det_qantK+k]*det_Hy[i*(NetSize)+j*det_qantK+k]-det_Ey[i*(NetSize)+j*det_qantK+k]*det_Hx[i*(NetSize)+j*det_qantK+k]);

						//fprintf(p_detP,"%12lf\t", (sqrt(pow(EH_i, 2)+pow(EH_j, 2)+pow(EH_k, 2)))*DET_MULT);
					}
					//fprintf(p_detEx,"\n");
					//fprintf(p_detEy,"\n");
					fprintf(p_detEz,"\n");
					fprintf(p_detHx,"\n");
					fprintf(p_detHy,"\n");
					//fprintf(p_detHz,"\n");

					fprintf(p_detP,"\n");
				}
				//fprintf(p_detEx,"\n");
				//fprintf(p_detEy,"\n");
				fprintf(p_detEz,"\n");
				fprintf(p_detHx,"\n");
				fprintf(p_detHy,"\n");
				//fprintf(p_detHz,"\n");

				fprintf(p_detP,"\n");

			}

	}

	if((SAVE_COMPONENTS==ENABLE)&&(useRegime.S2D==ENABLE))
	{
		for(k=0;k<(nMax*NetSize);k++)
		{
			detSum1=det_Ex[k]-infDet_Ex[k];
			infDet_Ex[k]=detSum1;

			detSum1=det_Ey[k]-infDet_Ey[k];
			infDet_Ey[k]=detSum1;

			detSum1=det_Ez[k]-infDet_Ez[k];
			infDet_Ez[k]=detSum1;

			detSum1=det_Hx[k]-infDet_Hx[k];
			infDet_Hx[k]=detSum1;

			detSum1=det_Hy[k]-infDet_Hy[k];
			infDet_Hy[k]=detSum1;

			detSum1=det_Hz[k]-infDet_Hz[k];
			infDet_Hz[k]=detSum1;
		}

			for(i=0;i<nMax;i++)
			{

				detSum=0; detSum2=0; detSum3=0; detSum4=0; detSum5=0; detSum6=0;

				for(k=0;k<(NetSize);k++)
					{
						//detSum+=infDet_Ex[i*(NetSize)+k];
						//detSum2+=infDet_Ey[i*(NetSize)+k];
						detSum3+=infDet_Ez[i*(NetSize)+k];
						detSum4+=infDet_Hx[i*(NetSize)+k];
						detSum5+=infDet_Hy[i*(NetSize)+k];
						//detSum6+=infDet_Hz[i*(NetSize)+k];
					}
					
			EH_i=(detSum2*detSum6-detSum3*detSum5);
			EH_j=-(detSum*detSum6-detSum3*detSum4);
			EH_k=(detSum*detSum5-detSum2*detSum4);

			fprintf(p_infDetEx,"%lf\t%12lf\t%12lf\t%12lf\t%12lf\t%12lf\t%12lf\t%12lf\n ",(dt*i)*1000000, detSum*DET_MULT, detSum2*DET_MULT, detSum3*DET_MULT, detSum4*DET_MULT, detSum5*DET_MULT, detSum6*DET_MULT, (sqrt(pow(EH_i, 2)+pow(EH_j, 2)+pow(EH_k, 2)))*DET_MULT);


			}
	}
//---------------------comp detect save----------------------------------------
			if(COMP_DETECT==ENABLE)
			{
				for(i=0;i<nMax;i++)
				{
					detSum=0;
					for(k=0;k<(qantI * qantK);k++)
						{
							detSum+=CompDet[i*(qantI * qantK)+k];
						}
						fprintf(pComDet,"%lf\t%12lf\n ",(dt*i)*1000000, detSum*DET_MULT);

	//					out<<detSum<<std::endl;
				}
			}


//-------------------------------------------------------------------------------
			if(LINE_D_ENABLE==ENABLE)
			{
				for(i=0;i<(nMax);i++)
				{
					fprintf(pLineDet,"%lf ", (dt*i)*1000000);
					for(j=0;j<(LINE_DET_A_QUANT);j++)
					{
						detSum=0;
						for(k=0;k<(LINE_DET_A_SIZE * LINE_DET_A_SIZE);k++)
						{
							detSum+=line_detect[i*(LINE_DET_A_QUANT * LINE_DET_A_SIZE * LINE_DET_A_SIZE)+j*(LINE_DET_A_SIZE * LINE_DET_A_SIZE)+k];
						}
						fprintf(pLineDet,"\t%12f ", detSum*DET_MULT);
					}
					fprintf(pLineDet,"\n");
				}
			}
//----------------------------------------------------------------------------------
			if(LINE_D2_ENABLE==ENABLE)
			{
				for(i=0;i<(nMax);i++)
				{
					fprintf(pLineDet2,"%lf ", (dt*i)*1000000);
					for(j=0;j<(LINE_DET_B_QUANT);j++)
					{
						detSum=0;
						for(k=0;k<(LINE_DET_B_SIZE * LINE_DET_B_SIZE);k++)
						{
							detSum+=line_detect2[i*(LINE_DET_B_QUANT * LINE_DET_B_SIZE * LINE_DET_B_SIZE)+j*(LINE_DET_B_SIZE * LINE_DET_B_SIZE)+k];
						}
						fprintf(pLineDet2,"\t%12f ", detSum*DET_MULT);
					}
					fprintf(pLineDet2,"\n");
				}
			}

//----------------------------------------------------------------------------------

			if(SAVE_LINE_ENABLE==ENABLE)
			{
				for(i=0;i<nMax;i++)
				{
					for(k=0;k<(Jmax-2);k++)
					{
						fprintf(pSLine,"%lf ", line_medium[(i*(Jmax-2))+k]*1000);
					}
					fprintf(pSLine,"\n ");
				}
			}

//----------------------------------------------------------------------------------
			if(NET2D_SAVE_ENABLE==ENABLE)
			{
				/*
				fstream outfile("CUDA_Spd.txt", fstream::out);
				for(i=0;i<nMax;i++)
				{					
					for(k=0;k<(RC_qantI * RC_qantK);k++)
						{
							 outfile<<PspdLine[i*(RC_qantI * RC_qantK)+k]<<" "; 
						}
						outfile<<std::endl;
				}
				outfile.close();
				*/
//--------------------+++++++++++++++++++++++++++++++++---------------
				RClineFilterPostTime(PspdLine, PspdFiltred, RCdetFilt.window);
//--------------------+++++++++++++++++++++++++++++++++---------------
				/*
				outfile.open("CUDA_SpdFltr.txt", fstream::out);
				for(i=0;i<nMax;i++)
				{					
					for(k=0;k<(RC_qantI * RC_qantK);k++)
						{
							 outfile<<PspdFiltred[i*(RC_qantI * RC_qantK)+k]<<" "; 
						}
						outfile<<std::endl;
				}
				outfile.close();
				*/

				CalculateRCfromPwrLines(PspdLine, PspdFiltred);
				CalculateTaufromPwrLines(PspdLine, PspdFiltred);
			}

//------------------------END SAVE TO FILE-----------------------------------

	printf("Done time-stepping...\n");


	// Display operating system-style date and time. 
	char tmpTime[128], tmpDate[128];
    _strtime_s( tmpTime, 128 );
    _strdate_s( tmpDate, 128 );

	runTime = clock() - runTime;

//plog=fopen("log.txt","a");

fopen_s(&plog,"log.txt","a");
fprintf(plog,"-------------------------------------------------------------------------\n");
    fprintf( plog,"OS date:\t\t\t\t%s\n", tmpDate );
	fprintf( plog,"OS time:\t\t\t\t%s\n", tmpTime );
fprintf(plog,"-------------------------------------------------------------------------\n");
fprintf(plog,"NET PARAMETERS \n Ex(%d, %d, %d) \n Ey(%d, %d, %d) \n Ez(%d, %d, %d) \n Hx(%d, %d, %d) \n Hy(%d, %d, %d) \n Hz(%d, %d, %d)\n\n(dx ,dy, dz) - dt: (%f mm, %f mm, %f mm) - %f ns\n", 
		Imax-1, Jmax, Kmax-1, Imax, Jmax-1, Kmax-1, Imax, Jmax, Kmax, Imax, Jmax-1, Kmax, Imax-1, Jmax, Kmax, Imax-1, Jmax-1, Kmax-1, dx*1000, dy*1000, dz*1000, dt*1e9);

fprintf(plog,"\nSIMULATION PARAMETERS\n Total Simulation time: %d time steps or %e seconds\n", nMax, nMax*dt );

fprintf(plog,"\nSAVE DATA PARAMETER\nSlises saved: %d\nData saved from time %d to time %d \nTime sieve parameter: %d \nNet sieve parameter: %d\nData multiplier: %f\n", 
		save_counter, FieldSaveParam.fromTime , FieldSaveParam.ToTime , FieldSaveParam.TimeSive, FieldSaveParam.FieldSieve, MAG);
fprintf(plog,"\nSORCE AND DETECTOR\n Source point: (%d, %d, %d)\n Detector point(%d, %d, %d)\n Detector size: %dx%d points", 
		SourceCen.I, SourceCen.J, SourceCen.K, DetectCen.I, DetectCen.J, DetectCen.K, DET_SIZE, DET_SIZE);

fprintf(plog,"\n\nCalculation on GPU\n Calculation time: %f seconds\n\n", (double)runTime/CLOCKS_PER_SEC);

//-----------------END of printing to LOG file-----------------------------------------------------


//---------------DEBUG--------------------------
FILE *debM;
deb = fopen ( "deb.txt" , "w");
//fprintf(deb,"Experiment \n");
debM = fopen ( "materials.txt" , "w");
//---------------END OF DEBUG-------------------
//---------------DEBUG--------------------------
//	sigma[i]

////fprintf(debM,"\nepsilon: ");
//for(k = 0; k < numMaterials; k++) 
//		{			
//			fprintf(debM,"%12.15f ", epsilon[k]);
//		}
//fprintf(debM,"\n");
////fprintf(debM,"\n  sigma: ");
//for(k = 0; k < numMaterials; k++) 
//		{						
//			fprintf(debM,"%12.15f ", sigma[k]);
//		}
//fprintf(debM,"\n");
////fprintf(debM,"\n  ne: ");
//for(k = 0; k < numMaterials; k++) 
//		{						
//			fprintf(debM,"%12.15f ", pne[k]);
//		}
//fprintf(debM,"\n");
////fprintf(debM,"\n\n     CA: ");
//for(k = 0; k < numMaterials; k++) 
//		{			
//			fprintf(debM,"%f ", CA[k]);
//		}
//fprintf(debM,"\n");
////fprintf(debM,"\n\n     CB: ");
//for(k = 0; k < numMaterials; k++) 
//		{						
//			fprintf(debM,"%f ", CB[k]);
//		}
//fprintf(debM,"\n");
////fprintf(debM,"\n\n     DA: ");
//for(k = 0; k < numMaterials; k++) 
//		{			
//			fprintf(debM,"%12.15f ", DA[k]);
//		}
//		fprintf(debM,"\n");
////fprintf(debM,"\n\n     DB: ");
//for(k = 0; k < numMaterials; k++) 
//		{						
//			fprintf(debM,"%12.15f ", DB[k]);
//		}
//		fprintf(debM,"\n");

//fprintf(deb,"\n\n");
//---------------END OF DEBUG-------------------
//---------------DEBUG--------------------------
//	k=12;		
	k=Kmax/2;		
		//for(k = 0; k < Kmax; k++) 
		{			
			//fprintf(deb,"slise num: %d \n", k);		    
			for(i = 1; i < Imax-1; i++) 
			{				
				for(j = 1; j < Jmax-1; j++)
				{
					fprintf(deb,"%d ", ID3[j*(Imax*Kmax)+(k*Imax)+i]);
				}
				fprintf(deb,"\n");
			}
			fprintf(deb,"\n\n");
		}
		fprintf(deb,"\n\n");
//---------------END OF DEBUG-------------------
//---------------DEBUG--------------------------
//	k=12;

//fprintf(deb,"\n\n");
//	j=J_DET;		
//		//for(k = 0; k < Kmax; k++) 
//		{			
//			fprintf(deb,"slise num: %d \n", k);		    
//			for(i = 1; i < Imax-1; i++) 
//			{				
//				for(k = 1; k < Kmax-1; k++)
//				{
//					fprintf(deb,"%d ", ID3[j*(Imax*Kmax)+(k*Imax)+i]);
//				}
//				fprintf(deb,"\n");
//			}
//			fprintf(deb,"\n\n");
//		}
//		fprintf(deb,"\n\n");
//---------------END OF DEBUG-------------------
//---------------DEBUG--------------------------
fclose(deb);
fclose(debM);
//---------------END OF DEBUG-------------------



	fclose(test);
	fclose(pEx);
	fclose(pEy);
	fclose(pEz);
	fclose(pHx);
	fclose(pHy);
	fclose(pHz);
	fclose(plog);
	fclose(pDet);
	if(LINE_D_ENABLE==ENABLE)
	fclose(pLineDet);
	if(LINE_D2_ENABLE==ENABLE)
	fclose(pLineDet2);
	if(SAVE_LINE_ENABLE==ENABLE)
	fclose(pSLine);
	fclose(pComDet);

	fclose(p_detAll);

	fclose(p_detEx);
	fclose(p_detEy);
	fclose(p_detEz);
	fclose(p_detHx);
	fclose(p_detHy);
	fclose(p_detHz);

	fclose(p_detP);

	fclose(p_infDetEx);
	
	out.close();

	//cudaStatus = cudaGetLastError();
	//printf("\nDEBUG last error: %s\n", cudaGetErrorString(cudaStatus));

	cudaStatus=DevMemoryFree();
	printf("\nGPU last error on end calculating: %s\n", cudaGetErrorString(cudaStatus));
	return cudaStatus;
}

__host__ cudaError_t DevMemoryFree()
{
	cudaError_t cudaStatus;
    // Allocate GPU buffers for E and H fields vectors
cudaStatus = 	cudaFree(DHx);
cudaStatus = 	cudaFree(DHy);
cudaStatus = 	cudaFree(DHz);

//-------------------------------------------------------------------------------------------------

cudaStatus = 	cudaFree(DEx);
cudaStatus = 	cudaFree(DEy);
cudaStatus = 	cudaFree(DEz);

//-------------------------------------------------------------------------------------------------

cudaStatus = 	cudaFree(DJx);
cudaStatus = 	cudaFree(DJy);
cudaStatus = 	cudaFree(DJz);

//-------------------------------------------------------------------------------------------------


cudaStatus = 	cudaFree(D_ID1);
cudaStatus = 	cudaFree(D_ID2);
cudaStatus = 	cudaFree(D_ID3);

//------------------DEN----------------------------------------------------------------------

cudaStatus =cudaFree(D_den_ex);		    
cudaStatus =cudaFree(D_den_hx);

cudaStatus =cudaFree(D_den_ey);
cudaStatus =cudaFree(D_den_hy);

cudaStatus =cudaFree(D_den_ez);
cudaStatus =cudaFree(D_den_hz);

//------------------PML coef---------------------------------------------------------------------

	    
cudaStatus =cudaFree(D_psi_Ezx_1); //!
cudaStatus =cudaFree(D_psi_Ezx_2);
cudaStatus =cudaFree(D_psi_Hyx_1);
cudaStatus =cudaFree(D_psi_Hyx_2);
cudaStatus =cudaFree(D_psi_Ezy_1);
cudaStatus =cudaFree(D_psi_Ezy_2);
cudaStatus =cudaFree(D_psi_Hxy_1);
cudaStatus =cudaFree(D_psi_Hxy_2);
cudaStatus =cudaFree(D_psi_Hxz_1);
cudaStatus =cudaFree(D_psi_Hxz_2);
cudaStatus =cudaFree(D_psi_Hyz_1);
cudaStatus =cudaFree(D_psi_Hyz_2);
cudaStatus =cudaFree(D_psi_Exz_1);
cudaStatus =cudaFree(D_psi_Exz_2);
cudaStatus =cudaFree(D_psi_Eyz_1);
cudaStatus =cudaFree(D_psi_Eyz_2);
cudaStatus =cudaFree(D_psi_Hzx_1);
cudaStatus =cudaFree(D_psi_Hzx_2);
cudaStatus =cudaFree(D_psi_Eyx_1);
cudaStatus =cudaFree(D_psi_Eyx_2);
cudaStatus =cudaFree(D_psi_Hzy_1);
cudaStatus =cudaFree(D_psi_Hzy_2);
cudaStatus =cudaFree(D_psi_Exy_1);
cudaStatus =cudaFree(D_psi_Exy_2);


//---------------------1D PML coeff------------------------------------------------------------------------

cudaStatus =cudaFree(D_be_x_1);
cudaStatus =cudaFree(D_ce_x_1);

cudaStatus =cudaFree(D_bh_x_1);
cudaStatus =cudaFree(D_ch_x_1);

cudaStatus =cudaFree(D_be_x_2);
cudaStatus =cudaFree(D_ce_x_2);
	    
cudaStatus =cudaFree(D_bh_x_2);
cudaStatus =cudaFree(D_ch_x_2);

cudaStatus =cudaFree(D_be_y_1);
cudaStatus =cudaFree(D_ce_y_1);

cudaStatus =cudaFree(D_bh_y_1);
cudaStatus =cudaFree(D_ch_y_1);

cudaStatus =cudaFree(D_be_y_2);
cudaStatus =cudaFree(D_ce_y_2);

cudaStatus =cudaFree(D_bh_y_2);
cudaStatus =cudaFree(D_ch_y_2);

cudaStatus =cudaFree(D_be_z_1);
cudaStatus =cudaFree(D_ce_z_1);

cudaStatus =cudaFree(D_bh_z_1);
cudaStatus =cudaFree(D_ch_z_1);

cudaStatus =cudaFree(D_be_z_2);
cudaStatus =cudaFree(D_ce_z_2);

cudaStatus =cudaFree(D_bh_z_2);
cudaStatus =cudaFree(D_ch_z_2);

//--------------------------------------------------

cudaStatus =cudaFree(D_CA);
cudaStatus =cudaFree(D_CB);

cudaStatus =cudaFree(D_DA);
cudaStatus =cudaFree(D_DB);

cudaStatus =cudaFree(D_NE);

cudaStatus =cudaFree(D_detect);

if(LINE_D_ENABLE==ENABLE)
cudaStatus =cudaFree(D_line_detect);
if(LINE_D2_ENABLE==ENABLE)
cudaStatus =cudaFree(D_line_detect2);

if(COMP_DETECT==ENABLE)
{
cudaStatus =cudaFree(D_arrayI);
cudaStatus =cudaFree(D_arrayK);

cudaStatus =cudaFree(Ddet_arrayI);
cudaStatus =cudaFree(Ddet_arrayK);
}

if(SAVE_COMPONENTS==ENABLE)
{
cudaStatus =cudaFree(D_det_Ex);
cudaStatus =cudaFree(D_det_Ey);
cudaStatus =cudaFree(D_det_Ez);

cudaStatus =cudaFree(D_det_Hx);
cudaStatus =cudaFree(D_det_Hy);
cudaStatus =cudaFree(D_det_Hz);
}


if(COMP_DETECT==ENABLE)
cudaStatus =cudaFree(D_CompDet);


if(SAVE_LINE_ENABLE==ENABLE)
cudaStatus =cudaFree(D_line_medium);

if(NET2D_SAVE_ENABLE == ENABLE)
{
cudaStatus =cudaFree(D_RCdetArrayI);
cudaStatus =cudaFree(D_RCdetArrayK);

cudaStatus =cudaFree(DPspdLine);
cudaStatus =cudaFree(DPspdFiltred);
cudaStatus =cudaFree(DedgeL);
}

	    
		cudaStatus = cudaGetLastError();
	 //   if (cudaStatus != cudaSuccess) {
		//   printf("Error while cudaFree exec");
  //      return cudaStatus;
    //}
	return cudaStatus;
}

__host__ cudaError_t CheckCoefInGPU(float *dCoef, int DCoef_size )						 
{
cudaError_t cudaStatus;
float *testField;
FILE *P_ind_test;
	P_ind_test=fopen("index_test.txt","w");

	testField = (float *)malloc(DCoef_size * sizeof(float));

	cudaStatus = cudaMemcpy(testField, dCoef, DCoef_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf( "\nCheckCoefInGPU cudaMemcpy failed!");
		return cudaStatus;
    }

	for(int i_index=0;i_index<DCoef_size;i_index++)
		{
			fprintf(P_ind_test,"%12.15f\t", testField[i_index]);
			if(i_index%32==0)
				fprintf(P_ind_test,"\n");
		}
	fclose(P_ind_test);
	free(testField);
	return cudaStatus;
}

__host__ cudaError_t CheckCoefInGPU(int *dCoef, int DCoef_size )						 
{
cudaError_t cudaStatus;
float *testField;
FILE *P_ind_test;
	P_ind_test=fopen("index_test.txt","w");

	testField = (float *)malloc(DCoef_size * sizeof(float));

	cudaStatus = cudaMemcpy(testField, dCoef, DCoef_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf( "\nCheckCoefInGPU cudaMemcpy failed!");
		return cudaStatus;
    }

	for(int i_index=0;i_index<DCoef_size;i_index++)
		{
			fprintf(P_ind_test,"%12.15f\t", testField[i_index]);
			if(i_index%32==0)
				fprintf(P_ind_test,"\n");
		}
	fclose(P_ind_test);
	free(testField);
	return cudaStatus;
}

__host__ void VriteSaveParameters( int SliceSaved )						 
{
	FILE *pSM;
	fopen_s(&pSM,"FieldSaveDat.txt","w");
	int countM=0;
	for(i=1;i<Imax-1;i+=FieldSaveParam.FieldSieve)
		countM++;

	fprintf(pSM,"%d %d",countM, SliceSaved);
	fclose(pSM);
}

__host__ cudaError_t CloseOpenDoor( int action )						 
{
	cudaError_t cudaStatus;

	if(action==1) //close
	{
	cudaStatus = cudaMemcpy(D_ID1, ID1_close, (Imax*Jmax*Kmax) * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(D_ID2, ID2_close, (Imax*Jmax*Kmax) * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(D_ID3, ID3_close, (Imax*Jmax*Kmax) * sizeof(int), cudaMemcpyHostToDevice);	
	}
	else //open
	{
	cudaStatus = cudaMemcpy(D_ID1, ID1, (Imax*Jmax*Kmax) * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(D_ID2, ID2, (Imax*Jmax*Kmax) * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(D_ID3, ID3, (Imax*Jmax*Kmax) * sizeof(int), cudaMemcpyHostToDevice);	
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
	{
		   printf("Change medium failed!");
        return cudaStatus;
	}
	return cudaStatus;
}

__host__ cudaError_t CheckDimConstInGPU( )						 
{
cudaError_t cudaStatus;
int hostData[3];

	//cudaStatus=cudaMemcpyToSymbol ( &hostData[0], DImax, sizeof ( int ), 0, cudaMemcpyDeviceToHost );    
	//cudaStatus=cudaMemcpyToSymbol ( &hostData[1], DJmax, sizeof ( int ), 0, cudaMemcpyDeviceToHost ); 
	//cudaStatus=cudaMemcpyToSymbol ( &hostData[2], DKmax, sizeof ( int ), 0, cudaMemcpyDeviceToHost );    
 //   if (cudaStatus != cudaSuccess) {
 //       printf("CheckDimConstInGPU failed!, %s\n", cudaGetErrorString(cudaStatus));
 //       return cudaStatus;
 //   }

	//printf(" \nSize on GPU const(%d %d %d)\n", hostData[0], hostData[1], hostData[2]);
cudaStatus=cudaSuccess;
	return cudaStatus;
}


//---------------------------------------------------------------------------------------
//	>>>>>>>>>>>>>>>>>	Additional host subsidiary function		<<<<<<<<<<<<<<<<<<<<<<<
//--------------------------------------------------------------------------------------

__host__ void findSpdLine(float *mas)
{
	FILE *pSpd, *pRc;
	int *ans, Rc=0;
	float *Fans;
	ans = (int *)malloc((nMax * sizeof(int)));
	Fans = (float *)malloc((nMax * sizeof(float)));
	fopen_s(&pSpd,"SpdLine.txt","w");
	fopen_s(&pRc,"RcData.txt","w");
//---------------------------------------------------------
	float edgePercent=0.01 , edgeL;
	int i, k;
	float temp=mas[0];

	temp=FindMax(mas, nMax*(Jmax-2));

	edgeL=temp*edgePercent;

	std::cout<<"temp = "<<temp<<" "<<"edgeL = "<<edgeL<<std::endl;

	for(i=0;i<nMax;i++)
		{
			ans[i]=0;
			k=(Jmax-2);
			do
			{
				k--;
				if((mas[(i*(Jmax-2))+k])<edgeL)
				ans[i]=k;
				
			}
			while(k>0 && (mas[(i*(Jmax-2))+k])<edgeL);
		}

	FilterPostTime(ans, Fans, nMax);
	
	//for(i=0;i<nMax;i++)
	//	{
	//		if(Fans[i]>Rc)
	//		Rc=Fans[i];
	//	}

	Rc=FindMax(Fans, nMax);

	std::cout<<"Rc= "<< Rc<<std::endl;
	fprintf(pRc,"%d ",Rc);

	for(i=0;i<nMax;i++)
	fprintf(pSpd,"%d ",ans[i]);

	fprintf(pSpd,"\n");

	for(i=0;i<nMax;i++)
	fprintf(pSpd,"%f ",Fans[i]);

	fclose(pSpd);
	fclose(pRc);
}

__host__ float FindMax(float *mas, int counts)
{
	float temp=mas[0];
	for(int i=0;i<counts;i++)
		{
			if(mas[i]>temp)
			temp=mas[i];
		}
	return temp;
}

__host__ void FilterPostTime(int *mas, float *ans, int counts)
{
	int w_beg=0,w_end=-200;
	float weight=1.0/abs(w_end);
	for(int i=0;i<counts;i++)
	{
		ans[i]=0;
		for(int j=w_beg;j>w_end;j--)
			if(j>0 && j<counts)
			ans[i]+=(float)mas[j]*weight;

		w_beg++;
		w_end++;
	}
}

__host__ void CalculateRCfromPwrLines(int *mas, float *fltr)
{
	int temp;
	float Ftemp;

	using namespace std;

	fstream outfile("RC_Line.txt", fstream::out);
		for(k=0;k<(RC_qantI * RC_qantK);k++)
		{
			temp=0;
			Ftemp=0;
			for(i=0;i<nMax;i++)
				{					 
					if(mas[i*(RC_qantI * RC_qantK)+k]>temp)
					temp=mas[i*(RC_qantI * RC_qantK)+k];
				}
			for(i=0;i<nMax;i++)
				{					 
					if(fltr[i*(RC_qantI * RC_qantK)+k]>Ftemp)
					Ftemp=fltr[i*(RC_qantI * RC_qantK)+k];
				}
			outfile<<temp<<" "<<Ftemp<<" "<<Ftemp*ds<<endl;
				
		}
		outfile.close();
} 

__host__ void CalculateTaufromPwrLines(int *mas, float *fltr)
{
	int temp;
	float Ftemp;

	using namespace std;

	fstream outfile("TAU_Line.txt", fstream::out);
		for(k=0;k<(RC_qantI * RC_qantK);k++)
		{
			temp=0;
			Ftemp=0;
			//i=2000;
			//do
			//{				
			//	temp = i*2;
			//	i++;
			//}
			//while(i<nMax && ((float)mas[i*(RC_qantI * RC_qantK)+k]-(float)mas[(i-1)*(RC_qantI * RC_qantK)+k]>=0));

			i=2000;
			do
			{				
				Ftemp = i*2;
				i++;
			}
			while(i<nMax && (fltr[i*(RC_qantI * RC_qantK)+k]-fltr[(i-1)*(RC_qantI * RC_qantK)+k]>=0));

			outfile<<Ftemp<<" "<<Ftemp*dt*TIME_PRINT_COEF<<endl;
				
		}
		outfile.close();
} 

__host__ void RClineFilterPostTime(int *mas, float *ans, int w_size)
{
	int w_beg ,w_end;
	float weight=1.0/abs(w_size);

	for(k=0;k<(RC_qantI * RC_qantK);k++)
	{
		w_beg=w_size/2;
		w_end=-w_size/2;
		for(int i=0;i<nMax;i++)
		{
			ans[i*(RC_qantI * RC_qantK)+k]=0;
			for(int j=w_beg;j>w_end;j--)
				if(j>0 && j<nMax)
				ans[i*(RC_qantI * RC_qantK)+k]+=(float)mas[j*(RC_qantI * RC_qantK)+k]*weight;

			w_beg++;
			w_end++;
		}
	}
}

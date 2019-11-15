#ifndef _SETTINGS_CUH_
#define _SETTINGS_CUH_

//51-126-26
#define GPU_DEVICE 0

#define M_PI 3.14

#define DS 1

#define TIME_COEF 0.5

#define TIME_PRINT_COEF 1e12
//-------------------------------------------
#define CALC_SOURCE_TO_DETECT_INFLUENCE 0

//--------------------------------------------

#define I_SIZE 550
#define J_SIZE 600
#define K_SIZE 1

#define PML_SIZE 11			
#define MAX_TIME_STEP 10000
#define MAX_TIME_NSECONDS 10

//--------Source-----------------------------
#define BUILD_DIPOLE 0
//25-63-12
#define I_SOURCE 300
#define J_SOURCE 20
#define K_SOURCE 1

#define I_SRC_SIZE 1
#define J_SRC_SIZE 1
#define K_SRC_SIZE 1

#define S_DIRECTION 'z'
#define DIPOL_DIRECTION 'y'

//--------AESA settings(in mm)-----------------
#define CALC_ARREY_POS 1

#define AA_SZ_I 40
#define AA_SZ_K 1

#define DI_AUTO 0

#define AA_DELTA_I 5
#define AA_DELTA_K 5

//---------AESA Detecttor settings(in mm)----------------
#define DD_SZ_I 40
#define DD_SZ_K 1

#define DD_AUTO 0

#define DD_DELTA_I 5
#define DD_DELTA_K 5

//---------AESA compensation detector settings----------------
#define COMP_DETECT 0

//--------detector-----------------------------
#define SAVE_COMPONENTS 1

#define DET_SIZE 10
#define I_DET 250
#define J_DET 30		//60
#define K_DET 1

#define DET_MULT 1000		//1000000

//-------------build antennas----------------
//-------------Source antenna----------------
#define ABUILD 0
#define ANTENNA_TUPE 5	// 1 - Square Tube; 2 - Hole in the wall; 3 - Square conical antenna; 4 - Hole after source; 5 - Square conical antenna; 6-closed square conical antenna;

#define A_Y_LENGTH 120
#define A_THICKNESS 3
#define A_SIZE 25
#define A_INITIAL_SIZE 10
//#define A_CURVE A_Y_LENGTH/120

#define A_MATERIAL 5

//----------------------------------------------
#define CLOSE_ENABLE 0
#define CLOSE_TIME 1000

//-------------Detector antenna----------------
#define DBUILD 0
#define D_ANTEN_TUPE 3	// 1 - Square Tube; 2 - Square Tube whith 45 grad border; 3 - Square conical antenna

#define D_Y_LENGTH 120
#define D_THICKNESS 3
#define D_SIZE 25
#define D_INITIAL_SIZE 10
//#define D_CURVE D_Y_LENGTH/120

#define D_MATERIAL 5

//------------STELL WALL-------------------
#define MAKEWALL 0

#define WALLPOS 950


//--------save data settings------------------
//--------2D slice settings------------>>>>>>>
#define		FROM_TIME 0
#define		TO_TIME 40000

#define	SAVE_APP_COUNT 240

#define SIEVE 1

#define	MAG 1 //множитель (домнажает сохраняемый результат при сохранении)

#define MATLAB_SAVE 1 //тип сохранения(1 - для автоматического построения в matlab), 0-для просмотра в текстовом редакторе и других целей.

//--------1D line settings------------>>>>>>>

#define		SAVE_LINE_ENABLE 0		//0-выключить, 1-включить. 		
#define		I_LINE	250
#define		K_LINE	40

#define		NET2D_SAVE_ENABLE 1		//0-выключить, 1-включить. 	RCdetect Enable
			
//--------line detector(speed measuring)-----------------------------
#define LINE_D_ENABLE 0

#define LINE_DET_A_SIZE 10

#define LINE_DET_A_STEP 170
#define LINE_DET_A_QUANT 3
#define I_LINE_DET_A 135
#define J_LINE_DET_A 70
#define K_LINE_DET_A 40

//--------line detector2(speed measuring)-----------------------------
#define LINE_D2_ENABLE 0

#define LINE_DET_B_SIZE 10

#define LINE_DET_B_STEP 170
#define LINE_DET_B_QUANT 3
#define I_LINE_DET_B 65
#define J_LINE_DET_B 70
#define K_LINE_DET_B 40

//--------KAMERA PARAMETERS--------------------------------
#define DRAW_2D_FIGURE 0

#define KAMERA_KONFIG 7
#define PATRUBOK_L 5


//--------MAIN MEDIUM--------------------------------
#define BUILD_SPHER_MEDIUM 0


#define SPHERE_MEDIUM_EQUOTION 2


#define M_COEF 100

#define M_NE0 1e+19
#define M_NEA 0

//-----Pulse parameters
#define P_PLATOW 2000
#define P_NAMPSIZE 2000

#define P_PLATOW_NS 3
#define P_NAMPSIZE_NS 3


#define FREQ_CHIRP 0

#define P_FREQ_A 28
#define P_FREQ_B 30


//-----center of elipse----------------
#define I_MEDIUM 275
#define J_MEDIUM 305
#define K_MEDIUM 1

//-----elipse coeff--------------------
#define A_COEF 1
#define B_COEF 1
#define C_COEF 1

//-----figure coeff--------------------
#define P_COEF 1
#define Q_COEF 1
#define N_COEF 1

//-----Ne equotin coefficients----------
#define	EQ_RADIUS 50
#define	EQ_ALPHA 2.0
#define	EQ_BETA 1.0
//-----Torus coeficients, if torus is used----------
#define EQ_RAD_TORUS 175
                               
//+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
//settings difines
#define ENABLE 1
#define DISABLE 0

/*
[Cubes]
Cudes_load = 1/0;
Cubes_save = 1/0; сохранять / не сохранять
Cubes_pos_i = ;
Cubes_pos_j = ; левый верхний угол блока кубиков
Cubes_pos_k = ;
Cubes_size_i = ;
Cubes_size_j = ; размеры блока
Cubes_size_k = ;
Cubes_time_from = ;
Cubes_time_to = ; сохранять от и до с прореживанием
Cubes_time_sieve = ;
*/


#endif //_SETTINGS_CUH_
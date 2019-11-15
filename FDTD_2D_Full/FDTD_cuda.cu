// CPML_LineIndex.cpp : Defines the entry point for the console application.
//

/*******************************************************************
//     3-D FD code with CPML absorbing boundary conditions - Dipole
//******************************************************************
//
//     This C code implements the finite-difference time-domain
//     solution of Maxwell's curl equations over a three-dimensional
//     Cartesian space lattice.  The grid is terminated by CPML ABCs.
//     The thickness of the PML in each Cartesian direction can be 
//     varied independently.
//
//******************************************************************/

#include"FDTD_cuda.cuh"
#include"Kernel.cuh"
//#include <fstream>

void main() {
	//freopen("outlog.txt", "w", stdout);
	initlimits();
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int readINIparam() {
	INIReader reader("Settings.ini");

	if (reader.ParseError() < 0) {
		std::cout << "Can't load 'Settings.ini'\n";
		return 1;
	}

	GPU_device = reader.GetInteger("hardware", "GPU_device", GPU_DEVICE);
	useRegime.S2D = reader.GetInteger("regimes", "SourceToDetect", CALC_SOURCE_TO_DETECT_INFLUENCE);

	ds=reader.GetReal("grid", "ds", DS);
	dt_coef=reader.GetReal("grid", "dt_coef", TIME_COEF);

	Imax = reader.GetInteger("grid", "Imax", I_SIZE);
	Jmax = reader.GetInteger("grid", "Jmax", J_SIZE);
	Kmax = K_SIZE;

	Exp_time_key=reader.GetInteger("grid", "Use_Exp_time", 0);
	nMax = reader.GetInteger("grid", "Max_time", MAX_TIME_STEP);
	Exp_time=reader.GetReal("grid", "Exp_time", MAX_TIME_NSECONDS);

	SourceCen.I=reader.GetInteger("grid", "Isource", I_SOURCE);		
	SourceCen.J=reader.GetInteger("grid", "Jsource", J_SOURCE);
	SourceCen.K=K_SOURCE;

	DetectCen.I=reader.GetInteger("grid", "Idetect", I_DET);		
	DetectCen.J=reader.GetInteger("grid", "Jdetect", J_DET);
	DetectCen.K=K_DET;

	SourceAESA.PI=reader.GetInteger("AESA", "SIsize", AA_SZ_I);	
	SourceAESA.PK=1;
	SourceAESA.deltaAuto=reader.GetInteger("AESA", "SAutoDelta", DI_AUTO);
	SourceAESA.DI=reader.GetInteger("AESA", "SIdelta", AA_DELTA_I);
	SourceAESA.DK=5;
	AESAsourceParam.Angle=reader.GetReal("AESA", "SPAngle", 0);


	DetectAESA.PI=reader.GetInteger("AESA", "DIsize", DD_SZ_I);
	DetectAESA.PK=1;
	DetectAESA.deltaAuto=reader.GetInteger("AESA", "SAutoDelta", DD_AUTO);
	DetectAESA.DI=reader.GetInteger("AESA", "DIdelta", DD_DELTA_I);
	DetectAESA.DK=5;

	SourcePulse.chirp=reader.GetInteger("PulseParam", "Chirp", FREQ_CHIRP);
	SourcePulse.freqA=reader.GetReal("PulseParam", "FreqA", P_FREQ_A);
	SourcePulse.freqB=reader.GetReal("PulseParam", "FreqB", P_FREQ_B);
	SourcePulse.Front=reader.GetInteger("PulseParam", "Fronts", P_NAMPSIZE);
	SourcePulse.Plato=reader.GetInteger("PulseParam", "Plato", P_PLATOW);

	Fronts_ns=reader.GetReal("PulseParam", "Fronts_ns", P_NAMPSIZE_NS);
	Plato_ns=reader.GetReal("PulseParam", "Plato_ns", P_PLATOW_NS);

	amp=reader.GetReal("PulseParam", "Pulse_amp", 1000); 

	FieldSaveParam.fromTime=reader.GetInteger("2Dsave", "From", FROM_TIME);
	FieldSaveParam.ToTime=reader.GetInteger("2Dsave", "To", TO_TIME);
	FieldSaveParam.TimeSive=reader.GetInteger("2Dsave", "TimeSieve", SAVE_APP_COUNT);			
	FieldSaveParam.FieldSieve=reader.GetInteger("2Dsave", "GridSieve", SIEVE);	

	Build_medium=reader.GetInteger("medium", "Build", BUILD_SPHER_MEDIUM);		
	MediumCen.I=reader.GetInteger("medium", "Icen", I_MEDIUM);		
	MediumCen.J=reader.GetInteger("medium", "Jcen", J_MEDIUM);
	MediumCen.K=K_MEDIUM;

	SphereCoef.A=reader.GetReal("medium", "A", A_COEF);
	SphereCoef.B=reader.GetReal("medium", "B", B_COEF);
	SphereCoef.C=reader.GetReal("medium", "C", C_COEF);
	SphereCoef.P=reader.GetInteger("medium", "P", P_COEF);
	SphereCoef.Q=reader.GetInteger("medium", "Q", Q_COEF);
	SphereCoef.N=reader.GetInteger("medium", "N", N_COEF);

	EqCoef.R=reader.GetInteger("medium", "Radius", EQ_RADIUS);
	EqCoef.Alpha=reader.GetReal("medium", "Alpha", EQ_ALPHA);
	EqCoef.Beta=reader.GetReal("medium", "Beta", EQ_BETA);
	EqCoef.ne0=reader.GetReal("medium", "ne0", M_NE0);
	EqCoef.neA=reader.GetReal("medium", "neA", M_NEA);

	Torus_big_r=reader.GetInteger("medium", "TorusBR", EQ_RAD_TORUS);

	Build_fluc_medium=reader.GetInteger("mediumFluc", "FlucBuild", 0);		
	MediumFlucCen.I=reader.GetInteger("mediumFluc", "FlucIcen", I_MEDIUM);		
	MediumFlucCen.J=reader.GetInteger("mediumFluc", "FlucJcen", J_MEDIUM);
	MediumFlucCen.K=K_MEDIUM;

	SphereFlucCoef.A=reader.GetReal("mediumFluc", "FlucA", A_COEF);
	SphereFlucCoef.B=reader.GetReal("mediumFluc", "FlucB", B_COEF);
	SphereFlucCoef.C=reader.GetReal("mediumFluc", "FlucC", C_COEF);
	SphereFlucCoef.P=reader.GetInteger("mediumFluc", "FlucP", P_COEF);
	SphereFlucCoef.Q=reader.GetInteger("mediumFluc", "FlucQ", Q_COEF);
	SphereFlucCoef.N=reader.GetInteger("mediumFluc", "FlucN", N_COEF);

	FlucAngle.I=reader.GetReal("mediumFluc", "FlucAngleI", 0);
	FlucAngle.J=reader.GetReal("mediumFluc", "FlucAngleJ", 0);
	FlucAngle.K=reader.GetReal("mediumFluc", "FlucAngleK", 0);
	flucSign=reader.GetInteger("mediumFluc", "FlucSign", 1);

	FlucEqCoef.R=reader.GetInteger("mediumFluc", "FlucRadius", 10);
	FlucEqCoef.Alpha=reader.GetReal("mediumFluc", "FlucAlpha", EQ_ALPHA);
	FlucEqCoef.Beta=reader.GetReal("mediumFluc", "FlucBeta", EQ_BETA);
	FlucEqCoef.ne0=reader.GetReal("mediumFluc", "FlucNe0", M_NE0);

	Kamera.Draw2D=reader.GetInteger("KameraConf", "Draw2D", DRAW_2D_FIGURE);
	Kamera.FigPreset=reader.GetInteger("KameraConf", "2Dpreset", KAMERA_KONFIG);
	Kamera.PatLength=reader.GetInteger("KameraConf", "PatrubokL", PATRUBOK_L);
	Kamera.DrawWall=reader.GetInteger("KameraConf", "DrawWall", MAKEWALL);
	Kamera.WallPos=reader.GetInteger("KameraConf", "WallPos", WALLPOS);

	RCdetectCen.I=reader.GetInteger("RC_calc", "Iline", I_SOURCE);
	RCdetect.PI=reader.GetInteger("RC_calc", "LIsize", DD_SZ_I);
	RCdetect.PK=1;
	//RCdetect.deltaAuto=reader.GetInteger("RC_calc", "SAutoDelta", DD_AUTO);
	RCdetect.DI=reader.GetInteger("RC_calc", "LIdelta", DD_DELTA_I);
	RCdetect.DK=5;
	RCdetFilt.level = reader.GetReal("RC_calc", "Rlevel", 0.1);
	RCdetFilt.window = reader.GetInteger("RC_calc", "FilterWindow", 200);

	Processing.perform			= reader.GetBoolean("Processing", "Perform", false);
	Processing.sec_der			= reader.GetBoolean("Processing", "SecDer", false);
	Processing.cf				= reader.GetBoolean("Processing", "CF", false);
	Processing.arc				= reader.GetBoolean("Processing", "ARC", false);
	Processing.save_filtered	= reader.GetBoolean("Processing", "SaveFiltered", false);
	Processing.save_original	= reader.GetBoolean("Processing", "SaveOriginal", false);
	Processing.sim_detect		= reader.GetBoolean("Processing", "DetSim", false);
	Processing.save_detector	= reader.GetBoolean("Processing", "DetSave", false);
	Processing.load_weights		= reader.GetBoolean("Processing", "DetWLoad", false);
	Processing.detector_size	= reader.GetInteger("Processing", "DetSize", 30);
	Processing.filter_type		= reader.GetInteger("Processing", "FilterType", 1);
	Processing.dim_sieve		= reader.GetInteger("Processing", "DimSieve", 1);
	Processing.time_sieve		= reader.GetInteger("Processing", "TimeSieve", 1);
	Processing.begin_after		= reader.GetInteger("Processing", "BeginAfter", 0);
	Processing.finish_at		= reader.GetInteger("Processing", "FinishAt", 0);
	Processing.pos_i			= reader.GetInteger("Processing", "PosI", 0);
	Processing.pos_j			= reader.GetInteger("Processing", "PosJ", 0);
	Processing.pos_k			= reader.GetInteger("Processing", "PosK", 0);
	Processing.size_i			= reader.GetInteger("Processing", "SizeI", 0);
	Processing.filter_win		= reader.GetInteger("Processing", "FilterWin", 300);
	Processing.filter_runs		= reader.GetInteger("Processing", "FilterRuns", 1);
	Processing.sdf_win			= reader.GetInteger("Processing", "SDFWin", Processing.filter_win);
	Processing.sdf_type			= reader.GetInteger("Processing", "SDFType", Processing.filter_type);
	Processing.sdf_runs			= reader.GetInteger("Processing", "SDFRuns", Processing.filter_runs);
	Processing.cf_tau			= reader.GetInteger("Processing", "CFTau", 300);
	Processing.arc_tau			= reader.GetInteger("Processing", "ARCTau", 300);
	Processing.tau_threshold	= reader.GetReal("Processing", "TauThreshold", 0);
	Processing.arc_k			= reader.GetReal("Processing", "ARCK", -0.5);
	Processing.cf_k				= reader.GetReal("Processing", "CFK", -0.5);

	logINIread();
	return 0;
}

int logINIread() {
	char tmpTime[128], tmpDate[128];
    _strtime_s( tmpTime, 128 );
    _strdate_s( tmpDate, 128 );

    std::ofstream finiout("iniLOG.txt", std::ios::app); // создаём объект класса ofstream для записи
	if(!finiout) return -1;

	finiout <<	"Config loaded from 'Settings.ini' on time " << tmpTime << " date " << tmpDate << ":\nHardware:" <<
				"\n GPU_device=" << GPU_device << "\nRegimes: " <<
				"\n SourceToDetect=" << useRegime.S2D << "\nGrid param: " <<	
				"\n Use_Exp_time=" << Exp_time_key << 
				"\n Imax=" << Imax << 
				"\n Jmax=" << Jmax << 
				"\n Max_time=" << nMax << 
				"\n Exp_time=" << Exp_time << 
				"\n ds=" << ds << 
				"\n dt_coef=" << dt_coef << 
				"\n Isource=" << SourceCen.I << 
				"\n Jsource=" << SourceCen.J << 
				"\n Idetect=" << DetectCen.I << 
				"\n Jdetect=" << DetectCen.J << 
				"\nAESA\n SIsize=" << SourceAESA.PI << 
				"\n SAutoDelta=" << SourceAESA.deltaAuto << 
				"\n SIdelta=" << SourceAESA.DI << 
				"\n SPAngle=" << AESAsourceParam.Angle << 			  
				"\n DIsize=" << DetectAESA.PI << 
				"\n DAutoDelta=" << DetectAESA.deltaAuto <<
				"\n DIdelta=" << DetectAESA.DI << "\nPulseParam:" <<
				"\n Chirp=" << SourcePulse.chirp <<
				"\n FreqA=" << SourcePulse.freqA <<
				"\n FreqB=" << SourcePulse.freqB <<
				"\n Fronts=" << SourcePulse.Front <<
				"\n Plato=" << SourcePulse.Plato << 
				"\n Fronts_ns=" << Fronts_ns <<
				"\n Plato_ns=" << Plato_ns <<
				"\n Pulse_amp=" << amp << "\n2Dsave:" <<
				"\n From=" << FieldSaveParam.fromTime << 
				"\n To=" << FieldSaveParam.ToTime << 
				"\n TimeSieve=" << FieldSaveParam.TimeSive << 
				"\n GridSieve=" << FieldSaveParam.FieldSieve << "\nmedium:" <<
				"\n Build=" << Build_medium << 
				"\n Icen=" << MediumCen.I << 
				"\n Jcen=" << MediumCen.J << 
				"\n\n A=" << SphereCoef.A << 
				"\n B=" << SphereCoef.B << 
				"\n C=" << SphereCoef.C << 
				"\n P=" << SphereCoef.P << 
				"\n Q=" << SphereCoef.Q << 
				"\n N=" << SphereCoef.N << 

				"\n\n Radius=" << EqCoef.R << 
				"\n Alpha=" << EqCoef.Alpha << 
				"\n Beta=" << EqCoef.Beta << 
				"\n ne0=" << EqCoef.ne0 << 
				"\n neA=" << EqCoef.neA << "\nmediumFluc:"

				"\n FlucBuild=" << Build_fluc_medium << 
				"\n FlucIcen=" << MediumFlucCen.I << 
				"\n FlucJcen=" << MediumFlucCen.J << 
				"\n\n FlucA=" << SphereFlucCoef.A << 
				"\n FlucB=" << SphereFlucCoef.B << 
				"\n FlucC=" << SphereFlucCoef.C << 
				"\n FlucP=" << SphereFlucCoef.P << 
				"\n FlucQ=" << SphereFlucCoef.Q << 
				"\n FlucN=" << SphereFlucCoef.N << 

				"\n\n FlucAngleI=" << FlucAngle.I << 
				"\n FlucAngleJ=" << FlucAngle.J << 
				"\n FlucAngleK=" << FlucAngle.K << 
				"\n FlucSign=" << flucSign << 

				"\n\n FlucRadius=" << FlucEqCoef.R << 
				"\n FlucAlpha=" << FlucEqCoef.Alpha << 
				"\n FlucBeta=" << FlucEqCoef.Beta << 
				"\n FlucNe0=" << FlucEqCoef.ne0 << 

				"\n TorusBR=" << Torus_big_r << "\nKameraConf:" <<
				"\n Draw2D=" << Kamera.Draw2D <<
				"\n 2Dpreset=" << Kamera.FigPreset <<
				"\n PatrubokL=" << Kamera.PatLength <<
				"\n DrawWall=" << Kamera.DrawWall <<
				"\n WallPos=" << Kamera.WallPos <<

				"\n\nRC_calc:"<<
				"\n Iline="<< RCdetectCen.I <<
				"\n LIsize="<< RCdetect.PI <<
				"\n LIdelta="<< RCdetect.DI <<
				"\n FilterWindow="<< RCdetFilt.window <<
				"\n Rlevel="<< RCdetFilt.level <<

				"\n\nProcessing:" <<
				"\n Perform=" << Processing.perform <<
				"\n SecDer=" << Processing.sec_der <<
				"\n SDFType=" << Processing.sdf_type <<
				"\n SDFWin=" << Processing.sdf_win <<
				"\n SDFRuns=" << Processing.sdf_runs <<
				"\n CF=" << Processing.cf <<
				"\n CFTau=" << Processing.cf_tau <<
				"\n CFK=" << Processing.cf_k <<
				"\n ARC=" << Processing.arc <<
				"\n ARCTau=" << Processing.arc_tau <<
				"\n ARCK=" << Processing.arc_k <<
				"\n TauThreshold=" << Processing.tau_threshold <<
				"\n FilterType=" << Processing.filter_type <<
				"\n FilterWin=" << Processing.filter_win <<
				"\n FilterRuns=" << Processing.filter_runs <<
				"\n SaveFiltered=" << Processing.save_filtered <<
				"\n SaveOriginal=" << Processing.save_original <<
				"\n PosI=" << Processing.pos_i <<
				"\n PosJ=" << Processing.pos_j <<
				"\n PosK=" << Processing.pos_k <<
				"\n SizeI=" << Processing.size_i <<
				"\n BeginAfter=" << Processing.begin_after <<
				"\n FinishAt=" << Processing.finish_at <<
				"\n TimeSieve=" << Processing.time_sieve <<
				"\n DimSieve=" << Processing.dim_sieve <<
				"\n SimDetect=" << Processing.sim_detect <<
				"\n DetSave=" << Processing.save_detector <<
				"\n DetSize=" << Processing.detector_size <<
				"\n DetWLoad=" << Processing.load_weights <<
				"\n-----------------------------------------------------------------------------\n";	

	finiout.close();
	return 0;
}

void recuclParametersToMetric()
{
	dx = ds*1E-3; 
    dy = ds*1E-3;
    dz = ds*1E-3; // cell size in each direction

	Imax = Imax * (1/ds); 
	Jmax = Jmax * (1/ds);

	SourceCen.I = SourceCen.I * (1/ds);
	SourceCen.J = SourceCen.J * (1/ds);

	DetectCen.I = DetectCen.I * (1/ds);
	DetectCen.J = DetectCen.J * (1/ds);

	MediumCen.I = MediumCen.I * (1/ds);
	MediumCen.J = MediumCen.J * (1/ds);
	MediumCen.K=K_MEDIUM;

	EqCoef.R = EqCoef.R * (1/ds);

	//----source reculc----------
	SourceAESA.PI = SourceAESA.PI * (1/ds);
	//SourceAESA.PK=1;

	SourceAESA.DI = SourceAESA.DI * (1/ds);
	//SourceAESA.DK=5;


	//----detector reculc----------
	DetectAESA.PI = DetectAESA.PI * (1/ds);
	//DetectAESA.PK=1;

	DetectAESA.DI = DetectAESA.DI * (1/ds);
	//DetectAESA.DK=5;

	//----fluc reculc-------
	MediumFlucCen.I = MediumFlucCen.I * (1/ds);
	MediumFlucCen.J = MediumFlucCen.J * (1/ds);
	MediumFlucCen.K=K_MEDIUM;

	FlucEqCoef.R = FlucEqCoef.R * (1/ds);

	// --------------------time reculculation---------------------

	dt = (0.99 / (C * sqrt(1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz))))*dt_coef;
	//delay
	tO = 4.0 * tw;

	// ---------------------RC recalc ------------------------------

	RCdetectCen.I /= ds;
	RCdetect.PI /= ds;
	//RCdetect.PK 
	
	RCdetect.DI /= ds;
	
	// ---------------------Processing calculation------------------

	Processing.pos_i /= ds;
	Processing.pos_j /= ds;
	Processing.pos_k /= ds;
	Processing.size_i /= ds;

	if(Exp_time_key == 1)
	{
		nMax = (Exp_time*1e-9)/dt;
		SourcePulse.Front = (Fronts_ns*1e-9)/dt;
		SourcePulse.Plato = (Plato_ns*1e-9)/dt;
		
		Processing.begin_after *= (1e-9/dt);
		Processing.finish_at *= (1e-9/dt);
		
		//Processing.arc_tau *= (1e-9/dt);
		//Processing.cf_tau *= (1e-9/dt);
	}

}

void initlimits() {
	readINIparam();
	recuclParametersToMetric();
	cudaSetDevice(GPU_device);

	if(useRegime.S2D == ENABLE) {
		int freeSpace = 24*2; //128*2

		ObjectsEnable = false;
		//------------------------------------------------------------------
		printf("\n//-------DEBUG--------//\n");
		printf("Size (%d %d %d),\nScen (%d %d %d),\nDcen (%d %d %d)", Imax, Jmax, Kmax, SourceCen.I, SourceCen.J, SourceCen.K, DetectCen.I, DetectCen.J, DetectCen.K );
		printf("\n//-----END-DEBUG------//\n");

		//------Source to Detect influence calculation--------------------
		Order = 1;

		initialize();
		setUp();
		buildObject();
		buildSourceProfile();
		initializeCPML();
		DevSourceToDetectInfluenceCalc();

		//------main field calculation------------------------------------
		ObjectsEnable = true;

		Order = 2;

		initialize();
		setUp();
		buildObject();
		buildSourceProfile();
		initializeCPML();		
		DevFieldCalculation();
		//----------------------------------------------------------------
	} else {
		Order = 1;
		ObjectsEnable = true;

		Build_Stell_wall = Kamera.DrawWall;

		//SourceCen.I=I_SOURCE;
		//SourceCen.J=J_SOURCE;
		SourceCen.K=K_SOURCE;

		//DetectCen.I=I_DET;
		//DetectCen.J=J_DET;
		DetectCen.K=K_DET;

		initialize();
		setUp();
		buildObject();
		buildSourceProfile();
		initializeCPML();
		DevFieldCalculation();		
	}
}

void initialize() {
	runTime = clock();
		
	if(CALC_ARREY_POS==DISABLE)	{
		detect = (float *)malloc((nMax * DET_SIZE* DET_SIZE) * sizeof(float));

		if(Order == 1) {
			infDet = (float *)malloc((nMax * DET_SIZE* DET_SIZE) * sizeof(float));

			if(SAVE_COMPONENTS == ENABLE) {
				det_Ex = (float *)malloc((nMax * DET_SIZE* DET_SIZE) * sizeof(float));
				det_Ey = (float *)malloc((nMax * DET_SIZE* DET_SIZE) * sizeof(float));
				det_Ez = (float *)malloc((nMax * DET_SIZE* DET_SIZE) * sizeof(float));
				det_Hx = (float *)malloc((nMax * DET_SIZE* DET_SIZE) * sizeof(float));
				det_Hy = (float *)malloc((nMax * DET_SIZE* DET_SIZE) * sizeof(float));
				det_Hz = (float *)malloc((nMax * DET_SIZE* DET_SIZE) * sizeof(float));

				if(useRegime.S2D == ENABLE) {
					infDet_Ex = (float *)malloc((nMax * DET_SIZE* DET_SIZE) * sizeof(float));
					infDet_Ey = (float *)malloc((nMax * DET_SIZE* DET_SIZE) * sizeof(float));
					infDet_Ez = (float *)malloc((nMax * DET_SIZE* DET_SIZE) * sizeof(float));
					infDet_Hx = (float *)malloc((nMax * DET_SIZE* DET_SIZE) * sizeof(float));
					infDet_Hy = (float *)malloc((nMax * DET_SIZE* DET_SIZE) * sizeof(float));
					infDet_Hz = (float *)malloc((nMax * DET_SIZE* DET_SIZE) * sizeof(float));
				}
			}
		}
	}

	if (Processing.perform) {
		//proc_line = new float[int(Processing.size_i / Processing.dim_sieve)];
		proc_line = new float[Processing.size_i];
		/* ******************** */
		//int temp_field_size = int((nMax - Processing.begin_after) / Processing.time_sieve);
		int temp_field_size = (Processing.finish_at > nMax ? nMax : Processing.finish_at) - Processing.begin_after;
		proc_field = new float*[temp_field_size];
		for (int i = 0; i < temp_field_size; i++) {
			//proc_field[i] = new float[int(Processing.size_i / Processing.dim_sieve)];
			proc_field[i] = new float[Processing.size_i];
		}
	}

	if(LINE_D_ENABLE == ENABLE) {
		line_detect = (float *)malloc((nMax * LINE_DET_A_QUANT * LINE_DET_A_SIZE * LINE_DET_A_SIZE)  * sizeof(float));
	}

	if(LINE_D2_ENABLE == ENABLE) {
		line_detect2 = (float *)malloc((nMax * LINE_DET_B_QUANT * LINE_DET_B_SIZE * LINE_DET_B_SIZE)  * sizeof(float));
	}

	line_medium = (float *)malloc((nMax * (Jmax-2))  * sizeof(float));
		
//***********************************************************************
//     Save parameters
//***********************************************************************

	mag = MAG; //множитель (домнажает сохраняемый результат при сохранении)

//***********************************************************************

	muO = 4.0 * pi * 1.0E-7;
	epsO = 1.0 / (C * C * muO);
		 
	//PML Layers (10 layers)
	nxPML_1 = PML_SIZE;
	nxPML_2 = PML_SIZE;
	nyPML_1 = PML_SIZE;
	nyPML_2 = PML_SIZE;
	nzPML_1 = PML_SIZE;
	nzPML_2 = PML_SIZE;

	//Dynamic memory allocation

	epsilon = (double *)malloc((numMaterials) * sizeof(double));
	for(i = 0; i < numMaterials; i++) {
		epsilon[i] = epsO;
	}

	mu = (double *)malloc((numMaterials) * sizeof(double));
	for(i = 0; i < numMaterials; i++) {
		mu[i] = muO;
	}

	sigma = (double *)malloc((numMaterials) * sizeof(double));
	for(i = 0; i < numMaterials; i++) {
		sigma[i] = 0.0;
	}

	Ro = (double *)malloc((numMaterials) * sizeof(double));
	for(i = 0; i < numMaterials; i++) {
		Ro[i] = 0.0;
	}

	CA = (float *)malloc((numMaterials) * sizeof(float));
	for(i = 0; i < numMaterials; i++) {
		CA[i] = 0.0;
	}

	CB = (float *)malloc((numMaterials) * sizeof(float));
	for(i = 0; i < numMaterials; i++) {
		CB[i] = 0.0;
	}

	DA = (float *)malloc((numMaterials) * sizeof(float));
	for(i = 0; i < numMaterials; i++) {
		DA[i] = 0.0;
	}

	DB = (float *)malloc((numMaterials) * sizeof(float));
	for(i = 0; i < numMaterials; i++) {
		DB[i] = 0.0;
	}

	pne = (float *)malloc((numMaterials) * sizeof(float));
	for(i = 0; i < numMaterials; i++) {
		pne[i] = 0.0;
	}

	//(Imax * Jmax * Kmax)
	Jz = (float *)malloc((Imax * Jmax * Kmax) * sizeof(float));
	for(i = 0; i < (Imax * Jmax * Kmax); i++) {
		Jz[i] = 0.0;
	}

	//(Imax * Jmax-1 * Kmax-1)
	Jy = (float *)malloc((Imax * (Jmax-1) * (Kmax-1)) * sizeof(float));
	for(i = 0; i < (Imax * (Jmax-1) * (Kmax-1)); i++) {
		Jy[i] = 0.0;
	}

	//(Imax-1 * Jmax * Kmax-1)
	Jx = (float *)malloc(((Imax-1) * Jmax * (Kmax-1)) * sizeof(float));
	for(i = 0; i < ((Imax-1) * Jmax * (Kmax-1)); i++) {
		Jx[i] = 0.0;
	}

	//(Imax * Jmax * Kmax)
	Ez = (float *)malloc((Imax * Jmax * Kmax) * sizeof(float));
	for(i = 0; i < (Imax * Jmax * Kmax); i++) {
		Ez[i] = 0.0;
	}

	//(Imax * Jmax-1 * Kmax-1)
	Ey = (float *)malloc((Imax * (Jmax-1) * (Kmax-1)) * sizeof(float));
	for(i = 0; i < (Imax * (Jmax-1) * (Kmax-1)); i++) {
		Ey[i] = 0.0;
	}

	//(Imax-1 * Jmax * Kmax-1)
	Ex = (float *)malloc(((Imax-1) * Jmax * (Kmax-1)) * sizeof(float));
	for(i = 0; i < ((Imax-1) * Jmax * (Kmax-1)); i++) {
		Ex[i] = 0.0;
	}

	//(Imax * Jmax-1 * Kmax)
	Hx = (float *)malloc((Imax * (Jmax-1) * Kmax) * sizeof(float));
	for(i = 0; i < (Imax * (Jmax-1) * Kmax); i++) {
		Hx[i] = 0.0;
	}

	//(Imax-1 * Jmax * Kmax)
	Hy = (float *)malloc(((Imax-1) * Jmax * Kmax) * sizeof(float));
	for(i = 0; i < ((Imax-1) * Jmax * Kmax); i++) {
		Hy[i] = 0.0;
	}

	//(Imax-1 * Jmax-1 * Kmax-1)
	Hz = (float *)malloc(((Imax-1) * (Jmax-1) * (Kmax-1)) * sizeof(float));
	for(i = 0; i < ((Imax-1) * (Jmax-1) * (Kmax-1)); i++) {
		Hz[i] = 0.0;
	}

//----------------------------ID for closing antenna------------------------------
	//ID1_close = (int *)malloc((Imax * Jmax * Kmax) * sizeof(int));
	//ID2_close = (int *)malloc((Imax * Jmax * Kmax) * sizeof(int));
	//ID3_close = (int *)malloc((Imax * Jmax * Kmax) * sizeof(int));

	ID1_close = (int *)malloc((1) * sizeof(int));
	ID2_close = (int *)malloc((1) * sizeof(int));
	ID3_close = (int *)malloc((1) * sizeof(int));
//---------------------------------------------------------------------------------

	//(Imax * Jmax * Kmax)
	ID1 = (int *)malloc((Imax * Jmax * Kmax) * sizeof(int));
	for(i = 0; i < (Imax * Jmax * Kmax); i++) {
		ID1[i] = 0;
		//ID1_close[i]=0;
	}

	//(Imax * Jmax * Kmax)
	ID2 = (int *)malloc((Imax * Jmax * Kmax) * sizeof(int));
	for(i = 0; i < (Imax * Jmax * Kmax); i++) {
		ID2[i] = 0;
		//ID2_close[i]=0;
	}

	//(Imax * Jmax * Kmax)
	ID3 = (int *)malloc((Imax * Jmax * Kmax) * sizeof(int));
	for(i = 0; i < (Imax * Jmax * Kmax); i++) {
		ID3[i] = 0;
		//ID3_close[i]=0;
	}
//----------------------------ID for closing antenna------------------------------ 

	//(nxPML_1 * Jmax * Kmax)
	psi_Ezx_1 = (float *)malloc((nxPML_1 * Jmax * Kmax) * sizeof(float));
	for(i = 0; i < (nxPML_1 * Jmax * Kmax); i++) {
		psi_Ezx_1[i] = 0.0;
	}

	//(nxPML_2 * Jmax * Kmax)
	psi_Ezx_2 = (float *)malloc((nxPML_2 * Jmax * Kmax) * sizeof(float));
	for(i = 0; i < (nxPML_2 * Jmax * Kmax); i++) {
		psi_Ezx_2[i] = 0.0;
	}

	//(nxPML_1-1 * Jmax * Kmax)
	psi_Hyx_1 = (float *)malloc(((nxPML_1-1) * Jmax * Kmax) * sizeof(float));
	for(i = 0; i < ((nxPML_1-1) * Jmax * Kmax); i++) {
		psi_Hyx_1[i] = 0.0;
	}

	//(nxPML_2-1 * Jmax * Kmax)
	psi_Hyx_2 = (float *)malloc(((nxPML_2-1) * Jmax * Kmax) * sizeof(float));
	for(i = 0; i < ((nxPML_2-1) * Jmax * Kmax); i++) {		    				
		psi_Hyx_2[i] = 0.0;
	}

	//(Imax * nyPML_1 * Kmax)
	psi_Ezy_1 = (float *)malloc((Imax * nyPML_1 * Kmax) * sizeof(float));
	for(i = 0; i < (Imax * nyPML_1 * Kmax); i++) {
		psi_Ezy_1[i] = 0.0;
	}

	//(Imax * nyPML_2 * Kmax)
	psi_Ezy_2 = (float *)malloc((Imax * nyPML_2 * Kmax) * sizeof(float));
	for(i = 0; i < (Imax * nyPML_2 * Kmax); i++) {
		psi_Ezy_2[i] = 0.0;
	}

	//(Imax * nyPML_1-1 * Kmax)
	psi_Hxy_1 = (float *)malloc((Imax * (nyPML_1-1) * Kmax) * sizeof(float));
	for(i = 0; i < (Imax * (nyPML_1-1) * Kmax); i++) {
		psi_Hxy_1[i] = 0.0;
	}

	//(Imax * nyPML_2-1 * Kmax)
	psi_Hxy_2 = (float *)malloc((Imax * (nyPML_2-1) * Kmax) * sizeof(float));
	for(i = 0; i < (Imax * (nyPML_2-1) * Kmax); i++) {
		psi_Hxy_2[i] = 0.0;
	}

	//(Imax * Jmax-1 * nzPML_1-1)
	psi_Hxz_1 = (float *)malloc((Imax * (Jmax-1) * (nzPML_1-1)) * sizeof(float));
	for(i = 0; i < (Imax * (Jmax-1) * (nzPML_1-1)); i++) {
		psi_Hxz_1[i] = 0.0;
	}

	//(Imax * Jmax-1 * nzPML_2-1) ???(Jmax)
	psi_Hxz_2 = (float *)malloc((Imax * (Jmax-1) * (nzPML_2-1)) * sizeof(float));
	for(i = 0; i < (Imax * (Jmax-1) * (nzPML_2-1)); i++) {
		psi_Hxz_2[i] = 0.0;
	}

	//(Imax-1 * Jmax * nzPML_1-1)
	psi_Hyz_1 = (float *)malloc(((Imax-1) * Jmax * (nzPML_1-1)) * sizeof(float));
	for(i = 0; i < ((Imax-1) * Jmax * (nzPML_1-1)); i++) {
		psi_Hyz_1[i] = 0.0;
	}

	//(Imax-1 * Jmax * nzPML_2-1)
	psi_Hyz_2 = (float *)malloc(((Imax-1) * Jmax * (nzPML_2-1)) * sizeof(float));
	for(i = 0; i < ((Imax-1) * Jmax * (nzPML_2-1)); i++) {
		psi_Hyz_2[i] = 0.0;
	}

	//(Imax-1 * Jmax * nzPML_1)
	psi_Exz_1 = (float *)malloc(((Imax-1) * Jmax * nzPML_1) * sizeof(float));
	for(i = 0; i < ((Imax-1) * Jmax * nzPML_1); i++) {
		psi_Exz_1[i] = 0.0;
	}

	//(Imax-1 * Jmax * nzPML_2)
	psi_Exz_2 = (float *)malloc(((Imax-1) * Jmax * nzPML_2) * sizeof(float));
	for(i = 0; i < ((Imax-1) * Jmax * nzPML_2); i++) {
		psi_Exz_2[i] = 0.0;
	}

	//(Imax-1 * Jmax-1 * nzPML_1) ???(Imax)
	psi_Eyz_1 = (float *)malloc(((Imax-1) * (Jmax-1) * nzPML_1) * sizeof(float));
	for(i = 0; i < ((Imax-1) * (Jmax-1) * nzPML_1); i++) {
		psi_Eyz_1[i] = 0.0;
	}

	//(Imax-1 * Jmax-1 * nzPML_2)
	psi_Eyz_2 = (float *)malloc(((Imax-1) * (Jmax-1) * nzPML_2) * sizeof(float));
	for(i = 0; i < ((Imax-1) * (Jmax-1) * nzPML_2); i++){
		psi_Eyz_2[i] = 0.0;
	}

	//(nxPML_1-1 * Jmax-1 * Kmax-1)
	psi_Hzx_1 = (float *)malloc(((nxPML_1-1) * (Jmax-1) * (Kmax-1)) * sizeof(float));
	for(i = 0; i < ((nxPML_1-1) * (Jmax-1) * (Kmax-1)); i++) {				
		psi_Hzx_1[i] = 0.0;
	}

	//(nxPML_2-1 * Jmax-1 * Kmax-1)
	psi_Hzx_2 = (float *)malloc(((nxPML_2-1) * (Jmax-1) * (Kmax-1)) * sizeof(float));
	for(i = 0; i < ((nxPML_2-1) * (Jmax-1) * (Kmax-1)); i++) {
		psi_Hzx_2[i] = 0.0;
	}

	//(nxPML_1 * Jmax-1 * Kmax-1)
	psi_Eyx_1 = (float *)malloc((nxPML_1 * (Jmax-1) * (Kmax-1)) * sizeof(float));
	for(i = 0; i < (nxPML_1 * (Jmax-1) * (Kmax-1)); i++) {
		psi_Eyx_1[i] = 0.0;
	}

	//(nxPML_2 * Jmax-1 * Kmax-1)
	psi_Eyx_2 = (float *)malloc((nxPML_2 * (Jmax-1) * (Kmax-1)) * sizeof(float));
	for(i = 0; i < (nxPML_2 * (Jmax-1) * (Kmax-1)); i++) {
		psi_Eyx_2[i] = 0.0;
	}

	//(Imax-1 * nyPML_2-1 * Kmax-1)
	psi_Hzy_1 = (float *)malloc(((Imax-1) * (nyPML_1-1) * (Kmax-1)) * sizeof(float));
	for(i = 0; i < ((Imax-1) * (nyPML_1-1) * (Kmax-1)); i++) {
		psi_Hzy_1[i] = 0.0;
	}

	//(Imax-1 * nyPML_2-1 * Kmax-1)
	psi_Hzy_2 = (float *)malloc(((Imax-1) * (nyPML_2-1) * (Kmax-1)) * sizeof(float));
	for(i = 0; i < ((Imax-1) * (nyPML_2-1) * (Kmax-1)); i++) {
		psi_Hzy_2[i] = 0.0;
	}

	//(Imax-1 * nyPML_1 * Kmax-1)
	psi_Exy_1 = (float *)malloc(((Imax-1) * nyPML_1 * (Kmax-1)) * sizeof(float));
	for(i = 0; i < ((Imax-1) * nyPML_1 * (Kmax-1)); i++) {
		psi_Exy_1[i] = 0.0;
	}

	//(Imax-1 * nyPML_2 * Kmax-1)
	psi_Exy_2 = (float *)malloc(((Imax-1) * nyPML_2 * (Kmax-1)) * sizeof(float));
	for(i = 0; i < ((Imax-1) * nyPML_2 * (Kmax-1)); i++) {
		psi_Exy_2[i] = 0.0;
	}

	den_ex = (float *)malloc((Imax-1) * sizeof(float));
	for(i = 0; i < Imax-1; i++) {
		den_ex[i] = 0.0;
	}

	den_hx = (float *)malloc((Imax-1) * sizeof(float));
	for(i = 0; i < Imax-1; i++) {
		den_hx[i] = 0.0;
	}

	den_ey = (float *)malloc((Jmax-1) * sizeof(float));
	for(i = 0; i < Jmax-1; i++) {
		den_ey[i] = 0.0;
	}

	den_hy = (float *)malloc((Jmax-1) * sizeof(float));
	for(i = 0; i < Jmax-1; i++) {
		den_hy[i] = 0.0;
	}

	den_ez = (float *)malloc((Kmax-1) * sizeof(float));
	for(i = 0; i < Kmax-1; i++) {
		den_ez[i] = 0.0;
	}

	den_hz = (float *)malloc((Kmax-1) * sizeof(float));
	for(i = 0; i < Kmax-1; i++) {
		den_hz[i] = 0.0;
	}

	be_x_1 = (float *)malloc((nxPML_1) * sizeof(float));
	for(i = 0; i < nxPML_1; i++) {
		be_x_1[i] = 0.0;
	}

	ce_x_1 = (float *)malloc((nxPML_1) * sizeof(float));
	for(i = 0; i < nxPML_1; i++) {
		ce_x_1[i] = 0.0;
	}

	alphae_x_PML_1 = (float *)malloc((nxPML_1) * sizeof(float));
	for(i = 0; i < nxPML_1; i++) {
		alphae_x_PML_1[i] = 0.0;
	}

	sige_x_PML_1 = (float *)malloc((nxPML_1) * sizeof(float));
	for(i = 0; i < nxPML_1; i++) {
		sige_x_PML_1[i] = 0.0;
	}

	kappae_x_PML_1 = (float *)malloc((nxPML_1) * sizeof(float));
	for(i = 0; i < nxPML_1; i++) {
		kappae_x_PML_1[i] = 0.0;
	}

	bh_x_1 = (float *)malloc((nxPML_1-1) * sizeof(float));
	for(i = 0; i < nxPML_1-1; i++) {
		bh_x_1[i] = 0.0;
	}

	ch_x_1 = (float *)malloc((nxPML_1-1) * sizeof(float));
	for(i = 0; i < nxPML_1-1; i++) {
		ch_x_1[i] = 0.0;
	}

	alphah_x_PML_1 = (float *)malloc((nxPML_1-1) * sizeof(float));
	for(i = 0; i < nxPML_1-1; i++) {
		alphah_x_PML_1[i] = 0.0;
	}

	sigh_x_PML_1 = (float *)malloc((nxPML_1-1) * sizeof(float));
	for(i = 0; i < nxPML_1-1; i++) {
		sigh_x_PML_1[i] = 0.0;
	}

	kappah_x_PML_1 = (float *)malloc((nxPML_1-1) * sizeof(float));
	for(i = 0; i < nxPML_1-1; i++) {
		kappah_x_PML_1[i] = 0.0;
	}

	be_x_2 = (float *)malloc((nxPML_2) * sizeof(float));
	for(i = 0; i < nxPML_2; i++) {
		be_x_2[i] = 0.0;
	}

	ce_x_2 = (float *)malloc((nxPML_2) * sizeof(float));
	for(i = 0; i < nxPML_2; i++) {
		ce_x_2[i] = 0.0;
	}

	alphae_x_PML_2 = (float *)malloc((nxPML_2) * sizeof(float));
	for(i = 0; i < nxPML_2; i++) {
		alphae_x_PML_2[i] = 0.0;
	}
	
	sige_x_PML_2 = (float *)malloc((nxPML_2) * sizeof(float));
	for(i = 0; i < nxPML_2; i++) {
		sige_x_PML_2[i] = 0.0;
	}

	kappae_x_PML_2 = (float *)malloc((nxPML_2) * sizeof(float));
	for(i = 0; i < nxPML_2; i++) {
		kappae_x_PML_2[i] = 0.0;
	}

	bh_x_2 = (float *)malloc((nxPML_2-1) * sizeof(float));
	for(i = 0; i < nxPML_2-1; i++) {
		bh_x_2[i] = 0.0;
	}

	ch_x_2 = (float *)malloc((nxPML_2-1) * sizeof(float));
	for(i = 0; i < nxPML_2-1; i++) {
		ch_x_2[i] = 0.0;
	}

	alphah_x_PML_2 = (float *)malloc((nxPML_2-1) * sizeof(float));
	for(i = 0; i < nxPML_2-1; i++) {
		alphah_x_PML_2[i] = 0.0;
	}

	sigh_x_PML_2 = (float *)malloc((nxPML_2-1) * sizeof(float));
	for(i = 0; i < nxPML_2-1; i++) {
		sigh_x_PML_2[i] = 0.0;
	}

	kappah_x_PML_2 = (float *)malloc((nxPML_2-1) * sizeof(float));
	for(i = 0; i < nxPML_1-1; i++) {
		kappah_x_PML_2[i] = 0.0;
	}

	be_y_1 = (float *)malloc((nyPML_1) * sizeof(float));
	for(i = 0; i < nyPML_1; i++) {
		be_y_1[i] = 0.0;
	}

	ce_y_1 = (float *)malloc((nyPML_1) * sizeof(float));
	for(i = 0; i < nyPML_1; i++) {
		ce_y_1[i] = 0.0;
	}

	alphae_y_PML_1 = (float *)malloc((nyPML_1) * sizeof(float));
	for(i = 0; i < nyPML_1; i++) {
		alphae_y_PML_1[i] = 0.0;
	}

	sige_y_PML_1 = (float *)malloc((nyPML_1) * sizeof(float));
	for(i = 0; i < nyPML_1; i++) {
		sige_y_PML_1[i] = 0.0;
	}

	kappae_y_PML_1 = (float *)malloc((nyPML_1) * sizeof(float));
	for(i = 0; i < nyPML_1; i++) {
		kappae_y_PML_1[i] = 0.0;
	}

	bh_y_1 = (float *)malloc((nyPML_1-1) * sizeof(float));
	for(i = 0; i < nyPML_1-1; i++) {
		bh_y_1[i] = 0.0;
	}

	ch_y_1 = (float *)malloc((nyPML_1-1) * sizeof(float));
	for(i = 0; i < nyPML_1-1; i++) {
		ch_y_1[i] = 0.0;
	}

	alphah_y_PML_1 = (float *)malloc((nyPML_1-1) * sizeof(float));
	for(i = 0; i < nyPML_1-1; i++) {
		alphah_y_PML_1[i] = 0.0;
	}

	sigh_y_PML_1 = (float *)malloc((nyPML_1-1) * sizeof(float));
	for(i = 0; i < nyPML_1-1; i++) {
		sigh_y_PML_1[i] = 0.0;
	}

	kappah_y_PML_1 = (float *)malloc((nyPML_1-1) * sizeof(float));
	for(i = 0; i < nyPML_1-1; i++) {
		kappah_y_PML_1[i] = 0.0;
	}

	be_y_2 = (float *)malloc((nyPML_2) * sizeof(float));
	for(i = 0; i < nyPML_2; i++) {
		be_y_2[i] = 0.0;
	}

	ce_y_2 = (float *)malloc((nyPML_2) * sizeof(float));
	for(i = 0; i < nyPML_2; i++) {
		ce_y_2[i] = 0.0;
	}

	alphae_y_PML_2 = (float *)malloc((nyPML_2) * sizeof(float));
	for(i = 0; i < nyPML_2; i++) {
		alphae_y_PML_2[i] = 0.0;
	}

	sige_y_PML_2 = (float *)malloc((nyPML_2) * sizeof(float));
	for(i = 0; i < nyPML_2; i++) {
		sige_y_PML_2[i] = 0.0;
	}

	kappae_y_PML_2 = (float *)malloc((nyPML_2) * sizeof(float));
	for(i = 0; i < nyPML_2; i++) {
		kappae_y_PML_2[i] = 0.0;
	}

	bh_y_2 = (float *)malloc((nyPML_2-1) * sizeof(float));
	for(i = 0; i < nyPML_2-1; i++) {
		bh_y_2[i] = 0.0;
	}

	ch_y_2 = (float *)malloc((nyPML_2-1) * sizeof(float));
	for(i = 0; i < nyPML_2-1; i++) {
		ch_y_2[i] = 0.0;
	}

	alphah_y_PML_2 = (float *)malloc((nyPML_2-1) * sizeof(float));
	for(i = 0; i < nyPML_2-1; i++) {
		alphah_y_PML_2[i] = 0.0;
	}

	sigh_y_PML_2 = (float *)malloc((nyPML_2-1) * sizeof(float));
	for(i = 0; i < nyPML_2-1; i++) {
		sigh_y_PML_2[i] = 0.0;
	}

	kappah_y_PML_2 = (float *)malloc((nyPML_2-1) * sizeof(float));
	for(i = 0; i < nyPML_1-1; i++) {
		kappah_y_PML_2[i] = 0.0;
	}

	be_z_1 = (float *)malloc((nzPML_1) * sizeof(float));
	for(i = 0; i < nzPML_1; i++) {
		be_z_1[i] = 0.0;
	}

	ce_z_1 = (float *)malloc((nzPML_1) * sizeof(float));
	for(i = 0; i < nzPML_1; i++) {
		ce_z_1[i] = 0.0;
	}

	alphae_z_PML_1 = (float *)malloc((nzPML_1) * sizeof(float));
	for(i = 0; i < nzPML_1; i++) {
		alphae_z_PML_1[i] = 0.0;
	}

	sige_z_PML_1 = (float *)malloc((nzPML_1) * sizeof(float));
	for(i = 0; i < nzPML_1; i++) {
		sige_z_PML_1[i] = 0.0;
	}

	kappae_z_PML_1 = (float *)malloc((nzPML_1) * sizeof(float));
	for(i = 0; i < nzPML_1; i++) {
		kappae_z_PML_1[i] = 0.0;
	}

	bh_z_1 = (float *)malloc((nzPML_1-1) * sizeof(float));
	for(i = 0; i < nzPML_1-1; i++) {
		bh_z_1[i] = 0.0;
	}

	ch_z_1 = (float *)malloc((nzPML_1-1) * sizeof(float));
	for(i = 0; i < nzPML_1-1; i++) {
		ch_z_1[i] = 0.0;
	}

	alphah_z_PML_1 = (float *)malloc((nzPML_1-1) * sizeof(float));
	for(i = 0; i < nzPML_1-1; i++) {
		alphah_z_PML_1[i] = 0.0;
	}

	sigh_z_PML_1 = (float *)malloc((nzPML_1-1) * sizeof(float));
	for(i = 0; i < nzPML_1-1; i++) {
		sigh_z_PML_1[i] = 0.0;
	}

	kappah_z_PML_1 = (float *)malloc((nzPML_1-1) * sizeof(float));
	for(i = 0; i < nzPML_1-1; i++) {
		kappah_z_PML_1[i] = 0.0;
	}

	be_z_2 = (float *)malloc((nzPML_2) * sizeof(float));
	for(i = 0; i < nzPML_2; i++) {
		be_z_2[i] = 0.0;
	}

	ce_z_2 = (float *)malloc((nzPML_2) * sizeof(float));
	for(i = 0; i < nzPML_2; i++) {
		ce_z_2[i] = 0.0;
	}

	alphae_z_PML_2 = (float *)malloc((nzPML_2) * sizeof(float));
	for(i = 0; i < nzPML_2; i++) {
		alphae_z_PML_2[i] = 0.0;
	}

	sige_z_PML_2 = (float *)malloc((nzPML_2) * sizeof(float));
	for(i = 0; i < nzPML_2; i++) {
		sige_z_PML_2[i] = 0.0;
	}

	kappae_z_PML_2 = (float *)malloc((nzPML_2) * sizeof(float));
	for(i = 0; i < nzPML_2; i++) {
		kappae_z_PML_2[i] = 0.0;
	}

	bh_z_2 = (float *)malloc((nzPML_2-1) * sizeof(float));
	for(i = 0; i < nzPML_2-1; i++) {
		bh_z_2[i] = 0.0;
	}

	ch_z_2 = (float *)malloc((nzPML_2-1) * sizeof(float));
	for(i = 0; i < nzPML_2-1; i++) {
		ch_z_2[i] = 0.0;
	}

	alphah_z_PML_2 = (float *)malloc((nzPML_2-1) * sizeof(float));
	for(i = 0; i < nzPML_2-1; i++) {
		alphah_z_PML_2[i] = 0.0;
	}

	sigh_z_PML_2 = (float *)malloc((nzPML_2-1) * sizeof(float));
	for(i = 0; i < nzPML_2-1; i++) {
		sigh_z_PML_2[i] = 0.0;
	}

	kappah_z_PML_2 = (float *)malloc((nzPML_2-1) * sizeof(float));
	for(i = 0; i < nzPML_1-1; i++) {
		kappah_z_PML_2[i] = 0.0;
	}
}

void setUp() {
	float ne0, nea;

	ne0=EqCoef.ne0;
	nea=EqCoef.neA;

	//Time step
	//dt = 0.99 / (C * sqrt(1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz)));



	//Specify the dipole size 
	//istart = 24;
	//iend = 26;
	//jstart = 55;
	//jend = 71;
	//kstart = 11;
	//kend = 13;

	//Material properties
	//Location '0' is for free space and '1' is for PEC
	epsilon[2] = 4.0 * epsO;
	sigma[2] = 0.005;
	epsilon[3] = 8.0 * epsO;
	sigma[3] = 3.96E7;// aluminum
	epsilon[4] = 9.5 * epsO;
	sigma[4] = 5.76E7;//copper
	epsilon[5] = 9.0 * epsO;
	sigma[5] = 2e6;//steel
	epsilon[6] = 2.1 * epsO;
	sigma[6] = 7.8e-4;//teflon
	epsilon[7] = 81 * epsO;
	sigma[7] = 1e-2;//water

	//------------Computing other materials------------------>>>>>>>>>>>>>>>
	float stepen;
	for (i = 8; i < numMaterials; i++) {
		//epsilon[i]=epsO*(1.0+((i-8)*1e-12));
		//mu[i]=muO*(1.0+((i-8)*1e-12));

		//epsilon[i]=epsO+(((i-8)*4e-12));
		//epsilon[i]=epsO;

		//epsilon[i] = 9.0 * epsO;
		//sigma[i] = 2e6;//steel

		//pne[i]=1E18+((i*0.025)*1E18);
		//pne[i]=((i*0.025)*1E18)*2;

		pne[i]=(((i-8)*0.01))*(ne0-nea)+nea;

		//mu[i]=muO+(((i-8)*1e-14));
				
		//epsilon[i]=epsO+(exp((i-8.)/5.)*(1e-14));
	}

	for(i = 8; i < numMaterials; i++) {
		//stepen=i/10;
		//sigma[i]=1*pow(10,stepen);
		//sigma[i]=((float)i/1000);
	}

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	//  COMPUTING FIELD UPDATE EQUATION COEFFICIENTS
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	//DA = 1.0;
	//DB = dt / muO;

	for (i = 0; i < numMaterials; ++i) {
		CA[i] = (1.0 - sigma[i] * dt / (2.0 * epsilon[i])) /
				(1.0 + sigma[i] * dt / (2.0 * epsilon[i]));
		CB[i] = (dt / (epsilon[i])) /
				(1.0 + sigma[i] * dt / (2.0 * epsilon[i]));

		DA[i]= (1.0 - Ro[i] * dt / (2.0 * mu[i])) / 
				(1.0 + Ro[i] * dt / (2.0 * mu[i]));
		DB[i] = (dt / (mu[i])) /
				(1.0 + Ro[i] * dt / (2.0 * mu[i]));
	}

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	//  PML parameters
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	sig_x_max = 0.75 * (0.8 * (m + 1) / (dx * sqrt(muO / (epsO * epsR))));
	sig_y_max = 0.75 * (0.8 * (m + 1) / (dy * sqrt(muO / (epsO * epsR))));
	sig_z_max = 0.75 * (0.8 * (m + 1) / (dz * sqrt(muO / (epsO * epsR))));
	alpha_x_max = 0.24;
	alpha_y_max = alpha_x_max;
	alpha_z_max = alpha_x_max;
	kappa_x_max = 15.0;
	kappa_y_max = kappa_x_max;
	kappa_z_max = kappa_x_max;
	printf("\nTIme step = %e", dt);
	printf("\n Number of steps = %d", nMax);
	printf("\n Total Simulation time = %e Seconds\n", nMax * dt);
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SET CPML PARAMETERS IN EACH DIRECTION
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

void initializeCPML() {

		for(i = 0; i < nxPML_1; ++i) {

		sige_x_PML_1[i] = sig_x_max * pow(((nxPML_1 - 1 - i)
							/ (nxPML_1 - 1.0)), m);
		alphae_x_PML_1[i] = alpha_x_max * pow(((float)i
			 							/ (nxPML_1 - 1.0)), ma);
		kappae_x_PML_1[i] = 1.0 + (kappa_x_max - 1.0)*
							pow((nxPML_1 - 1- i) / (nxPML_1 - 1.0), m);
		be_x_1[i] = exp(-(sige_x_PML_1[i] / kappae_x_PML_1[i] +
									alphae_x_PML_1[i]) * dt / epsO);

		if ((sige_x_PML_1[i] == 0.0) &&
			(alphae_x_PML_1[i] == 0.0) && (i == nxPML_1 - 1)) {

			ce_x_1[i] = 0.0;

		} else {

			ce_x_1[i] = sige_x_PML_1[i] * (be_x_1[i] - 1.0) /
				(sige_x_PML_1[i] + kappae_x_PML_1[i] * alphae_x_PML_1[i])
				/ kappae_x_PML_1[i];
		}
		}

	for(i = 0; i < nxPML_1 - 1; ++i) {

		sigh_x_PML_1[i] = sig_x_max * pow(((nxPML_1 - 1 - i - 0.5)
		  							/ (nxPML_1-1.0)), m);
		alphah_x_PML_1[i] = alpha_x_max * pow(((i + 1 -0.5)
		  								/ (nxPML_1-1.0)), ma);
		kappah_x_PML_1[i] = 1.0 + (kappa_x_max - 1.0) *
				pow(((nxPML_1 - 1 - i - 0.5) / (nxPML_1 - 1.0)), m);
		bh_x_1[i] = exp(-(sigh_x_PML_1[i] / kappah_x_PML_1[i] +
									alphah_x_PML_1[i]) * dt / epsO);
		ch_x_1[i] = sigh_x_PML_1[i] * (bh_x_1[i] - 1.0) /
					(sigh_x_PML_1[i] + kappah_x_PML_1[i] * alphah_x_PML_1[i])
					/ kappah_x_PML_1[i];
	}

	for(i = 0; i < nxPML_2; ++i) {

		sige_x_PML_2[i] = sig_x_max * pow(((nxPML_2 - 1 - i)
							        / (nxPML_2 - 1.0)), m);
		alphae_x_PML_2[i] = alpha_x_max * pow(((float)i
										/ (nxPML_2 - 1.0)), ma);
		kappae_x_PML_2[i] = 1.0 + (kappa_x_max - 1.0)*
						pow((nxPML_2 - 1- i) / (nxPML_2 - 1.0), m);
		be_x_2[i] = exp(-(sige_x_PML_2[i] / kappae_x_PML_2[i] +
									alphae_x_PML_2[i]) * dt / epsO);

		if ((sige_x_PML_2[i] == 0.0) &&
			(alphae_x_PML_2[i] == 0.0) && (i == nxPML_2 - 1)) {

			ce_x_2[i] = 0.0;

		} else {

			ce_x_2[i] = sige_x_PML_2[i] * (be_x_2[i] - 1.0) /
				(sige_x_PML_2[i] + kappae_x_PML_2[i] * alphae_x_PML_2[i])
				/ kappae_x_PML_2[i];
		}

	}

	for(i = 0; i < nxPML_2 - 1; ++i) {

		sigh_x_PML_2[i] = sig_x_max * pow(((nxPML_2 - 1 - i - 0.5)
		  							/ (nxPML_2-1.0)), m);
		alphah_x_PML_2[i] = alpha_x_max * pow(((i + 1 -0.5)
		  								/ (nxPML_2-1.0)), ma);
		kappah_x_PML_2[i] = 1.0 + (kappa_x_max - 1.0) *
					pow(((nxPML_2 - 1 - i - 0.5) / (nxPML_2 - 1.0)), m);
		bh_x_2[i] = exp(-(sigh_x_PML_2[i] / kappah_x_PML_2[i] +
									alphah_x_PML_2[i]) * dt / epsO);
		ch_x_2[i] = sigh_x_PML_2[i] * (bh_x_2[i] - 1.0) /
					(sigh_x_PML_2[i] + kappah_x_PML_2[i] * alphah_x_PML_2[i])
					/ kappah_x_PML_2[i];
	}

	for(j = 0; j < nyPML_1; ++j) {

		sige_y_PML_1[j] = sig_y_max * pow(((nyPML_1 - 1 - j)
		   							/ (nyPML_1 - 1.0)), m);
		alphae_y_PML_1[j] = alpha_y_max * pow(((float)j
		   								/ (nyPML_1 - 1.0)), ma);
		kappae_y_PML_1[j] = 1.0 + (kappa_y_max - 1.0)*
						pow((nyPML_1 - 1- j) / (nyPML_1 - 1.0), m);
		be_y_1[j] = exp(-(sige_y_PML_1[j] / kappae_y_PML_1[j] +
									alphae_y_PML_1[j]) * dt / epsO);

		if ((sige_y_PML_1[j] == 0.0) &&
			(alphae_y_PML_1[j] == 0.0) && (j == nyPML_1 - 1)) {

			ce_y_1[j] = 0.0;

		} else {

			ce_y_1[j] = sige_y_PML_1[j] * (be_y_1[j] - 1.0) /
				(sige_y_PML_1[j] + kappae_y_PML_1[j] * alphae_y_PML_1[j])
				/ kappae_y_PML_1[j];
		}
	}

	for(j = 0; j < nyPML_1 - 1; ++j) {

			sigh_y_PML_1[j] = sig_y_max * pow(((nyPML_1 - 1 - j - 0.5)
			 							/ (nyPML_1-1.0)), m);
			alphah_y_PML_1[j] = alpha_y_max * pow(((j + 1 -0.5)
			 								/ (nyPML_1-1.0)), ma);
			kappah_y_PML_1[j] = 1.0 + (kappa_y_max - 1.0) *
						pow(((nyPML_1 - 1 - j - 0.5) / (nyPML_1 - 1.0)), m);
			bh_y_1[j] = exp(-(sigh_y_PML_1[j] / kappah_y_PML_1[j] +
									alphah_y_PML_1[j]) * dt / epsO);
			ch_y_1[j] = sigh_y_PML_1[j] * (bh_y_1[j] - 1.0) /
						(sigh_y_PML_1[j] + kappah_y_PML_1[j] * alphah_y_PML_1[j])
						/ kappah_y_PML_1[j];
	}

	for(j = 0; j < nyPML_2; ++j) {

		sige_y_PML_2[j] = sig_y_max * pow(((nyPML_2 - 1 - j)
		   							/ (nyPML_2 - 1.0)), m);
		alphae_y_PML_2[j] = alpha_y_max * pow(((float)j
		   								/ (nyPML_2 - 1.0)), ma);
		kappae_y_PML_2[j] = 1.0 + (kappa_y_max - 1.0)*
						pow((nyPML_2 - 1- j) / (nyPML_2 - 1.0), m);
		be_y_2[j] = exp(-(sige_y_PML_2[j] / kappae_y_PML_2[j] +
									alphae_y_PML_2[j]) * dt / epsO);

		if ((sige_y_PML_2[j] == 0.0) &&
			(alphae_y_PML_2[j] == 0.0) && (j == nyPML_2 - 1)) {

			ce_y_2[j] = 0.0;

		} else {

			ce_y_2[j] = sige_y_PML_2[j] * (be_y_2[j] - 1.0) /
				(sige_y_PML_2[j] + kappae_y_PML_2[j] * alphae_y_PML_2[j])
				/ kappae_y_PML_2[j];
		}
	}

	for(j = 0; j < nyPML_2 - 1; ++j) 
	{

		sigh_y_PML_2[j] = sig_y_max * pow(((nyPML_2 - 1 - j - 0.5)
		 							/ (nyPML_2-1.0)), m);
		alphah_y_PML_2[j] = alpha_y_max * pow(((j + 1 -0.5)
		 								/ (nyPML_2-1.0)), ma);
		kappah_y_PML_2[j] = 1.0 + (kappa_y_max - 1.0) *
				pow(((nyPML_2 - 1 - j - 0.5) / (nyPML_2 - 1.0)), m);
		bh_y_2[j] = exp(-(sigh_y_PML_2[j] / kappah_y_PML_2[j] +
								alphah_y_PML_2[j]) * dt / epsO);
		ch_y_2[j] = sigh_y_PML_2[j] * (bh_y_2[j] - 1.0) /
					(sigh_y_PML_2[j] + kappah_y_PML_2[j] * alphah_y_PML_2[j])
					/ kappah_y_PML_2[j];
	}

	for(k = 0; k < nzPML_1; ++k) {

		sige_z_PML_1[k] = sig_z_max * pow(((nzPML_1 - 1 - k)
		   							/ (nzPML_1 - 1.0)), m);
		alphae_z_PML_1[k] = alpha_z_max * pow(((float)k
		   								/ (nzPML_1 - 1.0)), ma);
		kappae_z_PML_1[k] = 1.0 + (kappa_z_max - 1.0)*
						pow((nzPML_1 - 1- k) / (nzPML_1 - 1.0), m);
		be_z_1[k] = exp(-(sige_z_PML_1[k] / kappae_z_PML_1[k] +
									alphae_z_PML_1[k]) * dt / epsO);

		if ((sige_z_PML_1[k] == 0.0) &&
			(alphae_z_PML_1[k] == 0.0) && (k == nzPML_1 - 1)) {

			ce_z_1[k] = 0.0;

		} else {

			ce_z_1[k] = sige_z_PML_1[k] * (be_z_1[k] - 1.0) /
				(sige_z_PML_1[k] + kappae_z_PML_1[k] * alphae_z_PML_1[k])
				/ kappae_z_PML_1[k];
		}
	}

	for(k = 0; k < nzPML_1 - 1; ++k) {

		sigh_z_PML_1[k] = sig_z_max * pow(((nzPML_1 - 1 - k - 0.5)
		 							/ (nzPML_1-1.0)), m);
		alphah_z_PML_1[k] = alpha_z_max * pow(((k + 1 -0.5)
		 								/ (nzPML_1-1.0)), ma);
		kappah_z_PML_1[k] = 1.0 + (kappa_z_max - 1.0) *
					pow(((nzPML_1 - 1 - k - 0.5) / (nzPML_1 - 1.0)), m);
		bh_z_1[k] = exp(-(sigh_z_PML_1[k] / kappah_z_PML_1[k] +
								alphah_z_PML_1[k]) * dt / epsO);
		ch_z_1[k] = sigh_z_PML_1[k] * (bh_z_1[k] - 1.0) /
					(sigh_z_PML_1[k] + kappah_z_PML_1[k] * alphah_z_PML_1[k])
					/ kappah_z_PML_1[k];
	}

	for(k = 0; k < nzPML_2; ++k) {

		sige_z_PML_2[k] = sig_z_max * pow(((nzPML_2 - 1 - k)
		   							/ (nzPML_2 - 1.0)), m);
		alphae_z_PML_2[k] = alpha_z_max * pow(((float)k
		   						        / (nzPML_2 - 1.0)), ma);
		kappae_z_PML_2[k] = 1.0 + (kappa_z_max - 1.0)*
							pow((nzPML_2 - 1- k) / (nzPML_2 - 1.0), m);
		be_z_2[k] = exp(-(sige_z_PML_2[k] / kappae_z_PML_2[k] +
									alphae_z_PML_2[k]) * dt / epsO);

		if ((sige_z_PML_2[k] == 0.0) &&
			(alphae_z_PML_2[k] == 0.0) && (k == nzPML_2 - 1)) {

			ce_z_2[k] = 0.0;

		} else {

			ce_z_2[k] = sige_z_PML_2[k] * (be_z_2[k] - 1.0) /
				(sige_z_PML_2[k] + kappae_z_PML_2[k] * alphae_z_PML_2[k])
				/ kappae_z_PML_2[k];
		}
	}

	for(k = 0; k < nzPML_2 - 1; ++k) {

		sigh_z_PML_2[k] = sig_z_max * pow(((nzPML_2 - 1 - k - 0.5)
		 							/ (nzPML_2-1.0)), m);
		alphah_z_PML_2[k] = alpha_z_max * pow(((k + 1 -0.5)
		 							/ (nzPML_2-1.0)), ma);
		kappah_z_PML_2[k] = 1.0 + (kappa_z_max - 1.0) *
				pow(((nzPML_2 - 1 - k - 0.5) / (nzPML_2 - 1.0)), m);
		bh_z_2[k] = exp(-(sigh_z_PML_2[k] / kappah_z_PML_2[k] +
								alphah_z_PML_2[k]) * dt / epsO);
		ch_z_2[k] = sigh_z_PML_2[k] * (bh_z_2[k] - 1.0) /
					(sigh_z_PML_2[k] + kappah_z_PML_2[k] * alphah_z_PML_2[k])
					/ kappah_z_PML_2[k];
}

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	//  DENOMINATORS FOR FIELD UPDATES
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	   ii = nxPML_2 - 2;

	   for(i = 0; i < Imax-1; ++i) {

		  if (i < nxPML_1-1) {

			 den_hx[i] = 1.0 / (kappah_x_PML_1[i] * dx);

		  } else if (i >= Imax - nxPML_2) {

			 den_hx[i] = 1.0 / (kappah_x_PML_2[ii] * dx);
			 ii = ii - 1;

		  } else {

			 den_hx[i] = 1.0 / dx;
  		  }
	   }

	   jj = nyPML_2 - 2;

	   for(j = 0; j < Jmax-1; ++j) {

		  if (j < nyPML_1-1) {

			 den_hy[j] = 1.0 / (kappah_y_PML_1[j] * dy);

		  } else if (j >= Jmax - nyPML_2) {

			 den_hy[j] = 1.0 / (kappah_y_PML_2[jj] * dy);
			 jj = jj - 1;

		  } else {

			 den_hy[j] = 1.0 / dy;
  		  }
	   }

	   kk = nzPML_2 - 2;

	   for (k = 1; k < Kmax - 1; ++k){

		  if (k < nzPML_1) {

			 den_hz[k] = 1.0 / (kappah_z_PML_1[k-1] * dz);

		  } else if (k >= Kmax - nzPML_2) {

			 den_hz[k] = 1.0 / (kappah_z_PML_2[kk] * dz);
			 kk = kk - 1;

		  } else {

			 den_hz[k] = 1.0 / dz;
		  }
	   }

	   ii = nxPML_2 - 1;

	   for(i = 0; i < Imax - 1; ++i) {

		  if (i < nxPML_1) {

			 den_ex[i] = 1.0 / (kappae_x_PML_1[i] * dx);

		  } else if (i >= Imax - nxPML_2) {

			 den_ex[i] = 1.0 / (kappae_x_PML_2[ii] * dx);
			 ii = ii - 1;

		  } else{

			 den_ex[i] = 1.0 / dx;
		  }
	   }

	   jj = nyPML_2 - 1;

	   for(j = 0; j < Jmax - 1; ++j) {

		  if (j < nyPML_1) {

			 den_ey[j] = 1.0 / (kappae_y_PML_1[j] * dy);

		  } else if (j >= Jmax - nyPML_2) {

			 den_ey[j] = 1.0 / (kappae_y_PML_2[jj] * dy);
			 jj = jj - 1;

		  } else {

			 den_ey[j] = 1.0 / dy;
		  }
	   }

	   kk = nzPML_2 - 1;

	   for(k = 0; k < Kmax - 1; ++k) {

		  if (k < nzPML_1) {

			 den_ez[k] = 1.0 / (kappae_z_PML_1[k] * dz);

		  }else if (k >= Kmax - 1 - nzPML_2) {

			 den_ez[k] = 1.0 / (kappae_z_PML_2[kk] * dz);
			 kk = kk - 1;

		  } else {

			 den_ez[k] = 1.0 / dz;
		  }
	   }
	  }

void compute() {
	pEx = fopen ( "Ex.txt" , "w");
	pEy = fopen ( "Ey.txt" , "w");
	pEz = fopen ( "Ez.txt" , "w");

	pHx = fopen ( "Hx.txt" , "w");
	pHy = fopen ( "Hy.txt" , "w");
	pHz = fopen ( "Hz.txt" , "w");

	fprintf(pEx, "Experiment \n");
	fprintf(pEy, "Experiment \n");
	fprintf(pEz, "Experiment \n");
	fprintf(pHx, "Experiment \n");
	fprintf(pHy, "Experiment \n");
	fprintf(pHz, "Experiment \n");

//---------------DEBUG--------------------------
	deb = fopen ( "deb.txt" , "w");
//---------------END OF DEBUG-------------------
//---------------DEBUG--------------------------
	for(k = 0; k < numMaterials; k++) {			
		fprintf(deb,"%f ", CA[k]);
	}
	fprintf(deb,"\n\n");

	for(k = 0; k < numMaterials; k++) {						
		fprintf(deb,"%f ", CB[k]);
	}
//---------------END OF DEBUG-------------------

	FILE *test;
	fopen_s(&test, "test.txt", "w");

	int id;
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	//  BEGIN TIME STEP
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	printf("\nBegin time-stepping...\n");

	for(n = 1; n <= nMax; ++n) {
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		//  UPDATE Hx
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		//------------index calculation------------------------------------
		Im = Imax-1;
		Jm = Jmax-1;
		Km = Kmax-2;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip = 0;
			Jp = 0;
			Kp = 1; 

			Jindex = (l/(Im*Km));
			Kindex = ((l-(Jindex*(Im*Km)))/Im);
			Iindex = ((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i = Iindex+Ip;
			j = Jindex+Jp;
			k = Kindex+Kp;
			//-----------------------------------------------------------------
				  
			id = ID1[j*(Imax*Kmax)+(k*Imax)+i];
			Hx[j*(Imax*Kmax)+(k*Imax)+i] = DA[id] * Hx[j*(Imax*Kmax)+(k*Imax)+i] + DB[id] *
				((Ez[j*(Imax*Kmax)+(k*Imax)+i] - Ez[(j+1)*(Imax*Kmax)+(k*Imax)+i]) * den_hy[j]  +
				(Ey[j*(Imax*(Kmax-1))+(k*Imax)+i] - Ey[j*(Imax*(Kmax-1))+((k-1)*Imax)+i]) * den_hz[k] );      
		}

		//...............................................
		//  PML for bottom Hx, j-direction
		//...............................................
		//------------index calculation------------------------------------
		Im = Imax-1;
		Jm = nyPML_1-1;
		Km = Kmax-2;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip = 0;
			Jp = 0;
			Kp = 1; 

			Jindex = (l/(Im*Km));
			Kindex = ((l-(Jindex*(Im*Km)))/Im);
			Iindex = ((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i = Iindex+Ip;
			j = Jindex+Jp;
			k = Kindex+Kp;
			//-----------------------------------------------------------------
			id = ID1[j*(Imax*Kmax)+(k*Imax)+i];
			psi_Hxy_1[j*(Imax*Kmax)+(k*Imax)+i] = bh_y_1[j] * psi_Hxy_1[j*(Imax*Kmax)+(k*Imax)+i] + 
				ch_y_1[j] * (Ez[j*(Imax*Kmax)+(k*Imax)+i] - Ez[(j+1)*(Imax*Kmax)+(k*Imax)+i]) / dy;
			Hx[j*(Imax*Kmax)+(k*Imax)+i] = Hx[j*(Imax*Kmax)+(k*Imax)+i] + DB[id] * psi_Hxy_1[j*(Imax*Kmax)+(k*Imax)+i];
		}

		//....................................................
		//  PML for top Hx, j-direction
		//.....................................................
		//------------index calculation------------------------------------
		Im = Imax-1;
		Jm = nyPML_2-1;
		Km = Kmax-2;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=0;
			Jp=0;
			Kp=1; 

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			jj=(Jmax-2)-j;
			//-----------------------------------------------------------------
			id = ID1[jj*(Imax*Kmax)+(k*Imax)+i];
			psi_Hxy_2[j*(Imax*Kmax)+(k*Imax)+i] = bh_y_2[j] * psi_Hxy_2[j*(Imax*Kmax)+(k*Imax)+i] +
				ch_y_2[j] * (Ez[jj*(Imax*Kmax)+(k*Imax)+i] - Ez[(jj+1)*(Imax*Kmax)+(k*Imax)+i]) / dy;
			Hx[jj*(Imax*Kmax)+(k*Imax)+i] = Hx[jj*(Imax*Kmax)+(k*Imax)+i] + DB[id] * psi_Hxy_2[j*(Imax*Kmax)+(k*Imax)+i];
		}
				
		//....................................................
		//  PML for bottom Hx, k-direction
		//................................................
		//------------index calculation------------------------------------
		Im=Imax-1;
		Jm=Jmax-1;
		Km=nzPML_1-1;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=0;
			Jp=0;
			Kp=1; 

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			//-----------------------------------------------------------------
			id = ID1[j*(Imax*Kmax)+(k*Imax)+i];
			psi_Hxz_1[j*(Imax*Km)+((k-1)*Imax)+i] = bh_z_1[k-1] * psi_Hxz_1[j*(Imax*Km)+((k-1)*Imax)+i] +
				ch_z_1[k-1] * (Ey[j*(Imax*(Kmax-1))+(k*Imax)+i] - Ey[j*(Imax*(Kmax-1))+((k-1)*Imax)+i]) / dz;
			Hx[j*(Imax*Kmax)+(k*Imax)+i] = Hx[j*(Imax*Kmax)+(k*Imax)+i] + DB[id] * psi_Hxz_1[j*(Imax*Km)+((k-1)*Imax)+i];
		}

		//....................................................
		//  PML for top Hx, k-direction
		//...............................................
		//------------index calculation------------------------------------
		Im=Imax-1;
		Jm=Jmax-1;
		Km=nzPML_2-1;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=0;
			Jp=0;
			Kp=0;				//???? Check ?????

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			kk=(Kmax-2)-k;
			//-----------------------------------------------------------------
			id = ID1[j*(Imax*Kmax)+(kk*Imax)+i];
			psi_Hxz_2[j*(Imax*Km)+(k*Imax)+i] = bh_z_2[k] * psi_Hxz_2[j*(Imax*Km)+(k*Imax)+i]
				+ ch_z_2[k] * (Ey[j*(Imax*(Kmax-1))+(kk*Imax)+i] - Ey[j*(Imax*(Kmax-1))+((kk-1)*Imax)+i]) / dz;
			Hx[j*(Imax*Kmax)+(kk*Imax)+i] = Hx[j*(Imax*Kmax)+(kk*Imax)+i] + DB[id] * psi_Hxz_2[j*(Imax*Km)+(k*Imax)+i];		 		 			 
   		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		//  UPDATE Hy
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		//------------index calculation------------------------------------
		Im=Imax-1;
		Jm=Jmax-1;
		Km=Kmax-2;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=0;
			Jp=0;
			Kp=1; 

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			//-----------------------------------------------------------------
			id = ID2[j*(Imax*Kmax)+(k*Imax)+i];
			Hy[j*((Imax-1)*Kmax)+(k*(Imax-1))+i] = DA[id] * Hy[j*((Imax-1)*Kmax)+(k*(Imax-1))+i] + DB[id] *
				((Ez[j*(Imax*Kmax)+(k*Imax)+(i+1)] - Ez[j*(Imax*Kmax)+(k*Imax)+i]) * den_hx[i] +
				(Ex[j*((Imax-1)*(Kmax-1))+((k-1)*(Imax-1))+i] - Ex[j*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i]) * den_hz[k] );
		}
			  
		//.......................................................
		//  PML for bottom Hy, i-direction
		//.......................................................
		//------------index calculation------------------------------------
		Im=nxPML_1-1;
		Jm=Jmax-1;
		Km=Kmax-2;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=0;
			Jp=0;
			Kp=1; 

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			//-----------------------------------------------------------------
			id = ID2[j*(Imax*Kmax)+(k*Imax)+i];
				psi_Hyx_1[j*(Im*Kmax)+(k*Im)+i] = bh_x_1[i] * psi_Hyx_1[j*(Im*Kmax)+(k*Im)+i] +
					ch_x_1[i] * (Ez[j*(Imax*Kmax)+(k*Imax)+(i+1)] - Ez[j*(Imax*Kmax)+(k*Imax)+i]) / dx;
				Hy[j*((Imax-1)*Kmax)+(k*(Imax-1))+i] = Hy[j*((Imax-1)*Kmax)+(k*(Imax-1))+i] + DB[id] * psi_Hyx_1[j*(Im*Kmax)+(k*Im)+i];
		}

		//.........................................................
		//  PML for top Hy, i-direction
		//.........................................................
		//------------index calculation------------------------------------
		Im=nxPML_2-1;
		Jm=Jmax-1;
		Km=Kmax-2;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=0;
			Jp=0;
			Kp=1; 

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			ii=(Imax-2)-i;
			//-----------------------------------------------------------------
			id = ID2[j*(Imax*Kmax)+(k*Imax)+ii];
			psi_Hyx_2[j*(Im*Kmax)+(k*Im)+i] = bh_x_2[i] * psi_Hyx_2[j*(Im*Kmax)+(k*Im)+i]
				+ ch_x_2[i] * (Ez[j*(Imax*Kmax)+(k*Imax)+(ii+1)] - Ez[j*(Imax*Kmax)+(k*Imax)+ii]) / dx;
			Hy[j*((Imax-1)*Kmax)+(k*(Imax-1))+ii] = Hy[j*((Imax-1)*Kmax)+(k*(Imax-1))+ii] + DB[id] * psi_Hyx_2[j*(Im*Kmax)+(k*Im)+i];				
		}

		//.......................................................
		//  PML for bottom Hy, k-direction
		//......................................................
		//------------index calculation------------------------------------
		Im=Imax-1;
		Jm=Jmax-1;
		Km=nzPML_1-1;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=0;
			Jp=0;
			Kp=1; 

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			//-----------------------------------------------------------------
			id = ID2[j*(Imax*Kmax)+(k*Imax)+i];
			psi_Hyz_1[j*((Imax-1)*Km)+((k-1)*(Imax-1))+i] = bh_z_1[k-1] * psi_Hyz_1[j*((Imax-1)*Km)+((k-1)*(Imax-1))+i]
				+ ch_z_1[k-1] * (Ex[j*((Imax-1)*(Kmax-1))+((k-1)*(Imax-1))+i] - Ex[j*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i]) / dz;
			Hy[j*((Imax-1)*Kmax)+(k*(Imax-1))+i] = Hy[j*((Imax-1)*Kmax)+(k*(Imax-1))+i] + DB[id] * psi_Hyz_1[j*((Imax-1)*Km)+((k-1)*(Imax-1))+i];
		}

		//.......................................................
		//  PML for top Hy, k-direction
		//.........................................................
		//------------index calculation------------------------------------
		Im=Imax-1;
		Jm=Jmax-1;
		Km=nzPML_2-1;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=0;
			Jp=0;
			Kp=0; 

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			kk=(Kmax-2)-k;
		//-----------------------------------------------------------------
			id = ID2[j*(Imax*Kmax)+(kk*Imax)+i];
			psi_Hyz_2[j*((Imax-1)*Km)+(k*(Imax-1))+i] = bh_z_2[k] * psi_Hyz_2[j*((Imax-1)*Km)+(k*(Imax-1))+i]
					+ ch_z_2[k] * (Ex[j*((Imax-1)*(Kmax-1))+((kk-1)*(Imax-1))+i] - Ex[j*((Imax-1)*(Kmax-1))+(kk*(Imax-1))+i]) / dz;
			Hy[j*((Imax-1)*Kmax)+(kk*(Imax-1))+i] = Hy[j*((Imax-1)*Kmax)+(kk*(Imax-1))+i] + DB[id] * psi_Hyz_2[j*((Imax-1)*Km)+(k*(Imax-1))+i];
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		//  UPDATE Hz
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		//------------index calculation------------------------------------
		Im=Imax-1;
		Jm=Jmax-1;
		Km=Kmax-1;

		for(l = 0; l < Im*Jm*Km; ++l) {
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
			id = ID3[j*(Imax*Kmax)+(k*Imax)+i];
			Hz[j*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i] = DA[id] * Hz[j*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i] + DB[id]
					* ((Ey[j*(Imax*(Kmax-1))+(k*Imax)+i] - Ey[j*(Imax*(Kmax-1))+(k*Imax)+(i+1)]) * den_hx[i] +
				(Ex[(j+1)*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i] - Ex[j*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i]) * den_hy[j]);
		}

		//..........................................................
		//  PML for bottom Hz, x-direction
		//..........................................................
		//------------index calculation------------------------------------
		Im=nxPML_1-1;
		Jm=Jmax-1;
		Km=Kmax-1;

		for(l = 0; l < Im*Jm*Km; ++l) {
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
			id = ID3[j*(Imax*Kmax)+(k*Imax)+i];
			psi_Hzx_1[j*(Im*(Kmax-1))+(k*Im)+i] = bh_x_1[i] * psi_Hzx_1[j*(Im*(Kmax-1))+(k*Im)+i]
				+ ch_x_1[i] * (Ey[j*(Imax*(Kmax-1))+(k*Imax)+i] - Ey[j*(Imax*(Kmax-1))+(k*Imax)+(i+1)]) / dx;
			Hz[j*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i] = Hz[j*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i] + DB[id] * psi_Hzx_1[j*(Im*(Kmax-1))+(k*Im)+i];
		}

		//..........................................................
		//  PML for top Hz, x-direction
		//..........................................................
		//------------index calculation------------------------------------
		Im=nxPML_2-1;
		Jm=Jmax-1;
		Km=Kmax-1;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=0;
			Jp=0;
			Kp=0; 

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			ii=(Imax-2)-i;
			//-----------------------------------------------------------------
			id = ID3[j*(Imax*Kmax)+(k*Imax)+ii];
			psi_Hzx_2[j*(Im*(Kmax-1))+(k*Im)+i] = bh_x_2[i] * psi_Hzx_2[j*(Im*(Kmax-1))+(k*Im)+i]
				+ ch_x_2[i] * (Ey[j*(Imax*(Kmax-1))+(k*Imax)+ii] - Ey[j*(Imax*(Kmax-1))+(k*Imax)+(ii+1)])/dx;
			Hz[j*((Imax-1)*(Kmax-1))+(k*(Imax-1))+ii] = Hz[j*((Imax-1)*(Kmax-1))+(k*(Imax-1))+ii] + DB[id] * psi_Hzx_2[j*(Im*(Kmax-1))+(k*Im)+i];
		}
			  
		//........................................................
		//  PML for bottom Hz, y-direction
		//........................................................
		//------------index calculation------------------------------------
		Im=Imax-1;
		Jm=nyPML_1-1;
		Km=Kmax-1;

		for(l = 0; l < Im*Jm*Km; ++l) {
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
			id = ID3[j*(Imax*Kmax)+(k*Imax)+i];
			psi_Hzy_1[j*(Im*Km)+(k*Im)+i] = bh_y_1[j] * psi_Hzy_1[j*(Im*Km)+(k*Im)+i]
				+ ch_y_1[j] * (Ex[(j+1)*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i] - Ex[j*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i]) / dy;
			Hz[j*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i] = Hz[j*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i] + DB[id] *  psi_Hzy_1[j*(Im*Km)+(k*Im)+i];
		}

		//.........................................................
		//  PML for top Hz, y-direction
		//..........................................................
		//------------index calculation------------------------------------
		Im=Imax-1;
		Jm=nyPML_2-1;
		Km=Kmax-1;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=0;
			Jp=0;
			Kp=0; 

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			jj=(Jmax-2)-j;
		//-----------------------------------------------------------------
			id = ID3[jj*(Imax*Kmax)+(k*Imax)+i];
			psi_Hzy_2[j*(Im*Km)+(k*Im)+i] = bh_y_2[j] * psi_Hzy_2[j*(Im*Km)+(k*Im)+i]
				+ ch_y_2[j] * (Ex[(jj+1)*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i] - Ex[jj*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i]) / dy;
			Hz[jj*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i] = Hz[jj*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i] + DB[id] * psi_Hzy_2[j*(Im*Km)+(k*Im)+i];
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		//  UPDATE Ex
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		//------------index calculation------------------------------------
		Im=Imax-1;
		Jm=Jmax-2;
		Km=Kmax-1;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=0;
			Jp=1;
			Kp=0; 

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			//-----------------------------------------------------------------	

			id = ID1[j*(Imax*Kmax)+(k*Imax)+i];
			if(id == 1) { // PEC
				Ex[j*(Im*Km)+(k*Im)+i] = 0;
			} else {
				Ex[j*(Im*Km)+(k*Im)+i] = CA[id] * Ex[j*(Im*Km)+(k*Im)+i] + CB[id] *
					((Hz[j*(Im*Km)+(k*Im)+i] - Hz[(j-1)*(Im*Km)+(k*Im)+i]) * den_ey[j]  +
					(Hy[j*(Im*Kmax)+(k*Im)+i] - Hy[j*(Im*Kmax)+((k+1)*Im)+i]) * den_ez[k]);
			}
		}

		//..............................................................
		//  PML for bottom Ex, j-direction
		//..............................................................
		//------------index calculation------------------------------------
		Im=Imax-1;
		Jm=nyPML_1-1;
		Km=Kmax-1;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=0;
			Jp=1;
			Kp=0; 

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			//-----------------------------------------------------------------
			id = ID1[j*(Imax*Kmax)+(k*Imax)+i];
			psi_Exy_1[j*(Im*Km)+(k*Im)+i] = be_y_1[j] * psi_Exy_1[j*(Im*Km)+(k*Im)+i]
				+ ce_y_1[j] * (Hz[j*(Im*Km)+(k*Im)+i] - Hz[(j-1)*(Im*Km)+(k*Im)+i])/dy;
			Ex[j*(Im*Km)+(k*Im)+i] = Ex[j*(Im*Km)+(k*Im)+i] + CB[id] * psi_Exy_1[j*(Im*Km)+(k*Im)+i];
		}
		
		//.............................................................
		//  PML for top Ex, j-direction
		//.............................................................
		//------------index calculation------------------------------------
		Im=Imax-1;
		Jm=nyPML_2-1;
		Km=Kmax-1;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=0;
			Jp=1;
			Kp=0; 

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			jj=(Jmax-1)-j;
			//-----------------------------------------------------------------
			id = ID1[jj*(Imax*Kmax)+(k*Imax)+i];
			psi_Exy_2[j*(Im*Km)+(k*Im)+i] = be_y_2[j] * psi_Exy_2[j*(Im*Km)+(k*Im)+i]
				+ ce_y_2[j] * (Hz[jj*(Im*Km)+(k*Im)+i] - Hz[(jj-1)*(Im*Km)+(k*Im)+i]) / dy;
			Ex[jj*(Im*Km)+(k*Im)+i] = Ex[jj*(Im*Km)+(k*Im)+i] + CB[id] * psi_Exy_2[j*(Im*Km)+(k*Im)+i];					
		}

		//.............................................................
		//  PML for bottom Ex, k-direction
		//.............................................................
		//------------index calculation------------------------------------
		Im=Imax-1;
		Jm=Jmax-2;
		Km=nzPML_1;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=0;
			Jp=1;
			Kp=0; 

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			//-----------------------------------------------------------------
			id = ID1[j*(Imax*Kmax)+(k*Imax)+i];
			psi_Exz_1[j*(Im*Km)+(k*Im)+i] = be_z_1[k] * psi_Exz_1[j*(Im*Km)+(k*Im)+i]
				+ ce_z_1[k] * (Hy[j*(Im*Kmax)+(k*Im)+i] - Hy[j*(Im*Kmax)+((k+1)*Im)+i]) / dz;
			Ex[j*(Im*(Kmax-1))+(k*Im)+i] = Ex[j*(Im*(Kmax-1))+(k*Im)+i] + CB[id] * psi_Exz_1[j*(Im*Km)+(k*Im)+i];
		}

		//..............................................................
		//  PML for top Ex, k-direction
		//..............................................................
		//------------index calculation------------------------------------
		Im=Imax-1;
		Jm=Jmax-2;
		Km=nzPML_2;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=0;
			Jp=1;
			Kp=0;

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			kk=(Kmax-2)-k;
			//------------------------------------------------------------
			id = ID1[j*(Imax*Kmax)+(kk*Imax)+i];
			psi_Exz_2[j*(Im*Km)+(k*Im)+i] = be_z_2[k] * psi_Exz_2[j*(Im*Km)+(k*Im)+i]
				+ ce_z_2[k] * (Hy[j*(Im*Kmax)+(kk*Im)+i] - Hy[j*(Im*Kmax)+((kk+1)*Im)+i]) / dz;
			Ex[j*(Im*(Kmax-1))+(kk*Im)+i] = Ex[j*(Im*(Kmax-1))+(kk*Im)+i] + CB[id] * psi_Exz_2[j*(Im*Km)+(k*Im)+i];
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		//  UPDATE Ey
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		//------------index calculation------------------------------------
		Im=Imax-2;
		Jm=Jmax-1;
		Km=Kmax-1;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=1;
			Jp=0;
			Kp=0; 

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			//-----------------------------------------------------------------

			id = ID2[j*(Imax*Kmax)+(k*Imax)+i];
			if(id == 1) { // PEC
				Ey[j*(Imax*Km)+(k*Imax)+i] = 0;
			} else {
				Ey[j*(Imax*Km)+(k*Imax)+i] = CA[id] * Ey[j*(Imax*Km)+(k*Imax)+i] + CB[id] *
					((Hz[j*((Imax-1)*Km)+(k*(Imax-1))+(i-1)] - Hz[j*((Imax-1)*Km)+(k*(Imax-1))+i]) * den_ex[i] +
					(Hx[j*(Imax*Kmax)+((k+1)*Imax)+i] - Hx[j*(Imax*Kmax)+(k*Imax)+i]) * den_ez[k] );
			}
		}

		//...........................................................
		//  PML for bottom Ey, i-direction
		//...........................................................
		//------------index calculation------------------------------------
		Im=nxPML_1-1;
		Jm=Jmax-1;
		Km=Kmax-1;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=1;
			Jp=0;
			Kp=0; 

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			//----------------------------------------------------------------
			id = ID2[j*(Imax*Kmax)+(k*Imax)+i];
			psi_Eyx_1[j*(nxPML_1*Km)+(k*nxPML_1)+i] = be_x_1[i] * psi_Eyx_1[j*(nxPML_1*Km)+(k*nxPML_1)+i]
				+ ce_x_1[i] * (Hz[j*((Imax-1)*Km)+(k*(Imax-1))+(i-1)] - Hz[j*((Imax-1)*Km)+(k*(Imax-1))+i]) / dx;
			Ey[j*(Imax*Km)+(k*Imax)+i] = Ey[j*(Imax*Km)+(k*Imax)+i] + CB[id] * psi_Eyx_1[j*(nxPML_1*Km)+(k*nxPML_1)+i];
		}

		//............................................................
		//  PML for top Ey, i-direction
		//............................................................
		//------------index calculation------------------------------------
		Im=nxPML_2-1;
		Jm=Jmax-1;
		Km=Kmax-1;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=1;
			Jp=0;
			Kp=0;

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			ii=(Imax-1)-i;
			//----------------------------------------------------------------
			id = ID2[j*(Imax*Kmax)+(k*Imax)+ii];
			psi_Eyx_2[j*(nxPML_2*Km)+(k*nxPML_2)+i] = be_x_2[i] * psi_Eyx_2[j*(nxPML_2*Km)+(k*nxPML_2)+i]
				+ ce_x_2[i] * (Hz[j*((Imax-1)*Km)+(k*(Imax-1))+(ii-1)] - Hz[j*((Imax-1)*Km)+(k*(Imax-1))+ii]) / dx;
			Ey[j*(Imax*Km)+(k*Imax)+ii] = Ey[j*(Imax*Km)+(k*Imax)+ii] + CB[id] * psi_Eyx_2[j*(nxPML_2*Km)+(k*nxPML_2)+i];
		}

		//...........................................................
		//  PML for bottom Ey, k-direction
		//...........................................................
		//------------index calculation------------------------------------
		Im=Imax-2;
		Jm=Jmax-1;
		Km=nzPML_1;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=1;
			Jp=0;
			Kp=0; 

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			//----------------------------------------------------------------
			id = ID2[j*(Imax*Kmax)+(k*Imax)+i];
			psi_Eyz_1[j*((Imax-1)*Km)+(k*(Imax-1))+i] = be_z_1[k] * psi_Eyz_1[j*((Imax-1)*Km)+(k*(Imax-1))+i]
				+ ce_z_1[k] * (Hx[j*(Imax*Kmax)+((k+1)*Imax)+i] - Hx[j*(Imax*Kmax)+(k*Imax)+i]) / dz;
			Ey[j*(Imax*(Kmax-1))+(k*Imax)+i] = Ey[j*(Imax*(Kmax-1))+(k*Imax)+i] + CB[id] * psi_Eyz_1[j*((Imax-1)*Km)+(k*(Imax-1))+i];
		}	

		//...........................................................
		//  PML for top Ey, k-direction
		//............................................................
		//------------index calculation------------------------------------
		Im=Imax-2;
		Jm=Jmax-1;
		Km=nzPML_2;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=1;
			Jp=0;
			Kp=0;

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			kk=(Kmax-2)-k;
			//---------------------------------------------------------------
			id = ID2[j*(Imax*Kmax)+(kk*Imax)+i];
			psi_Eyz_2[j*((Imax-1)*Km)+(k*(Imax-1))+i] = be_z_2[k] * psi_Eyz_2[j*((Imax-1)*Km)+(k*(Imax-1))+i]
					+ ce_z_2[k] * (Hx[j*(Imax*Kmax)+((kk+1)*Imax)+i] - Hx[j*(Imax*Kmax)+(kk*Imax)+i]) / dz;
			Ey[j*(Imax*(Kmax-1))+(kk*Imax)+i] = Ey[j*(Imax*(Kmax-1))+(kk*Imax)+i] + CB[id] * psi_Eyz_2[j*((Imax-1)*Km)+(k*(Imax-1))+i];
		}

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
		//  UPDATE Ez
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		//------------index calculation------------------------------------
		Im=Imax-2;
		Jm=Jmax-2;
		Km=Kmax-2;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=1;
			Jp=1;
			Kp=1; 

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			//-----------------------------------------------------------------

			id = ID3[j*(Imax*Kmax)+(k*Imax)+i];
			if(id == 1) { // PEC
				Ez[j*(Imax*Kmax)+(k*Imax)+i] = 0;
			} else {
				Ez[j*(Imax*Kmax)+(k*Imax)+i] = CA[id] * Ez[j*(Imax*Kmax)+(k*Imax)+i] + CB[id]
						* ((Hy[j*((Imax-1)*Kmax)+(k*(Imax-1))+i] - Hy[j*((Imax-1)*Kmax)+(k*(Imax-1))+(i-1)]) * den_ex[i] +
					(Hx[(j-1)*(Imax*Kmax)+(k*Imax)+i] - Hx[j*(Imax*Kmax)+(k*Imax)+i]) * den_ey[j]);
			}
		}

		//............................................................
		//  PML for bottom Ez, x-direction
		//............................................................
		//------------index calculation------------------------------------
		Im=nxPML_1-1;
		Jm=Jmax-2;
		Km=Kmax-2;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=1;
			Jp=1;
			Kp=1; 

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			//-----------------------------------------------------------------
			id = ID3[j*(Imax*Kmax)+(k*Imax)+i];
			psi_Ezx_1[j*(nxPML_1*Kmax)+(k*nxPML_1)+i] = be_x_1[i] * psi_Ezx_1[j*(nxPML_1*Kmax)+(k*nxPML_1)+i]
				+ ce_x_1[i] * (Hy[j*((Imax-1)*Kmax)+(k*(Imax-1))+i] - Hy[j*((Imax-1)*Kmax)+(k*(Imax-1))+(i-1)]) / dx;
			Ez[j*(Imax*Kmax)+(k*Imax)+i] = Ez[j*(Imax*Kmax)+(k*Imax)+i] + CB[id] * psi_Ezx_1[j*(nxPML_1*Kmax)+(k*nxPML_1)+i];
		}

		//............................................................
		//  PML for top Ez, x-direction
		//............................................................
		//------------index calculation------------------------------------
		Im=nxPML_2-1;
		Jm=Jmax-2;
		Km=Kmax-2;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=1;
			Jp=1;
			Kp=1;

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			ii=(Imax-1)-i;
			//----------------------------------------------------------------
			id = ID3[j*(Imax*Kmax)+(k*Imax)+ii];
			psi_Ezx_2[j*(nxPML_2*Kmax)+(k*nxPML_2)+i] = be_x_2[i] * psi_Ezx_2[j*(nxPML_2*Kmax)+(k*nxPML_2)+i]
				+ ce_x_2[i] * (Hy[j*((Imax-1)*Kmax)+(k*(Imax-1))+ii] - Hy[j*((Imax-1)*Kmax)+(k*(Imax-1))+(ii-1)]) / dx;
			Ez[j*(Imax*Kmax)+(k*Imax)+ii] = Ez[j*(Imax*Kmax)+(k*Imax)+ii] + CB[id] * psi_Ezx_2[j*(nxPML_2*Kmax)+(k*nxPML_2)+i];
		}

		//..........................................................
		//  PML for bottom Ez, y-direction
		//..........................................................
		//------------index calculation------------------------------------
		Im=Imax-2;
		Jm=nyPML_1-1;
		Km=Kmax-2;

		for(l = 0; l < Im*Jm*Km; ++l) {
			Ip=1;
			Jp=1;
			Kp=1; 

			Jindex=(l/(Im*Km));
			Kindex=((l-(Jindex*(Im*Km)))/Im);
			Iindex=((l-(Jindex*(Im*Km)))-(Kindex*Im));

			i=Iindex+Ip;
			j=Jindex+Jp;
			k=Kindex+Kp;
			//----------------------------------------------------------------
			id = ID3[j*(Imax*Kmax)+(k*Imax)+i];
			psi_Ezy_1[j*(Imax*Kmax)+(k*Imax)+i] = be_y_1[j] * psi_Ezy_1[j*(Imax*Kmax)+(k*Imax)+i]
				+ ce_y_1[j] * (Hx[(j-1)*(Imax*Kmax)+(k*Imax)+i] - Hx[j*(Imax*Kmax)+(k*Imax)+i]) / dy;
			Ez[j*(Imax*Kmax)+(k*Imax)+i] = Ez[j*(Imax*Kmax)+(k*Imax)+i] + CB[id] * psi_Ezy_1[j*(Imax*Kmax)+(k*Imax)+i];
		}

		//............................................................
		//  PML for top Ez, y-direction
		//............................................................
		//------------index calculation------------------------------------
		Im=Imax-2;
		Jm=nyPML_2-1;
		Km=Kmax-2;

		for (l = 0; l < Im*Jm*Km; ++l) {
			Ip = 1;
			Jp = 1;
			Kp = 1;

			Jindex = (l / (Im*Km));
			Kindex = ((l - (Jindex*(Im*Km))) / Im);
			Iindex = ((l - (Jindex*(Im*Km))) - (Kindex*Im));

			i = Iindex + Ip;
			j = Jindex + Jp;
			k = Kindex + Kp;
			jj = (Jmax - 1) - j;
			//----------------------------------------------------------------
			id = ID3[jj*(Imax*Kmax) + (k*Imax) + i];
			psi_Ezy_2[j*(Imax*Kmax) + (k*Imax) + i] = be_y_2[j] * psi_Ezy_2[j*(Imax*Kmax) + (k*Imax) + i]
				+ ce_y_2[j] * (Hx[(jj - 1)*(Imax*Kmax) + (k*Imax) + i] - Hx[jj*(Imax*Kmax) + (k*Imax) + i]) / dy;
			Ez[jj*(Imax*Kmax) + (k*Imax) + i] = Ez[jj*(Imax*Kmax) + (k*Imax) + i] + CB[id] * psi_Ezy_2[j*(Imax*Kmax) + (k*Imax) + i];
		}

		//----------------------------------------------------------
		//   Apply a point source (Soft)
		//-----------------------------------------------------------
		i = 25;
		j = 63;
		k = 12;
		source = amp * -2.0 * ((n * dt - tO) / tw)
			* exp(-pow(((n * dt - tO) / tw), 2));
	 
		Ez[j*(Imax*Kmax)+(k*Imax)+i] = Ez[j*(Imax*Kmax)+(k*Imax)+i] - CB[ID3[j*(Imax*Kmax)+(k*Imax)+i]] * source;

		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
		//  WRITE TO OUTPUT FILES
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		if (n >= from_time && n <= to_time)	{
			if(save_app == save_app_count) {
				fprintf(test, "(Ex Ey Ez) (Hx Hy Hz) at time step %d at (25, 63, 12) :  %f  :  %f  :  %f  :  %f  :  %f  :  %f  \n", n, Ex[63*((Imax-1)*(Kmax-1))+(12*(Imax-1))+25], Ey[63*(Imax*(Kmax-1))+(12*Imax)+25], Ez[63*(Imax*Kmax)+(12*Imax)+25], Hx[63*(Imax*Kmax)+(12*Imax)+25], Hy[63*((Imax-1)*Kmax)+(12*(Imax-1))+25], Hz[63*((Imax-1)*(Kmax-1))+(12*(Imax-1))+25]);

				printf("\nsaving data ...\n");
				save_counter++;
				save_app=1;

				fprintf(pEx, "Time is=%d \n", n);
				fprintf(pEy, "Time is=%d \n", n);
				fprintf(pEz, "Time is=%d \n", n);
				fprintf(pHx, "Time is=%d \n", n);
				fprintf(pHy, "Time is=%d \n", n);
				fprintf(pHz, "Time is=%d \n", n);

				Sindex=Kmax/2;
				//k=Kmax/2;
				k=12;
				{
					fprintf(pEx,"slice num = %d \n",Sindex);
					fprintf(pEy,"slice num = %d \n",Sindex);
					fprintf(pEz,"slice num = %d \n",Sindex);
					fprintf(pHx,"slice num = %d \n",Sindex);
					fprintf(pHy,"slice num = %d \n",Sindex);
					fprintf(pHz,"slice num = %d \n",Sindex);

					for(i=1;i<Imax-1;i+=sieve) {
						for(j=1;j<Jmax-1;j+=sieve) {
		//--------------------------I-SLISE--------------------------------------------------------------------
							//fprintf(pEx,"%12f ", Ex[j*((Imax-1)*(Kmax-1))+(k*(Imax-1))+Sindex]*mag);
							//fprintf(pEy,"%12f ", Ey[j*((Imax)*(Kmax-1))+(k*Imax)+Sindex]*mag);
							//fprintf(pEz,"%12f ", Ez[j*((Imax)*(Kmax))+(k*(Imax))+Sindex]*mag);
							//fprintf(pHx,"%12f ", Hx[j*(Imax*Kmax)+(k*Imax)+Sindex]*mag);
							//fprintf(pHy,"%12f ", Hy[j*((Imax-1)*Kmax)+(k*(Imax-1))+Sindex]*mag);
							//fprintf(pHz,"%12f ", Hz[j*((Imax-1)*(Kmax-1))+(k*(Imax-1))+Sindex]*mag);
		//--------------------------K-SLISE-----------------------------------------------------------------------
							//fprintf(pEx,"%12f ", Ex[j*((Imax-1)*(Kmax-1))+(Sindex*(Imax-1))+i]*mag);
							//fprintf(pEy,"%12f ", Ey[j*((Imax)*(Kmax-1))+(Sindex*Imax)+i]*mag);
							//fprintf(pEz,"%12f ", Ez[j*((Imax)*(Kmax))+(Sindex*(Imax))+i]*mag);
							//fprintf(pHx,"%12f ", Hx[j*(Imax*Kmax)+(Sindex*Imax)+i]*mag);
							//fprintf(pHy,"%12f ", Hy[j*((Imax-1)*Kmax)+(Sindex*(Imax-1))+i]*mag);
							//fprintf(pHz,"%12f ", Hz[j*((Imax-1)*(Kmax-1))+(Sindex*(Imax-1))+i]*mag);
		//--------------------------J-SLISE----------------------------------------------------------------------
							fprintf(pEx,"%12f ", Ex[j*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i]*mag);
							fprintf(pEy,"%12f ", Ey[j*((Imax)*(Kmax-1))+(k*Imax)+i]*mag);
							fprintf(pEz,"%12f ", Ez[j*((Imax)*(Kmax))+(k*(Imax))+i]*mag);
							fprintf(pHx,"%12f ", Hx[j*(Imax*Kmax)+(k*Imax)+i]*mag);
							fprintf(pHy,"%12f ", Hy[j*((Imax-1)*Kmax)+(k*(Imax-1))+i]*mag);
							fprintf(pHz,"%12f ", Hz[j*((Imax-1)*(Kmax-1))+(k*(Imax-1))+i]*mag);
		//--------------------------------------------------------------------------------------------------
							//fprintf(pEx,"%8d ", Ex[j*(Imax*Kmax)+(k*Imax)+i]);
							//fprintf(pEy,"%8d ", Ey[j*(Imax*Kmax)+(k*Imax)+i]);
							//fprintf(pEz,"%8d ", Ez[j*(Imax*Kmax)+(k*Imax)+i]);
							//fprintf(pHx,"%8d ", Hx[j*(Imax*Kmax)+(k*Imax)+i]);
							//fprintf(pHy,"%8d ", Hy[j*(Imax*Kmax)+(k*Imax)+i]);
							//fprintf(pHz,"%8d ", Hz[j*(Imax*Kmax)+(k*Imax)+i]);

							//fprintf(pEx,"%d ", j*(Imax*Kmax)+(k*Imax)+i;
							//fprintf(pEy,"%d ", j*(Imax*Kmax)+(k*Imax)+i;
							//fprintf(pEz,"%d ", j*(Imax*Kmax)+(k*Imax)+i;
							//fprintf(pHx,"%d ", j*(Imax*Kmax)+(k*Imax)+i;
							//fprintf(pHy,"%d ", j*(Imax*Kmax)+(k*Imax)+i;
							//fprintf(pHz,"%d ", j*(Imax*Kmax)+(k*Imax)+i;
						}
						fprintf(pEx,"\n");
						fprintf(pEy,"\n");
						fprintf(pEz,"\n");
						fprintf(pHx,"\n");
						fprintf(pHy,"\n");
						fprintf(pHz,"\n");
					}
					fprintf(pEx,"\n");
					fprintf(pEy,"\n");
					fprintf(pEz,"\n");
					fprintf(pHx,"\n");
					fprintf(pHy,"\n");
					fprintf(pHz,"\n");
				}
				fprintf(pEx,"\n");
				fprintf(pEy,"\n");
				fprintf(pEz,"\n");
				fprintf(pHx,"\n");
				fprintf(pHy,"\n");
				fprintf(pHz,"\n");
			} else {
				save_app++;
			}
		}
	}
   	//  END TIME STEP

// ---------------------printf to LOG ile-------------------------------------------------------------------------
// Display operating system-style date and time. 
	char tmpTime[128], tmpDate[128];
    _strtime_s( tmpTime, 128 );   
    _strdate_s( tmpDate, 128 );

	runTime = clock() - runTime;

plog=fopen("log.txt", "a");
fprintf(plog,"-------------------------------------------------------------------------\n");
    fprintf( plog,"OS date:\t\t\t\t%s\n", tmpDate );
	fprintf( plog,"OS time:\t\t\t\t%s\n", tmpTime );
fprintf(plog,"-------------------------------------------------------------------------\n");
fprintf(plog,"NET PARAMETERS \n Ex(%d, %d, %d) \n Ey(%d, %d, %d) \n Ez(%d, %d, %d) \n Hx(%d, %d, %d) \n Hy(%d, %d, %d) \n Hz(%d, %d, %d) \n\n(dx ,dy, dz) - dt: (%f, %f, %f) - %f", 
		Imax-1, Jmax, Kmax-1, Imax, Jmax-1, Kmax-1, Imax, Jmax, Kmax, Imax, Jmax-1, Kmax, Imax-1, Jmax, Kmax, Imax-1, Jmax-1, Kmax-1, dx, dy, dz, dt);
fprintf(plog,"\nSAVE DATA PARAMETER\nSlise saved: %d\nData saved from time %d to time %d \nTime sieve parameter: %d \nNet sieve parameter: %dData multiplier: %f\n", 
		save_counter, from_time, to_time, save_app_count, sieve, mag);

fprintf(plog,"\n\nCalculation on CPU\n Calculation time: %f seconds\n\n", (double)runTime/CLOCKS_PER_SEC);
//-----------------END of printing to LOG file-----------------------------------------------------------------
    printf("Done time-stepping...\n");

	fclose(test);
	fclose(pEx);
	fclose(pEy);
	fclose(pEz);
	fclose(pHx);
	fclose(pHy);
	fclose(pHz);
	fclose(plog);

	//---------------DEBUG--------------------------
	fclose(deb);
	//---------------END OF DEBUG-------------------
}

	 //Builds an object
	void buildObject() 
	{

		if(ObjectsEnable==true)
		{

			if(Kamera.Draw2D==ENABLE)
			draw_2D_figure();

			if(Build_medium==1)
			{
				buildSphericalMedium();
			}
			if(Build_medium==2)
			{
				buildTorusMedium();
			}

			if(Build_fluc_medium!=0)
			{
				buildSphericalMediumFluc(Build_fluc_medium, MediumFlucCen.I, MediumFlucCen.J, MediumFlucCen.K, SphereFlucCoef.P, SphereFlucCoef.Q, SphereFlucCoef.N, SphereFlucCoef.A, SphereFlucCoef.B, SphereFlucCoef.C, FlucEqCoef.ne0, FlucEqCoef.R, FlucEqCoef.Alpha, FlucEqCoef.Beta ,FlucAngle.I ,FlucAngle.J ,FlucAngle.K ,flucSign);
			}
		}

		//buildMedium();
		//buildSphere();

		if(BUILD_DIPOLE==ENABLE)
		buildDipole();

		if(ABUILD==ENABLE)
		{
			buildCubikAntenna(ANTENNA_TUPE);
	//		buildCubikAntenna(2);
		
			//CopyMedium();
			//MakeCloseTheDoor();
		}
		if(DBUILD==ENABLE)
		{
			buildCubikDetector(D_ANTEN_TUPE) ;
		}

		if(Build_Stell_wall==ENABLE)
		{
			buildPlateMedium();
		}

		if(CALC_ARREY_POS==ENABLE)
		{
			calculate2Darrai();
			calculateDetect2Darrai();
		}

		if (NET2D_SAVE_ENABLE == ENABLE)
			calculateRCdetect2Darrai();

	}

	//Builds a sphere (Sample code - NOt used in this program)
	void buildSphere() {

		double dist;//distance
		double rad = 8;//(double)Imax / 5.0; // sphere radius
      	double sc = (double)Imax / 2.0;//sphere centre
		double rad2 = 0.3;//(double)Imax / 5.0 - 3.0; // sphere radius

      	for(i = 0; i < Imax; ++i) {

        	for(j = 0; j < Jmax; ++j) {

          		for(k = 0; k < Kmax; ++k) {

					//compute distance form centre to the point i, j, k
            		dist = sqrt((i + 0.5 - sc) * (i + 0.5 - sc) +
								(j + 0.5 - sc) * (j + 0.5 - sc) +
								(k + 0.5 - sc) * (k + 0.5 - sc));

					//if point is within the sphere
					if (dist <= rad) {
					   //set the material at that point
					   yeeCube (i, j, k, 6);

					}
  				}
            }
       }

	}

	//Builds a dipole
	void buildDipole() {

		int centre;
		char direct=DIPOL_DIRECTION;
		int jCentR=(1/DS-1);

		//Specify the dipole size 
		istart = SourceCen.I-1/DS;			//-1
		iend = SourceCen.I+1/DS;			//+1
		jstart = SourceCen.J-8/DS;
		jend = SourceCen.J+8/DS;
		kstart = SourceCen.K-1/DS;			//-1
		kend = SourceCen.K+1/DS;			//+1

		switch(direct)
		{
		case 'x':

			centre = (istart + iend) / 2;

      		for(i = istart; i <= iend+jCentR; ++i) {

        		for(j = jstart; j <= jend; ++j) {

          			for(k = kstart; k < kend; ++k) {
					
						if((i<centre)||(i>centre+(jCentR)))
						{
							yeeCube (i, j, k, 1);//PEC material
						}
					}
				}
			}
		break;

		case 'y':

			centre = (jstart + jend) / 2;

      		for(i = istart; i < iend; ++i) {

        		for(j = jstart; j <= jend+jCentR; ++j) {

          			for(k = kstart; k < kend; ++k) {
					
						if((j<centre)||(j>centre+(jCentR)))
						{
							yeeCube (i, j, k, 1);//PEC material
						}
					}
				}
			}
		break;

		case 'z':

			centre = (kstart + kend) / 2;

      		for(i = istart; i < iend; ++i) {

        		for(j = jstart; j < jend; ++j) {

          			for(k = kstart; k <= kend+jCentR; ++k) {
					
						if((k<centre)||(k>centre+(jCentR)))
						{
							yeeCube (i, j, k, 1);//PEC material
						}
					}
				}
			}
		break;

		}

	}

	//creates a dielctric cube (yee cell) made up of the selected material
    void yeeCube (int I, int J,int K, int mType) {

		   //set face 1 (for EX)
           //ID1[J*(Imax*Kmax)+(K*Imax)+I] = mType;
           //ID1[J*(Imax*Kmax)+((K+1)*Imax)+I] = mType;
           //ID1[(J+1)*(Imax*Kmax)+((K+1)*Imax)+I] = mType;
           //ID1[(J+1)*(Imax*Kmax)+(K*Imax)+I] = mType;

		   //set face 2 (for EY)
           //ID2[J*(Imax*Kmax)+(K*Imax)+I] = mType;
           //ID2[J*(Imax*Kmax)+(K*Imax)+(I+1)] = mType;
           //ID2[J*(Imax*Kmax)+((K+1)*Imax)+(I+1)] = mType;
           //ID2[J*(Imax*Kmax)+((K+1)*Imax)+I] = mType;

		   //set face 3 (for EZ)
           ID3[J*(Imax*Kmax)+(K*Imax)+I] = mType;
           ID3[J*(Imax*Kmax)+(K*Imax)+(I+1)] = mType;
           ID3[(J+1)*(Imax*Kmax)+(K*Imax)+(I+1)] = mType;
           ID3[(J+1)*(Imax*Kmax)+(K*Imax)+I] = mType;
      }

			//creates a dielctric cube (yee cell) made up of the selected material
	void yeeCubeFace (int I, int J,int K, int mType, char face) 
	{
		switch(face)
		{
		case 'x':
		   //set face 1 (for EX)
           ID1[J*(Imax*Kmax)+(K*Imax)+I] = mType;
//           ID1[J*(Imax*Kmax)+((K+1)*Imax)+I] = mType;
//           ID1[(J+1)*(Imax*Kmax)+((K+1)*Imax)+I] = mType;
           ID1[(J+1)*(Imax*Kmax)+(K*Imax)+I] = mType;
	    break;
		case 'y':
		   //set face 2 (for EY)
           ID2[J*(Imax*Kmax)+(K*Imax)+I] = mType;
           ID2[J*(Imax*Kmax)+(K*Imax)+(I+1)] = mType;
//           ID2[J*(Imax*Kmax)+((K+1)*Imax)+(I+1)] = mType;
//           ID2[J*(Imax*Kmax)+((K+1)*Imax)+I] = mType;
		break;
		case 'z':
		   //set face 3 (for EZ)
           ID3[J*(Imax*Kmax)+(K*Imax)+I] = mType;
           ID3[J*(Imax*Kmax)+(K*Imax)+(I+1)] = mType;
           ID3[(J+1)*(Imax*Kmax)+(K*Imax)+(I+1)] = mType;
           ID3[(J+1)*(Imax*Kmax)+(K*Imax)+I] = mType;
		break;
		}
      }


	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	// Saving Output Data to files
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	void writeField (int iteration) 
	{

		FILE *ptr;
		char step[10];
		char fileBaseName[] = "E_Field_";
		sprintf_s(step, "%d", iteration);
		strcat_s(fileBaseName, step);
		strcat_s(fileBaseName, ".txt");
		//strcat("E_Field_", step);
		//strcat("E_Field_", ".txt");

		fopen_s(&ptr,fileBaseName,"wt");

		for(i = 0 ; i < Imax-1 ; i++) {

			for(j = 0 ; j < Jmax-1 ; j++){
				// |E|
				fprintf(ptr, "%f\t", sqrt(pow(Ex[j*(Imax*Kmax)+(ksource*Imax)+i], 2) +
					pow(Ey[j*(Imax*Kmax)+(ksource*Imax)+i], 2) + pow( Ez[j*(Imax*Kmax)+(ksource*Imax)+i], 2)));
			
			//	fprintf(ptr, "%f\t", Ex[i][j][ksource]);//Ex
			//	fprintf(ptr, "%f\t", Ey[i][j][ksource]);//Ey
			//	fprintf(ptr, "%f\t", Ez[i][j][ksource]);//Ez
			//	fprintf(ptr, "%f\t", Hx[i][j][ksource]);//Hx
			//	fprintf(ptr, "%f\t", Hy[i][j][ksource]);//Hy
			//	fprintf(ptr, "%f\t", Hz[i][j][ksource]);//Hz

			// |H|
			//	fprintf(ptr, "%f\t", sqrt(pow(Hx[i][j][ksource], 2) +
			//		pow(Hy[i][j][ksource], 2) + pow( Hz[i][j][ksource], 2)));

			}
			fprintf(ptr, "\n");
		}

		fclose(ptr);

	}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// END OF PROGRAM CPMLFDTD3D
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	My functions
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	void addYeeCube (int I, int J,int K, int mType, char face) 
	{
		switch(face)
		{
		case 'x':
		   //set face 1 (for EX)
			//check limit of plasma
		   if(ID1[J*(Imax*Kmax)+(K*Imax)+I]==0)
			   ID1[J*(Imax*Kmax)+(K*Imax)+I]=8;

		   if((ID1[J*(Imax*Kmax)+(K*Imax)+I] + mType)>8)
			   ID1[J*(Imax*Kmax)+(K*Imax)+I] =ID1[J*(Imax*Kmax)+(K*Imax)+I] + mType;
		   else
			   ID1[J*(Imax*Kmax)+(K*Imax)+I]=8;
		   //if((ID1[J*(Imax*Kmax)+((K+1)*Imax)+I] + mType)>8)
     //      ID1[J*(Imax*Kmax)+((K+1)*Imax)+I] = ID1[J*(Imax*Kmax)+((K+1)*Imax)+I] + mType;

		   //if((ID1[(J+1)*(Imax*Kmax)+((K+1)*Imax)+I] + mType)>8)
     //      ID1[(J+1)*(Imax*Kmax)+((K+1)*Imax)+I] = ID1[(J+1)*(Imax*Kmax)+((K+1)*Imax)+I] + mType;

//-----------------------------------------------------------------------------------
		   	//check limit of plasma
		   if(ID1[(J+1)*(Imax*Kmax)+(K*Imax)+I]==0)
			   ID1[(J+1)*(Imax*Kmax)+(K*Imax)+I]=8;

		   if((ID1[(J+1)*(Imax*Kmax)+(K*Imax)+I] + mType)>8)
			   ID1[(J+1)*(Imax*Kmax)+(K*Imax)+I] = ID1[(J+1)*(Imax*Kmax)+(K*Imax)+I] + mType;
		   else
			   ID1[(J+1)*(Imax*Kmax)+(K*Imax)+I]=8;

		break;

		case 'y':
		   //set face 2 (for EY)
			//check limit of plasma
		   if(ID2[J*(Imax*Kmax)+(K*Imax)+I]==0)
			   ID2[J*(Imax*Kmax)+(K*Imax)+I]=8;

		   if((ID2[J*(Imax*Kmax)+(K*Imax)+I] + mType)>8)
			   ID2[J*(Imax*Kmax)+(K*Imax)+I] = ID2[J*(Imax*Kmax)+(K*Imax)+I] + mType;
		   else
			   ID2[J*(Imax*Kmax)+(K*Imax)+I]=8;
//-----------------------------------------------------------------------------------
		   //check limit of plasma
		   if(ID2[J*(Imax*Kmax)+(K*Imax)+(I+1)]==0)
			   ID2[J*(Imax*Kmax)+(K*Imax)+(I+1)]=8;

		   if((ID2[J*(Imax*Kmax)+(K*Imax)+(I+1)] + mType)>8)
				ID2[J*(Imax*Kmax)+(K*Imax)+(I+1)] = ID2[J*(Imax*Kmax)+(K*Imax)+(I+1)] + mType;
		   else
				ID2[J*(Imax*Kmax)+(K*Imax)+(I+1)]=8;
//-----------------------------------------------------------------------------------

		   //if((ID2[J*(Imax*Kmax)+((K+1)*Imax)+(I+1)] + mType)>8)
     //      ID2[J*(Imax*Kmax)+((K+1)*Imax)+(I+1)] = ID2[J*(Imax*Kmax)+((K+1)*Imax)+(I+1)] + mType;

		   //if((ID2[J*(Imax*Kmax)+((K+1)*Imax)+I] + mType)>8)
     //      ID2[J*(Imax*Kmax)+((K+1)*Imax)+I] = ID2[J*(Imax*Kmax)+((K+1)*Imax)+I] + mType;
		break;
		case 'z':
		   //set face 3 (for EZ)		
			//check limit of plasma
		   if(ID3[J*(Imax*Kmax)+(K*Imax)+I]==0)
			   ID3[J*(Imax*Kmax)+(K*Imax)+I]=8;

		   if((ID3[J*(Imax*Kmax)+(K*Imax)+I] + mType)>8)
			   ID3[J*(Imax*Kmax)+(K*Imax)+I] = ID3[J*(Imax*Kmax)+(K*Imax)+I] + mType;
		   else
			   ID3[J*(Imax*Kmax)+(K*Imax)+I]=8;
//-----------------------------------------------------------------------------------
			//check limit of plasma
		   if(ID3[J*(Imax*Kmax)+(K*Imax)+(I+1)]==0)
			   ID3[J*(Imax*Kmax)+(K*Imax)+(I+1)]=8;

		   if((ID3[J*(Imax*Kmax)+(K*Imax)+(I+1)] + mType)>8)
			   ID3[J*(Imax*Kmax)+(K*Imax)+(I+1)] = ID3[J*(Imax*Kmax)+(K*Imax)+(I+1)] + mType;
		   else
			   ID3[J*(Imax*Kmax)+(K*Imax)+(I+1)]=8;
//-----------------------------------------------------------------------------------
			//check limit of plasma
		   if(ID3[(J+1)*(Imax*Kmax)+(K*Imax)+(I+1)]==0)
			   ID3[(J+1)*(Imax*Kmax)+(K*Imax)+(I+1)]=8;

		   if((ID3[(J+1)*(Imax*Kmax)+(K*Imax)+(I+1)] + mType)>8)
			   ID3[(J+1)*(Imax*Kmax)+(K*Imax)+(I+1)] = ID3[(J+1)*(Imax*Kmax)+(K*Imax)+(I+1)] + mType;
		   else
			   ID3[(J+1)*(Imax*Kmax)+(K*Imax)+(I+1)]=8;
//-----------------------------------------------------------------------------------
			//check limit of plasma
		   if(ID3[(J+1)*(Imax*Kmax)+(K*Imax)+I]==0)
			   ID3[(J+1)*(Imax*Kmax)+(K*Imax)+I]=8;

		   if((ID3[(J+1)*(Imax*Kmax)+(K*Imax)+I] + mType)>8)
               ID3[(J+1)*(Imax*Kmax)+(K*Imax)+I] = ID3[(J+1)*(Imax*Kmax)+(K*Imax)+I] + mType;
		   else
			   ID3[(J+1)*(Imax*Kmax)+(K*Imax)+I]=8;

		break;
		}
      }

	
void ChangeYeeCube (int I, int J,int K, int mType) 
{
		   //set face 1 (for EX)
           ID1_close[J*(Imax*Kmax)+(K*Imax)+I] = mType;
           ID1_close[J*(Imax*Kmax)+((K+1)*Imax)+I] = mType;
           ID1_close[(J+1)*(Imax*Kmax)+((K+1)*Imax)+I] = mType;
           ID1_close[(J+1)*(Imax*Kmax)+(K*Imax)+I] = mType;

		   //set face 2 (for EY)
           ID2_close[J*(Imax*Kmax)+(K*Imax)+I] = mType;
           ID2_close[J*(Imax*Kmax)+(K*Imax)+(I+1)] = mType;
           ID2_close[J*(Imax*Kmax)+((K+1)*Imax)+(I+1)] = mType;
           ID2_close[J*(Imax*Kmax)+((K+1)*Imax)+I] = mType;

		   //set face 3 (for EZ)
           ID3_close[J*(Imax*Kmax)+(K*Imax)+I] = mType;
           ID3_close[J*(Imax*Kmax)+(K*Imax)+(I+1)] = mType;
           ID3_close[(J+1)*(Imax*Kmax)+(K*Imax)+(I+1)] = mType;
           ID3_close[(J+1)*(Imax*Kmax)+(K*Imax)+I] = mType;
}

void CopyMedium() 
{
    for(i = 0; i < (Imax * Jmax * Kmax); i++) 
		{
			ID1_close[i] = ID1[i];
			ID2_close[i] = ID2[i];
			ID3_close[i] = ID3[i];
		}
}

void buildMedium() 
{		

	int iMstart, jMstart, kMstart;
	int iMend, jMend, kMend;
	int sliceWidth, SliceMas[J_SIZE], Mindex=9, count=0;


	iMstart=1;
	jMstart=3*(Jmax/4);
	kMstart=1;
	iMend=Imax-1;
	jMend=Jmax-1;
	kMend=Kmax-1;


	sliceWidth=(jMend-jMstart)/(numMaterials-10);
	for(i=0; i<Jmax; i++)
	{
		if(i<jMstart)
			SliceMas[i]=0;
		else
		{
			SliceMas[i]=Mindex;
			count++;
			if(count>=sliceWidth)
			{
				Mindex++;
				count=0;
			}
		}
	}

	for(i = 0; i < Imax-1; ++i) 
		{
        	for(j = 0; j < Jmax-1; ++j) 
			{
          		for(k = 0; k < Kmax-1; ++k) 
				{										
					yeeCube (i, j, k, SliceMas[j]);//PEC material					
				}
			}
		}

  //    	for(i = iMstart; i < iMend; ++i) 
		//{
  //      	for(j = jMstart; j < jMend; ++j) 
		//	{
  //        		for(k = kMstart; k < kMend; ++k) 
		//		{										
		//			yeeCube (i, j, k, 49);//PEC material					
		//		}
		//	}
		//}


	// вставить пластину между источником и приемником 
	
	//for(i = Imax/2-1; i < Imax/2+1; ++i)
	//{
	//	for(j = 0; j < Jmax/6; ++j) 
	//	{
	//		for(k = 0; k < Kmax-1; ++k) 
	//		{										
	//			yeeCube (i, j, k, 5);//PEC material					
	//		}
	//	}
	//}

	}

void buildPlateMedium() 
{
	int iMstart, jMstart, kMstart;
	int iMend, jMend, kMend, Mat;



	iMstart=1;
	jMstart=Kamera.WallPos;
	kMstart=1;

	iMend=Imax-1;
	jMend=Jmax-1;
	kMend=Kmax-1;

//--------------------
	k=0;
//--------------------

	Mat=5;

		for(i = iMstart; i < iMend; ++i) 
		{
			//--------------------
//				jMstart++;
			//--------------------

        	for(j = jMstart; j < jMend; ++j) 
			{
//          		for(k = kMstart; k < kMend; ++k) 
				{										
					yeeCube (i, j, k, Mat);//PEC material					
				}
			}
		}
}

void buildSphericalMedium() 
{		
	int EqTupe = SPHERE_MEDIUM_EQUOTION;
	float chis, znam, ra, alpha, beta;
	int ne0;
	int mod_medium;

	ne0=M_COEF;
	ra=EqCoef.R;
	alpha=EqCoef.Alpha;
	beta=EqCoef.Beta;

	k=0;

	switch (EqTupe)
	{
//----------------exponetial growing-------------------------------
	case 1:
	for(i=0;i<Imax-1;i++)
	{
		for(j=0;j<Jmax-1;j++)
		{
//			for(k=0;k<Kmax;k++)
			{
				chis=exp(sqrt( ((SphereCoef.P*pow((float)i-MediumCen.I,2))/pow(SphereCoef.A,2))+((SphereCoef.Q*pow((float)j-MediumCen.J,2))/pow(SphereCoef.B,2))+((SphereCoef.N*pow((float)k-MediumCen.K,2))/pow(SphereCoef.C,2)) ));
				znam=pow(1+chis,2);
				mod_medium=floor(M_COEF*(chis/znam))+8;
				if(mod_medium==8){}
					//yeeCube (i, j, k, 0);
				else
				{}//yeeCube (i, j, k, mod_medium);
			}
		}
	}
	break;
	//----------------ne equotion-------------------------------
	case 2:
	for(i=0;i<Imax-1;i+=1)
	{
		for(j=0;j<Jmax-1;j+=2)
		{
			//for(k=0;k<Kmax;k++)
			{
				chis=sqrt(((SphereCoef.P*pow((float)i-MediumCen.I,2))/pow(SphereCoef.A,2))+((SphereCoef.Q*pow((float)j-MediumCen.J,2))/pow(SphereCoef.B,2))+((SphereCoef.N*pow((float)k-MediumCen.K,2))/pow(SphereCoef.C,2)) );
				//znam=pow(1+chis,2);
				//mod_medium=floor(M_COEF*(chis/znam))+8;
				mod_medium=(ne0*pow((1-pow(((chis)/ra),alpha)),beta))+8;				
				if(mod_medium<8){}
					//yeeCube (i, j, k, 0);
				else
					yeeCubeFace (i, j, k, mod_medium, 'x');
			}
		}
	}

	for(i=0;i<Imax-1;i+=2)
	{
		for(j=0;j<Jmax-1;j+=1)
		{
			//for(k=0;k<Kmax;k++)
			{
				chis=sqrt(((SphereCoef.P*pow((float)i-MediumCen.I,2))/pow(SphereCoef.A,2))+((SphereCoef.Q*pow((float)j-MediumCen.J,2))/pow(SphereCoef.B,2))+((SphereCoef.N*pow((float)k-MediumCen.K,2))/pow(SphereCoef.C,2)) );
				//znam=pow(1+chis,2);
				//mod_medium=floor(M_COEF*(chis/znam))+8;
				mod_medium=(ne0*pow((1-pow(((chis)/ra),alpha)),beta))+8;				
				if(mod_medium<8){}
					//yeeCube (i, j, k, 0);
				else
					yeeCubeFace (i, j, k, mod_medium, 'y');
			}
		}
	}

	for(i=0;i<Imax-1;i+=2)
	{
		for(j=0;j<Jmax-1;j+=2)
		{
			//for(k=0;k<Kmax;k++)
			{
				chis=sqrt(((SphereCoef.P*pow((float)i-MediumCen.I,2))/pow(SphereCoef.A,2))+((SphereCoef.Q*pow((float)j-MediumCen.J,2))/pow(SphereCoef.B,2))+((SphereCoef.N*pow((float)k-MediumCen.K,2))/pow(SphereCoef.C,2)) );
				//znam=pow(1+chis,2);
				//mod_medium=floor(M_COEF*(chis/znam))+8;
				mod_medium=(ne0*pow((1-pow(((chis)/ra),alpha)),beta))+8;				
				if(mod_medium<8){}
					//yeeCube (i, j, k, 0);
				else
					yeeCubeFace (i, j, k, mod_medium, 'z');
			}
		}
	}
	break;

	default:
	break;
	}
}

void buildSphericalMediumFluc(int EqTupe, int I_cen, int J_cen, int K_cen, int P_coef, int Q_coef, int N_coef, float A_coef, float B_coef, float C_coef, float pne0, float ra, float alpha, float beta, float angle_x, float angle_y, float angle_z, int sign) 
{		
	//в аргументы добавил следующие переменные
	// int angle_x, int angle_y,int angle_z, int sign
	// все углы измеряются в градусах
	// int sign - знак, в который направлено распределение. 1 или -1.

	int i_,j_,k_;
	float chis, znam;
	int mod_medium;
	int ne0;
	i=8;
	while((i<numMaterials)&&(pne[i]<pne0))
	{
		ne0=i-8; 
		i++;
	}

//пересчет в радианы	
	angle_x=angle_x*3.14159/180;
	angle_y=angle_y*3.14159/180;
	angle_z=angle_z*3.14159/180;
	
	k=0;

	switch (EqTupe)
	{
	//----------------ne equotion-------------------------------
	case 2:
	for(i=0;i<Imax-1;i+=1)
	{
		for(j=0;j<Jmax-1;j+=2)
		{
			//for(k=0;k<Kmax-1;k+=2)
			{
				i_=(k-K_cen)*sin(angle_y) + (i-I_cen)*cos(angle_y)*cos(angle_z) - (j-J_cen)*cos(angle_y)*sin(angle_z);
				j_=(i-I_cen)*(cos(angle_x)*sin(angle_z) + cos(angle_z)*sin(angle_x)*sin(angle_y)) + (j-J_cen)*(cos(angle_x)*cos(angle_z) - sin(angle_x)*sin(angle_y)*sin(angle_z)) - (k-K_cen)*cos(angle_y)*sin(angle_x);
				k_=(i-I_cen)*(sin(angle_x)*sin(angle_z) - cos(angle_x)*cos(angle_z)*sin(angle_y)) + (j-J_cen)*(cos(angle_z)*sin(angle_x) + cos(angle_x)*sin(angle_y)*sin(angle_z)) + (k-K_cen)*cos(angle_x)*cos(angle_y);
				
				chis=sqrt(((P_coef*pow((float)i_,2))/pow(A_coef,2))+((Q_coef*pow((float)j_,2))/pow(B_coef,2))+((N_coef*pow((float)k_,2))/pow(C_coef,2)) );

				mod_medium=floor(ne0*pow((1-pow(((chis)/(float)ra),alpha)),beta));			
				if(mod_medium<0){}
					//yeeCube (i, j, k, 0);
				else
					addYeeCube (i, j, k, sign*mod_medium, 'x');
			}
		}
	}

	for(i=0;i<Imax-1;i+=2)
	{
		for(j=0;j<Jmax-1;j+=1)
		{
			//for(k=0;k<Kmax-1;k+=2)
			{
				i_=(k-K_cen)*sin(angle_y) + (i-I_cen)*cos(angle_y)*cos(angle_z) - (j-J_cen)*cos(angle_y)*sin(angle_z);
				j_=(i-I_cen)*(cos(angle_x)*sin(angle_z) + cos(angle_z)*sin(angle_x)*sin(angle_y)) + (j-J_cen)*(cos(angle_x)*cos(angle_z) - sin(angle_x)*sin(angle_y)*sin(angle_z)) - (k-K_cen)*cos(angle_y)*sin(angle_x);
				k_=(i-I_cen)*(sin(angle_x)*sin(angle_z) - cos(angle_x)*cos(angle_z)*sin(angle_y)) + (j-J_cen)*(cos(angle_z)*sin(angle_x) + cos(angle_x)*sin(angle_y)*sin(angle_z)) + (k-K_cen)*cos(angle_x)*cos(angle_y);

				
				chis=sqrt(((P_coef*pow((float)i_,2))/pow(A_coef,2))+((Q_coef*pow((float)j_,2))/pow(B_coef,2))+((N_coef*pow((float)k_,2))/pow(C_coef,2)) );
				mod_medium=floor(ne0*pow((1-pow(((chis)/(float)ra),alpha)),beta));				
				if(mod_medium<0){}
					//yeeCube (i, j, k, 0);
				else
					addYeeCube (i, j, k, sign*mod_medium, 'y');
			}
		}
	}

	for(i=0;i<Imax-1;i+=2)
	{
		for(j=0;j<Jmax-1;j+=2)
		{
			//for(k=0;k<Kmax-1;k+=1)
			{	
				i_=(k-K_cen)*sin(angle_y) + (i-I_cen)*cos(angle_y)*cos(angle_z) - (j-J_cen)*cos(angle_y)*sin(angle_z);
				j_=(i-I_cen)*(cos(angle_x)*sin(angle_z) + cos(angle_z)*sin(angle_x)*sin(angle_y)) + (j-J_cen)*(cos(angle_x)*cos(angle_z) - sin(angle_x)*sin(angle_y)*sin(angle_z)) - (k-K_cen)*cos(angle_y)*sin(angle_x);
				k_=(i-I_cen)*(sin(angle_x)*sin(angle_z) - cos(angle_x)*cos(angle_z)*sin(angle_y)) + (j-J_cen)*(cos(angle_z)*sin(angle_x) + cos(angle_x)*sin(angle_y)*sin(angle_z)) + (k-K_cen)*cos(angle_x)*cos(angle_y);

				
				chis=sqrt(((P_coef*pow((float)i_,2))/pow(A_coef,2))+((Q_coef*pow((float)j_,2))/pow(B_coef,2))+((N_coef*pow((float)k_,2))/pow(C_coef,2)) );
				mod_medium=floor(ne0*pow((1-pow(((chis)/(float)ra),alpha)),beta));				
				if(mod_medium<0){}
					//yeeCube (i, j, k, 0);
				else
					addYeeCube (i, j, k, sign*mod_medium, 'z');
			}
		}
	}


	std::cout<<"\nne0="<<ne0<<std::endl;
	
	break;

	default:
	break;
	}
}


void buildTorusMedium() 
{
	int EqType = SPHERE_MEDIUM_EQUOTION, sign;		// можно ввести TORUS_MEDIUM_EQUOTION во избежание путаницы
	float chis, znam, a, alpha, beta, k, nea, Z0, R0, r_, p_x, p_y, p_z, phi, K;
	int ne0;		// странно что тут тип int, или я что то недопопял когда полумал что nea вроде как float значение? (строка взята из имующейся функции для сферы)
	int mod_medium;	// тут тоже int а используется с усечением float до int, хоть и знак в условии >=, но все же, путаницы добавляет
	
	R0 = Torus_big_r;
	Z0 = MediumCen.K;
	ne0 = M_COEF;
//	nea = SOME_NEA;			// я не знаю какая константа будет отвечать за "переход на ступеньку" и вероятно она должна быть типа int
	nea = 8;			// я не знаю какая константа будет отвечать за "переход на ступеньку" и вероятно она должна быть типа int
	alpha = EqCoef.Alpha;
	beta = EqCoef.Beta;
	a = EqCoef.R;
	K = SphereCoef.C;		// K заглавная т.к. k обьявлена в самом начале как int в качестве итератора циклов

	k=1;

	for (i = 0; i < Imax-1; i++) {
		for (j = 0; j < Jmax-1; j++) {
			//for (k = 0; k < Kmax; k++) 
			{
				p_x = i - MediumCen.I;		// p_* только для удобства введены
				p_y = j - MediumCen.J;		// перенос системы координат
				p_z = k - Z0;			// Z0 используется только тут, есть ли смысл вводить Z0 вообще?
				//p_z = k+1;			// 2D Var
				 
				if (p_x > 0) {
					phi = atan(p_y/p_x);
				} else {
					sign = (p_y >= 0) - (p_y < 0);
					if (p_x < 0) {
						phi = atan(p_y/p_x) + sign*M_PI;
					} else {
						phi = sign*(M_PI/2);
					}
				}
				
				p_x = p_x*cos(phi) + p_y*sin(phi) - R0;		// перенос и поворот СК; в р-те получаем эллипс в сечении
//				p_y = -p_x*sin(phi) + p_y*cos(phi);			// тора на развороте xOz с центром в начале новой СК		\ p_y дальше не участвует в вычислениях, лишняя операция
				p_z /= k;		// только для укорения, т.к. строчкой ниже отказался от pow()
				
				r_ = sqrt(p_x*p_x + p_z*p_z);		// не использую pow() специально - значительный прирост по быстродействию
				
				switch(EqType) {		// свич поместил внутрь циклов дабы избежать дублирования кода. На скорости выполнения не сказывается, проверял.
					case 1:
						chis = exp(r_);
						znam = (1 + chis)*(1 + chis);		// опять же при использовании pow() раза в 4 медленнее
						mod_medium = (int)floor(M_COEF*(chis/znam)) + 8;		// case 1: содрал с имеющейся функции для сферы
						if (mod_medium != 8) {
							yeeCube(i, j, k, mod_medium);
						}
					break;
					case 2:
						mod_medium = (ne0 - nea)*pow(1 - pow(r_/a, alpha), beta) + nea;
						if (mod_medium >= 8) {
							yeeCube(i, j, k, mod_medium);
						}
					break;
				}
			}
		}
	}
}

void calculate2Darrai()
{
	int sizeI=SourceAESA.PI/DS;
	int sizeK=SourceAESA.PK/DS;

	float C1=3e11;
	int di, dk;

	if(DI_AUTO==ENABLE)
	{
	di=floor((C1/(SourcePulse.freqA*1e9)/2)/DS);
	dk=di;
	}
	else
	{
	di=SourceAESA.DI/DS;
	dk=SourceAESA.DK/DS;
	}
	qantI=floor(sizeI/di)+1;
	qantK=floor(sizeK/dk)+1;

	int Istart=floor(SourceCen.I-sizeI/2);
	int Kstart=floor(SourceCen.K-sizeK/2);

	arrayI = (int *)malloc((qantI) * sizeof(int));
	arrayK = (int *)malloc((qantK) * sizeof(int));

	if(COMP_DETECT==ENABLE)	
	CompDet = (float *)malloc((nMax * qantI * qantK) * sizeof(float));

	printf("Source position:\n ");
	for(i = 0; i < qantI; i++) 
	{
		arrayI[i]=Istart+di*i;
		printf("%d ", arrayI[i]);
	}
	printf("\n");
	for(i = 0; i < qantK; i++) 
	{
		arrayK[i]=Kstart+dk*i;
		printf("%d ", arrayK[i]);
	}

}

void calculateDetect2Darrai()
{
	int sizeI=DetectAESA.PI/DS;
	int sizeK=DetectAESA.PK/DS;

	float C1=3e11;
	int di, dk;

	if(DD_AUTO==ENABLE)
	{
	di=floor((C1/(SourcePulse.freqA*1e9)/2)/DS);
	dk=di;
	}
	else
	{
	di=DetectAESA.DI/DS;
	dk=DetectAESA.DK/DS;
	}
	det_qantI=floor(sizeI/di)+1;
	det_qantK=floor(sizeK/dk)+1;

	detect = (float *)malloc((nMax * det_qantI * det_qantK) * sizeof(float));
	if(Order==1)
	{
		infDet = (float *)malloc((nMax * det_qantI * det_qantK) * sizeof(float));

		if(SAVE_COMPONENTS==ENABLE)
		{
			det_Ex = (float *)malloc((nMax * det_qantI * det_qantK) * sizeof(float));
			det_Ey = (float *)malloc((nMax * det_qantI * det_qantK) * sizeof(float));
			det_Ez = (float *)malloc((nMax * det_qantI * det_qantK) * sizeof(float));
			det_Hx = (float *)malloc((nMax * det_qantI * det_qantK) * sizeof(float));
			det_Hy = (float *)malloc((nMax * det_qantI * det_qantK) * sizeof(float));
			det_Hz = (float *)malloc((nMax * det_qantI * det_qantK) * sizeof(float));

			if(useRegime.S2D==ENABLE)
				{
					infDet_Ex = (float *)malloc((nMax * det_qantI * det_qantK) * sizeof(float));
					infDet_Ey = (float *)malloc((nMax * det_qantI * det_qantK) * sizeof(float));
					infDet_Ez = (float *)malloc((nMax * det_qantI * det_qantK) * sizeof(float));
					infDet_Hx = (float *)malloc((nMax * det_qantI * det_qantK) * sizeof(float));
					infDet_Hy = (float *)malloc((nMax * det_qantI * det_qantK) * sizeof(float));
					infDet_Hz = (float *)malloc((nMax * det_qantI * det_qantK) * sizeof(float));
				}
		}
	}

	int Istart=floor(DetectCen.I-sizeI/2);
	int Kstart=floor(DetectCen.K-sizeK/2);

	det_arrayI = (int *)malloc((det_qantI) * sizeof(int));
	det_arrayK = (int *)malloc((det_qantK) * sizeof(int));
	printf("\nDetector position:\n ");
	for(i = 0; i < det_qantI; i++) 
	{
		det_arrayI[i]=Istart+di*i;
		printf("%d ", det_arrayI[i]);
	}
	printf("\n");
	for(i = 0; i < det_qantK; i++) 
	{
		det_arrayK[i]=Kstart+dk*i;
		printf("%d ", det_arrayK[i]);
	}

}

void calculateRCdetect2Darrai()
{
	//std::ofstream outfile; 
	//outfile.open("RcDet_data.txt");

	using namespace std;

	fstream outfile("RcDet_data.txt", fstream::out);

	int sizeI=RCdetect.PI;
	int sizeK=RCdetect.PK;

	int di, dk;

	di=RCdetect.DI;
	dk=RCdetect.DK;

	RC_qantI=floor(sizeI/di)+1;
	RC_qantK=floor(sizeK/dk)+1;

	PspdLine = (int *)malloc ((nMax * RC_qantI * RC_qantK)*sizeof(int));
	PspdFiltred = (float *)malloc ((nMax * RC_qantI * RC_qantK)*sizeof(float));

	int Istart=floor(RCdetectCen.I-sizeI/2);
	int Kstart=floor(RCdetectCen.K-sizeK/2);

	RCdetArrayI = (int *)malloc((RC_qantI) * sizeof(int));
	RCdetArrayK = (int *)malloc((RC_qantK) * sizeof(int));
	printf("\nRC line position:\n ");
	for(i = 0; i < RC_qantI; i++) 
	{
		RCdetArrayI[i]=Istart+di*i;
		printf("%d ", RCdetArrayI[i]);
		outfile<<RCdetArrayI[i]<<" ";
	}
	printf("\n");
	outfile<<std::endl;
	for(i = 0; i < RC_qantK; i++) 
	{
		RCdetArrayK[i]=Kstart+dk*i;
		printf("%d ", RCdetArrayK[i]);
		outfile<<RCdetArrayK[i]<<" "; 
	}

	outfile.close();
}

void buildSourceProfile() 
{		

	SourceP = (float *)malloc((nMax) * sizeof(float));

	FILE *PsourseP;
	PsourseP=fopen("SourceP.txt","w");
	

	int platoW=SourcePulse.Plato;		//300
	int frontW=SourcePulse.Front/2;		//150
	float *Namp;
	Namp = (float *)malloc((SourcePulse.Front) * sizeof(float));

	double qa=1;

	//fCoef=2*3.14*(SourcePulse.freqA*1E9)*dt*DS;	//0.2416*DS ->20GHz
	//fCoef2=2*3.14*(SourcePulse.freqB*1E9)*dt*DS;	//0.2416*DS ->20GHz

	if(SourcePulse.chirp==1)
	{
		fCoef=SourcePulse.freqA*1E9;	
		fCoef2=SourcePulse.freqB*1E9;	
	}
	else
	{
		fCoef=SourcePulse.freqA*1E9;	
		fCoef2=SourcePulse.freqA*1E9;	
	}
	fPulseW=(platoW+2*frontW)*dt;

	
	double LocalCoef=0, UpperCoef=0, lowerCoef;
//Ramp=amp*(exp(-qa.*(time-frontW).^2/((frontW/2.5)^2)) );
	for(i=0;i<(2*frontW);i++) 
	{
		UpperCoef=pow(double(i-frontW),2.);
		LocalCoef=double(frontW)/double(2.5);
		lowerCoef=pow(LocalCoef,2.);
		Namp[i]=amp*(exp(-qa*UpperCoef/lowerCoef));
	}


//for i=1:(frontW)
//    Nes(i)=Ramp(i);
//end
//for i=frontW+1:(frontW+platoW)
//    Nes(i)=amp;
//end
//for i=(1):frontW
//    Nes(frontW+platoW+i)=Ramp((frontW-1)+i);
//end

	for(j=0,i=0;(i<frontW)&&(j<nMax);i++,j++) 
		SourceP[i]=Namp[i];
	for(i=frontW;(i<frontW+platoW)&&(j<nMax);i++,j++) 
		SourceP[i]=amp;
	for(i=0;(i<frontW)&&(j<nMax);i++,j++)
		SourceP[frontW+platoW+i]=Namp[(frontW)+i];

	

//-------------------------------------------------------
	//alpha=9.1;
	//beta=1.1;
	//pulseW=600;
	//fCoef=0.15;

	//float LocalCoef, TimeCoef=0;
	//float Namp[300], Nes[600];

	//for(i=0;i<nMax;i++) 
	//{
	//	SourceP[i]=0;
	//}

//Lamp=amp*abs((1-qa.*(time/a).^alpha).^beta);

	//for(i=0;i<(pulseW/2);i++) 
	//{
	//	TimeCoef=i/300.0;
	//	LocalCoef=pow(TimeCoef,alpha);
	//	Namp[i]=amp*abs(1-(pow(LocalCoef,beta)));
	//
	//}

//---------------DEBUG--------------------------
//for(k = 0; k < (pulseW/2); k++) 
//		{			
//			fprintf(PsourseP,"%f ", Namp[k]);
//		}
//fprintf(PsourseP,"\n");
//---------------END OF DEBUG-------------------


	//for(i=0;i<(pulseW/2);i++) 
	//{
	//	Nes[i]=Namp[((pulseW/2)-1)-i];
	//	Nes[(pulseW/2)+i]=Namp[i];
	//}

	//for(i=0;i<pulseW;i++) 
	//{
	//	SourceP[i]=Nes[i];
	//}


//---------------DEBUG--------------------------
for(k = 0; k < nMax; k++) 
		{			
			fprintf(PsourseP,"%f\n", SourceP[k]);
		}
//---------------END OF DEBUG-------------------

fclose(PsourseP);
free(Namp);
}

void buildCubikAntenna(int Antena_Tupe) 
{		
int YA_length=A_Y_LENGTH/DS;
int Ant_thickness=A_THICKNESS/DS;
int antenaSize=A_SIZE/DS;
int InitialSizeX=A_INITIAL_SIZE/DS;
//int XZ_Size=antenaSize*2;
int S_koef;
int Curve_koef=YA_length/(antenaSize-InitialSizeX);				//D_Y_LENGTH/(D_SIZE-D_INITIAL_SIZE)
int Beg_Antena=10/DS;			//Curve_koef*10
int Loc_count=0;

int RuporStart=SourceCen.J;

switch (Antena_Tupe)
{
//------------------Square Tube------------------------------------------
case 1:
//	for(i = SourceCen.I-antenaSize; i < SourceCen.I+antenaSize; i++) 
//		{			
//			//for(j = SourceCen.J-antenaSize; j < SourceCen.J+YA_length; j++) 
//			for(j = 0; j < SourceCen.J+YA_length; j++) 
//			{
//				for(k = SourceCen.K-antenaSize-Ant_thickness; (k < SourceCen.K-antenaSize) && (k<Kmax) && (k>0); k++) 
//					//k=SourceCen.K-antenaSize;
//					{
//						yeeCube (i, j, k, A_MATERIAL);
//					}
//					for(k = SourceCen.K+antenaSize; (k < SourceCen.K+antenaSize+Ant_thickness) && (k<Kmax) && (k>0); k++) 
//					//k=SourceCen.K+antenaSize;
//					{
//						yeeCube (i, j, k, A_MATERIAL);
//					}
//			}
//		}
//
//for(k = SourceCen.K-antenaSize; (k < SourceCen.K+antenaSize) && (k<Kmax) && (k>0); k++) 
		{			
			//for(j = SourceCen.J-antenaSize; j < SourceCen.J+YA_length; j++) 
			for(j = 0; j < SourceCen.J+YA_length; j++)
			{
					for(i = SourceCen.I-antenaSize-Ant_thickness; i < SourceCen.I-antenaSize; i++) 
					//i=SourceCen.I-antenaSize;
					{
						yeeCube (i, j, k, A_MATERIAL);
					}
					//i=SourceCen.I+antenaSize;
					for(i = SourceCen.I+antenaSize; i < SourceCen.I+antenaSize+Ant_thickness; i++) 
					{
						yeeCube (i, j, k, A_MATERIAL);
					}
			}
		}
	break;
//------------------Hole in the wall------------------------------------------------
case 2:
	for(i = PML_SIZE; i < Imax-PML_SIZE; i++) 
		{			
			//for(j = SourceCen.J-antenaSize; j < SourceCen.J+YA_length; j++) 
			for(j = PML_SIZE; j < SourceCen.J+YA_length; j++) 
			{
					for(k = PML_SIZE; k < Kmax-PML_SIZE; k++) 
					if((i > SourceCen.I-antenaSize)&&(i < SourceCen.I+antenaSize)&&(k > SourceCen.K-antenaSize)&&(k < SourceCen.K+antenaSize))
					{
						
					}
					else
					{
						yeeCube (i, j, k, A_MATERIAL);
					}
					
			}
		}
break;

//------------------Square conical antenna------------------------------------------------
case 3:
S_koef=antenaSize+Ant_thickness;
Loc_count=0;
//for(j = SourceCen.J-Beg_Antena; j < SourceCen.J+YA_length; j++) 
//		{		
//		if(S_koef>0 && Curve_koef==Loc_count)
//		{
//		S_koef--;
//		Loc_count=0;
//		}
//		else
//		{
//		Loc_count++;
//		}
//		for(i = SourceCen.I-antenaSize+S_koef; i < SourceCen.I+antenaSize-S_koef; i++) 
//			{			
//				for(k = SourceCen.K-antenaSize-Ant_thickness; k < SourceCen.K-antenaSize+S_koef; k++) 
//				{
//					yeeCube (i, j, k, A_MATERIAL);
//				}
//				for(k = SourceCen.K+antenaSize-S_koef; k < SourceCen.K+antenaSize+Ant_thickness; k++) 
//				{
//					yeeCube (i, j, k, A_MATERIAL);
//				}
//			}
//		}

S_koef=antenaSize+Ant_thickness;
Loc_count=0;
k=0;
for(j = SourceCen.J-Beg_Antena; j < SourceCen.J+YA_length; j++) 
		{		
		if(S_koef>0 && Curve_koef==Loc_count)
		{
		S_koef--;
		Loc_count=0;
		}
		else
		{
		Loc_count++;
		}
		//for(k = SourceCen.K-antenaSize+S_koef; k < SourceCen.K+antenaSize-S_koef; k++) 
			{
					for(i = SourceCen.I-antenaSize-Ant_thickness; i < SourceCen.I-antenaSize+S_koef; i++) 
					//i=SourceCen.I-antenaSize;
					{
						yeeCube (i, j, k, A_MATERIAL);
					}
					//i=SourceCen.I+antenaSize;
					for(i = SourceCen.I+antenaSize-S_koef; i < SourceCen.I+antenaSize+Ant_thickness; i++) 
					{
						yeeCube (i, j, k, A_MATERIAL);
					}
			}
		}
	break;

//-----------------Hole after source------------------------------------------------
case 4:
	for(i = PML_SIZE; i < Imax-PML_SIZE; i++) 
		{			
			//for(j = SourceCen.J-antenaSize; j < SourceCen.J+YA_length; j++) 
			for(j =SourceCen.J+YA_length-Ant_thickness ; j < SourceCen.J+YA_length; j++) 
			{
					for(k = PML_SIZE; k < Kmax-PML_SIZE; k++) 
					if((i > SourceCen.I-antenaSize)&&(i < SourceCen.I+antenaSize)&&(k > SourceCen.K-antenaSize)&&(k < SourceCen.K+antenaSize))
					{
								
					}
					else
					{
						yeeCube (i, j, k, A_MATERIAL);
					}
					
			}
		}
break;

//------------------Square conical antenna------------------------------------------------
case 5:
k=0;
//for(i = SourceCen.I-InitialSizeX; i < SourceCen.I+InitialSizeX; i++) 
//		{			
//			for(j = 0; j < SourceCen.J; j++) 
//			{
//					for(k = SourceCen.K-InitialSizeX-Ant_thickness; k < SourceCen.K-InitialSizeX; k++) 
//					{
//						yeeCube (i, j, k, A_MATERIAL);
//					}
//					for(k = SourceCen.K+InitialSizeX; k < SourceCen.K+InitialSizeX+Ant_thickness; k++) 
//					{
//						yeeCube (i, j, k, A_MATERIAL);
//					}
//			}
//		}
//
//for(k = SourceCen.K-InitialSizeX; k < SourceCen.K+InitialSizeX; k++) 
		{			
			//for(j = SourceCen.J-antenaSize; j < SourceCen.J+YA_length; j++) 
			for(j = 0; j < RuporStart; j++)
			{
					for(i = SourceCen.I-InitialSizeX-Ant_thickness; i < SourceCen.I-InitialSizeX; i++) 
					//i=SourceCen.I-antenaSize;
					{
						yeeCube (i, j, k, A_MATERIAL);
					}
					//i=SourceCen.I+antenaSize;
					for(i = SourceCen.I+InitialSizeX; i < SourceCen.I+InitialSizeX+Ant_thickness; i++) 
					{
						yeeCube (i, j, k, A_MATERIAL);
					}
			}
		}
//-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

//S_koef=(antenaSize+Ant_thickness)-A_INITIAL_SIZE;
S_koef=(antenaSize)-InitialSizeX;
Loc_count=0;

//for(j = SourceCen.J; j < SourceCen.J+YA_length; j++) 
//		{		
//		if(S_koef>0 && Curve_koef==Loc_count)
//		{
//		S_koef--;
//		Loc_count=0;
//		}
//		else
//		{
//		Loc_count++;
//		}
//		for(i = SourceCen.I-antenaSize+S_koef; i < SourceCen.I+antenaSize-S_koef; i++) 
//			{			
//				for(k = SourceCen.K-antenaSize-Ant_thickness; k < SourceCen.K-antenaSize+S_koef; k++) 
//				{
//					yeeCube (i, j, k, A_MATERIAL);
//				}
//				for(k = SourceCen.K+antenaSize-S_koef; k < SourceCen.K+antenaSize+Ant_thickness; k++) 
//				{
//					yeeCube (i, j, k, A_MATERIAL);
//				}
//			}
//		}

//S_koef=(antenaSize+Ant_thickness)-A_INITIAL_SIZE;
S_koef=(antenaSize)-InitialSizeX;
Loc_count=0;
for(j = RuporStart; j < RuporStart+YA_length; j++) 
		{		
		if(S_koef>0 && Curve_koef==Loc_count)
		{
		S_koef--;
		Loc_count=0;
		}
		else
		{
		Loc_count++;
		}
		//for(k = SourceCen.K-antenaSize+S_koef; k < SourceCen.K+antenaSize-S_koef; k++) 
			{
					for(i = SourceCen.I-antenaSize-Ant_thickness; i < SourceCen.I-antenaSize+S_koef; i++) 
					//i=SourceCen.I-antenaSize;
					{
						yeeCube (i, j, k, A_MATERIAL);
					}
					//i=SourceCen.I+antenaSize;
					for(i = SourceCen.I+antenaSize-S_koef; i < SourceCen.I+antenaSize+Ant_thickness; i++) 
					{
						yeeCube (i, j, k, A_MATERIAL);
					}
			}
		}
	break;

//------------------Square conical antenna closed------------------------------------------------
case 6:

for(i = SourceCen.I-InitialSizeX; i < SourceCen.I+InitialSizeX; i++) 
		{			
			for(j = 0; j < SourceCen.J; j++) 
			{
					for(k = SourceCen.K-InitialSizeX-Ant_thickness; k < SourceCen.K-InitialSizeX; k++) 
					{
						yeeCube (i, j, k, A_MATERIAL);
					}
					for(k = SourceCen.K+InitialSizeX; k < SourceCen.K+InitialSizeX+Ant_thickness; k++) 
					{
						yeeCube (i, j, k, A_MATERIAL);
					}
			}
		}

for(k = SourceCen.K-InitialSizeX; k < SourceCen.K+InitialSizeX; k++) 
		{			
			//for(j = SourceCen.J-antenaSize; j < SourceCen.J+YA_length; j++) 
			for(j = 0; j < SourceCen.J; j++)
			{
					for(i = SourceCen.I-InitialSizeX-Ant_thickness; i < SourceCen.I-InitialSizeX; i++) 
					//i=SourceCen.I-antenaSize;
					{
						yeeCube (i, j, k, A_MATERIAL);
					}
					//i=SourceCen.I+antenaSize;
					for(i = SourceCen.I+InitialSizeX; i < SourceCen.I+InitialSizeX+Ant_thickness; i++) 
					{
						yeeCube (i, j, k, A_MATERIAL);
					}
			}
		}

//-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
for(j = PML_SIZE+10; j < PML_SIZE+20; j++) 
		{			
			for(i = SourceCen.I-InitialSizeX; i < SourceCen.I+InitialSizeX; i++) 
			{
					for(k = SourceCen.K-InitialSizeX-Ant_thickness; k < SourceCen.K+InitialSizeX+Ant_thickness; k++) 
					{
						yeeCube (i, j, k, A_MATERIAL);
					}
			}
		}
//-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

//S_koef=(antenaSize+Ant_thickness)-A_INITIAL_SIZE;
S_koef=(antenaSize)-InitialSizeX;
Loc_count=0;

for(j = SourceCen.J; j < SourceCen.J+YA_length; j++) 
		{		
		if(S_koef>0 && Curve_koef==Loc_count)
		{
		S_koef--;
		Loc_count=0;
		}
		else
		{
		Loc_count++;
		}
		for(i = SourceCen.I-antenaSize+S_koef; i < SourceCen.I+antenaSize-S_koef; i++) 
			{			
				for(k = SourceCen.K-antenaSize-Ant_thickness; k < SourceCen.K-antenaSize+S_koef; k++) 
				{
					yeeCube (i, j, k, A_MATERIAL);
				}
				for(k = SourceCen.K+antenaSize-S_koef; k < SourceCen.K+antenaSize+Ant_thickness; k++) 
				{
					yeeCube (i, j, k, A_MATERIAL);
				}
			}
		}

//S_koef=(antenaSize+Ant_thickness)-A_INITIAL_SIZE;
S_koef=(antenaSize)-InitialSizeX;
Loc_count=0;
for(j = SourceCen.J; j < SourceCen.J+YA_length; j++) 
		{		
		if(S_koef>0 && Curve_koef==Loc_count)
		{
		S_koef--;
		Loc_count=0;
		}
		else
		{
		Loc_count++;
		}
		for(k = SourceCen.K-antenaSize+S_koef; k < SourceCen.K+antenaSize-S_koef; k++) 
			{
					for(i = SourceCen.I-antenaSize-Ant_thickness; i < SourceCen.I-antenaSize+S_koef; i++) 
					//i=SourceCen.I-antenaSize;
					{
						yeeCube (i, j, k, A_MATERIAL);
					}
					//i=SourceCen.I+antenaSize;
					for(i = SourceCen.I+antenaSize-S_koef; i < SourceCen.I+antenaSize+Ant_thickness; i++) 
					{
						yeeCube (i, j, k, A_MATERIAL);
					}
			}
		}
	break;

default:
	printf("\nunknown antenna tupe\n");
	break;
}
}

void MakeCloseTheDoor()
{		
int YA_length=120;
int Ant_thickness=20;
int antenaSize=5;
int S_koef;



//-----------------Close hole after source------------------------------------------------
	for(i = PML_SIZE; i < Imax-PML_SIZE; i++) 
		{			
			//for(j = SourceCen.J-antenaSize; j < SourceCen.J+YA_length; j++) 
			for(j =SourceCen.J+YA_length-Ant_thickness ; j < SourceCen.J+YA_length; j++) 
			{
					for(k = PML_SIZE; k < Kmax-PML_SIZE; k++) 
					if((i > SourceCen.I-antenaSize-1)&&(i < SourceCen.I+antenaSize+1)&&(k > SourceCen.K-antenaSize-1)&&(k < SourceCen.K+antenaSize+1))
					{
						ChangeYeeCube (i, j, k, 1);		
					}
					//else
					//{						
					//}
					
			}
		}
}


void buildCubikDetector(int Antena_Tupe) 
{		
int YA_length=D_Y_LENGTH/DS;
int Ant_thickness=D_THICKNESS/DS;
int antenaSize=D_SIZE/DS;
//int XZ_Size=antenaSize*2;
int InitialSizeX=D_INITIAL_SIZE/DS;
int S_koef;
int Curve_koef=YA_length/(antenaSize-InitialSizeX);
int Beg_Antena=Curve_koef*8;
int Loc_count=0;

switch (Antena_Tupe)
{
//------------------Square Tube------------------------------------------
case 1:
	for(i = DetectCen.I-antenaSize; i < DetectCen.I+antenaSize; i++) 
		{			
			//for(j = SourceCen.J-antenaSize; j < SourceCen.J+YA_length; j++) 
			for(j = 0; j < SourceCen.J+YA_length; j++) 
			{
					for(k = DetectCen.K-antenaSize-Ant_thickness; k < DetectCen.K-antenaSize; k++) 
					//k=SourceCen.K-antenaSize;
					{
						yeeCube (i, j, k, D_MATERIAL);
					}
					for(k = DetectCen.K+antenaSize; k < DetectCen.K+antenaSize+Ant_thickness; k++) 
					//k=SourceCen.K+antenaSize;
					{
						yeeCube (i, j, k, D_MATERIAL);
					}
			}
		}

for(k = DetectCen.K-antenaSize; k < DetectCen.K+antenaSize; k++) 
		{			
			//for(j = SourceCen.J-antenaSize; j < SourceCen.J+YA_length; j++) 
			for(j = 0; j < SourceCen.J+YA_length; j++)
			{
					for(i = DetectCen.I-antenaSize-Ant_thickness; i < DetectCen.I-antenaSize; i++) 
					//i=SourceCen.I-antenaSize;
					{
						yeeCube (i, j, k, D_MATERIAL);
					}
					//i=SourceCen.I+antenaSize;
					for(i = DetectCen.I+antenaSize; i < DetectCen.I+antenaSize+Ant_thickness; i++) 
					{
						yeeCube (i, j, k, D_MATERIAL);
					}
			}
		}
	for(i = DetectCen.I-antenaSize+1; i < DetectCen.I+antenaSize-1; i++) 
		{			
			for(j = 0; j < SourceCen.J+YA_length; j++) 
			{
					for(k = DetectCen.K-antenaSize+1; k < DetectCen.K+antenaSize-1; k++)
					{											
						yeeCube (i, j, k, 0);					
					}
			}
		}
	break;
//------------------Square Tube whith 45 grad border------------------------------------------
case 2:
S_koef=0;
Loc_count=0;
	for(i = DetectCen.I-(2*antenaSize+Ant_thickness); i < DetectCen.I; i++) 
		{
			S_koef++;
			for(k = DetectCen.K-S_koef; k < DetectCen.K+S_koef; k++) 
			{
				for(j = 0; j < SourceCen.J+YA_length; j++) 
					{
						yeeCube (i, j, k, D_MATERIAL);
					}
			}
		}
	S_koef=0;
	for(i = DetectCen.I+(2*antenaSize+Ant_thickness); i > DetectCen.I; i--) 
		{
			S_koef++;
			for(k = DetectCen.K-S_koef; k < DetectCen.K+S_koef; k++) 			
			{
				for(j = 0; j < SourceCen.J+YA_length; j++) 
					{
						yeeCube (i, j, k, D_MATERIAL);
					}
			}
		}


	for(i = DetectCen.I-antenaSize+1; i < DetectCen.I+antenaSize-1; i++) 
		{			
			for(j = 0; j < SourceCen.J+YA_length; j++) 
			{
					for(k = DetectCen.K-antenaSize+1; k < DetectCen.K+antenaSize-1; k++)
					{											
						yeeCube (i, j, k, 0);					
					}
			}
		}
	break;

//------------------Square conical detector antenna------------------------------------------------
case 3:

//for(i = DetectCen.I-InitialSizeX; i < DetectCen.I+InitialSizeX; i++) 
//		{			
//			for(j = 0; j < DetectCen.J; j++) 
//			{
//					for(k = DetectCen.K-InitialSizeX-Ant_thickness; k < DetectCen.K-InitialSizeX; k++) 
//					{
//						yeeCube (i, j, k, D_MATERIAL);
//					}
//					for(k = DetectCen.K+InitialSizeX; k < DetectCen.K+InitialSizeX+Ant_thickness; k++) 
//					{
//						yeeCube (i, j, k, D_MATERIAL);
//					}
//			}
//		}

//for(k = DetectCen.K-InitialSizeX; k < DetectCen.K+InitialSizeX; k++) 
		{			
			for(j = 0; j < DetectCen.J; j++)
			{
					for(i = DetectCen.I-InitialSizeX-Ant_thickness; i < DetectCen.I-InitialSizeX; i++) 
					//i=SourceCen.I-antenaSize;
					{
						yeeCube (i, j, k, D_MATERIAL);
					}
					//i=SourceCen.I+antenaSize;
					for(i = DetectCen.I+InitialSizeX; i < DetectCen.I+InitialSizeX+Ant_thickness; i++) 
					{
						yeeCube (i, j, k, D_MATERIAL);
					}
			}
		}
//-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

//S_koef=(antenaSize+Ant_thickness)-A_INITIAL_SIZE;
S_koef=(antenaSize)-InitialSizeX;
Loc_count=0;

//for(j = DetectCen.J; j < DetectCen.J+YA_length; j++) 
//		{		
//		if(S_koef>0 && Curve_koef==Loc_count)
//		{
//		S_koef--;
//		Loc_count=0;
//		}
//		else
//		{
//		Loc_count++;
//		}
//		for(i = DetectCen.I-antenaSize+S_koef; i < DetectCen.I+antenaSize-S_koef; i++) 
//			{			
//				for(k = DetectCen.K-antenaSize-Ant_thickness; k < DetectCen.K-antenaSize+S_koef; k++) 
//				{
//					yeeCube (i, j, k, D_MATERIAL);
//				}
//				for(k = DetectCen.K+antenaSize-S_koef; k < DetectCen.K+antenaSize+Ant_thickness; k++) 
//				{
//					yeeCube (i, j, k, D_MATERIAL);
//				}
//			}
//		}

//S_koef=(antenaSize+Ant_thickness)-A_INITIAL_SIZE;
S_koef=(antenaSize)-InitialSizeX;
Loc_count=0;
for(j = DetectCen.J; j < DetectCen.J+YA_length; j++) 
		{		
		if(S_koef>0 && Curve_koef==Loc_count)
		{
		S_koef--;
		Loc_count=0;
		}
		else
		{
		Loc_count++;
		}
		//for(k = DetectCen.K-antenaSize+S_koef; k < DetectCen.K+antenaSize-S_koef; k++) 
			{
					for(i = DetectCen.I-antenaSize-Ant_thickness; i < DetectCen.I-antenaSize+S_koef; i++) 
					//i=SourceCen.I-antenaSize;
					{
						yeeCube (i, j, k, D_MATERIAL);
					}
					//i=SourceCen.I+antenaSize;
					for(i = DetectCen.I+antenaSize-S_koef; i < DetectCen.I+antenaSize+Ant_thickness; i++) 
					{
						yeeCube (i, j, k, D_MATERIAL);
					}
			}
		}
	break;

//case 3:
//S_koef=antenaSize+Ant_thickness;
//Loc_count=0;
//for(j = SourceCen.J-Beg_Antena; j < SourceCen.J+YA_length; j++) 
//		{		
//		if(S_koef>0 && Curve_koef==Loc_count)
//		{
//		S_koef--;
//		Loc_count=0;
//		}
//		else
//		{
//		Loc_count++;
//		}
//		for(i = SourceCen.I-antenaSize+S_koef; i < SourceCen.I+antenaSize-S_koef; i++) 
//			{			
//				for(k = SourceCen.K-antenaSize-Ant_thickness; k < SourceCen.K-antenaSize+S_koef; k++) 
//				{
//					yeeCube (i, j, k, 5);
//				}
//				for(k = SourceCen.K+antenaSize-S_koef; k < SourceCen.K+antenaSize+Ant_thickness; k++) 
//				{
//					yeeCube (i, j, k, 5);
//				}
//			}
//		}
//
//S_koef=antenaSize+Ant_thickness;
//Loc_count=0;
//for(j = SourceCen.J-Beg_Antena; j < SourceCen.J+YA_length; j++) 
//		{		
//		if(S_koef>0 && Curve_koef==Loc_count)
//		{
//		S_koef--;
//		Loc_count=0;
//		}
//		else
//		{
//		Loc_count++;
//		}
//		for(k = SourceCen.K-antenaSize+S_koef; k < SourceCen.K+antenaSize-S_koef; k++) 
//			{
//					for(i = SourceCen.I-antenaSize-Ant_thickness; i < SourceCen.I-antenaSize+S_koef; i++) 
//					//i=SourceCen.I-antenaSize;
//					{
//						yeeCube (i, j, k, 5);
//					}
//					//i=SourceCen.I+antenaSize;
//					for(i = SourceCen.I+antenaSize-S_koef; i < SourceCen.I+antenaSize+Ant_thickness; i++) 
//					{
//						yeeCube (i, j, k, 5);
//					}
//			}
//		}
//	break;



default:
	printf("\nunknown detector antenna tupe\n");
	break;
}
}

void draw_2D_figure()
{
//----------circle-------------------
	int Circl_radius_B=270;
	int Circl_radius_S=60;

	int Patrubok_x=100;
	int Patrubok_y=Kamera.PatLength;

	int material1=5;
	int material2=5;

	switch(Kamera.FigPreset)
	{
	case 1:
//----------------Kamera----------------------------------
		draw_width_line(20, 180, Imax-20, 180, material1);
		draw_width_line(Imax-20, 180, Imax-20, Jmax-20, material1);
		draw_width_line(Imax-20, Jmax-20, 20, Jmax-20, material1);
		draw_width_line(20, Jmax-20, 20, 180, material1);

		draw_width_line(I_SOURCE-25, 180, I_SOURCE+25, 180, 0);
		//draw_width_line(I_DET-25, 180, I_DET+25, 180, 0);
	break;

	case 2:
//----------------Kamera----------------------------------
		draw_width_line(20, 180+Patrubok_y, Imax-20, 180+Patrubok_y, material1);
		draw_width_line(Imax-20, 180+Patrubok_y, Imax-20, Jmax-20, material1);
		draw_width_line(Imax-20, Jmax-20, 20, Jmax-20, material1);
		draw_width_line(20, Jmax-20, 20, 180+Patrubok_y, material1);
//-----------Patrubok-------------------------------------

		draw_width_line(600, 180+Patrubok_y, 600, 180, material2);
		draw_width_line(600, 180, 1000, 180, material2);
		draw_width_line(1000, 180, 1000, 180+Patrubok_y, material2);

		draw_width_line(600, 180+Patrubok_y, 1000, 180+Patrubok_y, 0);

//-----------antenna hole---------------------------------
		//draw_width_line(I_SOURCE-25, 180, I_SOURCE+25, 180, 0);
		Draw_Square(SourceCen.I-AA_SZ_I/2, 11, SourceCen.I+AA_SZ_I/2, 220, 0);
		//draw_width_line(I_DET-25, 180, I_DET+25, 180, 0);
	break;

//--------------------------------------------------------
	// NO BACK WALL
//--------------------------------------------------------
	case 3:
//----------------Kamera----------------------------------
		draw_width_line(20, 180+Patrubok_y, Imax-20, 180+Patrubok_y, material1);
		draw_width_line(Imax-20, 180+Patrubok_y, Imax-20, Jmax-20, material1);

		draw_width_line(20, Jmax-20, 20, 180+Patrubok_y, material1);
//-----------Patrubok-------------------------------------

		draw_width_line(600, 180+Patrubok_y, 600, 180, material2);
		draw_width_line(600, 180, 1000, 180, material2);
		draw_width_line(1000, 180, 1000, 180+Patrubok_y, material2);

		draw_width_line(600, 180+Patrubok_y, 1000, 180+Patrubok_y, 0);

//-----------antenna hole---------------------------------
		draw_width_line(I_SOURCE-25, 180, I_SOURCE+25, 180, 0);
		//draw_width_line(I_DET-25, 180, I_DET+25, 180, 0);
	break;

//--------------------------------------------------------
	// NO BACK, UP AND DOWN WALLS
//--------------------------------------------------------
	case 4:
//----------------Kamera----------------------------------
		draw_width_line(20, 180+Patrubok_y, Imax-20, 180+Patrubok_y, material1);
//-----------Patrubok-------------------------------------

		draw_width_line(600, 180+Patrubok_y, 600, 180, material2);
		draw_width_line(600, 180, 1000, 180, material2);
		draw_width_line(1000, 180, 1000, 180+Patrubok_y, material2);

		draw_width_line(600, 180+Patrubok_y, 1000, 180+Patrubok_y, 0);

//-----------antenna hole---------------------------------
		draw_width_line(I_SOURCE-25, 180, I_SOURCE+25, 180, 0);
		//draw_width_line(I_DET-25, 180, I_DET+25, 180, 0);
	break;

//--------------------------------------------------------
	// NO KAMERA
//--------------------------------------------------------
	case 5:
//----------------Kamera----------------------------------
		
//-----------Patrubok-------------------------------------

		draw_width_line(600, 180+Patrubok_y, 600, 180, material2);
		draw_width_line(600, 180, 1000, 180, material2);
		draw_width_line(1000, 180, 1000, 180+Patrubok_y, material2);

		draw_width_line(600, 180+Patrubok_y, 1000, 180+Patrubok_y, 0);

//-----------antenna hole---------------------------------
		draw_width_line(I_SOURCE-25, 180, I_SOURCE+25, 180, 0);
		//draw_width_line(I_DET-25, 180, I_DET+25, 180, 0);
	break;

//--------------------------------------------------------
	// NO UP AND DOWN WALLS
//--------------------------------------------------------
	case 6:
//----------------Kamera----------------------------------
		draw_width_line(20, 180+Patrubok_y, Imax-20, 180+Patrubok_y, material1);
		draw_width_line(Imax-20, Jmax-20, 20, Jmax-20, material1);
//-----------Patrubok-------------------------------------

		draw_width_line(600, 180+Patrubok_y, 600, 180, material2);
		draw_width_line(600, 180, 1000, 180, material2);
		draw_width_line(1000, 180, 1000, 180+Patrubok_y, material2);

		draw_width_line(600, 180+Patrubok_y, 1000, 180+Patrubok_y, 0);

//-----------antenna hole---------------------------------
		draw_width_line(I_SOURCE-25, 180, I_SOURCE+25, 180, 0);
		//draw_width_line(I_DET-25, 180, I_DET+25, 180, 0);
	break;

//--------------------------------------------------------
	// DRAW TOKAMAK IN EQUATORIAL SECTION
//--------------------------------------------------------
	case 7:
//----------------Kamera----------------------------------
		draw_Circl(Imax/2, Jmax/2+Patrubok_y, Circl_radius_B, 3, material1);
		draw_Circl(Imax/2, Jmax/2+Patrubok_y, Circl_radius_S, 4, material1);

		//Draw_Square(Imax/2-Patrubok_x/2, Jmax/2-Patrubok_y+15-Circl_radius_B, Imax/2+Patrubok_x/2, Jmax/2+Patrubok_y+15-Circl_radius_B, 0);
		Draw_Square(Imax/2-Patrubok_x/2, 11, Imax/2+Patrubok_x/2, Jmax/2+Patrubok_y+15-Circl_radius_B, 0);

//-----------Patrubok-------------------------------------
		draw_width_line(Imax/2-Patrubok_x/2, Jmax/2+Patrubok_y+4-Circl_radius_B, Imax/2-Patrubok_x/2, Jmax/2-Patrubok_y-Circl_radius_B, material2);
		draw_width_line(Imax/2-Patrubok_x/2, Jmax/2-Patrubok_y-Circl_radius_B, Imax/2+Patrubok_x/2, Jmax/2-Patrubok_y-Circl_radius_B, material2);
		draw_width_line(Imax/2+Patrubok_x/2, Jmax/2-Patrubok_y-Circl_radius_B, Imax/2+Patrubok_x/2, Jmax/2+Patrubok_y+4-Circl_radius_B, material2);

//-----------antenna hole---------------------------------
		//draw_width_line(I_SOURCE-25, Jmax/2-Patrubok_y-Circl_radius_B, I_SOURCE+25, Jmax/2-Patrubok_y-Circl_radius_B, 0);
		//draw_width_line(I_DET-25, 180, I_DET+25, 180, 0);
		Draw_Square(SourceCen.I-AA_SZ_I/2, 11, SourceCen.I+AA_SZ_I/2, Jmax/2+Patrubok_y+4-Circl_radius_B, 0);

	break;

	default:

	break;
	}
}


void draw_width_line(int x1, int y1, int x2, int y2, int matN)
{
int dx=x2-x1;
int dy=y2-y1;

if((dx>=0)&&(dy>=0))
{
    draw_line(x1, y1, x2, y2, matN);
    draw_line(x1, y1-1, x2, y2-1, matN);
}
if((dx<=0)&&(dy<=0))
{
    draw_line(x1, y1, x2, y2, matN);
    draw_line(x1, y1+1, x2, y2+1, matN);
}
if((dx<=0)&&(dy>=0))
{
	draw_line(x1, y1, x2, y2, matN);
    draw_line(x1, y1+1, x2, y2+1, matN);
}
if((dx>=0)&&(dy<=0))
{
    draw_line(x1, y1, x2, y2, matN);
    draw_line(x1, y1-1, x2, y2-1, matN);    
}
}

void draw_line(int x1, int y1, int x2, int y2, int matN)
{
int k=0, x=0, y=0, delta;

	if(y2!=y1)
	{
		if(y2>y1)
			delta=1;
		else
			delta=-1;

		for(y=y1;y!=y2;y+=delta)
		{
			x=(((x2-x1)*(y-y1))/(y2-y1))+x1;
			yeeCube (x, y, k, matN);	
		}
	}
	else
	{
		if(x2>x1)
			delta=1;
		else
			delta=-1;

		for(x=x1;x!=x2;x+=delta)
		{			
			yeeCube (x, y2, k, matN);	
		}

	}


}

void draw_Circl(int a, int b, int R, int width, int matN)
{
int k=0, x=0, y=0, delta;


for(x=0;x<Imax-1;x++)
{
	for(y=0;y<Jmax-1;y++)
	{
		if(((pow((x-a),2)+pow((y-b),2)))<=(pow(R,2)))
		yeeCube (x, y, k, matN);	
	}
}
		
for(x=0;x<Imax-1;x++)
{
	for(y=0;y<Jmax-1;y++)
	{
		if(((pow((x-a),2)+pow((y-b),2)))<=(pow(R-width,2)))
		yeeCube (x, y, k, 0);	
	}
}
}


void Draw_Square(int x1, int y1, int x2, int y2, int matN)
{
int k=0, x=0, y=0, delta;

for(x=0;x<Imax-1;x++)
{
	for(y=0;y<Jmax-1;y++)
	{
		if((x>x1)&&(x<x2)&&(y>y1)&&(y<y2))
		yeeCube (x, y, k, matN);	
	}
}


}
#ifndef _TAU_CALCULATION_CUH_
#define _TAU_CALCULATION_CUH_

#include <vector>

extern double dt;

struct Analysis { // только по Ez
	bool perform;			// выполнять обработку или нет
	// методы привязки
	bool sec_der;			// выполнять привязку методом перечения нуля второй производной
	int sdf_type;			// тип фильтра для склаживания дважды дифференцированного сигнала (ДДС): 1 - moving average, 2 - medfilt
	int sdf_win;			// ширина окна для сглажикания ДДС
	int sdf_runs;			// количество проходов фильтрации ДДС

	bool cf;				// выполнять привязку методом постоянной фракции (ПФ)
	int cf_tau;				// запаздывание в методе ПФ
	float cf_k;				// коэффициент аттеньюации в методе ПФ

	bool arc;				// выполнять ли привязку методом КАВ
	int arc_tau;			// запаздывание в методе КАВ
	float arc_k;			// коэффициент аттеньюации в методе КАВ

	float tau_threshold;	// пороговое ограничение тау (при 0.25 порог - 25% от максимума)

	// методы фильтрования
	int filter_type;		// тип фильтра при фильтрации сигнала с детектора: 1 - moving average, 2 - medfilt
	int filter_win;			// ширина окна фильтра сигнала с детектора
	int filter_runs;		// количество проходов фильтрации сигнала с детектора

	bool save_original;		// сохранять ли исходные данные с линейки детекторов
	bool save_filtered;		// сохранять ли фильтрованные данные с линейки детекторов
	
	int dim_sieve;			// пространственное прореживание
	int time_sieve;			// временное прореживание

	bool sim_detect;		// эмулировать ли поведение реального детектора (усреднение по площадке с заданными весами)
	bool save_detector;		// сохранять ли результат эмуляции детектора
	bool load_weights;		// загружать ли веса для реального детектора с файла "DetWeights.txt"
	int detector_size;		// размер реального детектора

	int begin_after;		// начать запоминание данных с линейки детекторов после временного шага
	int finish_at;			// закончить запоминание данных с линейки детекторов после временного шага
	int pos_i;				// 
	int pos_j;				// 
	int pos_k;				// 
	int size_i;				// положение и размеры линейки детекторов

	int times;				// уже не помню, но что-то дельное
};

typedef std::vector< float > FloatVector;
typedef std::vector< FloatVector > FloatMatrix;

// entering point
void processing_core(float ** field, const Analysis & A);

// filters
void filter(FloatVector & src, FloatVector & dst, int filter_type, int win_size, int iters = 1);
void moving_average(FloatVector & src, FloatVector & dst, int win_size);
void medfilt(FloatVector & src, FloatVector & dst, int win_size);

// tau search algorithms
int sd_method(FloatVector & src, const Analysis & A);
int cf_method(FloatVector & src, const Analysis & A);
int arc_method(FloatVector & src, const Analysis & A);

// additional
void fprintf(const char *fname, FloatVector & v, int sieve, int line_no_from);
void fprintf(const char *fname, FloatMatrix & m, int t_sieve, int d_sieve);
void sec_diff(FloatVector & src, FloatVector & dst);
float max(FloatVector & v);

#endif // _TAU_CALCULATION_CUH_
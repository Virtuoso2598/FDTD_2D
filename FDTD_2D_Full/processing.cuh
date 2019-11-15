#ifndef _TAU_CALCULATION_CUH_
#define _TAU_CALCULATION_CUH_

#include <vector>

extern double dt;

struct Analysis { // ������ �� Ez
	bool perform;			// ��������� ��������� ��� ���
	// ������ ��������
	bool sec_der;			// ��������� �������� ������� ��������� ���� ������ �����������
	int sdf_type;			// ��� ������� ��� ����������� ������ ������������������� ������� (���): 1 - moving average, 2 - medfilt
	int sdf_win;			// ������ ���� ��� ����������� ���
	int sdf_runs;			// ���������� �������� ���������� ���

	bool cf;				// ��������� �������� ������� ���������� ������� (��)
	int cf_tau;				// ������������ � ������ ��
	float cf_k;				// ����������� ����������� � ������ ��

	bool arc;				// ��������� �� �������� ������� ���
	int arc_tau;			// ������������ � ������ ���
	float arc_k;			// ����������� ����������� � ������ ���

	float tau_threshold;	// ��������� ����������� ��� (��� 0.25 ����� - 25% �� ���������)

	// ������ ������������
	int filter_type;		// ��� ������� ��� ���������� ������� � ���������: 1 - moving average, 2 - medfilt
	int filter_win;			// ������ ���� ������� ������� � ���������
	int filter_runs;		// ���������� �������� ���������� ������� � ���������

	bool save_original;		// ��������� �� �������� ������ � ������� ����������
	bool save_filtered;		// ��������� �� ������������� ������ � ������� ����������
	
	int dim_sieve;			// ���������������� ������������
	int time_sieve;			// ��������� ������������

	bool sim_detect;		// ����������� �� ��������� ��������� ��������� (���������� �� �������� � ��������� ������)
	bool save_detector;		// ��������� �� ��������� �������� ���������
	bool load_weights;		// ��������� �� ���� ��� ��������� ��������� � ����� "DetWeights.txt"
	int detector_size;		// ������ ��������� ���������

	int begin_after;		// ������ ����������� ������ � ������� ���������� ����� ���������� ����
	int finish_at;			// ��������� ����������� ������ � ������� ���������� ����� ���������� ����
	int pos_i;				// 
	int pos_j;				// 
	int pos_k;				// 
	int size_i;				// ��������� � ������� ������� ����������

	int times;				// ��� �� �����, �� ���-�� �������
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
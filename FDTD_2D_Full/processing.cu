#include "processing.cuh"
#include "Settings.cuh"
#include <algorithm>
#include <fstream>

void processing_core(float ** field, const Analysis & A) {
	printf("Processing...\n");
	int sz = A.size_i/* / A.dim_sieve*/;

	FloatMatrix source(sz), filtered;
	for (int i = 0; i < sz; i++) {
		source[i] = FloatVector(A.times);
		for (int t = 0; t < A.times; t++) {
			source[i][t] = fabs(field[t][i]);
		}
	}
	filtered = source;
	printf("Converting done.\n");
	
	// free memory
	delete[] field;

	printf("Executing of specified methods...\n");

	// *************** Perform processing for each coordinate ***************
	FloatVector sd_tau(sz), cf_tau(sz), arc_tau(sz);
	for (int i = 0; i < sz; i++) {
		//  **** filtering ****
		filter(source[i], filtered[i], A.filter_type, A.filter_win, A.filter_runs);

		// **** tau calculation ****
		if (A.sec_der) {
			sd_tau[i] = A.begin_after + sd_method(filtered[i], A);
		}

		if (A.cf) {
			cf_tau[i] = A.begin_after + cf_method(filtered[i], A);
		}

		if (A.arc) {
			arc_tau[i] = A.begin_after + arc_method(filtered[i], A);
		}
	}
	
	// save original field
	if (A.save_original) {
		fprintf("original_field.txt", source, A.time_sieve, A.dim_sieve);
		printf("Original field saved.\n");
	}

	// save filtered field
	if (A.save_filtered) {
		fprintf("filtered_field.txt", filtered, A.time_sieve, A.dim_sieve);
		printf("Filtered field saved.\n");
	}

	// save tau results in files
	if (A.sec_der) fprintf("SD.txt", sd_tau, A.dim_sieve, A.pos_i);
	if (A.cf) fprintf("CF.txt", cf_tau, A.dim_sieve, A.pos_i);
	if (A.arc) fprintf("ARC.txt", arc_tau, A.dim_sieve, A.pos_i);

	printf("TOFs saved in files.\n");

	// simulation of detector behaviour
	if (A.sim_detect) {
		printf("Processing simulation of detector behaviour...\n");
		FloatMatrix real_detector(A.size_i - A.detector_size, FloatVector(A.times, 0));
		filtered = real_detector;

		FloatVector det_weights(A.detector_size, 1.0/A.detector_size);
		float buf = 0;

		if (A.load_weights) {
			std::ifstream ist("DetWeights.txt");
			
			for (int i = 0; ist && i < A.detector_size; i++) {
				if (!ist.eof()) {
					ist >> det_weights[i];
				} else {
					det_weights = FloatVector(A.detector_size, 1);
					break;
				}
			}

			ist.close();
		}

		int half_size = A.detector_size / 2;

		for (int t = 0; t < A.times; t++) {
			for (int i = 0; i < sz - A.detector_size; i++) {
				buf = 0;
				for (int j = -half_size; j < A.detector_size - half_size; j++) {
					buf += source[i + half_size + j][t] * det_weights[j + half_size];
				}
				real_detector[i][t] = buf / A.detector_size;
			}
		}

		sd_tau = FloatVector(sz - A.detector_size), cf_tau = sd_tau, arc_tau = sd_tau;
		for (int i = 0; i < (sz - A.detector_size); i++) {
			//  **** filtering ****
			filter(real_detector[i], filtered[i], A.filter_type, A.filter_win, A.filter_runs);

			// **** tau calculation ****
			if (A.sec_der) {
				sd_tau[i] = A.begin_after + sd_method(filtered[i], A);
			}

			if (A.cf) {
				cf_tau[i] = A.begin_after + cf_method(filtered[i], A);
			}

			if (A.arc) {
				arc_tau[i] = A.begin_after + arc_method(filtered[i], A);
			}
		}

		printf("Detector simulation OK.\n");
		
		// save original field from detector sim
		if (A.save_detector && A.save_original) {
			fprintf("detector_original_field.txt", real_detector, A.time_sieve, A.dim_sieve);
			printf("Original real detector field saved.\n");

		}

		// save filtered field
		if (A.save_detector && A.save_filtered) {
			fprintf("detector_filtered_field.txt", filtered, A.time_sieve, A.dim_sieve);
			printf("Filtered real detector field saved.\n");
		}

		// save tau results in files
		if (A.sec_der) fprintf("det_SD.txt", sd_tau, A.dim_sieve, A.pos_i + half_size);
		if (A.cf) fprintf("det_CF.txt", cf_tau, A.dim_sieve, A.pos_i + half_size);
		if (A.arc) fprintf("det_ARC.txt", arc_tau, A.dim_sieve, A.pos_i + half_size);

		printf("Detector sim times saved in files.\n");
	}

	printf("Execution finished.\n");
	printf("Processing done.\n");
}

void filter(FloatVector & src, FloatVector & dst, int filter_type, int win_size, int iters) {
	FloatVector tmp = src;
	
	for (int i = 0; i < iters; i++) {
		switch (filter_type) {
			// moving average
			case 1:
				moving_average(tmp, dst, win_size);
				tmp = dst;
				break;

			// medfilt
			case 2:
				medfilt(tmp, dst, win_size);
				tmp = dst;
				break;

			// no_filter
			default:
				break;
		}
	}
}

void moving_average(FloatVector & src, FloatVector & dst, int win_size) {
	float buf = 0;
	int half_win = int(win_size / 2);
	int times = src.size();

	for (int t = 0; t < win_size - 1; t++) {
		buf += src[t];
	}

	for (int t = win_size - 1; t < times; t++) {
		buf += src[t];
		dst[t - half_win] = buf / win_size;
		buf -= src[t - win_size + 1];
	}

	for (int t = half_win; t > 0; t--) {
		dst[t] = dst[half_win - 1];
		dst[times - t] = dst[times - half_win - 1];
	}
}

void medfilt(FloatVector & src, FloatVector & dst, int win_size) {
	FloatVector buf, tmp;
	int half_win = int(win_size / 2);
	int times = src.size();

	for (int t = 0; t < win_size - 1; t++) {
		buf.push_back(src[t]);
	}

	for (int t = win_size - 1; t < times; t++) {
		buf.push_back(src[t]);
		tmp = buf;
		//sort(tmp);
		std::sort(tmp.begin(), tmp.end());
		dst[t - half_win] = tmp[half_win];
		buf.erase(buf.begin());
	}

	for (int t = half_win; t > 0; t--) {
		dst[t] = dst[half_win - 1];
		dst[times - t] = dst[times - half_win - 1];
	}
}

int sd_method(FloatVector & src, const Analysis & A) {
	FloatVector tmp1 = src, tmp2 = tmp1;
	float min_level = A.tau_threshold * max(src);

	sec_diff(src, tmp1);
	filter(tmp1, tmp2, A.sdf_type, A.sdf_win, A.sdf_runs);

	for (int i = 0; i < src.size() - 1; i++) {
		if (src[i] > min_level) {
			if (tmp2[i] * tmp2[i + 1] < 0) return i;
		}
	}
	return src.size() - 1;
}

int cf_method(FloatVector & src, const Analysis & A) {
	FloatVector tmp1 = src, tmp2 = tmp1;
	float min_level = A.tau_threshold * max(src), p1, p2;

	for (int t = 0; t < src.size() - (A.cf_tau + 1); t++) {
		if (src[t] > min_level) {
			p1 = src[t] + A.cf_k * src[t + A.cf_tau];
			p2 = src[t + 1] + A.cf_k * src[t + A.cf_tau + 1];
			// поиск значения
			if (p1 * p2 < 0) return t;
		}
	}

	return src.size() - 1;
}

int arc_method(FloatVector & src, const Analysis & A) {
	FloatVector tmp1 = src, tmp2 = tmp1;
	float min_level = A.tau_threshold * max(src), p1, p2;

	for (int t = 0; t < src.size() - (A.arc_tau + 1); t++) {
		if (src[t] > min_level) {
			p1 = src[t] + A.arc_k * src[t + A.arc_tau];
			p2 = src[t + 1] + A.arc_k * src[t + A.arc_tau + 1];
			// поиск значения
			if (p1 * p2 < 0) return t;
		}
	}

	return src.size() - 1;
}


void fprintf(const char *fname, FloatVector & v, int sieve = 1, int line_no_from = 0) {
	FILE * out = fopen(fname, "w");
	for (int i = 0; i < v.size(); i += sieve) {
		fprintf(out, "%d %lf %lf\n", line_no_from + i, v[i], (dt*v[i])*TIME_PRINT_COEF);
	}
	fclose(out);
}

void fprintf(const char *fname, FloatMatrix & m, int t_sieve = 1, int d_sieve = 1) {
	FILE * out = fopen(fname, "w");
	int cols = m.size();
	if (!cols) return;
	int rows = m[0].size();

	for (int t = 0; t < rows; t += t_sieve) {
		for (int i = 0; i < cols; i += d_sieve) {
			fprintf(out, "%f\t", m[i][t]);
		}
		fprintf(out, "\n");
	}
	fclose(out);
}

void sec_diff(FloatVector & src, FloatVector & dst) {
	dst[0] = dst[dst.size() - 1] = 0;
	for (int i = 1; i < src.size() - 1; i++) {
		dst[i] = src[i - 1] - 2 * src[i] + src[i + 1];
	}
}

float max(FloatVector & v) {
	if (!v.size()) return 0;

	float res = v[0];
	for (int i = 1; i < v.size(); i++) {
		if (v[i] > res) res = v[i];
	}
	return res;
}
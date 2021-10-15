#pragma once
#ifndef LLGMN_H
#define LLGMN_H
#include<vector>
#include<string>
using namespace std;
class llgmn {
private:
	vector<double> modified_input;
	const int modified_input_dim;
	const int input_dim;
	const int class_num;
	double initial_J;// �^�[�~�i���A�g���N�^�p
	const string time_prefix;
	// �w�K��
	const double learning_rate;
	// �N���X�̃R���|�[�l���g�̐��@�R���|�[�l���g�̐��̓N���X���Ƃɕς��̂��H
	// ����͓��͂��ʓ|�Ȃ̂Ŋe�N���X�ŃR���|�[�l���g���͓����Ƃ��Ď�������D
	//vector<int> component_num_for_class;
	const int component_size;
	// weight[����][�N���X][�R���|�[�l���g]
	vector<vector<vector<double>>> weight;
	// second_output[�N���X][�R���|�[�l���g]
	vector<vector<double>> second_output;
	// �N���X���̑傫��������
	vector<double> output;
	// ���t�f�[�^ �N���X���Ɠ����傫��
	vector<double> teacher_data;
	// �f�[�^�� �N���X�̒��Ŏ��K�v�����邩�H
	int data_num;
	// �덷�Z�o�p�@�~�ς���덷 �ǂ����ŏ��������Ȃ��Ⴂ���Ȃ�������K�X�@batch�̒��̑S�Ẵf�[�^�̍X�V�ʂ̘a
	double sum_error;
	// �X�V�ʂ̒~�� �d�ݍX�V�p batch�̒��̑S�Ẵf�[�^�̍X�V�ʂ̘a
	vector<vector<vector<double>>> sum_update_val;
	// function
	void init_weight();
	void input_loglinearization(const vector<double>& input);
	vector<vector<double>> calc_second_layer_input();
	void calc_second_layer_output(const vector<vector<double>>& second_input);
	void calc_third_layer_input();
	void pool_energy_func();
	void pool_update_val();
	void update_weight_patch();
	void init_sum_val();
	void set_teacher_data(const vector<double>& teacher_data);
	double terminal_attractor(double beta, const vector<vector<vector<double>>>& dj_dw, double learning_time);
	

	/*!----------------------------------------------------------------------------
	 @brif 1�f�[�^�����̑S�����v�Z���s���֐�

	 @param [in] data ���ʂ���f�[�^
	 @param [out] posterior_probability�@���ʌ��ʂ̊m��
	 @attention

	*/
	void one_forward(vector<double>& data, vector<double>& posterior_probability);
public:
	llgmn(const int input_dim, const int class_num, const int component_size, const double learning_rate, const string output_dir, const string data_name);
	virtual ~llgmn() {};
	// �O�����Ɍv�Z�����ăG�l���M�[�֐����o���D
	/*!----------------------------------------------------------------------------
	 @brif �O�����Ɍv�Z�����ăG�l���M�[�֐�������I�ɏo���D

	 @param [in] input_data ���ʂ���f�[�^
	 @param [out] teacher_data ���t�f�[�^
	 @attention

	*/
	void forward(const vector<vector<double>>& input_data, const vector<vector<double>>& teacher_data);
	// �덷�`��������
	void backward();
	// �G�l���M�[�֐��̒l���Ƃ�
	double get_error() const;
	vector<double> get_output() const;
	void set_initial_J();// �^�[�~�i���A�g���N�^�p�̍ŏ��̃G�l���M�[�֐�
	/*!----------------------------------------------------------------------------
	@brif �e�X�g���s���֐�

	 �w�K�����p�����[�^��p���ăe�X�g���s��
	@param [in] test_data(vector<vector<double>>&)   �e�X�g�f�[�^
	@param [in] class_label(const vector<vector<int>>&) �f�[�^��������N���X���������x��
	@param [in] output2csv (const bool)
	@param [in] filenum (const char) �e�X�g�t�@�C���̔ԍ�
	@return vector<vector<double>>&
	@attention

	*/
	vector<vector<double>> evaluate(vector<vector<double>>& test_data, const vector<vector<double>>& class_label, const bool output2csv, string filenum);
	/*!----------------------------------------------------------------------------
	@brif �l�b�g���[�N�p�����[�^��ۑ�����֐�

	 csv�t�@�C���ɃN���X��class_num, �R���|�[�l���g��component_num, ���͎�����data_size, ���w�K�N���X�̍����xzeta, ���U�̕␳���̏����linit_beta,
	 �G�l���M�[�֐���臒laccuracy_max, �w�K�萔xi, �]���ە��z�̕��U�����߂�epsilon, �w�K�񐔂̏��max_times���o�͂���
	@return �Ȃ�
	@attention

	*/
	void out_file_weight(const std::string file_suffix = "") const;

};
inline double llgmn::get_error() const {
	return sum_error;
}
inline void llgmn::set_teacher_data(const vector<double>& teacher_data) {
	this->teacher_data = teacher_data;
}
inline void llgmn::set_initial_J() {
	this->initial_J = this->sum_error;
}
inline vector<double> llgmn::get_output() const {
	return output;
}
#endif LLGMN_H
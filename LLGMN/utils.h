#pragma once
#ifndef __INC_UTILS_H
#define __INC_UTILS_H
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
/*!----------------------------------------------------------------------------
 @brif csv�t�@�C������f�[�^��ǂݍ��ފ֐�

  csv�t�@�C���̃f�[�^��T�^��2����vector�Ɋi�[����
 @param [in] filename(const string) csv�t�@�C����s
 @return vector<vector<int>> �ǂݍ��񂾃f�[�^
 @attention utils.cpp�̕��Ŏ�������ƃG���[�ɂȂ�...?

*/
template<class T>
std::vector<std::vector<T>> get_vector_from_file(const std::string filename) {
	std::ifstream ifs(filename);
	if (ifs.fail()) {
		std::cerr << "Can't open " + filename << std::endl;
		exit(-1);
	}
	std::string str, str1;
	std::vector<std::vector<T>> data;
	while (std::getline(ifs, str)) {
		std::stringstream ss{ str };
		std::vector<T> tmp;
		while (std::getline(ss, str1, ',')) {
			tmp.push_back(stod(str1));
		}
		data.push_back(tmp);
	}
	return data;
}

/*!----------------------------------------------------------------------------
 @brif �d�����Ȃ������𔭐�������֐�

  �����Z���k�c�C�X�^�[�𗐐��𔭐������Csort��unique�ŏd������菜��
 @param [in] size(const size_t) �������闐����̃T�C�Y
 @param [in] rand_min(int)�@�������闐����̍ŏ��l
 @param [in] rand_max(int)�@�������闐����̍ő�l
 @return vector<int> �d���Ȃ�������
 @attention �\�[�g����邩�珬�������ɂȂ�܂��B

*/
std::vector<int> make_rand_array_unique(const size_t size, int rand_min, int rand_max);

/*!----------------------------------------------------------------------------
 @brif �����𔭐�������֐�

  �����Z���k�c�C�X�^�[�𗐐��𔭐�������
 @param [in] rand_min(int)�@�������闐���̍ŏ��l
 @param [in] rand_max(int)�@�������闐���̍ő�l
 @return double ����
 @attention

*/
std::double_t make_rand(double rand_min, double rand_max);

/*!----------------------------------------------------------------------------
 @brif �������擾����֐�

  ���s���̓������擾
 @return string ����
 @attention

*/
std::string get_date_sec();

/*!----------------------------------------------------------------------------
 @brif �V���A���ʐM���s���֐�

  Arduino�ƃV���A���ʐM���s���A���ʌ��ʂ���Arduino�̐�����s��
 @param portnum(const int) �|�[�g�ԍ�
 @param label(vector<vector<double>>&) ���x���f�[�^
 @return �Ȃ�
 @attention

*/
//void serial_communication(const int portnum, const std::vector<std::vector<double>>& label);

/*!----------------------------------------------------------------------------
 @brif �V�O���C�h�֐�

  �D
 @param[in]   ����
 @return      �V�O���C�h�o��
 @attention

*/
double sigmoid(double x);

/*!----------------------------------------------------------------------------
 @brif �p�����[�^����̂��߂̃O���b�h�T�[�`���s���p�̃p�����[�^�����֐� �f�O

  static�ϐ��Ɍ��݂̃p�����[�^��ۑ����āC�֐����Ă΂�邽�тɎ��̃p�����[�^��n��
  �Ō�̃p�����[�^�𐶐�������ɌĂ΂ꂽ�ꍇ�͒���0�̔z���Ԃ��D
  ���g�͓K���ɏ��������ĂˁD
 @return map<string, double>& �p�����[�^���������}�b�v
 @attention

*/
//std::map<std::string, double>& generate_params();

#endif

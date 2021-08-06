#pragma once
#ifndef __INC_UTILS_H
#define __INC_UTILS_H
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
/*!----------------------------------------------------------------------------
 @brif csvファイルからデータを読み込む関数

  csvファイルのデータをT型の2次元vectorに格納する
 @param [in] filename(const string) csvファイル名s
 @return vector<vector<int>> 読み込んだデータ
 @attention utils.cppの方で実装するとエラーになる...?

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
 @brif 重複しない乱数を発生させる関数

  メルセンヌツイスターを乱数を発生させ，sortとuniqueで重複を取り除く
 @param [in] size(const size_t) 生成する乱数列のサイズ
 @param [in] rand_min(int)　生成する乱数列の最小値
 @param [in] rand_max(int)　生成する乱数列の最大値
 @return vector<int> 重複なし乱数列
 @attention ソートされるから小さい順になります。

*/
std::vector<int> make_rand_array_unique(const size_t size, int rand_min, int rand_max);

/*!----------------------------------------------------------------------------
 @brif 乱数を発生させる関数

  メルセンヌツイスターを乱数を発生させる
 @param [in] rand_min(int)　生成する乱数の最小値
 @param [in] rand_max(int)　生成する乱数の最大値
 @return double 乱数
 @attention

*/
std::double_t make_rand(double rand_min, double rand_max);

/*!----------------------------------------------------------------------------
 @brif 日時を取得する関数

  実行時の日時を取得
 @return string 日時
 @attention

*/
std::string get_date_sec();

/*!----------------------------------------------------------------------------
 @brif シリアル通信を行う関数

  Arduinoとシリアル通信を行い、識別結果からArduinoの制御を行う
 @param portnum(const int) ポート番号
 @param label(vector<vector<double>>&) ラベルデータ
 @return なし
 @attention

*/
//void serial_communication(const int portnum, const std::vector<std::vector<double>>& label);

/*!----------------------------------------------------------------------------
 @brif シグモイド関数

  ．
 @param[in]   入力
 @return      シグモイド出力
 @attention

*/
double sigmoid(double x);

/*!----------------------------------------------------------------------------
 @brif パラメータ決定のためのグリッドサーチを行う用のパラメータ生成関数 断念

  static変数に現在のパラメータを保存して，関数が呼ばれるたびに次のパラメータを渡す
  最後のパラメータを生成した後に呼ばれた場合は長さ0の配列を返す．
  中身は適当に書き換えてね．
 @return map<string, double>& パラメータが入ったマップ
 @attention

*/
//std::map<std::string, double>& generate_params();

#endif

// pcl_roi_based_compression_2.cpp: определяет точку входа для приложения.
//

#include "pcl_roi_based_compression_2.h"

using namespace std;

static const std::string ORIGIN_PATH = "F:/repos/prbc2/pcl_roi_based_compression_2/clouds";

static const std::string PLY_PATH = ORIGIN_PATH + "/bunny/scaled/bun_zipper_res2_10x.ply";
static const std::string FILE_SAVE_PATH = ORIGIN_PATH + "/bunny/processed.ply";

int main()
{
	
	// Считываем облако из .ply файла
	// Тип точек - XYZL (X, Y, Z, Label). Считаем, что Label = 0 по дефолту.
	pcl::PointCloud<pcl::PointXYZL>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZL>);
	pcl::PLYReader Reader;
	Reader.read(PLY_PATH, *cloud);

	// Обучим модель KDE на точках облака
	// Если облако слишком маленькое или слишком большое, возможны неадекватные значения
	// Нужно выставлять верный размер ядра.
	const double bandwidth = 0.75;
	mlpack::kde::KDE<mlpack::kernel::EpanechnikovKernel> kde(
		mlpack::kde::KDEDefaultParams::relError, 
		mlpack::kde::KDEDefaultParams::absError, 
		mlpack::kernel::EpanechnikovKernel(bandwidth)
	);
	arma::mat kdeRefData(cloud->size(), 3, arma::fill::none);

	for (int idx = 0; idx < cloud->size(); idx++) {
		auto& point = (*cloud)[idx];
		kdeRefData(idx, 0) = point.x;
		kdeRefData(idx, 1) = point.y;
		kdeRefData(idx, 2) = point.z;
	}
	
	arma::inplace_trans(kdeRefData);
	kde.Train(kdeRefData);
	assert(kde.IsTrained());

	// Расчитаем density для каждой точки
	arma::vec kdeDataEst;
	kde.Evaluate(kdeDataEst);

	// Запишем полученные значения в Label-ы точек облака
	for (int idx = 0; idx < kdeDataEst.n_rows; idx++) {
		auto& point = (*cloud)[idx];
		double _val = kdeDataEst(idx);
		std::cout << _val << std::endl;
		point.label = _val * 100;
	}

	// Сохраняем обработанное облако (облака) в файл(ы) 
	pcl::io::savePLYFile(FILE_SAVE_PATH, *cloud, true);
	return 0;

}

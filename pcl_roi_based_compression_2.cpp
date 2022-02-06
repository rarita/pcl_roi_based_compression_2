// pcl_roi_based_compression_2.cpp: определяет точку входа для приложения.
//

#include "pcl_roi_based_compression_2.h"

using namespace std;

static const std::string ORIGIN_PATH = "F:/repos/prbc2/pcl_roi_based_compression_2/clouds";

static const std::string PLY_PATH = ORIGIN_PATH + "/bunny/reconstruction/bun_zipper_res2.ply";
static const std::string FILE_SAVE_PATH = ORIGIN_PATH + "/bunny/processed.ply";

int main()
{
	
	// Считываем облако из .ply файла
	// Тип точек - XYZL (X, Y, Z, Label). Считаем, что Label = 0 по дефолту.
	pcl::PointCloud<pcl::PointXYZL>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZL>);
	pcl::PLYReader Reader;
	Reader.read(PLY_PATH, *cloud);

	KDE kdeModel;
	// Обучим модель KDE
	for (auto& p : *cloud) {
		// p.label = std::rand();
		std::vector<double> pointPosition{ p.x, p.y, p.z };
		kdeModel.add_data(pointPosition);
	}

	// Получим в каждой точке значения density
	for (int idx = 0; idx < (*cloud).size(); idx++) {

		auto& p = (*cloud)[idx];
		std::vector<double> pointPosition{ p.x, p.y, p.z };
		// Probability Density Function
		double pdf = kdeModel.pdf(pointPosition);
		p.label = pdf;

		if (idx % 250 == 0) {
			std::cout << "Processed " << idx << " points." << std::endl;
		}

	}

	// Сохраняем обработанное облако (облака) в файл(ы) 
	pcl::io::savePLYFile(FILE_SAVE_PATH, *cloud, true);
	return 0;

}

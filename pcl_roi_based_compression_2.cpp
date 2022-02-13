// pcl_roi_based_compression_2.cpp: определяет точку входа для приложения.
//

#include "pcl_roi_based_compression_2.h"

using namespace std;

static const std::string ORIGIN_PATH = "F:/repos/prbc2/pcl_roi_based_compression_2/clouds";

static const std::string PLY_PATH = ORIGIN_PATH + "/bunny/work/onplane_noisy_2.ply";
static const std::string FILE_SAVE_PATH = ORIGIN_PATH + "/bunny/processed.ply";

/**
* Создаем плоскую двумерную поверхность с заданными параметрами
* и сохраняем как облако точек в формате .ply
*/
void generateSurface(std::string fname, int dim, double spacing, double noisePower, double tileOffset) {

	pcl::PointCloud<pcl::PointXYZ> surf;
	surf.resize(dim, dim);

	noise::module::Perlin perlin;
	noise::utils::NoiseMap heightMap; 
	noise::utils::NoiseMapBuilderPlane heightMapBuilder;
	heightMapBuilder.SetSourceModule(perlin);
	heightMapBuilder.SetDestNoiseMap(heightMap);
	heightMapBuilder.SetDestSize(dim, dim);
	const double tileStart = 1.0;
	heightMapBuilder.SetBounds(
		tileStart, tileStart + tileOffset, 
		tileStart, tileStart + tileOffset
	);
	heightMapBuilder.Build();
	
	for (int idx = 0; idx < dim * dim; idx++) {
		auto& point = surf[idx];
		point.x = (idx / dim) * spacing;
		// perlin.GetValue((idx / dim) / 2, (idx % dim) / 2, 0.25) * noisePower;
		double noise = heightMap.GetValue((idx / dim), (idx % dim)) * noisePower;
		point.y = noise;
		// std::cout << point.y << std::endl;
		point.z = (idx % dim) * spacing;
	}

	pcl::io::savePLYFile(fname, surf, false);

}

void processCloud() {

	// Считываем облако из .ply файла
	// Тип точек - XYZL (X, Y, Z, Label). Считаем, что Label = 0 по дефолту.
	pcl::PointCloud<pcl::PointXYZL>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZL>);
	pcl::PLYReader Reader;
	int res = Reader.read(PLY_PATH, *cloud);
	assert(res == 0);
	std::cout << "Successfully read point cloud at " << PLY_PATH << std::endl;

	// Обучим модель KDE на точках облака
	// Если облако слишком маленькое или слишком большое, возможны неадекватные значения
	// Нужно выставлять верный размер ядра.
	const double bandwidth = 1.25;
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
	std::cout << "Gathered reference data from point cloud" << std::endl;

	arma::inplace_trans(kdeRefData);
	kde.Train(kdeRefData);
	assert(kde.IsTrained());
	std::cout << "Trained the model" << std::endl;

	// Расчитаем density для каждой точки
	arma::vec kdeDataEst;
	kde.Evaluate(kdeDataEst);
	std::cout << "Computed densities for all points" << std::endl;

	// Запишем полученные значения в Label-ы точек облака
	for (int idx = 0; idx < kdeDataEst.n_rows; idx++) {
		auto& point = (*cloud)[idx];
		double _val = kdeDataEst(idx);
		point.label = _val * 100;
	}

	// Сохраняем обработанное облако (облака) в файл(ы) 
	pcl::io::savePLYFile(FILE_SAVE_PATH, *cloud, true);
	std::cout << "Result saved to " << FILE_SAVE_PATH << std::endl;

}

int main()
{

	//generateSurface(ORIGIN_PATH + "/surf.ply", 140, 0.033, 0.35, 1.);
	processCloud();
	return 0;

}

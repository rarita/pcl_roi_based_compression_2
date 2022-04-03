// pcl_roi_based_compression_2.cpp: определяет точку входа для приложения.
//

#include "pcl_roi_based_compression_2.h"

using namespace std;

static const std::string ORIGIN_PATH = "F:/repos/prbc2/pcl_roi_based_compression_2/clouds";

static const std::string PLY_PATH = ORIGIN_PATH + "/reak/01_ss.ply";
// Путь и имя файла без расширения для дописывания разных суффиксов (bw)
static const std::string FILE_SAVE_PATH = ORIGIN_PATH + "/reak/processed/processed";

/**
* Создаем плоскую двумерную поверхность с заданными параметрами
* и сохраняем как облако точек в формате .ply
*/
void generateSurface(
	std::string fname, int dim, double spacing, 
	double noisePower, double tileOffset, int octaves) {

	pcl::PointCloud<pcl::PointXYZ> surf;
	surf.resize(dim, dim);

	noise::module::Perlin perlin;
	perlin.SetOctaveCount(octaves);
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

void _processCloudInternal(int bw, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {

	std::cout << "Started processing KDE with bw: " << bw << std::endl;
	mlpack::kde::KDE<mlpack::kernel::EpanechnikovKernel> kde(
		mlpack::kde::KDEDefaultParams::relError,
		mlpack::kde::KDEDefaultParams::absError,
		mlpack::kernel::EpanechnikovKernel((bw * 1.0) / 100)
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
		point.intensity = _val;
	}

	// Сохраняем обработанное облако (облака) в файл(ы) 
	const std::string path = FILE_SAVE_PATH + "-bw" + std::to_string(bw) + ".ply";
	pcl::io::savePLYFile(path, *cloud, true);
	std::cout << "Result saved to " << path << std::endl;
	std::cout << "Done processing KDE with bw: " << bw << std::endl;

}

void processCloud() {

	// Считываем облако из .ply файла
	// Тип точек - XYZL (X, Y, Z, Label). Считаем, что Label = 0 по дефолту.
	// todo надо сделать свой ТИП вроде PointXYZI, который в label пишет float64
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PLYReader Reader;
	int res = Reader.read(PLY_PATH, *cloud);
	assert(res == 0);
	std::cout << "Successfully read point cloud at " << PLY_PATH << std::endl;

	// Обучим модель KDE на точках облака
	// Если облако слишком маленькое или слишком большое, возможны неадекватные значения
	// Нужно выставлять верный размер ядра.
	const int minBandwidth = 25;
	const int maxBandwidth = 500;
	const int step = 20;
	assert(step < 1);
	
	// precise bandwidth
	#pragma omp parallel for
	for (int bw = minBandwidth; bw < 125; bw += step) {
		_processCloudInternal(bw, cloud);
	}

	// broad bandwidth
	#pragma omp parallel for
	for (int bw = 125; bw < maxBandwidth; bw += 100) {
		_processCloudInternal(bw, cloud);
	}

}

int main(int argc, char* argv[])
{
	/*
	if (argc != 3) {
		std::cout << "Program requires 2 arguments ;)";
		return 1;
	}
	*/
	//generateSurface(ORIGIN_PATH + "/surf.ply", 140, 0.033, 0.25, 3., 3);
	processCloud();
	return 0;

}

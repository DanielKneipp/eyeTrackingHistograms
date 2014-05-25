#include "opencv2/imgproc/imgproc.hpp"

/**
 * @brief	Função para fazer o Back Project em HSV para imagens em BGR.
 *			OpenCV_2_Computer_Vision_Application_Programming_Cookbook -> pg. 103
 *
 * @param	image				Imagem onde será gerado o Back Projection baseado
 *								no imageROI
 * @param	hist_imageROI_HSV	Histogramas da imagem de interesse nos canais HSV
 * @param	back_project		Vetor de imagens do resultado do Back Projection
 * @param	maskImage			Máscara para melhorar o resultado do Back Project
 */
void backProject(
	const cv::Mat& image, 
	const cv::MatND* hist_imageROI_HSV,
	cv::MatND* back_projections_HSV,
	const cv::Mat& maskImage);

/**
 * @brief	Usado para gerar os hitogramas necessários para a 
 *			função backProject
 *
 * @param	imageROI			Imagem de interesse em BGR
 * @param	hist_imageROI_HSV	Vetor com os histogramas nos canais HSV
 * @param	maskROI				Máscara usada para descartar pixel que não estejam
 *								na faixa de valores de imageROI 
 *								(gerado pela função mask_backProject)
 */
void hist_backProject (
	const cv::Mat& imageROI, 
	cv::MatND* hist_imageROI_HSV,
	const cv::Mat& maskROI);

/**
 * @brief	Gera a máscara que pode ser usada para melhorar os 
 *			resultados do Back Project
 *
 * @param	image			Imagem que será feita a máscara
 * @param	imageROI		Imagem de interesse que especifica a 
 *							variação em image para gerar a máscara
 * @param	fator_ajuste	Número utilizado para dividir o valor de ajuste
 *							usado para diminuir o valor máximo e aumentar o valor
 *							mínimo obtidos no imageROI para calcular o inRange.
 *							0 desativa o valor de ajuste
 *
 * @return	maskImage		Imagem que reprensenta a máscara feita
 */
cv::Mat mask_backProject(
	const cv::Mat image,
	const cv::Mat imageROI,
	const int* fator_ajuste = 0);
#include "opencv2/imgproc/imgproc.hpp"

/**
 * @brief	Fun��o para fazer o Back Project em HSV para imagens em BGR.
 *			OpenCV_2_Computer_Vision_Application_Programming_Cookbook -> pg. 103
 *
 * @param	image				Imagem onde ser� gerado o Back Projection baseado
 *								no imageROI
 * @param	hist_imageROI_HSV	Histogramas da imagem de interesse nos canais HSV
 * @param	back_project		Vetor de imagens do resultado do Back Projection
 * @param	maskImage			M�scara para melhorar o resultado do Back Project
 */
void backProject(
	const cv::Mat& image, 
	const cv::MatND* hist_imageROI_HSV,
	cv::MatND* back_projections_HSV,
	const cv::Mat& maskImage);

/**
 * @brief	Usado para gerar os hitogramas necess�rios para a 
 *			fun��o backProject
 *
 * @param	imageROI			Imagem de interesse em BGR
 * @param	hist_imageROI_HSV	Vetor com os histogramas nos canais HSV
 * @param	maskROI				M�scara usada para descartar pixel que n�o estejam
 *								na faixa de valores de imageROI 
 *								(gerado pela fun��o mask_backProject)
 */
void hist_backProject (
	const cv::Mat& imageROI, 
	cv::MatND* hist_imageROI_HSV,
	const cv::Mat& maskROI);

/**
 * @brief	Gera a m�scara que pode ser usada para melhorar os 
 *			resultados do Back Project
 *
 * @param	image			Imagem que ser� feita a m�scara
 * @param	imageROI		Imagem de interesse que especifica a 
 *							varia��o em image para gerar a m�scara
 * @param	fator_ajuste	N�mero utilizado para dividir o valor de ajuste
 *							usado para diminuir o valor m�ximo e aumentar o valor
 *							m�nimo obtidos no imageROI para calcular o inRange.
 *							0 desativa o valor de ajuste
 *
 * @return	maskImage		Imagem que reprensenta a m�scara feita
 */
cv::Mat mask_backProject(
	const cv::Mat image,
	const cv::Mat imageROI,
	const int* fator_ajuste = 0);
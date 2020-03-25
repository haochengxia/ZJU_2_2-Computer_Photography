
#include "BorderMatting.h"
using namespace cv;
#define PI  (3.14159)


BorderMatting::BorderMatting(const Mat& originImage, const Mat& mask)
{
	Initialize(originImage, mask);
}
void BorderMatting::drawContour()
{
	for (int i = 0; i < Image.rows; i++) {
		for (int j = 0; j < Image.cols; j++)
		{
			if (Edge.at<uchar>(i, j))
			{
				contourVector.push_back(point(j, i));
			}
		}
	}
}


void BorderMatting::computeNearestPoint()
{
	for (int i = 0; i < Image.rows; i++) {
		for (int j = 0; j < Image.cols; j++)
		{
			if (Edge.at<uchar>(i, j) == 0 && Mask.at<uchar>(i, j) == 1)
			{
				point p(j, i);
				double mindis = INFINITY;
				int max = -1;
				int id = 0;
				for (int k = 0; k < contourVector.size(); k++) {
					double dis = p.distance(contourVector[k].pointInfo);
					if(dis < mindis){
						mindis = dis;
						id = k;
					}
				}
				if (mindis > 3) {
					continue;
				}
				else {
					p.dis = mindis;
					contourVector[id].neighbor.push_back(p);
				}
			}
		}
	}
}




void BorderMatting::Initialize(const Mat& originImage, const Mat& mask, int threadshold_1, int threadshold_2)
{
	mask.copyTo(this->Mask);
	Mask = Mask & 1;
	originImage.copyTo(this->Image);
	Canny(Mask, Edge, threadshold_1, threadshold_2);   //get the edge
	drawContour(); // construct the outline according to the edge information
	computeNearestPoint();
	haveEdge = true;
}


int BorderMatting::computeEdgeDistance(point p)
{
	for (int i = 0; i < contourVector.size(); i++)
	{
		if (p.distance(contourVector[i].pointInfo) < edgeRadius) // 3
			return p.distance(contourVector[i].pointInfo);
	}
	return -1;
}




//The Gaussian parametersμt(α), Σt(α), α = 0, 1 for 
//foregroundand background are estimated as the sample mean and
//covariancefrom each of the regionsFtandBtdefined asFt = St∩TFandBt = St∩TB, 
//whereStis a square region of sizeL×Lpixels centred onthe segmentation boundaryCatt(and we takeL =41）
void BorderMatting::computeMeanVariance(point p, Info &res)
{
	const int halfL = 20; 
	Vec3b backMean, FrontMean;
	double backVariance = 0, frontVariance = 0;
	int frontCounter = 0, backCounter = 0;
	int x = (p.x - halfL < 0) ? 0 : p.x - halfL;
	int width = (x + 2 * halfL + 1 <= Image.cols) ? halfL * 2 + 1 : Image.cols - x;
	int y = (p.y - halfL < 0) ? 0 : p.y - halfL;
	int height = (y + 2 * halfL + 1 <= Image.rows) ? halfL * 2 + 1 : Image.rows - y;
	Mat neiborPixels = Image(Rect(x,y, width, height));
	for (int i = 0; i < neiborPixels.rows; i++) {
		for (int j = 0; j < neiborPixels.cols; j++)
		{
			Vec3b pixelColor = neiborPixels.at<Vec3b>(i, j);
			if (Edge.at<uchar>(y + i, x + j) == 1)
			{
				FrontMean += pixelColor;
				frontCounter++;
			}
			else
			{
				backMean += pixelColor;;
				backCounter++;
			}
		}
	}

	if (frontCounter > 0) {
		FrontMean = FrontMean / frontCounter;
	}
	else {
		FrontMean = 0;
	}
	if (backCounter > 0) {
		backMean = backMean / backCounter;
	}
	else {
		backMean = 0;
	}

	for (int i = 0; i < neiborPixels.rows; i++) {
		for (int j = 0; j < neiborPixels.cols; j++)
		{
			Vec3b pixelColor = neiborPixels.at<Vec3b>(i, j);
			if (Edge.at<uchar>(y + i, x + j) == 1)
				frontVariance += (FrontMean - pixelColor).dot(FrontMean - pixelColor);
			else
				backVariance += (pixelColor - backMean).dot(pixelColor - backMean);
		}
	}

	if (frontCounter >0) {
		frontVariance = frontVariance / frontCounter;
	}
	else {
		frontVariance = 0;
	}
	if (backCounter > 0) {
		backVariance = backVariance / backCounter;
	}
	else {
		backVariance = 0;
	}
	res.backMean = backMean;
	res.backVar = backVariance;
	res.frontMean = FrontMean;
	res.frontVar = frontVariance;
}


double BorderMatting::Gaussian(double x, double mean, double sigma) {
	double res = 1.0 / (pow(sigma, 0.5)*pow(2.0*PI, 0.5))* exp(-(pow(x - mean, 2.0) / (2.0*sigma)));
	return res;
}


//eq 15（1）
double BorderMatting::Mmean(double alfa, double Fmean, double Bmean) {
	return (1.0 - alfa)*Bmean + alfa*Fmean;
}


//eq 15（2）
double BorderMatting::Mvar(double alfa, double Fvar, double Bvar) {
	return (1.0 - alfa)*(1.0 - alfa)*Bvar + alfa*alfa*Fvar;
}



//sigmoidfunction as soft step-function
double BorderMatting::Sigmoid(double dis, double deltaCenter, double sigma) {
	if (dis + deltaCenter -edgeRadius/2 < (deltaCenter - sigma / 2))
		return 0;
	if (dis + deltaCenter - edgeRadius / 2 >= (deltaCenter + sigma / 2))
		return 1;
	double res = -(dis  - edgeRadius / 2) / sigma;

	//double res = -(dis - deltaCenter) / sigma;
	res = exp(res);
	res = 1.0 / (1.0 + res);
	//if (res != 0) std::cout << res << std::endl;
	return res;
}


double BorderMatting::dataTerm(point p, uchar z, int delta, int sigma, Info &para) {
	double alpha = Sigmoid(p.dis,delta,sigma);
	double MmeanValue = Mmean(alpha, valueColor2Gray(para.frontMean), valueColor2Gray(para.backMean));
	double MvarValue  = Mvar(alpha, para.frontVar, para.backVar);
	double D = Gaussian(z, MmeanValue, MvarValue);
	D = -log(D) / log(2.0);
	return D;
}



uchar BorderMatting::valueColor2Gray(Vec3b color)
{ 
	// Y <- 0.299R + 0.587G + 0.114B
	return (color[2] * 299 + color[1] * 587 + color[0] * 114 ) / 1000; 
}


// According to the description in the paper, the level of delta is 30, the level of sigma is 10, traversal is D, and delta and sigma when D is minimized.
void BorderMatting::Run()
{
	
	int delta = MAXDELTA / 2; 
	int sigma = MAXSIGMA / 2;

	for (int i = 0; i < contourVector.size(); i++)
	{
		Info info;
		computeMeanVariance(contourVector[i].pointInfo, info);
		contourVector[i].pointInfo.nearbyInfo = info;
		for (int j = 0; j < contourVector[i].neighbor.size(); j++)
		{
			point &p = contourVector[i].neighbor[j];
			computeMeanVariance(p, info);
			p.nearbyInfo = info;
		}

		double min = INFINITY;
		for (int deltalevel = 0; deltalevel < 30; deltalevel++) {
			for (int sigmalevel = 0; sigmalevel < 10; sigmalevel++)
			{
				double grayValue = valueColor2Gray(Image.at<Vec3b>(contourVector[i].pointInfo.y, contourVector[i].pointInfo.x));
				double D = dataTerm(contourVector[i].pointInfo, grayValue, deltalevel, sigmalevel, contourVector[i].pointInfo.nearbyInfo);
				for (int j = 0; j < contourVector[i].neighbor.size(); j++)
				{
					point &p = contourVector[i].neighbor[j];
					D += dataTerm(p, valueColor2Gray(Image.at<Vec3b>(p.y, p.x)), deltalevel, sigmalevel, p.nearbyInfo);
				}
				double V = lamda1 * (deltalevel - delta)*(deltalevel - delta) + lamda2 * (sigma - sigmalevel)*(sigma - sigmalevel); // 按照论文公式 13
				if (D + V < min)
				{
					min = D + V;
					contourVector[i].pointInfo.delta = deltalevel;
					contourVector[i].pointInfo.sigma = sigmalevel;
				}
			}
		}
		sigma = contourVector[i].pointInfo.sigma;
		delta = contourVector[i].pointInfo.delta;
		std::cout << sigma << std::endl;
		std::cout << delta << std::endl;
		contourVector[i].pointInfo.alpha = Sigmoid(0, delta, sigma);
		for (int j = 0; j < contourVector[i].neighbor.size(); j++)
		{
			point &p = contourVector[i].neighbor[j];
			p.alpha = Sigmoid(p.dis, delta, sigma);
		}
	}

	Mat alphaMask = Mat(Mask.size(), CV_32FC1, Scalar(0));
	for (int i = 0; i < Mask.rows; i++) 
	{
		for (int j = 0; j < Mask.cols; j++) 
		{
			alphaMask.at<float>(i, j) = Mask.at<uchar>(i, j);
		}
	}

	for (int i = 0; i < contourVector.size(); i++)
	{
		alphaMask.at<float>(contourVector[i].pointInfo.y, contourVector[i].pointInfo.x) = contourVector[i].pointInfo.alpha;
		for (int j = 0; j < contourVector[i].neighbor.size(); j++)
		{
			point &p = contourVector[i].neighbor[j];
			alphaMask.at<float>(p.y, p.x) = p.alpha;
			//if (p.alpha != 0 && p.alpha != 1) std::cout <<"p"<< p.alpha << std::endl;
		}
	}
	Mat rst = Mat(Image.size(), CV_8UC4);
	for (int i = 0; i < rst.rows; i++) {
		for (int j = 0; j < rst.cols; j++)
		{
			if (alphaMask.at<float>(i, j) * 255)
			{	if (alphaMask.at<float>(i, j)!=1 && alphaMask.at<float>(i, j) != 0)  std::cout << (alphaMask.at<float>(i, j) ) << std::endl;
				rst.at<Vec4b>(i, j) = Vec4b(Image.at<Vec3b>(i, j)[0], Image.at<Vec3b>(i, j)[1], Image.at<Vec3b>(i, j)[2], (int)(alphaMask.at<float>(i, j) * 255)); //*255
				//if ((unsigned char)(alphaMask.at<float>(i, j) * 255) && (int)(alphaMask.at<float>(i, j) * 255) != 255) std::cout << (int)(alphaMask.at<float>(i, j) * 255) << std::endl;
			}
			else {
				rst.at<Vec4b>(i, j) = Vec4b(0, 0, 0, 0);
			}
		}
	}
	imshow("bordingmatting running", rst);imwrite("b image.png", rst);
	std::cout << "bordingmatting done!" << std::endl;
}

void BorderMatting::showEdge() {
	if (!haveEdge) {
		return;
	}
	imshow("canny",Edge);
	
}
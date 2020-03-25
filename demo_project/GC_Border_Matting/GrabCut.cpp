
#include <chrono>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "GrabCut.h"
#include "GMM.h"
#include "graph.h"
#include <time.h>
using namespace std;

GrabCut2D::~GrabCut2D(void)
{
}

//一.参数解释：
//输入：
//cv::InputArray _img,     :输入的color图像(类型-cv:Mat)
//cv::Rect rect            :在图像上画的矩形框（类型-cv:Rect) 
//int iterCount :           :每次分割的迭代次数（类型-int)


//中间变量
//cv::InputOutputArray _bgdModel ：   背景模型（推荐GMM)（类型-13*n（组件个数）个double类型的自定义数据结构，可以为cv:Mat，或者Vector/List/数组等）
//cv::InputOutputArray _fgdModel :    前景模型（推荐GMM) （类型-13*n（组件个数）个double类型的自定义数据结构，可以为cv:Mat，或者Vector/List/数组等）


//输出:
//cv::InputOutputArray _mask  : 输出的分割结果 (类型： cv::Mat)

//二. 伪代码流程：
//1.Load Input Image: 加载输入颜色图像;
//2.Init Mask: 用矩形框初始化Mask的Label值（确定背景：0， 确定前景：1，可能背景：2，可能前景：3）,矩形框以外设置为确定背景，矩形框以内设置为可能前景;
//3.Init GMM: 定义并初始化GMM(其他模型完成分割也可得到基本分数，GMM完成会加分）
//4.Sample Points:前背景颜色采样并进行聚类（建议用kmeans，其他聚类方法也可)
//5.Learn GMM(根据聚类的样本更新每个GMM组件中的均值、协方差等参数）
//6.Construct Graph（计算t-weight(数据项）和n-weight（平滑项））
//7.Estimate Segmentation(调用maxFlow库进行分割)
//8.Save Result输入结果（将结果mask输出，将mask中前景区域对应的彩色图像保存和显示在交互界面中）




//Calculate beta, which is the beta of the exponential term in the second term(smooth term) 
//of the Gibbs energy term, used to adjust the difference between two neighborhood pixels when
//adjusting for high or low contrast, such as at low contrast.The difference between the two neighborhood 
//pixels may be small.In this case, you need to multiply by a larger beta to amplify the difference.In high contrast, 
//you need to narrow the difference.So we need to analyze the contrast of the entire image to determine the parameter beta, 
//see the paper formula(5).
/*
Calculate beta - parameter of GrabCut algorithm.
beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
*/

double GrabCut2D::computeBeta(const cv::Mat &img)
{
	double beta = 0;
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			cv::Vec3d color = img.at<cv::Vec3b>(y, x);
			if (x > 0) // left
			{
				cv::Vec3d diff = color - (cv::Vec3d) img.at<cv::Vec3b>(y, x - 1);
				beta += diff.dot(diff);
			}
			if (y > 0 && x > 0) // upleft
			{
				cv::Vec3d diff = color - (cv::Vec3d) img.at<cv::Vec3b>(y - 1, x - 1);
				beta += diff.dot(diff);
			}
			if (y > 0) // up
			{
				cv::Vec3d diff = color - (cv::Vec3d) img.at<cv::Vec3b>(y - 1, x);
				beta += diff.dot(diff);
			}
			if (y > 0 && x < img.cols - 1) // upright
			{
				cv::Vec3d diff = color - (cv::Vec3d) img.at<cv::Vec3b>(y - 1, x + 1);
				beta += diff.dot(diff);
			}
		}
	}
	if (beta <= std::numeric_limits<double>::epsilon())
		beta = 0;
	else
		beta = 1.f / (2 * beta / (4 * img.cols * img.rows - 3 * img.cols - 3 * img.rows + 2));

	return beta;
}

void GrabCut2D::GrabCut(const cv::Mat &img, cv::Mat &mask, cv::Rect rect,
                        cv::Mat &bgdModel, cv::Mat &fgdModel,
                        int iterCount, int mode)
{
	if (iterCount <= 0)
		return;

	clock_t start, end;
	start = clock();


	// Init GMM 
    GMM bgdGMM, fgdGMM; // Prospect Gaussian model and background Gaussian model

    if (mode == GC_WITH_RECT || mode == GC_WITH_MASK) {
		if (mode == GC_WITH_RECT) {
			initMaskWithRect(mask, img.size(), rect); // Frame image and get the rect
		}
        initGMMs(img, mask, bgdGMM, fgdGMM);
    }
    else if(mode == GC_CUT)
    {
        bgdGMM = GMM::matToGMM(bgdModel); // copy GMM model
        fgdGMM = GMM::matToGMM(fgdModel);
    }


	//Construct Graph(Calculate t-weight (data term) and n-weight (smooth term)
	cv::Mat leftW, upleftW, upW, uprightW;
    const double gamma = 50;
    const double lambda = 9 * gamma;
    const double beta = computeBeta(img);
    computeEdgeWeights(img, leftW, upleftW, upW, uprightW, beta, gamma);
    int vertexCount = img.cols * img.rows;// 顶点数
    int edgeCount = 2 * (4 * img.cols * img.rows - 3 * (img.cols + img.rows) + 2); 
	cv::Mat gaussComponentIdMask(img.size(), CV_32SC1); 
    for (int i = 0; i < iterCount; i++) {
        Graph<double, double, double> graph(vertexCount, edgeCount);
		// Calculate a new mask for each pixel's Gaussian component and store it in gaussComponentIdMask
        assignGMMsComponents(img, mask, bgdGMM, fgdGMM, gaussComponentIdMask);
        learnGMMs(img, mask, gaussComponentIdMask, bgdGMM, fgdGMM); 
        buildGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph);
		//Estimate Segmentation
        estimateSegmentation(graph, mask);// Call the maxflow library to cut, update the mask, and finally output the mask
    }

	//Save Result Input result (output the result mask, save and display the color image corresponding to the foreground area in the mask in the interactive interface)
    fgdModel = fgdGMM.GMMtoMat(); // update model
    bgdModel = bgdGMM.GMMtoMat();

    end = clock();
    cout << "time cost: " << (double)(end - start)/CLOCKS_PER_SEC << "s\n";
}


void GrabCut2D::initMaskWithRect(cv::Mat &mask, cv::Size size, cv::Rect rect)
{
    mask.create(size, CV_8UC1);
    mask.setTo(cv::GC_BGD);
    rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);
    rect.width = std::min(rect.width, size.width - rect.x);
    rect.height = std::min(rect.height, size.height - rect.y);
    (mask(rect)).setTo(cv::Scalar(cv::GC_PR_FGD));
}



/*****
* 建立背景高斯模型和前景高斯模型
* 2.Init Mask: 用矩形框初始化Mask的Label值（确定背景：0， 确定前景：1，可能背景：2，可能前景：3）,矩形框以外设置为确定背景，矩形框以内设置为可能前景;
* 3.Init GMM: 定义并初始化GMM(其他模型完成分割也可得到基本分数，GMM完成会加分）
*/


void GrabCut2D::initGMMs(const cv::Mat &img, const cv::Mat &mask, GMM &fgdGMM, GMM &bgdGMM)
{
    std::vector<cv::Vec3f> backgroundPixel;
    std::vector<cv::Vec3f> frontgroundPixel;
	cv::Mat bgdLabels;
	cv::Mat fgdLabels;

	vector<cv::Vec3b> meansrecord;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
			if (mask.at<uchar>(i, j) == cv::GC_BGD || mask.at<uchar>(i, j) == cv::GC_PR_BGD) {
				backgroundPixel.push_back((cv::Vec3f) img.at<cv::Vec3b>(i, j));
			}
			else {
				frontgroundPixel.push_back((cv::Vec3f) img.at<cv::Vec3b>(i, j));
			}
        }
    }
    cv::Mat bgdSampleMat( backgroundPixel.size(), 3, CV_32FC1, &backgroundPixel[0][0]);
    cv::Mat fgdSampleMat( frontgroundPixel.size(), 3, CV_32FC1, &frontgroundPixel[0][0]);
	// kmeans
    cv::kmeans(bgdSampleMat, bgdGMM.getComponentsCount(), bgdLabels, // 5 components
               cv::TermCriteria(cv::TermCriteria::MAX_ITER, 10, 0.0), 0, cv::KMEANS_PP_CENTERS);
    cv::kmeans(fgdSampleMat, fgdGMM.getComponentsCount(), fgdLabels,
               cv::TermCriteria(cv::TermCriteria::MAX_ITER, 10, 0.0), 0, cv::KMEANS_PP_CENTERS);

	// Learn GMM
    bgdGMM.initLearning();
    for (int i = 0; i < backgroundPixel.size(); i++) {
        bgdGMM.addPixel(bgdLabels.at<int>(i, 0), backgroundPixel[i]); 
    }
    bgdGMM.endLearning();

    fgdGMM.initLearning();
    for (int i = 0; i < frontgroundPixel.size(); i++) {
        fgdGMM.addPixel(fgdLabels.at<int>(i, 0), frontgroundPixel[i]); 
    }
    fgdGMM.endLearning();
}



/************
* edge weight
*/
void GrabCut2D::computeEdgeWeights(const cv::Mat &img, cv::Mat &leftW, cv::Mat &upleftW,
                             cv::Mat &upW, cv::Mat &uprightW, double beta, double gamma)
{
    leftW.create(img.rows, img.cols, CV_64FC1);
    upleftW.create(img.rows, img.cols, CV_64FC1);
    upW.create(img.rows, img.cols, CV_64FC1);
    uprightW.create(img.rows, img.cols, CV_64FC1);
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            cv::Vec3d color = img.at<cv::Vec3b>(y, x);
            if (x > 0) // left
            {
                cv::Vec3d diff = color - (cv::Vec3d) img.at<cv::Vec3b>(y, x - 1);
                leftW.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));
            }
            else
                leftW.at<double>(y, x) = 0;
            if (x > 0 && y > 0) // upleft
            {
                cv::Vec3d diff = color - (cv::Vec3d) img.at<cv::Vec3b>(y - 1, x - 1);
                upleftW.at<double>(y, x) = gamma / sqrt(2.0) * exp(-beta * diff.dot(diff));
            }
            else
                upleftW.at<double>(y, x) = 0;
            if (y > 0) // up
            {
                cv::Vec3d diff = color - (cv::Vec3d) img.at<cv::Vec3b>(y - 1, x);
                upW.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));
            }
            else
                upW.at<double>(y, x) = 0;
            if (x < img.cols - 1 && y > 0) // upright
            {
                cv::Vec3d diff = color - (cv::Vec3d) img.at<cv::Vec3b>(y - 1, x + 1);
                uprightW.at<double>(y, x) = gamma / sqrt(2.0) * exp(-beta * diff.dot(diff));
            }
            else
                uprightW.at<double>(y, x) = 0;
        }
    }

}

/****
* Calculate a mask of img size to store which Gaussian component each pixel belongs to
*/
void GrabCut2D::assignGMMsComponents(const cv::Mat &img, const cv::Mat &mask,
                                     const GMM &bgdGMM, const GMM &fgdGMM, cv::Mat &gaussCompIdMask)
{
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            cv::Vec3d color = img.at<cv::Vec3b>(y, x);
			if (mask.at<uchar>(y, x) == cv::GC_BGD || mask.at<uchar>(y, x) == cv::GC_PR_BGD) {
				gaussCompIdMask.at<int>(y, x) = bgdGMM.mostPossibleComponent(color); 
			}
			else {
				gaussCompIdMask.at<int>(y, x) = fgdGMM.mostPossibleComponent(color);
			}
        }
    }
}

// according Mask, retrain
void GrabCut2D::learnGMMs(const cv::Mat &img, const cv::Mat &mask, const cv::Mat &gaussCompIdMask,
                          GMM &bgdGMM, GMM &fgdGMM)
{
    bgdGMM.initLearning();
    fgdGMM.initLearning();
    for (int i = 0; i < bgdGMM.getComponentsCount(); i++) {
        for (int y = 0; y < img.rows; y++) {
            for (int x = 0; x < img.cols; x++) {
                if (gaussCompIdMask.at<int>(y, x) == i) {
                    if (mask.at<uchar>(y, x) == cv::GC_BGD || mask.at<uchar>(y, x) == cv::GC_PR_BGD)
                        bgdGMM.addPixel(i, img.at<cv::Vec3b>(y, x));
                    else
                        fgdGMM.addPixel(i, img.at<cv::Vec3b>(y, x));
                }
            }
        }
    }
    bgdGMM.endLearning();
    fgdGMM.endLearning();
}



/***
* Construct GCGraph
*/
void GrabCut2D::buildGraph(const cv::Mat &img, const cv::Mat &mask, const GMM &bgdGMM,
                                 const GMM &fgdGMM, double lambda, const cv::Mat &leftW,
                                 const cv::Mat &upleftW, const cv::Mat &upW, const cv::Mat &uprightW,
                                 Graph<double, double, double> &graph)
{
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            // add node
            int vertexId = graph.add_node();
            cv::Vec3b color = img.at<cv::Vec3b>(y, x);
            // set t-weights
            double fromSource, toSink;
            if (mask.at<uchar>(y, x) == cv::GC_PR_BGD || mask.at<uchar>(y, x) == cv::GC_PR_FGD) {
                fromSource = -log(bgdGMM(color));
                toSink = -log(fgdGMM(color));
            }
            else if (mask.at<uchar>(y, x) == cv::GC_BGD) {
                fromSource = 0;
                toSink = lambda;
            }
            else // GC_FGD
            {
                fromSource = lambda;
                toSink = 0;
            }
            graph.add_tweights(vertexId, fromSource, toSink);

            // set n-weights
			double edgeWeight;
            if (x > 0) { // left
                edgeWeight = leftW.at<double>(y, x);
                graph.add_edge(vertexId, vertexId - 1, edgeWeight, edgeWeight);
            }
            if (x > 0 && y > 0) { // upleft
                edgeWeight = upleftW.at<double>(y, x);
                graph.add_edge(vertexId, vertexId - img.cols - 1, edgeWeight, edgeWeight);
            }
            if (y > 0) { // up
                edgeWeight = upW.at<double>(y, x);
                graph.add_edge(vertexId, vertexId - img.cols, edgeWeight, edgeWeight);
            }
            if (x < img.cols - 1 && y > 0) { // upright
                edgeWeight = uprightW.at<double>(y, x);
                graph.add_edge(vertexId, vertexId - img.cols + 1, edgeWeight, edgeWeight);
            }
        }
    }
}

/***
* Estimate segmentation using MaxFlow algorithm 
*/
void GrabCut2D::estimateSegmentation(Graph<double, double, double> &graph, cv::Mat &mask)
{
    graph.maxflow();
    for (int y = 0; y < mask.rows; y++) {
        for (int x = 0; x < mask.cols; x++) {
            if (mask.at<uchar>(y, x) == cv::GC_PR_BGD || mask.at<uchar>(y, x) == cv::GC_PR_FGD) {
                if (graph.what_segment(y * mask.cols + x ) == Graph<double, double, double>::SOURCE)
                    mask.at<uchar>(y, x) = cv::GC_PR_FGD;
                else
                    mask.at<uchar>(y, x) = cv::GC_PR_BGD;
            }
        }
    }
}




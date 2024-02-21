#ifndef LINEAR_OSMOSIS
#define LINEAR_OSMOSIS

#include <opencv2/highgui.hpp>
#include <petsc.h>


class LinearOsmosis
{
    public:
    
        LinearOsmosis(cv:: Mat u);
        void computeOsmosisWeights(double hx, double hy, cv::Mat d1, cv::Mat d2);
        void performOsmosis(cv:: Mat &u, long time_step_size, long max_epoch);

    private:
        long sizeX;
        long sizeY;
        cv::Mat oneDimIndices;
        Mat petscA;
        KSP petscKsp;   
};

#endif
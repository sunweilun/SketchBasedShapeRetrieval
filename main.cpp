/* 
 * File:   main.cpp
 * Author: swl
 *
 * Created on November 29, 2015, 11:52 AM
 */

#include <cstdlib>
#include <opencv2/highgui/highgui.hpp>
#include "SketchRenderer.h"
#include "Retriever.h"
#include "utils.h"

using namespace std;

int main(int argc, char** argv) 
{
    std::string libPath = "/home/swl/Documents/Data/psb/";
    Retriever retriever;
    retriever.init();
    retriever.setLibPath(libPath);
    retriever.train(libPath+"cluster.bin");
    
    retriever.saveTrainingData(libPath+"sbsr.bin");
    retriever.loadTrainingData(libPath+"sbsr.bin");
    
    cv::Mat sketch = cv::imread("/home/swl/Documents/Data/test_sketches/lamp", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat1f input = sketch / 255.0;
    std::vector<Retriever::RetrievalInfo> retrievalList = retriever.retrieveAll(input);
    for(unsigned i=0; i<retrievalList.size(); i++)
    {
        printf("%s -- %f\n", retrievalList[i].path.c_str(), retrievalList[i].score);
        SketchRenderer::load(retrievalList[i].path.c_str());
        cv::Mat1f sketch = SketchRenderer::genSketch(
                retrievalList[i].front, 
                retrievalList[i].up
                );
        dbg(sketch);
    }
    
    return 0;
}


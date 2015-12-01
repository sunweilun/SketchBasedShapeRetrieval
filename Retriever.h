/* 
 * File:   Retriever.h
 * Author: swl
 *
 * Created on November 30, 2015, 11:32 AM
 */

#ifndef RETRIEVER_H
#define	RETRIEVER_H

#include <string>
#include "SketchRenderer.h"

class Retriever
{
protected:
    bool nnApprox;
    float patchAreaRatio;
    unsigned dictSize;
    unsigned patchNum;
    unsigned tileNum;
    unsigned thetaNum;
    unsigned phiNum;
    unsigned clusterFeatNum;
    unsigned oriNum;
    std::string libPath;
    cv::Mat1f dict;
    cv::KDTree dictTree;
    std::vector<float> idfWeights;
    
    
    struct ViewInfo
    {
        glm::vec3 front;
        glm::vec3 up;
        std::vector<unsigned> hist;
        void write(FILE*& file);
        void read(FILE*& file);
    };
    struct ModelInfo
    {
        std::string path;
        std::vector<ViewInfo> views;
        void write(FILE*& file);
        void read(FILE*& file);
    };
    struct RetrievalIndexInfo
    {
        float score;
        unsigned modelIndex;
        unsigned viewIndex;
        bool operator<(const RetrievalIndexInfo& info) const
        {
            return score > info.score;
        }
    };
    std::vector<ModelInfo> models;
    void findModelFilesInLibrary(const std::string& root,
        std::vector<std::string>& pathList);
    virtual ModelInfo generateModelInfo(const std::string& path);
public:
    struct RetrievalInfo
    {
        float score;
        std::string path;
        glm::vec3 front;
        glm::vec3 up;
    };
    void init();
    void setLibPath(const std::string& libPath) { this->libPath = libPath; }
    void train(const std::string& libPath);
    void saveTrainingData(const std::string& path);
    void loadTrainingData(const std::string& path);
    virtual cv::Mat1f extractFeatures(const cv::Mat1f& sketch);
    std::vector<RetrievalInfo> retrieveAll(cv::Mat1f& input);
    virtual cv::Mat1f getClusterFeatures();
    virtual void filterFeatures(cv::Mat1f& features);
    virtual void makeDict(const cv::Mat1f& clusterFeatures);
    std::vector<unsigned> getHistFromFeatures(const cv::Mat1f& features);
    std::vector<unsigned> getHistFromSketch(const cv::Mat1f& sketch);
    cv::Mat1f hist2vec(const std::vector<unsigned>& hist);
};

#endif	/* RETRIEVER_H */


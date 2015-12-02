#include "Retriever.h"
#include "utils.h"
#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <iostream>
#include <set>

void Retriever::init()
{
    SketchRenderer::init();
    thetaNum = 8;
    phiNum = 7;
    patchNum = 32;
    tileNum = 4;
    dictSize = 1000;
    clusterFeatNum = 1e5;
    patchAreaRatio = 0.1;
    oriNum = 4;
    nnApprox = true;
}

void
 Retriever::findModelFilesInLibrary(const std::string& root,
        std::vector<std::string>& pathList)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir((libPath+root).c_str())) == NULL) {
        std::cout << "Error(" << errno << ") opening " << libPath
                << root << std::endl;

    }

    while ((dirp = readdir(dp)) != NULL) {
        std::string path = std::string(dirp->d_name);
        if(path == "." || path == "..")
            continue;
        if(int(dirp->d_type) == 4)
        {
            findModelFilesInLibrary(root+path+"/", pathList);
            continue;
        }
        
        if(path.substr(path.length()-4, 4) == ".off")
        {
            pathList.push_back(root+path);
        } 
    }
    closedir(dp);
}

Retriever::ModelInfo 
Retriever::generateModelInfo(const std::string& path)
{
    ModelInfo mi;
    mi.path = path;
    for(unsigned p=0; p<phiNum; p++)
    {
        for(unsigned t=0; t<thetaNum; t++)
        {
        
            float theta = t / float(thetaNum) * 2 * M_PI;
            float phi = (p+0.5) / float(phiNum) * M_PI;
            ViewInfo vi;
            vi.front.x = sin(phi)*cos(theta);
            vi.front.y = cos(phi);
            vi.front.z = sin(phi)*sin(theta);
            vi.up = glm::vec3(0, 1, 0);
            vi.hist.resize(dictSize);
            mi.views.push_back(vi);
        }
    }
    return mi;
}

std::vector<unsigned> Retriever::getHistFromSketch(const cv::Mat1f& sketch)
{
    cv::Mat1f blurred;
    cv::GaussianBlur(sketch, blurred, cv::Size(0, 0), 1.0, 0.0);
    cv::Mat1f features = extractFeatures(blurred);
    filterFeatures(features);
    return getHistFromFeatures(features);
}

std::vector<unsigned> Retriever::getHistFromFeatures(const cv::Mat1f& features)
{
    std::vector<unsigned> hist(dictSize, 0.0);
    for(unsigned f=0; f<features.rows; f++)
    {
        unsigned index = 0;
        float min_dist = 1e5;
        
        if(nnApprox)
        {
            cv::Mat neighborIndex;
            cv::Mat dist;
            dictTree.knnSearch(features.row(f), neighborIndex, dist, 1);
            index = neighborIndex.at<int>(0, 0);
        }
        else
        {
            for(unsigned d=0; d<dict.rows; d++)
            {
                float dist = cv::norm(features.row(f)-dict.row(d));
                if(dist < min_dist)
                {
                    min_dist = dist;
                    index = d;
                }
            }
        }
        hist[index] ++;
    }
    /*for(unsigned t=0; t<dictSize; t++)
        printf("%0.f ", hist[t]);
    printf("\n");*/
    return hist;
}

void Retriever::train(const std::string& clusterMatPath, const std::string& dictMatPath)
{
    std::vector<std::string> pathList;
    findModelFilesInLibrary("", pathList);
    
    models.resize(pathList.size());
    
    for(unsigned i=0; i<pathList.size(); i++)
    {
        ModelInfo mi = generateModelInfo(pathList[i]);
        models[i] = mi;
    }
    
    cv::Mat1f clusterFeatures;
    
    if( access(clusterMatPath.c_str(), F_OK) != -1 ) 
    {
    // file exists
        readMat(clusterMatPath.c_str(), clusterFeatures);
    } 
    else 
    {
    // file doesn't exist
        clusterFeatures = getClusterFeatures();
        writeMat(clusterMatPath.c_str(), clusterFeatures);
    }
    
    if( access(dictMatPath.c_str(), F_OK) != -1 ) 
    {
    // file exists
        readMat(dictMatPath.c_str(), dict);
    } 
    else 
    {
    // file doesn't exist
        printf("Making dictionary...\n");
        makeDict(clusterFeatures);
        printf("Done.\n");
        writeMat(dictMatPath.c_str(), dict);
    }
    
    if(nnApprox)
    {
        printf("Building tree...\n");
        dictTree.build(dict, cv::flann::KDTreeIndexParams(), 
                cvflann::FLANN_DIST_EUCLIDEAN);
        printf("Done.\n");
    }
    
    std::vector<unsigned> freq(dictSize);
    
    idfWeights.resize(dictSize);
    
    unsigned viewSize = 0;
    
    for(unsigned i=0; i<models.size(); i++)
    {
        ModelInfo& mi = models[i];
        std::string path = libPath + models[i].path;
        SketchRenderer::load(path.c_str());
        printf("Processing model #%u / %u\n", i+1, models.size());
        for(int k=0; k<mi.views.size(); k++)
        {
            cv::Mat1f sketch = SketchRenderer::genSketch(
                    mi.views[k].front, mi.views[k].up);
            
            mi.views[k].hist = getHistFromSketch(sketch);
            
            for(unsigned t=0; t<dictSize; t++)
            {
                if(mi.views[k].hist[t] > 0)
                    freq[t]++;
            }
            
            viewSize++;
        }
    }
    
    for(int i=0; i<dictSize; i++)
    {
        idfWeights[i] = log(viewSize/float(freq[i]));
    }
}

void Retriever::filterFeatures(cv::Mat1f& features)
{
    unsigned j=0;
    for(unsigned i=0; i<features.rows; i++)
    {
        double min, max;
        cv::minMaxLoc(features.row(i), &min, &max);
        if(-min >= 1e-5 || max >= 1e-5)
        {
            features.row(i).copyTo(features.row(j));
            cv::normalize(features.row(j), features.row(j));
            j++;
        }
    }
    
    cv::Mat1f subFeatures = features.rowRange(0, j);
    subFeatures.copyTo(features);
}

void Retriever::makeDict(const cv::Mat1f& clusterFeatures)
{
    cv::Mat labels;
    cv::TermCriteria criteria(cv::TermCriteria::COUNT, 100, 1);
    cv::kmeans(clusterFeatures, dictSize, labels, criteria, 1, 
            cv::KMEANS_RANDOM_CENTERS, dict);
}

cv::Mat1f Retriever::getClusterFeatures()
{
    cv::Mat1f clusterFeatures(clusterFeatNum, tileNum*tileNum*oriNum);
    
    unsigned seed = 323;
    unsigned featureCount = 0;
    
    for(unsigned i=0; i<models.size(); i++)
    {
        const ModelInfo& mi = models[i];
        std::string path = libPath + models[i].path;
        SketchRenderer::load(path.c_str());
        printf("Processing model #%u / %u\n", i+1, models.size());
        for(unsigned k=0; k<mi.views.size(); k++)
        {
            cv::Mat1f sketch = SketchRenderer::genSketch(
                    mi.views[k].front, mi.views[k].up);
            
            cv::GaussianBlur(sketch, sketch, cv::Size(0, 0), 1.0, 0.0);
            cv::Mat1f features = extractFeatures(sketch);
            filterFeatures(features);
            
            for(unsigned f=0; f<features.rows; f++)
            {
                if(featureCount < clusterFeatNum)
                {
                    features.row(f).copyTo(clusterFeatures.row(featureCount));
                }
                else
                {
                    unsigned targetIndex = urand(seed) % featureCount;
                    if(targetIndex < clusterFeatNum)
                        features.row(f).copyTo(clusterFeatures.row(targetIndex));
                }
                featureCount++;
            }
        }
        
    }
    
    return clusterFeatures;
}

cv::Mat1f Retriever::extractFeatures(const cv::Mat1f& sketch)
{
    cv::Mat1f features(patchNum*patchNum, tileNum*tileNum*oriNum);
    features.setTo(0.0);
    unsigned kernel_size = 20;
    float sigma = 1.0;
    float lambda = 2.0;
    float gamma = 0.3;
    float psi = 0.0;
    
#pragma omp parallel for
    for(int k=0; k<oriNum; k++)
    {
        float theta = 2.0*M_PI*k/oriNum;
        cv::Mat1f kernel = cv::getGaborKernel(
                cv::Size(kernel_size, kernel_size), 
                sigma, theta, lambda, gamma, psi);
        cv::Mat1f filtered(sketch.rows, sketch.rows);
        cv::filter2D(sketch, filtered, CV_32F, kernel);
        float patchLength = sqrt(patchAreaRatio)*sketch.rows;
        float tileLength = patchLength/tileNum;
        float patchStep = (sketch.rows - patchLength) / patchNum;
        cv::blur(filtered, filtered, cv::Size(tileLength, tileLength));
        for(unsigned i=0; i<patchNum; i++)
        {
            for(unsigned j=0; j<patchNum; j++)
            {
                for(unsigned x=0; x<tileNum; x++)
                {
                    for(unsigned y=0; y<tileNum; y++)
                    {
                        cv::Point2f pt;
                        pt.x = i*patchStep+(x+0.5)*tileLength;
                        pt.y = j*patchStep+(y+0.5)*tileLength;
                        cv::Mat pixel;
                        cv::getRectSubPix(filtered, cv::Size(1,1), pt, pixel);
                        float value = pixel.at<float>(0,0);
                        features.at<float>(i*patchNum+j, 
                                k*tileNum*tileNum+x*tileNum+y) = value;
                    }
                }
            }
        }
    }
    return features;
}



void Retriever::saveTrainingData(const std::string& path)
{
    FILE* file = fopen(path.c_str(), "wb");
    
    // write other parameters
    writeType<bool>(file, nnApprox);
    writeType<float>(file, patchAreaRatio);
    writeType<unsigned>(file, dictSize);
    writeType<unsigned>(file, patchNum);
    writeType<unsigned>(file, tileNum);
    writeType<unsigned>(file, thetaNum);
    writeType<unsigned>(file, phiNum);
    writeType<unsigned>(file, clusterFeatNum);
    writeType<unsigned>(file, oriNum);
    
    // write model info
    writeType<unsigned>(file, models.size());
    for(unsigned i=0; i<models.size(); i++)
        models[i].write(file);
    
    // write dictionary
    writeMat(file, dict);
    
    // write idf weights
    writeVec<float>(file, idfWeights);
    
    writeString(file, libPath);
    
    fclose(file);
}

void Retriever::loadTrainingData(const std::string& path)
{
    FILE* file = fopen(path.c_str(), "rb");
    
    // read parameters
    readType<bool>(file, nnApprox);
    readType<float>(file, patchAreaRatio);
    readType<unsigned>(file, dictSize);
    readType<unsigned>(file, patchNum);
    readType<unsigned>(file, tileNum);
    readType<unsigned>(file, thetaNum);
    readType<unsigned>(file, phiNum);
    readType<unsigned>(file, clusterFeatNum);
    readType<unsigned>(file, oriNum);
    
    // read model info
    unsigned size;
    readType<unsigned>(file, size);
    
    models.resize(size);
    for(unsigned i=0; i<models.size(); i++)
        models[i].read(file);
    
    // read dictionary
    readMat(file, dict);
    
    // read idf weights
    readVec<float>(file, idfWeights);
    
    if(nnApprox)
    {
        printf("Building tree...\n");
        dictTree.build(dict, cv::flann::KDTreeIndexParams(), 
                cvflann::FLANN_DIST_EUCLIDEAN);
        printf("Done.\n");
    }
    
    readString(file, libPath);
    
    fclose(file);
}

cv::Mat1f Retriever::hist2vec(const std::vector<unsigned>& hist)
{
    cv::Mat1f vec(1, hist.size());
    for(unsigned i=0; i<hist.size(); i++)
        vec.at<float>(0, i) = hist[i] * idfWeights[i];
    cv::normalize(vec, vec);
    return vec;
}

void Retriever::buildTopKDTree()
{
    modelVecs = cv::Mat1f(thetaNum*phiNum*models.size(), dictSize);
    
    modelVecs.setTo(0.0f);
    for(unsigned i=0; i<models.size(); i++)
    {
        const ModelInfo& mi = models[i];
        for(unsigned j=0; j<mi.views.size(); j++)
        {
            hist2vec(mi.views[j].hist).copyTo(
                    modelVecs.row(i*thetaNum*phiNum+j));
        }
    }
   
    vecTree.build(modelVecs, cv::flann::KDTreeIndexParams(), 
                cvflann::FLANN_DIST_EUCLIDEAN);
}

std::vector<Retriever::RetrievalInfo> 
Retriever::retrieveTop(const cv::Mat1f& input, unsigned k)
{
    std::vector<Retriever::RetrievalInfo> retrievalList(k);
    std::vector<unsigned> hist = getHistFromSketch(input);
    
    cv::Mat1f vec = hist2vec(hist);
    
    cv::Mat neighborIndices;
    cv::Mat dist;
    
    vecTree.knnSearch(vec, neighborIndices, dist, k*3,
            cv::flann::SearchParams(512));
    
    std::set<unsigned> modelIndices;
    unsigned j=0;
    for(unsigned i=0; i<neighborIndices.cols; i++)
    {
        int index = neighborIndices.at<int>(i);
        printf("index = %u\n", index);
        unsigned mid = index / (thetaNum*phiNum);
        if(modelIndices.find(mid) == modelIndices.end())
        {
            modelIndices.insert(mid);
            neighborIndices.at<int>(j) = neighborIndices.at<int>(i);
            dist.at<float>(j) = dist.at<float>(i);
            j++;
        }
    }
    
    for(unsigned i=0; i<k; i++)
    {
        int index = neighborIndices.at<int>(i);
        printf("index = %u\n", index);
        unsigned mid = index / (thetaNum*phiNum);
        const ModelInfo& mi = models[mid];
        unsigned vid = index % (thetaNum*phiNum);
        const ViewInfo& vi = mi.views[vid];
        retrievalList[i].path = libPath+mi.path;
        retrievalList[i].front = vi.front;
        retrievalList[i].up = vi.up;
        retrievalList[i].score = dist.at<float>(i);
    }
    return retrievalList;
}

std::vector<Retriever::RetrievalInfo> 
Retriever::retrieveAll(const cv::Mat1f& input)
{
    std::vector<unsigned> hist = getHistFromSketch(input);
    cv::Mat1f vec = hist2vec(hist);
    
    std::vector<RetrievalIndexInfo> retrievalIndexList(models.size());
    for(unsigned i=0; i<models.size(); i++)
    {
        const ModelInfo& mi = models[i];
        retrievalIndexList[i].score = 0.0;
        retrievalIndexList[i].modelIndex = i;
        for(unsigned j=0; j<mi.views.size(); j++)
        {
            float score = vec.dot(hist2vec(mi.views[j].hist));
            if(score > retrievalIndexList[i].score)
            {
                retrievalIndexList[i].score = score;
                retrievalIndexList[i].viewIndex = j;
            }
        }
    }
    std::sort(retrievalIndexList.begin(), retrievalIndexList.end());
    
    std::vector<RetrievalInfo> retrievalList(retrievalIndexList.size());
    
    for(unsigned i=0; i<retrievalList.size(); i++)
    {
        const ModelInfo& mi = models[retrievalIndexList[i].modelIndex];
        retrievalList[i].path = libPath+mi.path;
        const ViewInfo& vi = mi.views[retrievalIndexList[i].viewIndex];
        retrievalList[i].front = vi.front;
        retrievalList[i].up = vi.up;
        retrievalList[i].score = retrievalIndexList[i].score;
    }
   
    return retrievalList;
}

void Retriever::ModelInfo::write(FILE*& file)
{
    writeString(file, path);
    writeType<unsigned>(file, views.size());
    for(unsigned i=0; i<views.size(); i++)
        views[i].write(file);
}

void Retriever::ModelInfo::read(FILE*& file)
{
    readString(file, path);
    unsigned size;
    readType<unsigned>(file, size);
    views.resize(size);
    for(unsigned i=0; i<views.size(); i++)
        views[i].read(file);
}

void Retriever::ViewInfo::write(FILE*& file)
{
    writeType<glm::vec3>(file, front);
    writeType<glm::vec3>(file, up);
    writeVec<unsigned>(file, hist);
}

void Retriever::ViewInfo::read(FILE*& file)
{
    readType<glm::vec3>(file, front);
    readType<glm::vec3>(file, up);
    readVec<unsigned>(file, hist);
}
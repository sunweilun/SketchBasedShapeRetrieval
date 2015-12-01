/* 
 * File:   utils.h
 * Author: swl
 *
 * Created on December 1, 2015, 10:11 AM
 */

#ifndef UTILS_H
#define	UTILS_H
#include <string>
#include <vector>

inline void dbg(const cv::Mat1f& m)
{
    cv::namedWindow("Image");
    double min, max;
    cv::minMaxLoc(m, &min, &max);
    cv::Mat1b d = (m-min)/(max-min)*255.0;
    cv::imshow("Image", d);
    cv::waitKey();
}

inline unsigned urand(unsigned& seed)
{
    seed = seed*32452843;
    seed += 492876863;
    return seed;
}

inline void writeMat(FILE*& file, const cv::Mat1f& mat)
{
    fwrite(&mat.rows, 1, sizeof(mat.rows), file);
    fwrite(&mat.cols, 1, sizeof(mat.cols), file);
    fwrite(mat.data, mat.rows*mat.cols, sizeof(float), file);
}

inline void readMat(FILE*& file, cv::Mat1f& mat)
{
    int rows, cols;
    fread(&rows, 1, sizeof(rows), file);
    fread(&cols, 1, sizeof(cols), file);
    mat = cv::Mat1f(rows, cols);
    fread(mat.data, mat.rows*mat.cols, sizeof(float), file);
}

inline void writeMat(const char* path, const cv::Mat1f& mat)
{
    FILE* file = fopen(path, "wb");
    writeMat(file, mat);
    fclose(file);
}

inline void readMat(const char* path, cv::Mat1f& mat)
{
    FILE* file = fopen(path, "rb");
    readMat(file, mat);
    fclose(file);
}

template<typename T> void writeVec(FILE*& file, const std::vector<T>& vec)
{
    unsigned size = vec.size();
    fwrite(&size, 1, sizeof(unsigned), file);
    fwrite(vec.data(), size, sizeof(T), file);
}

template<typename T> void readVec(FILE*& file, std::vector<T>& vec)
{
    unsigned size;
    fread(&size, 1, sizeof(unsigned), file);
    vec.resize(size);
    fread(vec.data(), size, sizeof(unsigned), file);
}

template<typename T> void writeType(FILE*& file, T t)
{
    fwrite(&t, 1, sizeof(T), file);
}

template<typename T> void readType(FILE*& file, T& t)
{
    fread(&t, 1, sizeof(T), file);
}

inline void writeString(FILE*& file, const std::string& t)
{
    writeType<unsigned>(file, t.length());
    fwrite(t.data(), t.length(), sizeof(char), file);
}

inline void readString(FILE*& file, std::string& t)
{
    unsigned len;
    readType<unsigned>(file, len);
    char* str = new char[len+1];
    fread(str, len, sizeof(char), file);
    str[len] = 0;
    t = str;
    delete[] str;
}

#endif	/* UTILS_H */


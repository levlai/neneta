#pragma once

#include <string>
//#define HAVE_BOOLEAN
#define XMD_H
#define cimg_use_jpeg
#include <CImg/CImg.h>
#include <Types.h>

namespace neneta
{
namespace gpu
{
class OpenCLExecutionPlan;
}

namespace ip
{

static const cmn::GPUFLOAT PI = 3.1415926;

template<typename T, typename Label>
class Image
{
public:
    Image(const std::string& imgPath, bool gs = false);
    Image(const T* data, const int size);
    Image(const T* data, const int width, const int height, Label label);    
    Image(const Image& img);
    Image(const cimg_library::CImg<T>& img, Label label);
    template<typename U>
    Image& operator=(const Image<U, Label>& img);
    Image(Image&& img);
    Image& operator=(Image&& img);
    ~Image();

    bool resizeWithCrop(const int sizexy);
    bool resize(const int sizexy);
    void zeroPad(const int sizexy);
    void normalize(const int min, const int max);
    void show() const;
    void showAndWait() const;
    size_t width() const;
    size_t height() const;
    size_t channels() const;    
    const cimg_library::CImg<T>& getCImg() const;
    T*  data(unsigned int channel = 0);
    Image<cmn::GPUFLOAT, Label> convertToComplex(const std::uint16_t maxInt) const;
    Image<cmn::GPUFLOAT, Label> convertToRealFloat(const std::uint16_t maxInt) const;
    void printImage(unsigned int z, unsigned int channel);
    size_t getLabel() const { return static_cast<size_t>(m_label); }

private:
    cimg_library::CImg<T> m_image;
    Label m_label;
};

}
}

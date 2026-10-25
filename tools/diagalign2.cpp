// 验证逐字符对齐后的残余偏移
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif
#include "chardet.hpp"
#include "common/console_utf8.hpp"
namespace fs=std::filesystem;
namespace{
std::string w2u(const std::wstring& w){if(w.empty())return{};int n=::WideCharToMultiByte(CP_UTF8,0,w.data(),(int)w.size(),nullptr,0,nullptr,nullptr);std::string o(n,'\0');::WideCharToMultiByte(CP_UTF8,0,w.data(),(int)w.size(),o.data(),n,nullptr,nullptr);return o;}
fs::path u8p(const std::string& s){return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()),s.size()));}
bool rd(const std::string& p,cv::Mat& o){std::ifstream f(u8p(p),std::ios::binary);if(!f)return false;f.seekg(0,std::ios::end);auto n=f.tellg();f.seekg(0,std::ios::beg);if(n<=0)return false;std::vector<uchar> b((size_t)n);f.read((char*)b.data(),n);if(!f)return false;o=cv::imdecode(b,cv::IMREAD_GRAYSCALE);return !o.empty();}
}
int main(){
    markloc::enableUtf8Console();
    int wc=0; LPWSTR* wv=::CommandLineToArgvW(::GetCommandLineW(),&wc);
    std::vector<std::string> a; if(wv){for(int i=0;i<wc;++i)a.push_back(w2u(wv[i]));::LocalFree(wv);}
    if(a.size()<3)return 2;
    chardet::Model cm; if(!chardet::loadModel(a[1],cm))return 1;

    for(size_t fi=2;fi<a.size();++fi){
        cv::Mat g; if(!rd(a[fi],g)||g.size()!=cm.freq.size())continue;
        cv::Mat sig=chardet::strokeSignal(g,cv::Rect(0,0,g.cols,g.rows),cm.config,false);
        if(!cm.ref.empty()){
            cv::Mat r32,s32; cm.ref.convertTo(r32,CV_32F); sig.convertTo(s32,CV_32F);
            auto sh=cv::phaseCorrelate(s32,r32,cv::noArray());
            if(std::abs(sh.x)<60&&std::abs(sh.y)<60&&(std::abs(sh.x)>0.01||std::abs(sh.y)>0.01)){
                cv::Mat M=(cv::Mat_<double>(2,3)<<1,0,-sh.x,0,1,-sh.y);
                cv::warpAffine(sig,sig,M,sig.size(),cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_REPLICATE);
            }
        }
        auto units=chardet::toUnits(cm,0);
        fs::path fp=u8p(a[fi]);
        std::printf("%s\n",std::string((const char*)fp.filename().u8string().c_str()).c_str());
        for(size_t i=0;i<units.size()&&i<3;++i){
            cv::Rect box=units[i].bbox;
            constexpr int kSearch=12;
            cv::Rect search(box.x-kSearch,box.y-kSearch,box.width+2*kSearch,box.height+2*kSearch);
            search&=cv::Rect(0,0,sig.cols,sig.rows);
            if(search.width<box.width||search.height<box.height)continue;
            cv::Mat mS,tS;
            cm.freq(search).convertTo(mS,CV_32F,1.0/255.0);
            sig(search).convertTo(tS,CV_32F);
            cv::Rect tr(box.x-search.x,box.y-search.y,box.width,box.height);
            cv::Mat res; cv::matchTemplate(tS,mS(tr),res,cv::TM_CCOEFF_NORMED);
            cv::Point mLoc; double mVal=0; cv::minMaxLoc(res,nullptr,&mVal,nullptr,&mLoc);
            int dx=mLoc.x-kSearch, dy=mLoc.y-kSearch;
            // 对齐后再测残余
            cv::Rect moved(box.x+dx,box.y+dy,box.width,box.height);
            cv::Mat tSig2; cv::compare(sig(moved),cm.config.strokeHi,tSig2,cv::CMP_GT);
            cv::Mat mF; cm.freq(box).convertTo(mF,CV_32F,1.0/255);
            cv::Mat mBin; cv::compare(mF,0.5,mBin,cv::CMP_GT);
            int mi=cv::countNonZero(mBin), ti=cv::countNonZero(tSig2);
            std::printf("  #%zu 需偏移 dx=%+d dy=%+d (峰值%.2f) → 对齐后 模型墨%d 待检墨%d 比%.0f%%\n",
                i,dx,dy,mVal,mi,ti,100.0*ti/std::max(1,mi));
        }
        std::printf("\n");
    }
    return 0;
}
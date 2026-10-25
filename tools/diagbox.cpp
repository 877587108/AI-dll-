// 直接算：对齐后框内，模型"必然有墨"处待检是否也有墨
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
    chardet::Model cm; if(!chardet::loadModel(a[1],cm))return 1;
    std::vector<chardet::CharUnit> all;
    for(size_t ri=0;ri<cm.regions.size();++ri){auto u=chardet::toUnits(cm,(int)ri);for(auto&x:u)all.push_back(x);}
    cv::Rect only=all[a.size()>3?std::atoi(a[3].c_str()):0].bbox;

    for(size_t fi=2;fi<a.size()&&fi<3;++fi){
        cv::Mat g; if(!rd(a[fi],g))continue;
        cv::Mat sig=chardet::strokeSignal(g,cv::Rect(0,0,g.cols,g.rows),cm.config,false);
        if(!cm.ref.empty()){cv::Mat r32,s32;cm.ref.convertTo(r32,CV_32F);sig.convertTo(s32,CV_32F);auto sh=cv::phaseCorrelate(s32,r32,cv::noArray());if(std::abs(sh.x)<60&&std::abs(sh.y)<60){cv::Mat M=(cv::Mat_<double>(2,3)<<1,0,-sh.x,0,1,-sh.y);cv::warpAffine(sig,sig,M,sig.size(),cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_REPLICATE);}}
        for(auto& u:all){
            cv::Rect box=u.bbox&cv::Rect(0,0,sig.cols,sig.rows);
            if(box.width<=4||box.height<=4)continue;
            constexpr int kS=12;
            cv::Rect search(box.x-kS,box.y-kS,box.width+2*kS,box.height+2*kS);
            search&=cv::Rect(0,0,sig.cols,sig.rows);
            cv::Mat mS,tS; cm.freq(search).convertTo(mS,CV_32F,1.0/255.0); sig(search).convertTo(tS,CV_32F);
            cv::Mat res; cv::matchTemplate(tS,mS(cv::Rect(box.x-search.x,box.y-search.y,box.width,box.height)),res,cv::TM_CCOEFF_NORMED);
            cv::Point mLoc; double mVal=0; cv::minMaxLoc(res,nullptr,&mVal,nullptr,&mLoc);
            int dx=mLoc.x-kS,dy=mLoc.y-kS;
            u.bbox=cv::Rect(box.x+dx,box.y+dy,box.width,box.height);
        }
        fs::path fp=u8p(a[fi]);
        std::printf("%s\n",std::string((const char*)fp.filename().u8string().c_str()).c_str());
        for(size_t i=0;i<all.size()&&i<4;++i){
            cv::Rect box=all[i].bbox;
            cv::Mat mf,tf; cm.freq(box).convertTo(mf,CV_32F,1.0/255.0); sig(box).convertTo(tf,CV_32F);
            double mMax=0; cv::minMaxLoc(mf,nullptr,&mMax);
            double inkThr=std::max(0.50,mMax*0.75);
            cv::Mat si,sb,tb;
            cv::compare(mf,inkThr,si,cv::CMP_GT);
            cv::compare(mf,cm.config.blankFreq,sb,cv::CMP_LT);
            cv::compare(tf,cm.config.strokeHi,tb,cv::CMP_GT);
            cv::Mat nt; cv::bitwise_not(tb,nt);
            cv::Mat miss; cv::bitwise_and(si,nt,miss);
            int nInk=cv::countNonZero(si), missN=cv::countNonZero(miss);
            // 待检信号在 sureInk 处的中位数
            std::vector<float> vals;
            for(int y=0;y<box.height;++y){const float* ps=tf.ptr<float>(y);const uchar* pm=si.ptr<uchar>(y);for(int x=0;x<box.width;++x)if(pm[x])vals.push_back(ps[x]);}
            double med=0; if(!vals.empty()){std::nth_element(vals.begin(),vals.begin()+vals.size()/2,vals.end());med=vals[vals.size()/2];}
            std::printf("  #%zu mMax=%.2f inkThr=%.2f nInk=%d 待检墨=%d 缺=%d(%.0f%%) 待检中位强度=%.2f\n",
                i,mMax,inkThr,nInk,cv::countNonZero(tb),missN,100.0*missN/std::max(1,nInk),med);
        }
        std::printf("\n");
    }
    return 0;
}
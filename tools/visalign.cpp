// 可视化验证：对齐前后的字符框位置
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
    if(a.size()<4){std::printf("用法: markloc_visalign <模型> <图片> <输出png>\n");return 2;}
    chardet::Model cm; if(!chardet::loadModel(a[1],cm))return 1;
    cv::Mat g; if(!rd(a[2],g))return 1;
    cv::Mat sig=chardet::strokeSignal(g,cv::Rect(0,0,g.cols,g.rows),cm.config,false);
    if(!cm.ref.empty()){
        cv::Mat r32,s32; cm.ref.convertTo(r32,CV_32F); sig.convertTo(s32,CV_32F);
        auto sh=cv::phaseCorrelate(s32,r32,cv::noArray());
        if(std::abs(sh.x)<60&&std::abs(sh.y)<60&&(std::abs(sh.x)>0.01||std::abs(sh.y)>0.01)){
            cv::Mat M=(cv::Mat_<double>(2,3)<<1,0,-sh.x,0,1,-sh.y);
            cv::warpAffine(sig,sig,M,sig.size(),cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_REPLICATE);
        }
    }
    cv::Mat vis; cv::cvtColor(g, vis, cv::COLOR_GRAY2BGR);

    std::vector<chardet::CharUnit> all;
    for(size_t ri=0;ri<cm.regions.size();++ri){auto u=chardet::toUnits(cm,(int)ri);for(auto&x:u)all.push_back(x);}

    int nmoved=0;
    for(auto& u : all){
        cv::Rect box=u.bbox&cv::Rect(0,0,sig.cols,sig.rows);
        if(box.width<=4||box.height<=4)continue;
        constexpr int kS=12;
        cv::Rect search(box.x-kS,box.y-kS,box.width+2*kS,box.height+2*kS);
        search&=cv::Rect(0,0,sig.cols,sig.rows);
        if(search.width<box.width||search.height<box.height)continue;
        cv::Mat mS,tS;
        cm.freq(search).convertTo(mS,CV_32F,1.0/255.0);
        sig(search).convertTo(tS,CV_32F);
        cv::Mat res;
        cv::matchTemplate(tS,mS(cv::Rect(box.x-search.x,box.y-search.y,box.width,box.height)),res,cv::TM_CCOEFF_NORMED);
        cv::Point mLoc; double mVal=0; cv::minMaxLoc(res,nullptr,&mVal,nullptr,&mLoc);
        if(mVal<0.30)continue;
        int dx=mLoc.x-kS, dy=mLoc.y-kS;
        cv::Rect moved(box.x+dx,box.y+dy,box.width,box.height);
        if((moved&cv::Rect(0,0,sig.cols,sig.rows))!=moved)continue;

        // 原框：红；对齐后：绿
        cv::rectangle(vis,box,cv::Scalar(0,0,255),1);
        cv::rectangle(vis,moved,cv::Scalar(0,255,0),2);
        if(dx!=0||dy!=0)++nmoved;
    }
    cv::imwrite(a[3],vis);
    std::printf("字符 %zu 个，需移动 %d 个\n输出: %s\n",all.size(),nmoved,a[3].c_str());
    return 0;
}
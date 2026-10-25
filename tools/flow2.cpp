// 完整流程图：每步 原图|效果 对比 + 参数标注
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
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
cv::Mat gray3(const cv::Mat& m){cv::Mat o;if(m.channels()==1)cv::cvtColor(m,o,cv::COLOR_GRAY2BGR);else o=m.clone();return o;}
// 一张小图：左上角标题，下方放图
cv::Mat thumb(const cv::Mat& img,const std::string& t,int W,int H){
    cv::Mat b=gray3(img);
    double s=std::min((W-6.0)/b.cols,(H-24.0)/b.rows);
    cv::Mat r; cv::resize(b,r,cv::Size(),s,s);
    cv::Mat o=cv::Mat::zeros(H,W,CV_8UC3);
    r.copyTo(o(cv::Rect(3,22,std::min(r.cols,W-6),std::min(r.rows,H-24))));
    cv::putText(o,t,cv::Point(4,16),cv::FONT_HERSHEY_SIMPLEX,0.42,cv::Scalar(0,255,255),1,cv::LINE_AA);
    return o;
}
// 一行：步骤号+参数说明 | 原图 | 效果图
cv::Mat row(const std::string& step,const std::string& params,const cv::Mat& before,const cv::Mat& after,int W=1240,int H=230){
    cv::Mat head=cv::Mat::zeros(H,330,CV_8UC3);
    cv::putText(head,step,cv::Point(6,24),cv::FONT_HERSHEY_SIMPLEX,0.55,cv::Scalar(0,220,255),1,cv::LINE_AA);
    int y=50;
    std::string cur;
    for(size_t i=0;i<params.size();++i){
        char c=params[i];
        if(c=='\n'||i==params.size()-1){ if(c!='\n')cur+=c;
            cv::putText(head,cur,cv::Point(8,y),cv::FONT_HERSHEY_SIMPLEX,0.36,cv::Scalar(210,210,210),1,cv::LINE_AA); cur.clear(); y+=19; if(y>H-6)break; }
        else cur+=c;
    }
    cv::Mat b=thumb(before,"BEFORE",(W-330)/2,H);
    cv::Mat a=thumb(after,"AFTER",(W-330)/2,H);
    cv::Mat line; cv::hconcat(std::vector<cv::Mat>{head,b,a},line);
    return line;
}
}
int main(){
    markloc::enableUtf8Console();
    int wc=0; LPWSTR* wv=::CommandLineToArgvW(::GetCommandLineW(),&wc);
    std::vector<std::string> a; if(wv){for(int i=0;i<wc;++i)a.push_back(w2u(wv[i]));::LocalFree(wv);}
    if(a.size()<4){std::printf("usage: flow2 <model> <img> <out>\n");return 2;}
    chardet::Model cm; if(!chardet::loadModel(a[1],cm))return 1;
    cv::Mat gray; if(!rd(a[2],gray))return 1;
    chardet::Config cfg=cm.config;

    std::vector<cv::Mat> rows;
    char buf[512];

    // 1 原图
    rows.push_back(row("STEP 1  RAW INPUT",
        "file: xxx_00.tif (view v00)\nsize: 1688 x 1460\ntype: 8-bit grayscale",
        gray,gray));

    // 2 裁到塑封体
    cv::Rect body(300,225,1065,915);
    rows.push_back(row("STEP 2  CROP CHIP BODY",
        "roi: x300 y225 1065x915\nexclude: leads + substrate\n(reason: unrelated to marking)",
        gray,gray(body)));

    // 3 笔画信号
    cv::Mat sig=chardet::strokeSignal(gray,cv::Rect(0,0,gray.cols,gray.rows),cfg,false);
    cv::Mat sigVis; cv::normalize(sig,sigVis,0,255,cv::NORM_MINMAX,CV_8U);
    snprintf(buf,sizeof(buf),"gauss sigma = %.0f\nnormalize: q99.5*0.55+q90*0.45\n=> stroke/bg contrast boosted",cfg.sigma);
    rows.push_back(row("STEP 3  STROKE SIGNAL",buf,gray,sigVis));

    // 4 二值化
    cv::Mat bin; cv::compare(sig,cfg.strokeHi,bin,cv::CMP_GT);
    snprintf(buf,sizeof(buf),"threshold S > %.2f\n(strokeHi)\nink pixels = %d",cfg.strokeHi,cv::countNonZero(bin));
    rows.push_back(row("STEP 4  BINARY INK",buf,sigVis,bin));

    // 5 模型频率图
    snprintf(buf,sizeof(buf),"model = avg of 84 OK samples\nfreq>%.2f => 'must have ink'\nfreq<%.2f => 'must be blank'",cfg.missFreq,cfg.blankFreq);
    rows.push_back(row("STEP 5  MODEL FREQ MAP",buf,bin,cm.freq));

    // 6 字符框
    std::vector<chardet::CharUnit> all;
    for(size_t ri=0;ri<cm.regions.size();++ri){auto u=chardet::toUnits(cm,(int)ri);for(auto&x:u)all.push_back(x);}
    cv::Mat s6=gray3(gray);
    for(auto&u:all)cv::rectangle(s6,u.bbox,cv::Scalar(0,255,0),2);
    for(size_t ri=0;ri<cm.regions.size();++ri)cv::rectangle(s6,cm.regions[ri].rect,cv::Scalar(255,80,0),2);
    snprintf(buf,sizeof(buf),"regions: %zu  chars: %zu\nx-split: colThrRatio=%.2f\ngapMergeRatio=%.2f",cm.regions.size(),all.size(),cfg.colThrRatio,cfg.gapMergeRatio);
    rows.push_back(row("STEP 6  CHAR BOXES",buf,cm.freq,s6));

    // 7 整图配准（字符区）
    cv::Mat sigReg=sig.clone();
    cv::Mat r32,s32; cm.ref.convertTo(r32,CV_32F); sig.convertTo(s32,CV_32F);
    cv::Rect ia;
    for(auto&u:all){ia=ia.area()>0?(ia|u.bbox):u.bbox;}
    ia.x=std::max(0,ia.x-20); ia.y=std::max(0,ia.y-20);
    ia.width=std::min(sig.cols-ia.x,ia.width+40); ia.height=std::min(sig.rows-ia.y,ia.height+40);
    cv::Mat rC,sC; r32(ia).copyTo(rC); s32(ia).copyTo(sC);
    cv::Point2d sh=cv::phaseCorrelate(sC,rC,cv::noArray());
    cv::Mat sigW=sig.clone();
    if(std::abs(sh.x)<60&&std::abs(sh.y)<60){
        cv::Mat M=(cv::Mat_<double>(2,3)<<1,0,-sh.x,0,1,-sh.y);
        cv::warpAffine(sig,sigW,M,sig.size(),cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_REPLICATE);
    }
    cv::Mat shVis; cv::normalize(sigW,shVis,0,255,cv::NORM_MINMAX,CV_8U);
    snprintf(buf,sizeof(buf),"phaseCorrelate on CHAR AREA only\nshift = (%+.1f, %+.1f)\nNOT full image (leads dominate)",-sh.x,-sh.y);
    rows.push_back(row("STEP 7  REGISTER",buf,sigVis,shVis));

    // 8 逐字符对齐
    cv::Mat s8=gray3(gray);
    int moved=0;
    for(auto&u:all){
        cv::Rect box=u.bbox&cv::Rect(0,0,sigW.cols,sigW.rows);
        if(box.width<=4||box.height<=4)continue;
        constexpr int KS=12;
        cv::Rect sr(box.x-KS,box.y-KS,box.width+2*KS,box.height+2*KS);
        sr&=cv::Rect(0,0,sigW.cols,sigW.rows);
        if(sr.width<box.width||sr.height<box.height)continue;
        cv::Mat mS,tS; cm.freq(sr).convertTo(mS,CV_32F,1.0/255.0); sigW(sr).convertTo(tS,CV_32F);
        cv::Mat res; cv::matchTemplate(tS,mS(cv::Rect(box.x-sr.x,box.y-sr.y,box.width,box.height)),res,cv::TM_CCOEFF_NORMED);
        cv::Point mL; double mv=0; cv::minMaxLoc(res,nullptr,&mv,nullptr,&mL);
        if(mv<0.30)continue;
        cv::Rect nb(box.x+mL.x-KS,box.y+mL.y-KS,box.width,box.height);
        if((nb&cv::Rect(0,0,sigW.cols,sigW.rows))!=nb)continue;
        cv::rectangle(s8,u.bbox,cv::Scalar(0,0,255),1);   // 原框 红
        cv::rectangle(s8,nb,cv::Scalar(0,255,0),2);       // 对齐后 绿
        u.bbox=nb; ++moved;
    }
    snprintf(buf,sizeof(buf),"per-char matchTemplate\nsearch +-%d px, subpixel\nmoved %d / %zu boxes",12,moved,all.size());
    rows.push_back(row("STEP 8  PER-CHAR ALIGN",buf,s6,s8));

    // 9 检测
    cv::Mat vis=gray3(gray);
    int nmiss=0,nextra=0;
    for(auto&u:all){
        auto nc=chardet::diffNcc(sigW,cm.freq,u,cfg);
        if(!nc.valid)continue;
        if(nc.missRatio>0.10){cv::rectangle(vis,u.bbox,cv::Scalar(0,0,255),2);++nmiss;}
        else if(nc.extraRatio>0.10){cv::rectangle(vis,u.bbox,cv::Scalar(255,120,0),2);++nextra;}
    }
    auto bd=chardet::diffBlocks(sigW,cm.freq,all,cfg,8);
    for(auto&b:bd.extraBlocks)cv::rectangle(vis,cv::Rect(b.x0,b.y0,b.x1-b.x0,b.y1-b.y0),cv::Scalar(0,220,220),2);
    snprintf(buf,sizeof(buf),"char miss=%d extra=%d\nblock extra=%d\nNG if any > 0",nmiss,nextra,bd.extraCount);
    rows.push_back(row("STEP 9  DETECT",buf,s8,vis));

    cv::Mat out;
    cv::vconcat(rows,out);
    cv::imwrite(a[3],out);
    std::printf("saved %s (%dx%d)\n",a[3].c_str(),out.cols,out.rows);
    return 0;
}